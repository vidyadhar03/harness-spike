from harness.memory import caption_eval as ce
from harness.memory.ingest import PROMPTS, load_prompt
from harness.memory.ports import Blob, ImageHit, MemoryImages, Text
from harness.memory.schemas import CaptionsOut, OutCaption


def hit(i: int) -> ImageHit:
    return ImageHit(title=f"File {i}.jpg", page_url=f"https://commons/{i}", image_url=f"https://upload/{i}.jpg",
                    preview_url=f"https://preview/{i}.jpg", description=f"desc {i}", license="CC BY-SA 4.0",
                    attribution="A. Photographer")


def fixture(n: int) -> ce.Fixture:
    return ce.Fixture(location="Devgram", context_md="# Devgram\nA village on a gorge.",
                      hits=[hit(i) for i in range(n)], origins=[f"term {i}" for i in range(n)],
                      included=["File 9.jpg"], captured_at="2026-09-14T00:00:00+00:00", references_version="v")


def images_for(fx: ce.Fixture) -> MemoryImages:
    return MemoryImages({}, {}, {h.preview_url: f"bytes {h.title}".encode() for h in fx.hits})


class FakeLLM:
    model_id = "fake"

    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def generate(self, *, system, parts, schema, fast=False, thinking_level=None):
        indices = [int(p.text.split("]")[0][1:]) for p in parts if isinstance(p, Text) and p.text.startswith("[")]
        self.calls.append((system, schema, indices, sum(isinstance(p, Blob) for p in parts)))
        return self.respond(schema, indices)


def keep_even(schema, indices):
    if schema is CaptionsOut:
        return CaptionsOut(captions=[OutCaption(index=i, caption=f"Caption {i}.", facet="terrain")
                                     for i in indices if i % 2 == 0])
    return ce.VerdictsOut(verdicts=[ce.Verdict(index=i, keep=i % 2 == 0, reason=f"reason {i}") for i in indices])


def test_fixture_round_trips(tmp_path):
    fx = fixture(3)
    path = tmp_path / "evals" / "fx.json"
    ce.save_fixture(path, fx)
    assert ce.load_fixture(path) == fx
    assert b"bytes" not in path.read_bytes()             # metadata only, never image binaries


def test_production_run_batches_exactly_like_references():
    fx = fixture(10)
    llm = FakeLLM(keep_even)
    kept, warnings = ce.production_run(llm, images_for(fx), fx)

    assert [c[2] for c in llm.calls] == [list(range(8)), [8, 9]]     # CAPTION_BATCH, in retrieval order
    assert all(c[0] == load_prompt("caption_references") and c[1] is CaptionsOut for c in llm.calls)
    assert kept == {i: ("terrain", f"Caption {i}.") for i in range(0, 10, 2)}
    assert warnings == []


def test_unfetchable_candidate_is_skipped_not_fatal():
    fx = fixture(3)
    images = images_for(fx)
    images.blobs.pop("https://preview/1.jpg")
    llm = FakeLLM(keep_even)
    kept, warnings = ce.production_run(llm, images, fx)
    assert llm.calls[0][2:] == ([0, 2], 2) and set(kept) == {0, 2}
    assert any("could not fetch image 1" in w for w in warnings)


def test_diagnostic_is_separate_from_the_production_prompt():
    fx = fixture(3)
    llm = FakeLLM(keep_even)
    verdicts, _ = ce.diagnostic_run(llm, images_for(fx), fx)
    system, schema = llm.calls[0][:2]
    assert schema is ce.VerdictsOut and system == load_prompt("caption_references") + ce.DIAGNOSTIC
    assert set(verdicts) == {0, 1, 2}

    shipped = (PROMPTS / "caption_references.md").read_text()
    assert "Diagnostic" not in shipped and "rule_quoted" not in shipped and "verdict" not in shipped.lower()


def test_aggregation_separates_stable_flipped_and_disagreeing_images():
    fx = fixture(5)
    runs = [
        {0: ("place", "a"), 1: ("terrain", "b"), 3: ("terrain", "d")},
        {0: ("place", "a"), 1: ("material", "b2")},
        {0: ("place", "a"), 1: ("terrain", "b"), 3: ("terrain", "d")},
    ]
    verdicts = {0: ce.Verdict(index=0, keep=True), 2: ce.Verdict(index=2, keep=True),
                3: ce.Verdict(index=3, keep=False)}
    report = ce.aggregate(fx, runs, verdicts)

    assert [r.index for r in report.by_status("kept")] == [0, 1]
    assert [r.index for r in report.by_status("dropped")] == [2, 4]
    assert [r.index for r in report.by_status("flipped")] == [3]
    assert [r.index for r in report.facet_flips] == [1]
    assert [r.index for r in report.disagreements] == [2, 3]   # 2: diag keeps, never kept; 3: kept 2/3, diag drops
    assert report.images[1].captions == ["b", "b2"]
    assert report.headline() == ("1 of 5 images flipped across 3 runs (2 always kept, 2 always dropped; "
                                 "facet changed on 1 always-kept); diagnostic disagreed with the production "
                                 "majority on 2 of 3")
    text = ce.render(report)
    assert text.index("[3] FLIPPED 2/3") < text.index("[0] KEPT 3/3")
    assert "[DISAGREES with production majority]" in text


def test_run_eval_repeats_production_and_diagnoses_once():
    fx = fixture(3)
    llm = FakeLLM(keep_even)
    report, _ = ce.run_eval(llm, images_for(fx), fx, n=4)
    assert [c[1] for c in llm.calls] == [CaptionsOut] * 4 + [ce.VerdictsOut]
    assert report.runs == 4 and report.headline().startswith("0 of 3 images flipped across 4 runs")

    report, _ = ce.run_eval(FakeLLM(keep_even), images_for(fx), fx, n=2, reasons=False)
    assert not report.diagnosed and "diagnostic" not in report.headline()


def test_cached_images_fetch_once(tmp_path):
    fx = fixture(1)
    inner = images_for(fx)
    fetched = []
    original = inner.fetch
    inner.fetch = lambda url: fetched.append(url) or original(url)
    cached = ce.CachedImages(inner, tmp_path / "cache")
    assert cached.fetch("https://preview/0.jpg") == cached.fetch("https://preview/0.jpg") == b"bytes File 0.jpg"
    assert fetched == ["https://preview/0.jpg"]
