import io
import json
import time

import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from harness.memory.config import Settings
from harness.memory.ingest import Ctx, LockRenewer, digest_version, ingest_source, register_file
from harness.memory.models import PROJECT_SCOPE, Applicability, Location, Project, Scene
from harness.memory.ports import Blob, MemoryBlobs, MemoryStore, OutputTruncated, Text
from harness.memory.schemas import (
    ClassifyOut, ImageOut, NotesOut, OutLocation, OutNote, OutReference, OutScene, RosterOut, gemini_schema,
)

PID = "prj_test"


class FakeLLM:
    model_id = "fake"

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def generate(self, *, system, parts, schema, fast=False):
        self.calls.append(schema.__name__)
        return self.handler(schema, system, parts)


def make_ctx(handler, **settings):
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    s = Settings(gcp_project="test", bucket="bucket", **settings)
    return Ctx(store, MemoryBlobs(), FakeLLM(handler), s)


def text_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for body in pages:
        y = 800
        for line in body.split("\n"):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def visual_pdf(n: int) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(n):
        c.rect(50 + i * 5, 300, 400, 300, fill=1)
        c.showPage()
    c.save()
    return buf.getvalue()


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "brown").save(buf, "PNG")
    return buf.getvalue()


def all_text(parts):
    return "\n".join(p.text for p in parts if isinstance(p, Text))


FILLER = "\n".join(["The wind moves through the dry fields and the village sleeps under a low moon."] * 6)
SCRIPT = [
    f"1 EXT. DEVGRAM WELL - NIGHT\nPrasad walks to the well. A stone well.\n{FILLER}",
    f"Continued at the well.\n{FILLER}",
    f"2 EXT. TEMPLE COURTYARD - DAY\nSlate floor, uneven.\n{FILLER}",
    f"Continued in the courtyard.\n{FILLER}",
]


def script_handler(schema, system, parts):
    if schema is ClassifyOut:
        return ClassifyOut(doc_type="script")
    if schema is RosterOut:
        return RosterOut(
            locations=[OutLocation(name="Devgram well", aliases=["the well"], inside="Devgram"),
                       OutLocation(name="Temple courtyard")],
            scenes=[OutScene(number="1", heading="EXT. DEVGRAM WELL - NIGHT", locations=["Devgram well"]),
                    OutScene(number="2", heading="EXT. TEMPLE COURTYARD - DAY", locations=["Temple courtyard"])],
        )
    if schema is NotesOut:
        text = all_text(parts)
        if "- 1 | EXT. DEVGRAM WELL - NIGHT" in text:
            return NotesOut(notes=[
                OutNote(kind="description", body="Stone well, waist high, rope worn smooth.",
                        owner="the well", page=1, quote="A stone well."),
                OutNote(kind="constraint", body="No electricity poles anywhere in Devgram.",
                        project_wide=True, page=2),
                OutNote(kind="description", body="Old banyan tree with a deep fissure.",
                        owner="Old banyan tree", only_during_scene="99", page=2),
                OutNote(kind="description", body="Floating fact with no owner.", page=1),
            ])
        return NotesOut(notes=[
            OutNote(kind="description", body="Courtyard paved with uneven slate.",
                    owner="Temple courtyard", only_during_scene="Scene 2", page=3,
                    mentions=["Devgram well"]),
            OutNote(kind="description", body="Stone well, waist high, rope worn smooth.",
                    owner="Devgram well", page=4),
        ])
    raise AssertionError(schema)


def run_script(ctx):
    src, created = register_file(ctx, PID, text_pdf(SCRIPT), "scripts/Dehleez_Ep1.pdf")
    assert created
    return src, ingest_source(ctx, PID, src.id)


def by_name(ctx):
    return {e.name: e for e in ctx.store.list_entities(PID)}


def test_register_is_idempotent_and_never_rejects():
    ctx = make_ctx(script_handler)
    data = text_pdf(SCRIPT)
    a, created_a = register_file(ctx, PID, data, "a.pdf")
    b, created_b = register_file(ctx, PID, data, "copy-of-a.pdf")
    assert (created_a, created_b) == (True, False) and a.id == b.id
    audio, _ = register_file(ctx, PID, b"ID3\x04fake mp3", "recce/ambience.mp3")
    assert audio.status == "unsupported" and audio.kind == "audio"
    assert ingest_source(ctx, PID, audio.id).status == "unsupported"
    assert audio.storage_path in ctx.blobs.objects


def test_script_pdf_builds_roster_notes_and_provenance():
    ctx = make_ctx(script_handler, script_chunk_chars=500)
    src, report = run_script(ctx)
    assert report.status == "digested" and report.doc_type == "script"

    ents = by_name(ctx)
    assert {n for n, e in ents.items() if isinstance(e, Location)} == {
        "Devgram well", "Temple courtyard", "Old banyan tree", "Devgram"}
    well, courtyard = ents["Devgram well"], ents["Temple courtyard"]
    scene1 = next(e for e in ents.values() if isinstance(e, Scene) and e.number == "1")
    scene2 = next(e for e in ents.values() if isinstance(e, Scene) and e.number == "2")
    assert scene1.location_ids == [well.id]
    # the roster proposed containment; it stays proposed until a human confirms it
    assert well.containment.parent_id == ents["Devgram"].id and well.containment.status == "proposed"

    notes = {n.body: n for n in ctx.store.list_notes(PID)}
    assert len(notes) == 4
    well_note = notes["Stone well, waist high, rope worn smooth."]
    assert well_note.owner_id == well.id                         # alias resolved
    assert well_note.applicability == Applicability()
    assert [p.page for p in well_note.provenance] == [1, 4]      # one note, two occurrences
    assert well_note.provenance[0].quote == "A stone well."
    assert all(p.extracted_body == well_note.body for p in well_note.provenance)
    assert notes["No electricity poles anywhere in Devgram."].owner_id == PROJECT_SCOPE
    assert notes["Old banyan tree with a deep fissure."].owner_id == ents["Old banyan tree"].id
    courtyard_note = notes["Courtyard paved with uneven slate."]
    assert courtyard_note.owner_id == courtyard.id
    assert courtyard_note.applicability.scene_id == scene2.id    # conditional, not general
    assert courtyard_note.mentions == [well.id]                  # mention never owns
    assert all(n.origin.digest_version == digest_version(ctx.llm) for n in notes.values())
    assert any("unknown scene '99'" in w for w in report.warnings)
    assert any("no owner" in w for w in report.warnings)
    assert report.warn_counts == {"unowned_notes": 1}

    stored = ctx.store.get_source(PID, src.id)
    assert stored.derived.page_count == 4
    assert b"<<<PAGE 3>>>" in ctx.blobs.get(stored.derived.text_path)


def test_same_version_skips_and_force_redigest_keeps_reviewed_notes():
    ctx = make_ctx(script_handler, script_chunk_chars=500)
    src, _ = run_script(ctx)
    calls = len(ctx.llm.calls)
    assert ingest_source(ctx, PID, src.id).status == "skipped"
    assert len(ctx.llm.calls) == calls

    confirmed = next(n for n in ctx.store.list_notes(PID) if n.body.startswith("Stone well"))
    ctx.store.put_notes(PID, [confirmed.model_copy(update={
        "status": "confirmed", "reviewed_revision": confirmed.revision, "reviewed_by": "vd"})])

    report = ingest_source(ctx, PID, src.id, force=True)
    assert report.status == "digested"
    # unchanged candidates keep their ids; the confirmed one is reused, not rewritten
    assert (report.entities_created, report.notes_replaced, report.notes_written,
            report.notes_reused) == (0, 0, 3, 1)
    notes = ctx.store.list_notes(PID)
    assert len(notes) == 4
    assert [n.status for n in notes if n.body.startswith("Stone well")] == ["confirmed"]
    assert len(ctx.store.list_entities(PID)) == 6


def test_visual_pdf_renders_pages_and_offsets_page_numbers():
    def handler(schema, system, parts):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="lookbook")
        return NotesOut(references=[
            OutReference(caption="Mud-plastered wall at dusk.", location="Devgram well", page=2),
            OutReference(caption="Out of range page.", page=9),
        ], notes=[OutNote(kind="tone", body="Muted ochre palette.", project_wide=True, page=1)])

    ctx = make_ctx(handler, visual_chunk_pages=6)
    src, _ = register_file(ctx, PID, visual_pdf(8), "lookbook.pdf")
    report = ingest_source(ctx, PID, src.id)
    assert report.status == "digested"
    stored = ctx.store.get_source(PID, src.id)
    assert stored.derived.text_path is None
    assert sum(k.startswith(stored.derived.pages_prefix) for k in ctx.blobs.objects) == 8
    refs = [n for n in ctx.store.list_notes(PID) if n.kind == "reference_image"]
    assert sorted(n.provenance[0].page for n in refs) == [2, 8]
    tone = [n for n in ctx.store.list_notes(PID) if n.kind == "tone"]
    assert len(tone) == 1 and [p.page for p in tone[0].provenance] == [1, 7]
    assert sum("without a valid page" in w for w in report.warnings) == 2


def test_image_resolves_merged_and_skips_rejected_locations():
    target = Location(name="Devgram well", status="confirmed", author="user")
    old = Location(name="Old well", status="merged", merged_into=target.id, author="agent")
    bad = Location(name="Haunted palace", status="rejected", author="agent")

    def handler(schema, system, parts):
        assert isinstance(parts[-1], Blob) and parts[-1].mime_type == "image/png"
        assert "Old well" not in all_text(parts) and "Devgram well" in all_text(parts)
        return ImageOut(doc_type="recce",
                        references=[OutReference(caption="Well at dusk, rope coiled on the rim.", location="old well")],
                        notes=[OutNote(kind="description", body="Palace gate carved sandstone.", owner="Haunted palace")])

    ctx = make_ctx(handler)
    ctx.store.put_entities(PID, [target, old, bad])
    src, _ = register_file(ctx, PID, png(), "devgram_well/IMG_0012.png")
    report = ingest_source(ctx, PID, src.id)
    notes = ctx.store.list_notes(PID)
    assert report.doc_type == "recce" and len(notes) == 1
    assert notes[0].kind == "reference_image" and notes[0].owner_id == target.id
    assert notes[0].provenance[0].page is None
    assert report.entities_created == 0


def test_llm_failure_marks_source_failed_and_retry_succeeds():
    state = {"fail": True}

    def handler(schema, system, parts):
        if state["fail"] and schema is NotesOut:
            raise RuntimeError("quota")
        return script_handler(schema, system, parts)

    ctx = make_ctx(handler)
    src, report = run_script(ctx)
    assert report.status == "failed" and "quota" in report.error
    assert ctx.store.get_source(PID, src.id).status == "failed"
    assert ctx.store.list_notes(PID) == []
    state["fail"] = False
    assert ingest_source(ctx, PID, src.id).status == "digested"
    assert len([e for e in ctx.store.list_entities(PID) if isinstance(e, Location)]) == 4


def test_markdown_notes_have_no_pages():
    def handler(schema, system, parts):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="notes")
        return NotesOut(notes=[OutNote(kind="constraint", body="Shoot the well only after sunset.",
                                       owner="Devgram well", page=3)])

    ctx = make_ctx(handler)
    src, _ = register_file(ctx, PID, b"# Director notes\n\nWell scenes after sunset.", "notes/director.md")
    assert src.mime_type == "text/markdown"
    ingest_source(ctx, PID, src.id)
    (note,) = ctx.store.list_notes(PID)
    assert note.provenance[0].page is None


@pytest.mark.parametrize("schema", [ClassifyOut, RosterOut, NotesOut, ImageOut])
def test_gemini_schema_is_self_contained(schema):
    dumped = json.dumps(gemini_schema(schema))
    assert "$ref" not in dumped and "$defs" not in dumped and '"default"' not in dumped


# --- patch 1: truncation is never accepted -------------------------------------

def test_gemini_adapter_rejects_non_stop_responses():
    from types import SimpleNamespace
    from google.genai import types
    from harness.memory.gcp import GeminiLLM

    def llm_returning(reason, text='{"doc_type": "script"}'):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm._s, llm.model_id = Settings(gcp_project="p", bucket="b"), "x"
        llm.accumulated_usage = {}

        def gen(model, contents, config):
            assert config.max_output_tokens == 65_536
            return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=reason)])
        llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
        return llm

    call = dict(system="s", parts=[Text("x")], schema=ClassifyOut)
    with pytest.raises(OutputTruncated):
        llm_returning(types.FinishReason.MAX_TOKENS, '{"doc_type": "scr').generate(**call)
    with pytest.raises(RuntimeError, match="SAFETY"):
        llm_returning(types.FinishReason.SAFETY).generate(**call)
    assert llm_returning(types.FinishReason.STOP).generate(**call).doc_type == "script"


def test_truncated_unit_is_split_and_retried():
    def handler(schema, system, parts):
        if schema is NotesOut:
            text = all_text(parts)
            if "- 1 | " in text and "- 2 | " in text:
                raise OutputTruncated("too long")
            n = "1" if "- 1 | " in text else "2"
            owner = "Devgram well" if n == "1" else "Temple courtyard"
            return NotesOut(notes=[OutNote(kind="description", body=f"Fact for scene {n}.",
                                           owner=owner, only_during_scene=n, page=1)])
        return script_handler(schema, system, parts)

    ctx = make_ctx(handler)  # default budget packs both scenes into one unit
    src, report = run_script(ctx)
    assert report.status == "digested"
    assert sorted(n.body for n in ctx.store.list_notes(PID)) == ["Fact for scene 1.", "Fact for scene 2."]
    assert any("truncated; split into 2" in w for w in report.warnings)


def test_unsplittable_truncation_fails_the_source():
    def handler(schema, system, parts):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="notes")
        raise OutputTruncated("too long")

    ctx = make_ctx(handler)
    src, _ = register_file(ctx, PID, b"Short note about the well.", "notes/one-liner.md")
    report = ingest_source(ctx, PID, src.id)
    assert report.status == "failed" and "cannot be split further" in report.error
    assert ctx.store.list_notes(PID) == []


def test_roster_truncation_fails_with_clear_error():
    def handler(schema, system, parts):
        if schema is RosterOut:
            raise OutputTruncated("too long")
        return script_handler(schema, system, parts)

    ctx = make_ctx(handler)
    _, report = run_script(ctx)
    assert report.status == "failed" and "rolling roster" in report.error


# --- patch 3: chunks hold whole scenes ------------------------------------------

def test_text_script_units_cut_at_sluglines_mid_page():
    pages = [
        f"1 EXT. DEVGRAM WELL - NIGHT\nPrasad at the well.\n{FILLER}",
        f"The rope creaks.\n{FILLER}\n2 EXT. TEMPLE COURTYARD – DAY\nSlate floor.",
        f"The bell rings.\n{FILLER}",
    ]
    seen = []

    def handler(schema, system, parts):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="script")
        if schema is RosterOut:
            return RosterOut(scenes=[
                OutScene(number="1", heading="EXT. DEVGRAM WELL - NIGHT", start_page=1, locations=["Devgram well"]),
                OutScene(number="2", heading="EXT. TEMPLE COURTYARD - DAY", start_page=1, locations=["Temple courtyard"]),
            ])
        seen.append(parts[-1].text)
        return NotesOut()

    ctx = make_ctx(handler, script_chunk_chars=100)
    src, _ = register_file(ctx, PID, text_pdf(pages), "script.pdf")
    report = ingest_source(ctx, PID, src.id)
    assert report.status == "digested" and len(seen) == 2
    first, second = seen
    assert "<<<PAGE 2>>>" in first and "TEMPLE COURTYARD" not in first
    assert second.startswith("<<<PAGE 2>>>\n2 EXT. TEMPLE COURTYARD")
    assert "<<<PAGE 3>>>" in second and "Prasad" not in second
    assert not any("slugline not found" in w for w in report.warnings)  # wrong start_page ignored


def test_scanned_script_uses_start_pages_and_drops_out_of_scope_notes():
    contexts = []

    def handler(schema, system, parts):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="script")
        if schema is RosterOut:
            assert isinstance(parts[-1], Blob)
            return RosterOut(scenes=[
                OutScene(number="1", heading="EXT. DEVGRAM WELL - NIGHT", start_page=1, locations=["Devgram well"]),
                OutScene(number="2", heading="EXT. TEMPLE COURTYARD - DAY", start_page=4, locations=["Temple courtyard"]),
            ])
        text = all_text(parts)
        contexts.append(text)
        if "- 2 | " in text:
            return NotesOut(notes=[OutNote(kind="description", body="Courtyard slate.",
                                           owner="Temple courtyard", only_during_scene="2", page=1)])
        return NotesOut(notes=[OutNote(kind="description", body="Courtyard slate, seen early.",
                                       owner="Temple courtyard", only_during_scene="2", page=4)])

    ctx = make_ctx(handler, scanned_script_chunk_pages=3)
    src, _ = register_file(ctx, PID, visual_pdf(6), "scanned_script.pdf")
    report = ingest_source(ctx, PID, src.id)
    assert report.status == "digested"
    assert "pages 1-4 of the original" in contexts[0] and "pages 4-6 of the original" in contexts[1]
    (note,) = ctx.store.list_notes(PID)
    assert note.body == "Courtyard slate." and note.provenance[0].page == 4
    assert any("outside the excerpt's scope" in w for w in report.warnings)


def test_unit_splits_keep_pages_and_continuation():
    from harness.memory.chunks import MARKER_RULE, PdfUnit, SceneRef, TextUnit

    s1, s2 = SceneRef("1", "EXT. WELL - NIGHT"), SceneRef("2", "EXT. TEMPLE - DAY")
    text = "<<<PAGE 5>>>\n" + "a" * 3000 + "\n\nmore\n<<<PAGE 6>>>\n" + "b" * 3000
    left, right = TextUnit(text, MARKER_RULE, [s1], [0]).split()
    assert right.text.startswith("<<<PAGE 6>>>") and right.continuation == s1 and left.continuation is None

    left, right = PdfUnit(0, 6, [s1, s2], [0, 3]).split()
    assert (left.a, left.b, right.a, right.b) == (0, 4, 3, 6)
    left, right = PdfUnit(3, 6, [s2], [3]).split()
    assert (left.b, right.a, right.continuation) == (4, 4, s2)
    assert PdfUnit(2, 3, [s2], [2]).split() is None


def test_a_named_owner_beats_the_project_wide_flag():
    """A model that sets both is describing a place, not a production-wide rule."""
    def handler(schema, system, parts):
        if schema is NotesOut:
            if "- 1 | " in all_text(parts):
                return NotesOut(notes=[
                    OutNote(kind="description", body="The village is prosperous.",
                            owner="Devgram well", project_wide=True,
                            applies_to_places_within=True, page=1),
                    OutNote(kind="tone", body="Longer lenses throughout.", project_wide=True,
                            applies_to_places_within=True, page=1),
                ])
            return NotesOut()
        return script_handler(schema, system, parts)

    ctx = make_ctx(handler)
    _, report = run_script(ctx)
    notes = {n.body: n for n in ctx.store.list_notes(PID)}
    village = notes["The village is prosperous."]
    assert village.owner_id != PROJECT_SCOPE and village.applicability.include_descendants
    lenses = notes["Longer lenses throughout."]
    assert lenses.owner_id == PROJECT_SCOPE
    assert not lenses.applicability.include_descendants      # project notes apply everywhere already


# --- LockRenewer: the shared periodic-renewal primitive behind both the CLI's
# drop/ingest loops and the API's JobRunner._run_ingest -------------------------------

def test_lock_renewer_keeps_a_held_lock_alive_on_a_fixed_schedule():
    store = MemoryStore()
    pid = "prj_renewer_test"
    token = store.acquire_lock(pid, "holder", stale_after_s=0.15)
    assert token is not None

    renewer = LockRenewer(store, pid, token, interval_s=0.03).start()
    try:
        time.sleep(0.3)  # 2x the staleness window, renewed ~10 times at 0.03s
        assert store.acquire_lock(pid, "competitor", 0.15) is None  # still held
        assert not renewer.lost.is_set()
    finally:
        renewer.stop()
    store.release_lock(pid, token)


def test_lock_renewer_detects_and_reports_lost_ownership():
    store = MemoryStore()
    pid = "prj_renewer_test2"
    token = store.acquire_lock(pid, "holder", stale_after_s=0.15)
    assert token is not None

    renewer = LockRenewer(store, pid, token, interval_s=0.03).start()
    try:
        store.release_lock(pid, token)
        thief_token = store.acquire_lock(pid, "thief", 0.15)
        assert thief_token is not None

        assert renewer.lost.wait(timeout=1), "renewer never noticed the stolen lock"
        # the thief's own lock is untouched - the renewer only ever tries its own token
        assert store.acquire_lock(pid, "someone-else", 0.15) is None
        store.release_lock(pid, thief_token)
    finally:
        renewer.stop()  # idempotent even though the renewer already stopped itself


def test_lock_renewer_stop_is_prompt_and_idempotent():
    store = MemoryStore()
    pid = "prj_renewer_test3"
    token = store.acquire_lock(pid, "holder", stale_after_s=60)
    assert token is not None

    renewer = LockRenewer(store, pid, token, interval_s=60).start()  # would not tick for 60s
    t0 = time.monotonic()
    renewer.stop()
    assert time.monotonic() - t0 < 1, "stop() should not wait out the renewal interval"
    renewer.stop()  # idempotent - must not raise or hang
    store.release_lock(pid, token)
