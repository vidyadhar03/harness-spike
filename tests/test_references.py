import hashlib
import re

import pytest

from harness.memory.config import Settings
from harness.memory.models import Location, Note, NoteOrigin, Project, Provenance, Scene
from harness.memory.ports import ImageHit, MemoryBlobs, MemoryImages, MemoryStore, TermHit, Text
from harness.memory.references import RefCtx, suggest_references
from harness.memory.retrieval import get_context, render_context_md
from harness.memory.schemas import CaptionsOut, CurateOut, OutCaption, OutDirection, OutTerm, VocabularyOut

PID = "prj_ref"

TERMS = ["kath kuni architecture", "deodar forest", "slate roof", "terraced fields", "jharokha", "made up term"]
VERIFIED = {t: TermHit(title=t.title(), url=f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}")
            for t in TERMS if t != "made up term"}


def hit(n: int, term: str) -> ImageHit:
    return ImageHit(title=f"File {term} {n}.jpg", page_url=f"https://commons.wikimedia.org/wiki/File:{term}{n}",
                    image_url=f"https://upload/{term}{n}.jpg", preview_url=f"https://preview/{term}{n}.jpg",
                    description=f"{term} example {n}", license="CC BY-SA 4.0", attribution="A. Photographer",
                    mime_type="image/jpeg")


@pytest.fixture
def world():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    well = Location(name="Devgram well", aliases=["the well"], status="confirmed", author="agent")
    scene = Scene(name="EXT. DEVGRAM WELL - NIGHT", number="1", location_ids=[well.id], author="agent")
    store.put_entities(PID, [well, scene])
    store.put_notes(PID, [Note(kind="description", body="Stone well in a hill village, slate roofs behind.",
                              owner_id=well.id, author="user")])

    images = {t: [hit(i, t.split()[0]) for i in range(4)] for t in VERIFIED}
    blobs = {h.preview_url: f"bytes-{h.preview_url}".encode()
             for hits in images.values() for h in hits}
    return store, well, MemoryImages({k.lower(): v for k, v in VERIFIED.items()},
                                     {k.lower(): v for k, v in images.items()}, blobs)


def make_ctx(store, images, handler):
    class LLM:
        model_id = "fake"

        def __init__(self):
            self.calls = []

        def generate(self, *, system, parts, schema, fast=False):
            self.calls.append((schema.__name__, system, parts))
            return handler(schema, system, parts)

    return RefCtx(store, MemoryBlobs(), LLM(), images, Settings(gcp_project="t", bucket="b"))


def default_handler(schema, system, parts):
    if schema is VocabularyOut:
        return VocabularyOut(
            script_phrases=["stone well", "slate roofs"],
            description="A stone well in a hill village.",
            terms=[OutTerm(term=t, kind="technique", why="w") for t in TERMS]
            + [OutTerm(term="Kath Kuni Architecture", kind="technique")],  # duplicate, different case
        )
    if schema is CaptionsOut:
        indices = [int(p.text.split("]")[0][1:]) for p in parts if isinstance(p, Text) and p.text.startswith("[")]
        return CaptionsOut(captions=[OutCaption(index=i, caption=f"Caption {i}.") for i in indices[:5]])
    if schema is CurateOut:
        indices = [int(i) for p in parts if isinstance(p, Text) for i in re.findall(r"^\[(\d+)\]", p.text, re.M)]
        a, b = indices[:2], indices[2:4]
        return CurateOut(
            directions=[OutDirection(name="Weathered slate and dark timber", why="Older, heavier.", images=a),
                        OutDirection(name="Open terraced slopes", why="Wider, greener.", images=b),
                        OutDirection(name="Named but empty", images=[])],
        )
    raise AssertionError(schema)


def test_vocabulary_is_verified_before_any_image_search(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    report = suggest_references(ctx, PID, "the well", per_term=2, max_images=6)

    assert report.terms_proposed == 6                      # duplicate term folded
    assert report.terms_dropped == ["made up term"]
    assert "made up term" not in images.searched           # unverified terms are never searched
    assert set(images.searched) == set(VERIFIED)
    context_sent = ctx.llm.calls[0][2][0].text
    assert "Stone well in a hill village" in context_sent


def test_images_are_grouped_stored_and_written_back(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    assert report.images_found == 10 and report.images_kept == 5   # nothing captioned is lost
    assert report.directions == [("Weathered slate and dark timber", 2),
                                 ("Open terraced slopes", 2), ("Unsorted", 1)]
    assert report.images_uncaptioned == 1
    assert any("named no usable images" in w for w in report.warnings)
    assert any("kept under 'Unsorted'" in w for w in report.warnings)
    assert any("judged irrelevant" in w for w in report.warnings)

    notes = store.list_notes(PID)
    refs = [n for n in notes if n.kind == "reference_image"]
    vocab = [n for n in notes if n.kind == "vocabulary"]
    assert len(refs) == 5 and len(vocab) == 1
    assert {n.group for n in refs} == {"Place"}           # the facet is the heading
    assert all(n.status == "proposed" and n.owner_id == well.id for n in refs)
    assert all(n.origin.producer == "references" and n.origin.scope == well.id for n in refs + vocab)
    assert sorted(n.body for n in refs) == ["Caption 4.",   # the direction rides with the caption; Unsorted has none
                                            "Open terraced slopes — Caption 2.", "Open terraced slopes — Caption 3.",
                                            "Weathered slate and dark timber — Caption 0.",
                                            "Weathered slate and dark timber — Caption 1."]
    assert vocab[0].provenance[0].url.startswith("https://en.wikipedia.org/wiki/")
    assert vocab[0].provenance[0].source_id is None
    assert "made up term" not in vocab[0].body and "stone well" in vocab[0].body

    stored = {s.id: s for s in store.list_sources(PID)}
    assert len(stored) == 5
    src = stored[refs[0].provenance[0].source_id]
    assert src.kind == "image" and src.doc_type == "reference" and src.license == "CC BY-SA 4.0"
    assert src.attribution == "A. Photographer" and src.origin_url.startswith("https://commons.wikimedia.org/")
    assert src.id == hashlib.sha256(ctx.blobs.get(src.storage_path)).hexdigest()


def test_round_robin_spreads_across_terms(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6, dry_run=True)
    caption_call = next(parts for name, _, parts in ctx.llm.calls if name == "CaptionsOut")
    shown = [p.text for p in caption_call if isinstance(p, Text) and p.text.startswith("[")]
    terms = [t.split()[2] for t in shown]                 # term of each candidate, in order
    assert len(shown) == 6 and len(set(terms[:5])) == 5   # one from each term before any second
    assert [n.kind for n in store.list_notes(PID)] == ["description"]   # dry run wrote nothing
    assert store.list_sources(PID) == []


def test_rerun_replaces_proposals_but_keeps_reviewed_images(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    refs = [n for n in store.list_notes(PID) if n.kind == "reference_image"]
    keep, drop = refs[0], refs[1]
    store.put_notes(PID, [
        keep.model_copy(update={"status": "confirmed", "reviewed_revision": keep.revision,
                                "reviewed_by": "vd"}),
        drop.model_copy(update={"status": "rejected", "reviewed_revision": drop.revision,
                                "reviewed_by": "vd", "review_reason": "not_useful"})])

    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6)
    after = [n for n in store.list_notes(PID) if n.kind == "reference_image"]
    assert report.notes_replaced == 4                      # three proposed images plus the vocabulary note
    assert {n.status for n in after} == {"confirmed", "rejected", "proposed"}
    assert len(after) == 5                                 # the rejected image is not proposed again
    assert sum(n.status == "proposed" for n in after) == 3


def test_context_md_shows_directions_with_licence(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)
    md = render_context_md(get_context(store, PID, well.id))
    assert "## Search vocabulary" in md and "kath kuni architecture (technique)" in md
    assert "### Place" in md
    assert "- [proposed] Weathered slate and dark timber — Caption 0." in md
    assert "\n  Verified terms: kath kuni architecture (technique);" in md
    assert "_(A. Photographer, CC BY-SA 4.0 <https://commons.wikimedia.org/" in md


def test_scene_scope_is_rejected(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    with pytest.raises(ValueError, match="per location"):
        suggest_references(ctx, PID, "Scene 1")


def test_unfetchable_image_is_skipped_not_fatal(world):
    store, well, images = world
    images.blobs.pop("https://preview/kath0.jpg")
    ctx = make_ctx(store, images, default_handler)
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6)
    assert any("could not fetch image" in w for w in report.warnings)
    assert report.images_kept == 5 and report.notes_written == 6   # one fetch failed, rest survive


# --- region-scoped retrieval ------------------------------------------------------------

REGION_TERMS = {"kinnaur": TermHit(title="Kinnaur district", url="https://en.wikipedia.org/wiki/Kinnaur_district"),
                "chamba": TermHit(title="Chamba, Himachal Pradesh", url="https://en.wikipedia.org/wiki/Chamba"),
                "river gorge": TermHit(title="Canyon", url="https://en.wikipedia.org/wiki/Canyon"),
                "kath kuni": TermHit(title="Kath-kuni", url="https://en.wikipedia.org/wiki/Kath-kuni")}


def region_world(store, well, scoped, *, regions=("kinnaur", "chamba")):
    terms = [OutTerm(term=r, kind="region") for r in regions] + [
        OutTerm(term="river gorge", kind="landform", needs_region=True),
        OutTerm(term="kath kuni", kind="technique")]
    unscoped = {t: [hit(i, t.replace(" ", "")) for i in range(3)] for t in ("river gorge", "kath kuni")}
    scoped = {**{(r, r): [hit(i, r) for i in range(3)] for r in regions}, **scoped}
    blobs = {h.preview_url: b"x" for hits in [*unscoped.values(), *scoped.values()] for h in hits}
    images = MemoryImages({k: v for k, v in REGION_TERMS.items() if k in {t.term for t in terms}},
                          unscoped, blobs, scoped)

    def handler(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=terms)
        return default_handler(schema, system, parts)
    return make_ctx(store, images, handler), images


def test_generic_term_is_searched_in_each_region_before_unscoped(world):
    store, well, _ = world
    scoped = {("river gorge", "chamba"): [hit(i, "gorgechamba") for i in range(2)]}
    ctx, images = region_world(store, well, scoped)
    suggest_references(ctx, PID, well.id, per_term=3, max_images=20, dry_run=True)
    gorge = [q for q in images.queries if q[0] == "river gorge"]
    assert gorge == [("river gorge", "kinnaur"), ("river gorge", "chamba")]   # Chamba had hits: no bare search

    ctx, images = region_world(store, well, {})
    report = suggest_references(ctx, PID, well.id, per_term=3, max_images=20, dry_run=True)
    gorge = [q for q in images.queries if q[0] == "river gorge"]
    assert gorge == [("river gorge", "kinnaur"), ("river gorge", "chamba"), ("river gorge", None)]
    assert "'river gorge' returned no licensed images" not in report.warnings   # the fallback found some


def test_scoped_hits_from_several_regions_are_merged(world):
    store, well, _ = world
    scoped = {("river gorge", "kinnaur"): [hit(i, "gorgekinnaur") for i in range(3)],
              ("river gorge", "chamba"): [hit(i, "gorgechamba") for i in range(3)]}
    ctx, images = region_world(store, well, scoped)
    report = suggest_references(ctx, PID, well.id, per_term=3, max_images=20, dry_run=True)
    assert report.images_found == 4 * 3                    # capped per term, not per region
    titles = " ".join(p.text for name, _, parts in ctx.llm.calls if name == "CaptionsOut"
                      for p in parts if isinstance(p, Text) and p.text.startswith("["))
    assert "gorgekinnaur 0" in titles and "gorgechamba 0" in titles and "gorgekinnaur 1" in titles
    assert "gorgechamba 1" not in titles                   # regions interleaved within the term's cap
    assert ("File gorgekinnaur 0.jpg", "river gorge (in kinnaur (3), chamba (3))") in report.kept
    assert ("File kathkuni 0.jpg", "kath kuni (unscoped)") in report.kept


def test_specific_term_is_searched_once_unscoped(world):
    store, well, _ = world
    ctx, images = region_world(store, well, {})
    suggest_references(ctx, PID, well.id, per_term=3, max_images=20, dry_run=True)
    assert [q for q in images.queries if q[0] == "kath kuni"] == [("kath kuni", None)]
    assert ("kinnaur", "kinnaur") in images.queries and ("chamba", "chamba") in images.queries


def test_no_region_terms_behaves_as_before(world):
    store, well, _ = world
    ctx, images = region_world(store, well, {}, regions=())
    suggest_references(ctx, PID, well.id, per_term=3, max_images=20, dry_run=True)
    assert images.queries == [("river gorge", None), ("kath kuni", None)]   # needs_region, but nowhere to scope


def test_round_robin_holds_with_scoped_queries(world):
    store, well, _ = world
    scoped = {("river gorge", "kinnaur"): [hit(i, "gorgekinnaur") for i in range(3)]}
    ctx, images = region_world(store, well, scoped)
    suggest_references(ctx, PID, well.id, per_term=3, max_images=8, dry_run=True)
    caption_call = next(parts for name, _, parts in ctx.llm.calls if name == "CaptionsOut")
    terms = [p.text.split()[2] for p in caption_call if isinstance(p, Text) and p.text.startswith("[")]
    assert terms[:4] == ["kinnaur", "chamba", "gorgekinnaur", "kathkuni"]   # one per term before any second
    assert terms[4:8] == ["kinnaur", "chamba", "gorgekinnaur", "kathkuni"]


# --- Wikimedia query shapes, offline ----------------------------------------------------

def commons_page(title):
    return {"title": f"File:{title}", "imageinfo": [{
        "mime": "image/jpeg", "url": f"https://upload/{title}", "thumburl": f"https://thumb/{title}",
        "descriptionurl": f"https://commons/{title}",
        "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}


class FakeCommons:
    def __init__(self, categories):
        self.categories, self.calls = categories, []

    def __call__(self, endpoint, **params):
        self.calls.append(params)
        if params.get("list") == "search":                  # category namespace search
            return {"query": {"search": [{"title": t} for t in self.categories]}}
        return {"query": {"pages": [commons_page(params["gsrsearch"] + ".jpg")]}}


def test_region_resolves_to_a_category_matching_the_verified_page():
    from harness.memory.wikimedia import WikimediaImages

    w = WikimediaImages()
    w._api = fake = FakeCommons(["Category:Kinnaur Kailash", "Category:Kinnaur district"])
    own = w.search_images("Kinnaur", 5, region="Kinnaur", region_title="Kinnaur district")
    assert len(own) == 1
    assert fake.calls[-1]["gsrsearch"].startswith('deepcat:"Kinnaur district" ')   # not the better-ranked mountain

    scoped = w.search_images("river gorge", 5, region="Kinnaur", region_title="Kinnaur district")
    assert fake.calls[-1]["gsrsearch"].startswith('river gorge deepcat:"Kinnaur district"')
    assert len(scoped) == 1
    assert sum(c.get("srnamespace") == 14 for c in fake.calls) == 1    # resolution is cached per region


def test_region_without_matching_category_falls_back_to_free_text():
    from harness.memory.wikimedia import WikimediaImages

    w = WikimediaImages()
    w._api = fake = FakeCommons(["Category:Chamba, Uttarakhand", "Category:Chamba Valley temples"])
    w.search_images("river gorge", 5, region="Chamba", region_title="Chamba, Himachal Pradesh")
    assert fake.calls[-1]["gsrsearch"].startswith("river gorge Chamba ")
    w.search_images("Chamba", 5, region="Chamba", region_title="Chamba, Himachal Pradesh")
    assert fake.calls[-1]["generator"] == "search" and fake.calls[-1]["gsrsearch"].startswith("Chamba ")
    assert not any("deepcat" in c.get("gsrsearch", "") for c in fake.calls)
