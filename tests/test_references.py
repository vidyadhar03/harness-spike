import hashlib

import pytest

from harness.memory.config import Settings
from harness.memory.models import Location, Note, NoteOrigin, Project, Provenance, Scene
from harness.memory.ports import ImageHit, MemoryBlobs, MemoryImages, MemoryStore, TermHit, Text
from harness.memory.references import RefCtx, suggest_references
from harness.memory.retrieval import get_context, render_context_md
from harness.memory.schemas import CurateOut, OutCaption, OutDirection, OutTerm, VocabularyOut

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
    if schema is CurateOut:
        indices = [int(p.text.split("]")[0][1:]) for p in parts if isinstance(p, Text) and p.text.startswith("[")]
        a, b = indices[:2], indices[2:4]
        return CurateOut(
            directions=[OutDirection(name="Weathered slate and dark timber", why="Older, heavier.", images=a),
                        OutDirection(name="Open terraced slopes", why="Wider, greener.", images=b),
                        OutDirection(name="Named but empty", images=[])],
            captions=[OutCaption(index=i, caption=f"Caption {i}.") for i in indices[:5]],
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
    assert {n.group for n in refs} == {"Weathered slate and dark timber — Older, heavier.",
                                       "Open terraced slopes — Wider, greener.", "Unsorted"}
    assert all(n.status == "proposed" and n.owner_id == well.id for n in refs)
    assert all(n.origin.producer == "references" and n.origin.scope == well.id for n in refs + vocab)
    assert refs[0].body.startswith("Caption ")           # the rationale lives on the group, not every caption
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
    shown = [p.text for p in ctx.llm.calls[-1][2] if isinstance(p, Text) and p.text.startswith("[")]
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
    assert "### Weathered slate and dark timber — Older, heavier." in md
    assert "### Open terraced slopes — Wider, greener." in md
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
