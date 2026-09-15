import hashlib
from pathlib import Path
import re

import pytest

from harness.memory.config import Settings
from harness.memory.models import Location, Note, NoteOrigin, Project, Provenance, Scene, Source
from harness.memory.cli import print_references
from harness.memory.curate import review_note
from harness.memory.ports import ImageHit, MemoryBlobs, MemoryImages, MemoryStore, ReplacementTooLarge, TermHit, Text, VerificationServiceError
from harness.memory.references import RefCtx, ReferenceReport, RunLocalImageCache, export_references_html, suggest_references
from harness.memory.retrieval import get_context, get_references_context, render_context_md, render_references_context_md
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


# --- Visual references pipeline improvements tests -------------------------------------

def test_references_context_filters_unselected_and_prior_vocabulary(world):
    store, well, _ = world
    from harness.memory.models import Source
    source = Source(
        id="b" * 64,
        filename="slate0.jpg",
        size_bytes=1024,
        kind="image",
        doc_type="reference",
        storage_path="gs://b/refs/slate0.jpg",
        mime_type="image/jpeg",
        origin_url="https://commons.wikimedia.org/wiki/File:Slate0.jpg",
        license="CC BY-SA 4.0",
        attribution="Photographer A",
    )
    store.put_source(PID, source)

    # Script notes
    store.put_notes(PID, [
        Note(kind="constraint", body="Do not use concrete or modern bricks.",
             owner_id=well.id, author="user"),
        # Prior generated vocabulary note
        Note(kind="vocabulary", body="kath kuni architecture; slate roof",
             owner_id=well.id, author="agent",
             provenance=[Provenance(url="https://en.wikipedia.org/wiki/Kath-kuni")],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
        # Unselected proposed reference
        Note(kind="reference_image", body="Open terraced slopes — Caption 1.",
             owner_id=well.id, author="agent", status="proposed",
             provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:Terrace.jpg")],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
        # Rejected reference
        Note(kind="reference_image", body="Flat roof — Caption 2.",
             owner_id=well.id, author="user", status="rejected",
             provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:Flat.jpg")],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
        # Confirmed reference with director guidance
        Note(kind="reference_image", body="Weathered slate — Caption 0.",
             owner_id=well.id, author="user", status="confirmed",
             guidance="Match courtyard scale and rough slate overhang",
             direction="Weathered slate and dark timber", group="Architecture",
             provenance=[Provenance(source_id=source.id, url=source.origin_url)],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
    ])

    ref_pack = get_references_context(store, PID, well.id)
    notes_in_pack = ref_pack.notes
    kinds_in_pack = [n.kind for n in notes_in_pack]

    # Description and constraint are preserved
    assert "description" in kinds_in_pack
    assert "constraint" in kinds_in_pack

    # Prior vocabulary is excluded
    assert "vocabulary" not in kinds_in_pack

    # Proposed and rejected references are excluded from notes
    assert not any(n.kind == "reference_image" and n.status != "confirmed" for n in notes_in_pack)

    # Confirmed references are in reference_images
    assert len(ref_pack.reference_images) == 1
    assert ref_pack.reference_images[0].guidance == "Match courtyard scale and rough slate overhang"

    # Rendered markdown checks
    md = render_references_context_md(ref_pack)
    assert "Do not use concrete or modern bricks." in md
    assert "Stone well in a hill village" in md
    assert "## Director-approved reference inspiration" in md
    assert "not script requirements, geography, or geometry" in md
    assert "Match courtyard scale and rough slate overhang" in md
    assert "Weathered slate — Caption 0." in md
    assert "Open terraced slopes" not in md
    assert "Flat roof" not in md
    assert "## Search vocabulary" not in md


def test_suggest_references_passes_filtered_context_to_all_steps(world):
    store, well, images = world
    # Seed prior vocabulary and proposed reference
    store.put_notes(PID, [
        Note(kind="vocabulary", body="old prior vocabulary terms",
             owner_id=well.id, author="agent",
             provenance=[Provenance(url="https://en.wikipedia.org/wiki/OldVocab")],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
        Note(kind="reference_image", body="prior proposed image caption",
             owner_id=well.id, author="agent", status="proposed",
             provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:Old.jpg")],
             origin=NoteOrigin(producer="references", scope=well.id, digest_version="1")),
    ])
    ctx = make_ctx(store, images, default_handler)
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    for call_name, system, parts in ctx.llm.calls:
        text_parts = " ".join(p.text for p in parts if isinstance(p, Text))
        assert "prior proposed image caption" not in text_parts
        assert "old prior vocabulary terms" not in text_parts
        assert "## Search vocabulary" not in text_parts


def test_run_local_image_cache_bounds_and_single_download():
    class CountingImages(MemoryImages):
        def __init__(self):
            super().__init__({}, {}, {})
            self.fetch_calls = []

        def fetch(self, url: str) -> bytes:
            self.fetch_calls.append(url)
            return f"bytes-for-{url}".encode()

    raw_images = CountingImages()
    # Cache bounded to 50 bytes total
    cache = RunLocalImageCache(raw_images, max_bytes=50)

    # First fetch: should download
    d1 = cache.fetch("http://img1.jpg")
    assert d1 == b"bytes-for-http://img1.jpg"
    assert len(raw_images.fetch_calls) == 1
    assert cache.downloads == 1 and cache.hits == 0

    # Second fetch of same URL: should hit cache
    d2 = cache.fetch("http://img1.jpg")
    assert d2 == b"bytes-for-http://img1.jpg"
    assert len(raw_images.fetch_calls) == 1
    assert cache.downloads == 1 and cache.hits == 1

    # Cleanup clears cache and temp dir
    cache.cleanup()
    assert cache._tmpdir is None
    assert len(cache._cache) == 0


def test_suggest_references_tracks_cache_hits_and_downloads(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    # Images evaluated are downloaded once and reused for hashing/writing
    assert report.cache_downloads == 6
    assert report.cache_hits >= 5   # at least reused when preparing blob writes
    assert report.images_found == 10


def test_safe_replacement_preserves_concurrently_confirmed_notes(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    # First run generates proposed references
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    refs = [n for n in store.list_notes(PID) if n.kind == "reference_image"]
    assert len(refs) == 5
    to_confirm = refs[0]

    # Simulate director confirming this note during/before the second run
    confirmed_note = to_confirm.touch(
        status="confirmed",
        author="user",
        guidance="Director explicitly locked this reference",
    )
    store.put_notes(PID, [confirmed_note])

    # Second run
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    notes_after = store.list_notes(PID)
    matching = [n for n in notes_after if n.id == to_confirm.id]
    assert len(matching) == 1
    assert matching[0].status == "confirmed"
    assert matching[0].guidance == "Director explicitly locked this reference"


def test_safe_replacement_atomic_failure_leaves_previous_intact(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    initial_notes = {n.id: n for n in store.list_notes(PID)}
    assert len(initial_notes) > 1

    # Simulate atomic replacement store failure
    class FailingStore(MemoryStore):
        def replace_notes(self, project_id, to_delete, to_put, expected_status=None):
            raise RuntimeError("Atomic replacement failed")

    failing_store = FailingStore()
    failing_store.projects = store.projects
    failing_store.entities = store.entities
    failing_store.notes = dict(store.notes)
    failing_store.sources = dict(store.sources)

    failing_ctx = RefCtx(failing_store, ctx.blobs, ctx.llm, images, Settings(gcp_project="t", bucket="b"))
    with pytest.raises(RuntimeError, match="Atomic replacement failed"):
        suggest_references(failing_ctx, PID, well.id, per_term=2, max_images=6)

    # Previous notes in store must remain completely intact
    after_notes = {n.id: n for n in failing_store.list_notes(PID)}
    assert after_notes == initial_notes


def test_replace_notes_skips_concurrently_reviewed_note(world):
    """Adapter-level test: MemoryStore.replace_notes must skip deletion of a
    note whose status no longer matches expected_status (simulating a concurrent
    review between the pipeline's read and the replacement call)."""
    store, well, _ = world
    note_a = Note(kind="description", body="Will be reviewed", owner_id=well.id, author="user")
    note_b = Note(kind="description", body="Will stay proposed", owner_id=well.id, author="user")
    store.put_notes(PID, [note_a, note_b])

    # Simulate: pipeline read both as "proposed" earlier, but before replace_notes
    # is called, note_a was confirmed by a director.
    confirmed = note_a.touch(status="confirmed", reviewed_by="vd",
                             reviewed_revision=note_a.revision)
    store.put_notes(PID, [confirmed])

    # Now call replace_notes with expected_status reflecting the stale read
    replacement = Note(kind="description", body="Fresh replacement", owner_id=well.id, author="user")
    store.replace_notes(PID,
                        to_delete=[note_a.id, note_b.id],
                        to_put=[replacement],
                        expected_status={note_a.id: "proposed", note_b.id: "proposed"})

    remaining = {n.id: n for n in store.list_notes(PID)}
    # note_a must survive because its status changed from "proposed" to "confirmed"
    assert note_a.id in remaining
    assert remaining[note_a.id].status == "confirmed"
    # note_b was still "proposed" so it should be deleted
    assert note_b.id not in remaining
    # replacement was added
    assert replacement.id in remaining


def test_atomic_replacement_too_large_raises_error(world):
    store, well, _ = world
    to_delete = [f"del_{i}" for i in range(300)]
    to_put = [Note(kind="description", body=f"note {i}", owner_id=well.id, author="user") for i in range(250)]

    # 300 + 250 = 550 > 500 limit
    with pytest.raises(ReplacementTooLarge) as exc_info:
        store.replace_notes(PID, to_delete, to_put)

    assert exc_info.value.needed == 550
    assert exc_info.value.limit == 500


def test_performance_metrics_recorded_in_report(world, capsys):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6)

    timing_names = {st.name for st in report.stage_timings}
    assert {"total", "retrieval", "captioning", "grouping", "persistence"}.issubset(timing_names)
    assert report.images_found > 0
    assert report.images_kept == 5

    # print_references prints performance metrics and cache stats cleanly
    print_references(report)
    out = capsys.readouterr().out
    assert "timing:" in out
    assert "cache:" in out
    assert "downloads" in out


def test_retrieval_provenance_multi_origins_and_structured_fields(world):
    store, well, _ = world
    # Same image hit returned for two different terms
    shared_hit = hit(99, "shared")
    images_map = {
        "term1": [shared_hit],
        "term2": [shared_hit],
    }
    blobs = {shared_hit.preview_url: b"preview-data"}
    mem_images = MemoryImages({"term1": TermHit(title="T1", url="http://wiki/T1"),
                               "term2": TermHit(title="T2", url="http://wiki/T2")},
                              images_map, blobs)

    def handler(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=[OutTerm(term="term1", kind="technique"),
                                        OutTerm(term="term2", kind="landform")])
        if schema is CaptionsOut:
            return CaptionsOut(captions=[OutCaption(index=0, caption="Shared image caption", facet="material")])
        if schema is CurateOut:
            return CurateOut(directions=[OutDirection(name="Timber structure", why="Traditional joinery", images=[0])])
        raise AssertionError(schema)

    ctx = make_ctx(store, mem_images, handler)
    suggest_references(ctx, PID, well.id, per_term=1, max_images=2)

    refs = [n for n in store.list_notes(PID) if n.kind == "reference_image"]
    assert len(refs) == 1
    ref = refs[0]

    # Structured direction and facet fields
    assert ref.direction == "Timber structure"
    assert ref.direction_rationale == "Traditional joinery"
    assert ref.group == "Material"

    # Multi query origins recorded in provenance
    assert len(ref.provenance) == 1
    prov = ref.provenance[0]
    assert len(prov.retrieval_origins) == 2
    terms_in_origins = [o.term for o in prov.retrieval_origins]
    assert "term1" in terms_in_origins
    assert "term2" in terms_in_origins


def test_verification_service_failure_vs_rejection(world):
    store, well, _ = world

    class FlakyImages(MemoryImages):
        def verify_term(self, term: str) -> TermHit | None:
            if term == "broken service":
                raise VerificationServiceError("Wikipedia API 503 error")
            if term == "dropped term":
                return None
            return super().verify_term(term)

    images = FlakyImages({"good term": TermHit(title="Good", url="http://wiki/good")},
                         {"good term": [hit(1, "good")]},
                         {hit(1, "good").preview_url: b"good-bytes"})

    def handler(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=[
                OutTerm(term="broken service", kind="technique"),
                OutTerm(term="dropped term", kind="technique"),
                OutTerm(term="good term", kind="technique"),
            ])
        return default_handler(schema, system, parts)

    ctx = make_ctx(store, images, handler)
    report = suggest_references(ctx, PID, well.id, per_term=1, max_images=2, dry_run=True)

    # Verification network failure is captured under verification_failures
    assert "broken service" in report.verification_failures
    # Normal 404 / no match is dropped
    assert "dropped term" in report.terms_dropped
    # Good term succeeds
    assert "good term" in report.terms_verified


def test_bypass_term_verification(world):
    store, well, images = world
    ctx = make_ctx(store, images, default_handler)
    # With bypass_verification=True, "made up term" is NOT dropped even though not verified
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6, dry_run=True,
                                bypass_verification=True)

    assert "made up term" not in report.terms_dropped
    assert "made up term" in images.searched


def test_bypass_canonical_region_and_unmatched_queries(world):
    store, well, _ = world
    terms = [
        OutTerm(term="kinnaur", kind="region"),
        OutTerm(term="shrine enclosed by roots", kind="technique", needs_region=True),
        OutTerm(term="unknown valley", kind="region"),
    ]
    verified_terms = {
        "kinnaur": TermHit(title="Kinnaur district", url="https://en.wikipedia.org/wiki/Kinnaur_district"),
    }
    unscoped = {
        "kinnaur": [hit(0, "kinnaur")],
        "shrine enclosed by roots": [hit(0, "shrine")],
        "unknown valley": [hit(0, "valley")],
    }
    scoped = {
        ("kinnaur", "kinnaur"): [hit(0, "kinnaur")],
        ("shrine enclosed by roots", "kinnaur"): [hit(0, "roots_kinnaur")],
    }
    blobs = {h.preview_url: b"bytes" for hits in [*unscoped.values(), *scoped.values()] for h in hits}
    images = MemoryImages(verified_terms, unscoped, blobs, scoped)

    def handler(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=terms)
        return default_handler(schema, system, parts)

    ctx = make_ctx(store, images, handler)
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=6, dry_run=True,
                                bypass_verification=True)

    # 1. Unmatched search term was not rejected despite having no Wikipedia hit
    assert "shrine enclosed by roots" not in report.terms_dropped
    assert "shrine enclosed by roots" in images.searched

    # 2. Canonical title for Kinnaur was preserved and passed to search_images
    kinnaur_calls = [c for c in images.search_calls if c["region"] == "kinnaur"]
    assert len(kinnaur_calls) > 0
    assert all(c["region_title"] == "Kinnaur district" for c in kinnaur_calls)

    # 3. Unresolved region term triggers a specific warning
    assert any("region 'unknown valley' lacks canonical resolution" in w for w in report.warnings)



def test_export_references_html_and_escaping(world, tmp_path):
    store, well, _ = world
    blobs = MemoryBlobs()
    storage_path = "gs://b/refs/test_img.jpg"
    data = b"test-image-bytes"
    blobs.put(storage_path, data, "image/jpeg")

    from harness.memory.models import Source
    source_id = "a" * 64
    source = Source(
        id=source_id,
        filename="test.jpg",
        size_bytes=len(data),
        kind="image",
        doc_type="reference",
        storage_path=storage_path,
        mime_type="image/jpeg",
        origin_url="https://commons.wikimedia.org/wiki/File:Test.jpg",
        license="CC BY-SA 4.0",
        attribution="Photographer <XSS & Tag>",
    )
    store.put_source(PID, source)

    store.put_notes(PID, [
        Note(
            kind="reference_image",
            owner_id=well.id,
            status="confirmed",
            author="user",
            direction="Weathered slate <script>",
            direction_rationale="Older & heavier <rationale>",
            group="Architecture",
            body="Slate overhang <alert>",
            guidance="Director's guidance with <tags> & 'quotes'",
            provenance=[Provenance(
                source_id=source.id,
                url=source.origin_url,
            )],
            origin=NoteOrigin(producer="references", scope=well.id, digest_version="1"),
        ),
        Note(
            kind="reference_image",
            owner_id=well.id,
            status="proposed",
            author="agent",
            direction="Terraced slopes",
            group="Place",
            body="Open mountain terraces",
            provenance=[Provenance(
                source_id=source.id,
                url=source.origin_url,
            )],
            origin=NoteOrigin(producer="references", scope=well.id, digest_version="1"),
        ),
    ])

    out_dir = tmp_path / "ref_export"
    html_file, ref_count, hint = export_references_html(store, blobs, PID, well.id, out_dir)

    assert ref_count == 2  # confirmed + proposed, neither rejected
    assert hint is None
    assert html_file.exists()
    content = html_file.read_text()

    # Verify text escaping
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "<alert>" not in content
    assert "&lt;alert&gt;" in content
    assert "<tags>" not in content
    assert "&lt;tags&gt;" in content
    assert "<XSS & Tag>" not in content
    assert "&lt;XSS &amp; Tag&gt;" in content

    # Verify local image download
    exported_img = out_dir / "images" / f"{source.id}.jpg"
    assert exported_img.exists()
    assert exported_img.read_bytes() == b"test-image-bytes"

    # Verify confirmed_only filter
    confirmed_out_dir = tmp_path / "confirmed_export"
    confirmed_html, confirmed_count, confirmed_hint = export_references_html(
        store, blobs, PID, well.id, confirmed_out_dir, confirmed_only=True)
    confirmed_content = confirmed_html.read_text()
    assert confirmed_count == 1
    assert confirmed_hint is None
    assert "Slate overhang" in confirmed_content
    assert "Open mountain terraces" not in confirmed_content


def test_export_references_html_empty_state(world, tmp_path):
    """export_references_html returns count=0 and a distinguishing hint in both
    empty cases: no stored references at all, and references excluded by filters."""
    store, well, _ = world
    blobs = MemoryBlobs()

    # --- Case 1: no references stored yet ---
    no_refs_dir = tmp_path / "no_refs"
    html_path, count, hint = export_references_html(store, blobs, PID, well.id, no_refs_dir)
    assert count == 0
    assert hint is not None
    assert "Run" in hint  # actionable instruction
    content = html_path.read_text()
    assert "No references have been generated" in content  # hint appears in the HTML body

    # --- Case 2: references stored but confirmed_only and nothing confirmed ---
    store.put_notes(PID, [
        Note(
            kind="reference_image",
            owner_id=well.id,
            status="proposed",
            author="agent",
            body="A proposed place",
            provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:A.jpg")],
            origin=NoteOrigin(producer="references", scope=well.id, digest_version="1"),
        ),
    ])
    filtered_dir = tmp_path / "filtered"
    html_path2, count2, hint2 = export_references_html(
        store, blobs, PID, well.id, filtered_dir, confirmed_only=True)
    assert count2 == 0
    assert hint2 is not None
    assert "confirmed" in hint2.lower()
    content2 = html_path2.read_text()
    assert "No confirmed references" in content2


def test_review_note_guidance(world):
    store, well, _ = world
    note = Note(kind="reference_image", body="Stone texture", owner_id=well.id,
                status="proposed", author="agent",
                provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:Stone.jpg")])
    store.put_notes(PID, [note])

    # --- Confirm with guidance: revision bumps, reviewed_revision tracks it ---
    res = review_note(store, PID, note.id, decision="confirmed", reviewer="user",
                      guidance="Focus on stone mortar and rough dressing")

    assert res.note.status == "confirmed"
    assert res.note.guidance == "Focus on stone mortar and rough dressing"
    assert res.note.reviewed_by == "user"
    assert res.note.revision == note.revision + 1       # guidance bumped revision
    assert res.note.reviewed_revision == res.note.revision  # confirmation points at new revision

    # Verify in store
    in_store = [n for n in store.list_notes(PID) if n.id == note.id][0]
    assert in_store.status == "confirmed"
    assert in_store.guidance == "Focus on stone mortar and rough dressing"
    assert in_store.revision == note.revision + 1

    # --- Re-confirm without guidance: existing guidance is preserved ---
    # Need to reset to proposed to re-confirm (simulate a pipeline re-proposal with same id)
    re_proposed = in_store.touch(status="proposed", reviewed_revision=None,
                                 reviewed_by=None, reviewed_at=None)
    store.put_notes(PID, [re_proposed])

    res2 = review_note(store, PID, note.id, decision="confirmed", reviewer="vd")
    assert res2.note.guidance == "Focus on stone mortar and rough dressing"  # preserved
    assert res2.note.revision == re_proposed.revision  # no bump when guidance omitted
    assert res2.note.reviewed_revision == re_proposed.revision


def test_strip_heading_prefix_and_html_composition(world, tmp_path):
    from harness.memory.references import strip_heading_prefix

    # Unit checks on strip_heading_prefix
    assert strip_heading_prefix("Roots & Stone — Deep roots wrapping walls", "Roots & Stone") == "Deep roots wrapping walls"
    assert strip_heading_prefix("Masonry - Rough dry stone", "Masonry") == "Rough dry stone"
    assert strip_heading_prefix("Architecture: Traditional tower", "Architecture") == "Traditional tower"
    assert strip_heading_prefix("Already plain caption", "Different Heading") == "Already plain caption"
    assert strip_heading_prefix("Unsorted caption", "Unsorted") == "Unsorted caption"

    # HTML export composition checks
    store, well, _ = world
    blobs = MemoryBlobs()
    sid = "a" * 64
    source = Source(
        id=sid, filename="tree_shrine.jpg", mime_type="image/jpeg",
        kind="image", doc_type="reference", size_bytes=100,
        storage_path=f"gs://bucket/projects/prj_1/sources/{sid}/original.jpg",
        status="digested", origin_url="https://commons.wikimedia.org/wiki/File:tree_shrine.jpg",
    )
    store.put_source(PID, source)
    blobs.put(source.storage_path, b"dummy-bytes", source.mime_type)

    note = Note(
        kind="reference_image",
        owner_id=well.id,
        status="proposed",
        author="agent",
        direction="Engulfing Root Canopies",
        body="Engulfing Root Canopies — Massive roots encasing ancient stone sanctum.",
        group="Architecture",
        provenance=[Provenance(source_id=source.id, url=source.origin_url)],
        origin=NoteOrigin(producer="references", scope=well.id, digest_version="1"),
    )
    store.put_notes(PID, [note])

    out_dir = tmp_path / "comp_export"
    html_file, count, hint = export_references_html(store, blobs, PID, well.id, out_dir)
    assert count == 1
    content = html_file.read_text()

    # Preserves image composition with contain
    assert "object-fit: contain;" in content
    # Clickable link to local preview image
    assert f'<a class="img-link" href="images/{source.id}.jpg" target="_blank" rel="noopener"' in content
    # Repeated prefix is removed from displayed paragraph
    assert "<p>Massive roots encasing ancient stone sanctum.</p>" in content
    assert "<p>Engulfing Root Canopies — Massive roots" not in content
    # Note in store is preserved with original body
    stored = [n for n in store.list_notes(PID) if n.id == note.id][0]
    assert stored.body == "Engulfing Root Canopies — Massive roots encasing ancient stone sanctum."


def test_clarified_candidate_counts(capsys):
    r = ReferenceReport(
        scope_id="loc_1",
        location="Tree Temple",
        terms_proposed=10,
        terms_verified=["roots", "shrine"],
        candidates_retrieved=54,
        candidates_evaluated=32,
        images_kept=13,
        candidates_omitted=19,
        failed_downloads=0,
        notes_written=14,
        notes_replaced=14,
    )
    print_references(r)
    out = capsys.readouterr().out
    assert "13/32 evaluated kept" in out
    assert "13/54" not in out
    assert "candidates: 54 retrieved, 32 evaluated, 13 kept, 19 omitted, 0 failed downloads" in out


def test_print_references_bypass_and_logical_model_calls(capsys):
    r = ReferenceReport(
        scope_id="loc_1",
        location="Tree temple",
        terms_proposed=9,
        terms_verified=[f"term_{i}" for i in range(8)],
        terms_bypassed=["roots absorbing masonry"],
        model_calls=6,
        http_retries=1,
    )
    print_references(r)
    out = capsys.readouterr().out
    assert "Tree temple: 8/9 terms verified, 1 accepted without match" in out
    assert "9/9 terms verified" not in out
    assert "bypassed: roots absorbing masonry" in out
    assert "6 logical calls (1 HTTP retry)" in out


def test_saved_candidates_evaluation(tmp_path):
    import json
    fixture_path = Path("evals/devgram_candidates.json")
    if not fixture_path.exists():
        return
    data = json.loads(fixture_path.read_text())
    assert "hits" in data
    assert len(data["hits"]) > 0
    # Verify hits can be converted to ImageHit
    hits = [ImageHit(**h) for h in data["hits"]]
    assert len(hits) == len(data["hits"])


def test_bypass_persistence_e2e_unmatched_term_and_cli_failure(world, monkeypatch):
    from unittest.mock import Mock
    from harness.memory.cli import main

    store, well, _ = world
    unmatched = "roots absorbing masonry"
    terms = [
        OutTerm(term=unmatched, kind="technique"),
        OutTerm(term="slate roof", kind="material"),
    ]
    verified_terms = {
        "slate roof": TermHit(title="Slate Roof", url="https://en.wikipedia.org/wiki/Slate_roof"),
    }
    unscoped = {
        unmatched: [hit(0, "roots")],
        "slate roof": [hit(0, "slate")],
    }
    blobs = {h.preview_url: b"image-data" for hits in unscoped.values() for h in hits}
    images = MemoryImages(verified_terms, unscoped, blobs)

    def handler(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=terms)
        return default_handler(schema, system, parts)

    ctx = make_ctx(store, images, handler)

    # 1. Run with bypass_verification=True, dry_run=False
    report = suggest_references(ctx, PID, well.id, per_term=2, max_images=4, dry_run=False,
                                bypass_verification=True)

    # Persistence succeeds
    assert report.notes_written > 0
    stored_notes = store.list_notes(PID)
    refs = [n for n in stored_notes if n.kind == "reference_image"]
    vocab = [n for n in stored_notes if n.kind == "vocabulary"]

    assert len(refs) > 0
    assert len(vocab) == 1

    # Image reference is stored in sources and notes
    stored_sources = store.list_sources(PID)
    assert len(stored_sources) > 0
    assert refs[0].provenance[0].source_id in {s.id for s in stored_sources}
    assert refs[0].provenance[0].url is not None
    assert refs[0].provenance[0].url != ""

    # Vocabulary note has provenance ONLY from hits with real, nonempty URLs
    for p in vocab[0].provenance:
        assert p.url is not None and p.url != ""
    assert all(p.title != unmatched for p in vocab[0].provenance)
    assert any(p.title == "Slate Roof" for p in vocab[0].provenance)

    # Ensure no empty-URL provenance is created anywhere
    for note in stored_notes:
        for p in note.provenance:
            assert p.url != ""

    # Also test all-unmatched run: vocabulary note has empty provenance and succeeds
    all_unmatched_terms = [OutTerm(term=unmatched, kind="technique")]
    images_unmatched = MemoryImages({}, {unmatched: [hit(1, "roots2")]}, {hit(1, "roots2").preview_url: b"data"})
    def handler_all_unmatched(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(terms=all_unmatched_terms)
        return default_handler(schema, system, parts)
    ctx_unmatched = make_ctx(store, images_unmatched, handler_all_unmatched)
    rep2 = suggest_references(ctx_unmatched, PID, well.id, per_term=2, max_images=4, dry_run=False,
                              bypass_verification=True)
    assert rep2.notes_written > 0
    v_note = [n for n in store.list_notes(PID) if n.kind == "vocabulary"][0]
    assert v_note.provenance == []

    # 2. Assert that a persistence failure returns a nonzero CLI exit status
    monkeypatch.setenv("GCP_PROJECT", "test-prj")
    monkeypatch.setenv("MEMORY_BUCKET", "test-bucket")
    monkeypatch.setattr("harness.memory.cli.build_ref_ctx", lambda settings: ctx)
    monkeypatch.setattr(store, "replace_notes", Mock(side_effect=RuntimeError("persistence write failed")))
    exit_code = main(["references", PID, well.id, "--bypass-term-verification"])
    assert exit_code != 0


