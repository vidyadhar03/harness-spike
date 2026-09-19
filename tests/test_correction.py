"""Tests for the human note-correction/split workflow (harness.memory.curate.correct_note,
harness.api.routes.references's POST .../notes/{id}/correct).

Every test uses MemoryStore/MemoryBlobs - no real GCP or Gemini call is made anywhere
in this file, except the fake-LLM re-ingestion test, which uses the same FakeLLM
pattern as tests/test_ingest.py (no real model call there either).
"""
from __future__ import annotations

import threading

import pytest
from starlette.testclient import TestClient

from harness.api.main import create_app
from harness.api.settings import ApiSettings
from harness.memory.concepts import (
    CORE_NOTES_CAVEAT, approval_staleness_reasons, build_approval_snapshot, lock_approval,
    upload_concept_version,
)
from harness.memory.config import Settings
from harness.memory.curate import NoteSuccessorSpec, correct_note
from harness.memory.ingest import Ctx as IngestCtx, ingest_source, register_file
from harness.memory.models import Applicability, Location, Note, Project, Provenance, Scene, Source
from harness.memory.ports import MemoryBlobs, MemoryImages, MemoryStore, NoteReviewConflict
from harness.memory.retrieval import get_context
from harness.memory.schemas import ClassifyOut, NotesOut, OutLocation, OutNote, OutScene, RosterOut

PID = "prj_correction_test"


@pytest.fixture
def env():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Correction Test Film"))
    house = Location(name="Anshul's House", status="confirmed", author="agent")
    store.put_entities(PID, [house])
    scene4 = Scene(name="4. INT. ANSHUL'S HOUSE - NIGHT", number="4", location_ids=[house.id], author="agent")
    scene7 = Scene(name="7. INT./EXT. PANDIT'S DREAM - NIGHT", number="7", location_ids=[house.id], author="agent")
    other_loc = Location(name="River Bend", status="confirmed", author="agent")
    other_scene = Scene(name="99. EXT. RIVER BEND - DAY", number="99", location_ids=[other_loc.id], author="agent")
    store.put_entities(PID, [scene4, scene7, other_loc, other_scene])

    src = Source(id="a" * 64, filename="Ep1.pdf", mime_type="application/pdf", kind="document",
                doc_type="script", size_bytes=100, storage_path=f"gs://b/{'a' * 64}.pdf", status="digested")
    store.put_source(PID, src)
    mixed = Note(
        kind="description",
        body="Family photographs decorate the space, specifically one of a younger Pandit "
             "beside another man whose face is completely obscured by reflected light.",
        owner_id=house.id, author="agent",
        provenance=[Provenance(source_id=src.id, page=6,
                               quote="a younger Pandit beside another man. Reflected light hides the second man's face.")],
    )
    store.put_notes(PID, [mixed])

    blobs = MemoryBlobs()
    settings = Settings(gcp_project="test-gcp", bucket="test-bucket")
    images = MemoryImages({}, {}, {})
    return store, blobs, images, settings, house, scene4, scene7, other_loc, other_scene, mixed


def make_client(store, blobs, images):
    app = create_app(
        settings=Settings(gcp_project="t", bucket="b"),
        api_settings=ApiSettings(allowed_hosts=("testserver", "127.0.0.1", "localhost")),
        store=store, blobs=blobs, images=images,
    )
    return TestClient(app, base_url="http://testserver")


# ==========================================================================================
# Plain correction (1 successor): text/kind fix
# ==========================================================================================

def test_correct_text_preserves_provenance_and_marks_original_corrected(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    client = make_client(store, blobs, images)

    r = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [{"kind": "description", "body": "A framed photograph is present in the house."}],
        "expectedRevision": mixed.revision, "expectedStatus": mixed.status, "by": "director",
    })
    assert r.status_code == 200
    body = r.json()

    assert body["original"]["id"] == mixed.id
    assert body["original"]["status"] == "rejected"
    assert body["original"]["reviewReason"] == "corrected"

    assert len(body["newNotes"]) == 1
    new = body["newNotes"][0]
    assert new["body"] == "A framed photograph is present in the house."
    assert new["status"] == "proposed"          # never inherits the original's review
    assert new["sceneId"] is None

    # Citations/provenance preserved verbatim - same source, page, and quote.
    assert new["citations"][0]["quote"] == mixed.provenance[0].quote
    assert new["citations"][0]["page"] == 6

    stored_new = store.notes[(PID, new["id"])]
    assert stored_new.provenance[0].quote == mixed.provenance[0].quote   # unchanged evidence
    assert stored_new.body != mixed.provenance[0].quote                  # edited wording, not a fake quote
    assert stored_new.supersedes_note_id == mixed.id
    assert stored_new.author == "user"
    assert stored_new.origin.producer == "user"

    stored_original = store.notes[(PID, mixed.id)]
    assert stored_original.superseded_by_note_ids == [new["id"]]
    assert stored_original.revision == mixed.revision   # never bumped - it wasn't edited in place


def test_correct_kind(env):
    store, blobs, images, _, house, *_rest, mixed = env
    result = correct_note(store, PID, mixed.id,
                          [NoteSuccessorSpec(kind="constraint", body="Keep a framed photograph in the room.")],
                          reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status)
    assert result.new_notes[0].kind == "constraint"
    assert result.original.status == "rejected"


# ==========================================================================================
# Applicability change: standing -> scene, and scene -> standing
# ==========================================================================================

def test_change_applicability_to_scene(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    result = correct_note(
        store, PID, mixed.id,
        [NoteSuccessorSpec(kind="description", body="Reflected light hides a face in one photograph.",
                          scene_id=scene7.id)],
        reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status,
    )
    new = result.new_notes[0]
    assert new.applicability.scene_id == scene7.id
    assert new.owner_id == house.id   # ownership unchanged


def test_change_applicability_to_standing(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    scoped = mixed.touch(applicability=Applicability(scene_id=scene7.id))
    store.put_note_if_current(PID, scoped, expected_revision=mixed.revision, expected_status=mixed.status)

    result = correct_note(
        store, PID, mixed.id,
        [NoteSuccessorSpec(kind="description", body="A framed photograph is present.", scene_id=None)],
        reviewer="director", expected_revision=scoped.revision, expected_status=scoped.status,
    )
    assert result.new_notes[0].applicability.scene_id is None


def test_scene_must_belong_to_the_same_location(env):
    store, blobs, images, _, house, scene4, scene7, other_loc, other_scene, mixed = env
    with pytest.raises(ValueError, match="not a scene linked to"):
        correct_note(
            store, PID, mixed.id,
            [NoteSuccessorSpec(kind="description", body="x", scene_id=other_scene.id)],
            reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status,
        )
    # Never partially applied - the original is untouched.
    assert store.notes[(PID, mixed.id)].status == "proposed"


def test_unknown_scene_id_is_rejected(env):
    store, blobs, images, _, house, *_rest, mixed = env
    with pytest.raises(ValueError, match="not a scene linked to"):
        correct_note(store, PID, mixed.id,
                     [NoteSuccessorSpec(kind="description", body="x", scene_id="scn_doesnotexist")],
                     reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status)


# ==========================================================================================
# Split into standing + scene-specific (2 successors) - the Anshul's House example
# ==========================================================================================

def test_split_into_standing_and_scene_specific(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    client = make_client(store, blobs, images)

    r = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [
            {"kind": "description", "body": "A framed photograph is present in the house."},
            {"kind": "description", "body": "Reflected light obscures a face in the photograph.",
             "sceneId": scene7.id},
        ],
        "expectedRevision": mixed.revision, "expectedStatus": mixed.status, "by": "director",
    })
    assert r.status_code == 200
    new_notes = r.json()["newNotes"]
    assert len(new_notes) == 2
    standing, scene_specific = new_notes
    assert standing["sceneId"] is None
    assert scene_specific["sceneId"] == scene7.id

    # Atomic: both new notes and the rejected original are present together.
    assert store.notes[(PID, mixed.id)].status == "rejected"
    assert store.notes[(PID, standing["id"])] is not None
    assert store.notes[(PID, scene_specific["id"])] is not None
    for n in new_notes:
        assert store.notes[(PID, n["id"])].provenance[0].quote == mixed.provenance[0].quote


def test_split_requires_one_or_two_successors(env):
    store, blobs, images, _, house, *_rest, mixed = env
    with pytest.raises(ValueError, match="1 successor .* or 2"):
        correct_note(store, PID, mixed.id, [], reviewer="director",
                     expected_revision=mixed.revision, expected_status=mixed.status)
    with pytest.raises(ValueError, match="1 successor .* or 2"):
        correct_note(store, PID, mixed.id, [
            NoteSuccessorSpec(kind="description", body="a"),
            NoteSuccessorSpec(kind="description", body="b"),
            NoteSuccessorSpec(kind="description", body="c"),
        ], reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status)


# ==========================================================================================
# Scope validation
# ==========================================================================================

def test_unsupported_kind_is_rejected(env):
    store, blobs, images, _, house, *_rest, mixed = env
    with pytest.raises(ValueError, match="not a supported kind"):
        correct_note(store, PID, mixed.id, [NoteSuccessorSpec(kind="vocabulary", body="x")],
                     reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status)


def test_original_kind_not_supported_is_rejected(env):
    store, blobs, images, _, house, *_rest, mixed = env
    vocab = Note(kind="vocabulary", body="temple bell", owner_id=house.id, author="agent")
    store.put_notes(PID, [vocab])
    with pytest.raises(ValueError, match="only description/constraint/tone notes"):
        correct_note(store, PID, vocab.id, [NoteSuccessorSpec(kind="description", body="x")],
                     reviewer="director", expected_revision=vocab.revision, expected_status=vocab.status)


def test_already_rejected_note_cannot_be_corrected(env):
    store, blobs, images, _, house, *_rest, mixed = env
    rejected = mixed.touch(status="rejected", review_reason="not_useful")
    store.put_note_if_current(PID, rejected, expected_revision=mixed.revision, expected_status=mixed.status)
    with pytest.raises(ValueError, match="already rejected"):
        correct_note(store, PID, mixed.id, [NoteSuccessorSpec(kind="description", body="x")],
                     reviewer="director", expected_revision=rejected.revision, expected_status="rejected")


def test_unknown_note_is_404(env):
    store, blobs, images, _, house, *_rest, mixed = env
    client = make_client(store, blobs, images)
    r = client.post(f"/projects/{PID}/notes/note_doesnotexist/correct", json={
        "successors": [{"kind": "description", "body": "x"}],
        "expectedRevision": 1, "expectedStatus": "proposed",
    })
    assert r.status_code == 404


def test_owner_no_longer_live_is_rejected(env):
    store, blobs, images, _, house, *_rest, mixed = env
    store.put_entities(PID, [house.touch(status="rejected")])
    with pytest.raises(ValueError, match="no longer a live entity"):
        correct_note(store, PID, mixed.id, [NoteSuccessorSpec(kind="description", body="x")],
                     reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status)


# ==========================================================================================
# Concurrency: stale edits 409, atomic writes, concurrent corrections
# ==========================================================================================

def test_stale_expected_revision_returns_409_and_writes_nothing(env):
    store, blobs, images, _, house, *_rest, mixed = env
    client = make_client(store, blobs, images)

    r = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [{"kind": "description", "body": "wrong draft"}],
        "expectedRevision": 999, "expectedStatus": "proposed",
    })
    assert r.status_code == 409

    # Nothing was written - the original is untouched and no stray note was created.
    assert store.notes[(PID, mixed.id)].status == "proposed"
    assert store.notes[(PID, mixed.id)].revision == mixed.revision
    assert len(store.notes) == 1

    # The caller's draft still works once resubmitted with fresh values.
    r2 = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [{"kind": "description", "body": "wrong draft"}],
        "expectedRevision": mixed.revision, "expectedStatus": mixed.status,
    })
    assert r2.status_code == 200


def test_correcting_an_already_corrected_note_conflicts(env):
    store, blobs, images, _, house, *_rest, mixed = env
    correct_note(store, PID, mixed.id, [NoteSuccessorSpec(kind="description", body="first correction")],
                reviewer="a", expected_revision=mixed.revision, expected_status=mixed.status)
    # A second, independent correction attempt using the now-stale original values.
    with pytest.raises(NoteReviewConflict):
        correct_note(store, PID, mixed.id, [NoteSuccessorSpec(kind="description", body="second correction")],
                    reviewer="b", expected_revision=mixed.revision, expected_status=mixed.status)
    # Only the first correction's successor exists.
    assert len(store.notes) == 2


def test_concurrent_corrections_of_the_same_note_one_wins_one_conflicts(env):
    store, blobs, images, _, house, *_rest, mixed = env
    results = {}
    barrier = threading.Barrier(2)

    def attempt(name, body):
        barrier.wait()
        try:
            results[name] = ("ok", correct_note(
                store, PID, mixed.id, [NoteSuccessorSpec(kind="description", body=body)],
                reviewer=name, expected_revision=mixed.revision, expected_status=mixed.status,
            ))
        except NoteReviewConflict as exc:
            results[name] = ("conflict", exc)

    t1 = threading.Thread(target=attempt, args=("a", "correction A"))
    t2 = threading.Thread(target=attempt, args=("b", "correction B"))
    t1.start(); t2.start()
    t1.join(timeout=2); t2.join(timeout=2)

    outcomes = sorted([results["a"][0], results["b"][0]])
    assert outcomes == ["conflict", "ok"]
    # Exactly one successor note exists - no partial/double split.
    assert len(store.notes) == 2
    assert store.notes[(PID, mixed.id)].status == "rejected"


# ==========================================================================================
# Retrieval reflects the correction via existing supersession/visibility rules
# ==========================================================================================

def test_retrieval_shows_corrected_notes_not_the_original(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    correct_note(
        store, PID, mixed.id,
        [NoteSuccessorSpec(kind="description", body="A framed photograph is present."),
         NoteSuccessorSpec(kind="description", body="Reflected light obscures a face.", scene_id=scene7.id)],
        reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status,
    )

    pack = get_context(store, PID, house.id, include_proposed=True)
    body_texts = {n.body for n in pack.notes}
    assert mixed.body not in body_texts
    assert "A framed photograph is present." in body_texts

    conditional_bodies = {n.body for c in pack.conditional for n in c.notes}
    assert "Reflected light obscures a face." in conditional_bodies


# ==========================================================================================
# Approval staleness surfaces a correction; the stored package stays immutable/readable
# ==========================================================================================

def test_correction_surfaces_as_approval_staleness_and_package_stays_immutable(env):
    store, blobs, images, settings, house, scene4, scene7, _, _, mixed = env
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (40, 40), "gray").save(buf, format="JPEG")
    version, _ = upload_concept_version(store, blobs, settings, PID, house.id, buf.getvalue(), "v.jpg")

    snapshot = build_approval_snapshot(store, PID, house.id, version.id, [])
    approval, created = lock_approval(store, PID, house.id, concept_version_id=version.id, reference_ids=[],
                                      context_token=snapshot.context_token, expected_revision=0,
                                      locked_by="director")
    assert created is True
    assert any(n.id == mixed.id for n in approval.brief_notes)

    result = correct_note(
        store, PID, mixed.id,
        [NoteSuccessorSpec(kind="description", body="A framed photograph is present.")],
        reviewer="director", expected_revision=mixed.revision, expected_status=mixed.status,
    )

    # The stored package is exactly as it was - never rewritten.
    current = store.get_current_approval(PID, house.id)
    assert current.id == approval.id
    assert any(n.id == mixed.id and n.body == mixed.body for n in current.brief_notes)

    reasons = approval_staleness_reasons(store, PID, current)
    assert any("was corrected since this was approved" in r and result.new_notes[0].id in r for r in reasons)
    assert not any("was removed since this was approved" in r for r in reasons)


# ==========================================================================================
# Re-ingestion cannot resurrect or overwrite a human correction
# ==========================================================================================

class _FakeLLM:
    model_id = "fake"

    def __init__(self, handler):
        self.handler = handler

    def generate(self, *, system, parts, schema, fast=False, thinking_level=None):
        return self.handler(schema)


def test_reingestion_does_not_resurrect_or_overwrite_a_correction():
    """Self-contained (doesn't use the `env` fixture, which pre-seeds an unrelated
    note with a similar body) - a fresh project/location/scene, one note produced by a
    real ingest_source run against a fake LLM, corrected, then re-ingested."""
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Correction Reingest Test"))
    house = Location(name="Anshul's House", status="confirmed", author="agent")
    store.put_entities(PID, [house])
    blobs = MemoryBlobs()
    settings = Settings(gcp_project="test-gcp", bucket="test-bucket")

    note_body = "Family photographs decorate the space; reflected light hides a face."
    note_quote = "Reflected light hides the second man's face."

    def handler(schema):
        if schema is ClassifyOut:
            return ClassifyOut(doc_type="script")
        if schema is RosterOut:
            return RosterOut(locations=[OutLocation(name="Anshul's House")],
                             scenes=[OutScene(number="4", heading="INT. ANSHUL'S HOUSE - NIGHT",
                                             locations=["Anshul's House"])])
        if schema is NotesOut:
            return NotesOut(notes=[OutNote(kind="description", body=note_body, owner="Anshul's House",
                                          page=6, quote=note_quote)])
        raise AssertionError(schema)

    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    filler = "\n".join(["The wind moves through the dry fields."] * 6)
    c.drawString(40, 800, f"4 INT. ANSHUL'S HOUSE - NIGHT {note_body} {filler}")
    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()

    ctx = IngestCtx(store=store, blobs=blobs, llm=_FakeLLM(handler), settings=settings)
    src, created = register_file(ctx, PID, pdf_bytes, "Ep1.pdf")
    assert created is True
    report1 = ingest_source(ctx, PID, src.id)
    assert report1.status == "digested"

    ingested_notes = [n for n in store.list_notes(PID) if n.owner_id == house.id]
    assert len(ingested_notes) == 1
    original = ingested_notes[0]
    assert original.origin.producer == "ingest"

    # Human correction.
    result = correct_note(
        store, PID, original.id,
        [NoteSuccessorSpec(kind="description", body="A framed photograph is present in the house.")],
        reviewer="director", expected_revision=original.revision, expected_status=original.status,
    )
    assert result.original.status == "rejected"

    # Re-ingest the SAME source (force=True bypasses the "already digested" skip) -
    # the fake LLM re-extracts the identical original wording, matching the rejected
    # note's unchanged assertion exactly.
    report2 = ingest_source(ctx, PID, src.id, force=True)
    assert report2.status == "digested"

    # The rejected original is not resurrected, and no duplicate proposed note with
    # the old wording reappears.
    still_rejected = store.notes[(PID, original.id)]
    assert still_rejected.status == "rejected"
    assert still_rejected.review_reason == "corrected"
    matching_body = [n for n in store.list_notes(PID) if n.owner_id == house.id and n.body == original.body]
    assert len(matching_body) == 1
    assert matching_body[0].id == original.id

    # The human correction's successor note is completely untouched.
    corrected = store.notes[(PID, result.new_notes[0].id)]
    assert corrected.status == "proposed"
    assert corrected.revision == result.new_notes[0].revision
    assert corrected.body == "A framed photograph is present in the house."


# ==========================================================================================
# Caveat wording no longer points at a nonexistent path
# ==========================================================================================

def test_core_notes_caveat_names_a_real_action():
    assert "Correct this note" in CORE_NOTES_CAVEAT
    assert "supersession path" not in CORE_NOTES_CAVEAT


# ==========================================================================================
# Current location detail exposes scene-scoped notes (sceneRequirements)
# ==========================================================================================

def _detail(client, house):
    r = client.get(f"/projects/{PID}/locations/{house.id}")
    assert r.status_code == 200
    return r.json()


def test_location_detail_lists_every_linked_scene_with_conditional_notes(env):
    store, blobs, images, _, house, scene4, scene7, other_loc, other_scene, mixed = env
    scoped = Note(kind="constraint", body="The house tilts toward the water.", owner_id=house.id,
                  author="user", applicability=Applicability(scene_id=scene7.id),
                  provenance=[Provenance(source_id="a" * 64, page=13, quote="tilts toward the water")])
    # scene-conditional notes of non-brief kinds and an out-of-roster scene id
    stray = Note(kind="description", body="Something about a scene not linked here.", owner_id=house.id,
                 author="user", applicability=Applicability(scene_id=other_scene.id))
    store.put_notes(PID, [scoped, stray])
    client = make_client(store, blobs, images)
    d = _detail(client, house)

    assert [s["sceneId"] for s in d["sceneRequirements"]] == [scene4.id, scene7.id]   # full roster
    by_id = {s["sceneId"]: s for s in d["sceneRequirements"]}
    assert by_id[scene4.id]["notes"] == []                     # roster scene, nothing extracted
    assert all(s["linked"] for s in d["sceneRequirements"])
    assert by_id[scene7.id]["number"] == "7"
    assert by_id[scene7.id]["heading"] == scene7.name
    n = by_id[scene7.id]["notes"][0]
    assert (n["id"], n["kind"], n["revision"], n["status"]) == (scoped.id, "constraint", 1, "proposed")
    assert n["sceneId"] == scene7.id and n["includeDescendants"] is False
    assert n["citations"][0]["quote"] == "tilts toward the water"
    assert (n["ownerId"], n["owned"], n["inheritedFrom"], n["editable"]) == (house.id, True, None, True)

    # not implied to be a linked scene
    assert [s["sceneId"] for s in d["outOfRosterSceneRequirements"]] == [other_scene.id]
    oor = d["outOfRosterSceneRequirements"][0]
    assert oor["linked"] is False and oor["number"] is None
    assert oor["notes"][0]["id"] == stray.id
    # scene-scoped notes never leak into the standing lists
    standing = {x["id"] for k in ("descriptionNotes", "constraintNotes", "toneNotes") for x in d[k]}
    assert scoped.id not in standing and stray.id not in standing


def test_standing_note_corrected_into_scene_shows_in_detail_and_can_be_corrected_back(env):
    store, blobs, images, _, house, scene4, scene7, _, _, mixed = env
    client = make_client(store, blobs, images)
    assert mixed.id in {n["id"] for n in _detail(client, house)["descriptionNotes"]}

    r1 = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [{"kind": "description", "body": "Reflected light hides a face.", "sceneId": scene7.id}],
        "expectedRevision": 1, "expectedStatus": "proposed"})
    assert r1.status_code == 200
    scoped = r1.json()["newNotes"][0]

    d = _detail(client, house)
    assert mixed.id not in {n["id"] for n in d["descriptionNotes"]}
    s7 = next(s for s in d["sceneRequirements"] if s["sceneId"] == scene7.id)
    shown = next(n for n in s7["notes"] if n["id"] == scoped["id"])
    assert shown["editable"] is True and shown["revision"] == 1 and shown["status"] == "proposed"

    # correct it back to standing, using only what the detail response told us
    r2 = client.post(f"/projects/{PID}/notes/{shown['id']}/correct", json={
        "successors": [{"kind": "description", "body": "Reflected light hides a face."}],
        "expectedRevision": shown["revision"], "expectedStatus": shown["status"]})
    assert r2.status_code == 200
    back = r2.json()["newNotes"][0]
    d2 = _detail(client, house)
    assert back["id"] in {n["id"] for n in d2["descriptionNotes"]}
    assert shown["id"] not in {n["id"] for s in d2["sceneRequirements"] for n in s["notes"]}

    # provenance retained unchanged through both hops; lineage chains original -> scene -> standing
    quote = mixed.provenance[0].quote
    for note_id in (scoped["id"], back["id"]):
        prov = store.notes[(PID, note_id)].provenance[0]
        assert (prov.quote, prov.page, prov.source_id) == (quote, 6, "a" * 64)
    assert store.notes[(PID, scoped["id"])].supersedes_note_id == mixed.id
    assert store.notes[(PID, back["id"])].supersedes_note_id == scoped["id"]
    assert store.notes[(PID, mixed.id)].superseded_by_note_ids == [scoped["id"]]
    assert store.notes[(PID, scoped["id"])].superseded_by_note_ids == [back["id"]]
    assert store.notes[(PID, scoped["id"])].review_reason == "corrected"


# ==========================================================================================
# A superseded note cannot be reactivated through the ordinary review endpoint
# ==========================================================================================

def test_review_endpoint_cannot_reactivate_a_superseded_note(env):
    store, blobs, images, _, house, *_rest, mixed = env
    client = make_client(store, blobs, images)
    new_id = client.post(f"/projects/{PID}/notes/{mixed.id}/correct", json={
        "successors": [{"kind": "description", "body": "A framed photograph is present."}],
        "expectedRevision": 1, "expectedStatus": "proposed"}).json()["newNotes"][0]["id"]

    for decision, extra in (("confirmed", {}), ("rejected", {"reason": "not_useful"})):
        r = client.post(f"/projects/{PID}/notes/{mixed.id}/review", json={
            "decision": decision, "expectedRevision": 1, "expectedStatus": "rejected", **extra})
        assert r.status_code == 400 and "replaced by a correction" in r.json()["detail"]
    # a client still holding the old live view gets the ordinary stale-edit conflict
    r = client.post(f"/projects/{PID}/notes/{mixed.id}/review", json={
        "decision": "confirmed", "expectedRevision": 1, "expectedStatus": "proposed"})
    assert r.status_code == 409

    kept = store.notes[(PID, mixed.id)]
    assert (kept.status, kept.review_reason, kept.superseded_by_note_ids) == ("rejected", "corrected", [new_id])
    # the replacement is reviewable as normal
    r = client.post(f"/projects/{PID}/notes/{new_id}/review", json={
        "decision": "confirmed", "expectedRevision": 1, "expectedStatus": "proposed"})
    assert r.status_code == 200
