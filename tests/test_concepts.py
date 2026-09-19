"""Tests for concept-art versions and their explicit visual-direction approval
(harness.memory.concepts, harness.api.routes.concepts).

Every test uses MemoryStore/MemoryBlobs/MemoryImages - no real GCP or Gemini call is
made anywhere in this file. Mirrors the fixture/make_client style of
tests/test_reference_upload.py.
"""
from __future__ import annotations

import hashlib
import io
import threading

import pytest
from PIL import Image
from starlette.testclient import TestClient

from harness.api.main import create_app
from harness.api.settings import ApiSettings
from harness.memory.concepts import (
    CURRENT_SNAPSHOT_SCHEMA_VERSION, approval_staleness_reasons, build_approval_snapshot,
    lock_approval, promote_reference_to_concept, upload_concept_version,
)
from harness.memory.config import Settings
from harness.memory.curate import review_note
from harness.memory.ingest import Ctx as IngestCtx, register_file
from harness.memory.models import (
    ConceptApproval, ConditionalNoteSnapshot, Location, Note, Project, Provenance, Scene, Source,
)
from harness.memory.ports import ApprovalConflict, MemoryBlobs, MemoryImages, MemoryStore
from harness.memory.references import RefCtx, suggest_references, upload_reference_image

PID = "prj_concepts_test"


def _make_image(fmt="JPEG", size=(80, 80), color="blue"):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def _confirm(store, note, **extra):
    updated = note.touch(status="confirmed", reviewed_by="director", reviewed_at=note.created_at, **extra)
    store.put_note_if_current(PID, updated, expected_revision=note.revision, expected_status=note.status)
    return updated


@pytest.fixture
def harness_env():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Concept Test Film"))
    loc = Location(name="Old Mill", status="confirmed", author="agent")
    other_loc = Location(name="River Bend", status="confirmed", author="agent")
    store.put_entities(PID, [loc, other_loc])

    store.put_notes(PID, [
        Note(kind="description", body="A weathered stone mill beside the river.",
            owner_id=loc.id, author="user"),
        Note(kind="constraint", body="Keep the wheel intact.", owner_id=loc.id, author="user"),
    ])

    blobs = MemoryBlobs()
    ref_src = Source(id="a" * 64, filename="mill_ref.jpg", mime_type="image/jpeg", kind="image",
                     doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'a' * 64}.jpg",
                     status="digested", source_purpose="reference")
    store.put_source(PID, ref_src)
    blobs.put(ref_src.storage_path, b"ref-bytes", ref_src.mime_type)
    ref_note = Note(kind="reference_image", body="Stone mill with wheel.", owner_id=loc.id,
                    author="user", status="proposed", provenance=[Provenance(source_id=ref_src.id)])
    store.put_notes(PID, [ref_note])

    settings = Settings(gcp_project="test-gcp", bucket="test-bucket")
    images = MemoryImages({}, {}, {})
    return store, blobs, images, settings, loc, other_loc, ref_note


def make_client(store, blobs, images, *, max_upload_bytes=10_000_000):
    app = create_app(
        settings=Settings(gcp_project="t", bucket="b"),
        api_settings=ApiSettings(
            allowed_hosts=("testserver", "127.0.0.1", "localhost"),
            max_upload_bytes=max_upload_bytes,
        ),
        store=store, blobs=blobs, images=images,
    )
    return TestClient(app, base_url="http://testserver")


# ==========================================================================================
# Upload / list / preview
# ==========================================================================================

def test_upload_list_preview_lifecycle(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    img = _make_image(color="brown")

    r = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=mill_v1.jpg", content=img)
    assert r.status_code == 201
    body = r.json()
    version_id = body["id"]
    assert body["created"] is True
    assert body["approved"] is False
    assert body["image"] == f"/projects/{PID}/concepts/{version_id}/image"

    r_img = client.get(body["image"])
    assert r_img.status_code == 200
    assert r_img.content == img
    assert r_img.headers["content-type"] == "image/jpeg"

    r_list = client.get(f"/projects/{PID}/locations/{loc.id}/concepts")
    assert r_list.status_code == 200
    listing = r_list.json()
    assert listing["approvedVersionId"] is None
    assert [v["id"] for v in listing["versions"]] == [version_id]
    assert listing["versions"][0]["approved"] is False

    r_prev = client.get(
        f"/projects/{PID}/locations/{loc.id}/approval/preview",
        params={"conceptVersionId": version_id, "referenceIds": [confirmed.id]},
    )
    assert r_prev.status_code == 200
    preview = r_prev.json()
    assert preview["conceptVersionId"] == version_id
    assert len(preview["coreNotes"]) == 1       # the description note
    assert len(preview["physicalNotes"]) == 1   # the constraint note
    assert preview["coreNotesCaveat"]           # always present, static text
    assert [r["noteId"] for r in preview["references"]] == [confirmed.id]
    assert preview["references"][0]["status"] == "confirmed"
    assert isinstance(preview["contextToken"], str) and preview["contextToken"]


# ==========================================================================================
# Project / location isolation
# ==========================================================================================

def test_project_and_location_isolation(harness_env):
    store, blobs, images, _, loc, other_loc, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="green")

    r = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=a.jpg", content=img)
    version_id = r.json()["id"]

    # Not visible under a different location in the same project
    r_other = client.get(f"/projects/{PID}/locations/{other_loc.id}/concepts")
    assert r_other.json()["versions"] == []

    # Not fetchable image-wise from an unknown project (store keys by (project, id))
    r_bad_project = client.get(f"/projects/unknown_prj/concepts/{version_id}/image")
    assert r_bad_project.status_code == 404

    # Unknown location -> 404
    r_bad_loc = client.post(
        f"/projects/{PID}/locations/unknown_loc/concepts/upload?filename=a.jpg", content=img,
    )
    assert r_bad_loc.status_code == 404


# ==========================================================================================
# Duplicate retries / dedup
# ==========================================================================================

def test_duplicate_upload_is_idempotent_no_new_version(harness_env):
    store, blobs, images, _, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="gray")

    r1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=first.jpg", content=img)
    v1 = r1.json()["id"]

    r2 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=second.jpg", content=img)
    assert r2.json()["created"] is False
    assert r2.json()["id"] == v1

    r_list = client.get(f"/projects/{PID}/locations/{loc.id}/concepts")
    assert len(r_list.json()["versions"]) == 1


def test_duplicate_upload_after_approval_preserves_approval(harness_env):
    store, blobs, images, _, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="orange")

    r1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=first.jpg", content=img)
    v1 = r1.json()["id"]

    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": []}).json()
    lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })
    assert lock.status_code == 201

    # Re-upload identical bytes: idempotent, approval untouched
    r2 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=dup.jpg", content=img)
    assert r2.json()["created"] is False
    assert r2.json()["approved"] is True

    approval_after = client.get(f"/projects/{PID}/locations/{loc.id}/approval").json()
    assert approval_after["revision"] == 1
    assert approval_after["approval"]["conceptVersionId"] == v1


# ==========================================================================================
# Candidate changes preserve prior approval
# ==========================================================================================

def test_new_candidate_upload_preserves_prior_approval(harness_env):
    store, blobs, images, _, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img1, img2 = _make_image(color="red"), _make_image(color="blue")

    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg", content=img1).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": []}).json()
    client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [], "contextToken": prev["contextToken"], "expectedRevision": 0,
    })

    # Upload a second, different candidate
    v2 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v2.jpg", content=img2).json()["id"]
    assert v2 != v1

    listing = client.get(f"/projects/{PID}/locations/{loc.id}/concepts").json()
    assert listing["approvedVersionId"] == v1
    ids = {v["id"]: v["approved"] for v in listing["versions"]}
    assert ids[v1] is True
    assert ids[v2] is False

    approval = client.get(f"/projects/{PID}/locations/{loc.id}/approval").json()
    assert approval["approval"]["conceptVersionId"] == v1
    assert approval["revision"] == 1


# ==========================================================================================
# Issue 1: reference selection must be confirmed + applicable
# ==========================================================================================

def test_selecting_a_proposed_reference_is_rejected(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    client = make_client(store, blobs, images)
    assert ref_note.status == "proposed"
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]

    r_prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                        params={"conceptVersionId": v1, "referenceIds": [ref_note.id]})
    assert r_prev.status_code == 400
    assert "not confirmed" in r_prev.json()["detail"]

    r_lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [ref_note.id], "contextToken": "irrelevant",
        "expectedRevision": 0,
    })
    assert r_lock.status_code == 400
    assert "not confirmed" in r_lock.json()["detail"]

    # Never auto-confirmed by the attempt
    assert store.notes[(PID, ref_note.id)].status == "proposed"


def test_selecting_a_rejected_reference_is_rejected(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    rejected = ref_note.touch(status="rejected", review_reason="not_useful")
    store.put_note_if_current(PID, rejected, expected_revision=ref_note.revision, expected_status="proposed")
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]

    r_prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                        params={"conceptVersionId": v1, "referenceIds": [ref_note.id]})
    assert r_prev.status_code == 400
    assert "not confirmed" in r_prev.json()["detail"]


def test_selecting_an_unrelated_reference_is_rejected(harness_env):
    """A reference confirmed for a *different* location must not be selectable here -
    ownership/inheritance rules still apply on top of the confirmed-status check."""
    store, blobs, images, _, loc, other_loc, _ = harness_env
    other_src = Source(id="b" * 64, filename="river.jpg", mime_type="image/jpeg", kind="image",
                       doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'b' * 64}.jpg",
                       status="digested", source_purpose="reference")
    store.put_source(PID, other_src)
    other_ref = Note(kind="reference_image", body="River bend view.", owner_id=other_loc.id,
                     author="user", status="confirmed", provenance=[Provenance(source_id=other_src.id)])
    store.put_notes(PID, [other_ref])

    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]

    r_prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                        params={"conceptVersionId": v1, "referenceIds": [other_ref.id]})
    assert r_prev.status_code == 400
    assert "not a reference applicable" in r_prev.json()["detail"]


def test_confirmed_and_applicable_reference_is_accepted_and_never_auto_confirmed_elsewhere(harness_env):
    """Confirming happens once, independently, through the normal review endpoint -
    selecting a reference for approval never confirms a second, different note."""
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    # A second, still-proposed reference on the same location must stay untouched.
    other_src = Source(id="c" * 64, filename="mill2.jpg", mime_type="image/jpeg", kind="image",
                       doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'c' * 64}.jpg",
                       status="digested", source_purpose="reference")
    store.put_source(PID, other_src)
    bystander = Note(kind="reference_image", body="Second angle.", owner_id=loc.id, author="user",
                     status="proposed", provenance=[Provenance(source_id=other_src.id)])
    store.put_notes(PID, [bystander])

    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    r_prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                        params={"conceptVersionId": v1, "referenceIds": [confirmed.id]})
    assert r_prev.status_code == 200
    r_lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id],
        "contextToken": r_prev.json()["contextToken"], "expectedRevision": 0,
    })
    assert r_lock.status_code == 201
    assert store.notes[(PID, bystander.id)].status == "proposed"   # untouched


# ==========================================================================================
# Issue 1: a saved historical approval stays readable when a selected reference
# is later rejected/removed/loses applicability - isStale + reasons, not a failure
# ==========================================================================================

def test_approval_stays_readable_after_selected_reference_is_rejected(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })
    assert lock.status_code == 201

    # The reference is later rejected outright.
    r_rej = client.post(f"/projects/{PID}/notes/{confirmed.id}/review", json={
        "decision": "rejected", "reason": "not_useful", "by": "director",
        "expectedRevision": confirmed.revision, "expectedStatus": "confirmed",
    })
    assert r_rej.status_code == 200

    state = client.get(f"/projects/{PID}/locations/{loc.id}/approval")
    assert state.status_code == 200          # never fails the whole retrieval
    body = state.json()
    assert body["approval"]["id"] == lock.json()["approval"]["id"]
    assert body["approval"]["references"][0]["status"] == "confirmed"   # stored snapshot unchanged
    assert body["isStale"] is True
    assert any("no longer confirmed" in r and confirmed.id in r for r in body["staleReasons"])


def test_approval_stays_readable_after_location_rejected_or_reference_removed(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })

    # The reference note is deleted outright (loses applicability entirely).
    store.delete_notes(PID, [confirmed.id])
    reasons = approval_staleness_reasons(store, PID, store.get_current_approval(PID, loc.id))
    assert any("no longer applies" in r for r in reasons)

    state = client.get(f"/projects/{PID}/locations/{loc.id}/approval")
    assert state.status_code == 200
    assert state.json()["isStale"] is True


def test_approval_stays_readable_after_location_rejected(harness_env):
    store, blobs, images, _, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": []}).json()
    client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })

    rejected_loc = loc.touch(status="rejected")
    store.put_entities(PID, [rejected_loc])

    reasons = approval_staleness_reasons(store, PID, store.get_current_approval(PID, loc.id))
    assert reasons == ["the approved location was rejected"]


# ==========================================================================================
# Issue 2: validation-vs-commit boundary
# ==========================================================================================

def test_stale_context_token_returns_409(harness_env):
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image(color="purple")).json()["id"]

    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    stale_token = prev["contextToken"]

    # A brief note changes underneath (a new one is added).
    store.put_notes(PID, [Note(kind="tone", body="Melancholy, sun-bleached.", owner_id=loc.id, author="user")])

    r = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": stale_token,
        "expectedRevision": 0,
    })
    assert r.status_code == 409

    # A fresh preview reflects the new brief and locks cleanly
    fresh = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                       params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    assert len(fresh["coreNotes"]) == 2          # description + the new tone note
    assert len(fresh["physicalNotes"]) == 1      # the constraint note
    r2 = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": fresh["contextToken"],
        "expectedRevision": 0,
    })
    assert r2.status_code == 201


def test_idempotent_lock_retry_is_a_no_op(harness_env):
    store, blobs, _, settings, loc, _, _ = harness_env
    img = _make_image(color="cyan")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")

    snapshot = build_approval_snapshot(store, PID, loc.id, version.id, [])
    approval1, created1 = lock_approval(store, PID, loc.id, concept_version_id=version.id,
                                        reference_ids=[], context_token=snapshot.context_token,
                                        expected_revision=0, locked_by="director")
    assert created1 is True
    assert approval1.revision == 1

    # Retry with the same (now stale) expectedRevision=0 the client started with -
    # succeeds as a no-op because the content is identical to what's already current.
    # The idempotency check runs inside the same atomic operation as the CAS (see
    # ports.Store.put_approval_if_current) - never a separate pre-check outside it.
    approval2, created2 = lock_approval(store, PID, loc.id, concept_version_id=version.id,
                                        reference_ids=[], context_token=snapshot.context_token,
                                        expected_revision=0, locked_by="director")
    assert created2 is False
    assert approval2.id == approval1.id
    assert approval2.revision == 1


def test_idempotency_check_verifies_the_actual_token_not_just_the_revision(harness_env):
    """put_approval_if_current's idempotency short-circuit must compare the current
    approval's own context_token against the *submitted* approval's token, not merely
    assume equality from the caller's expected_revision. Two different DIFFERENT
    packages submitted with revision=0 must not both be treated as idempotent."""
    store, blobs, _, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    img = _make_image(color="magenta")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")

    snap_a = build_approval_snapshot(store, PID, loc.id, version.id, [])
    snap_b = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id])
    assert snap_a.context_token != snap_b.context_token

    approval_a, created_a = lock_approval(store, PID, loc.id, concept_version_id=version.id,
                                          reference_ids=[], context_token=snap_a.context_token,
                                          expected_revision=0, locked_by="a")
    assert created_a is True

    # Submitting the DIFFERENT package b with the now-stale expected_revision=0 must
    # be a genuine conflict, not a false "idempotent success" just because revision=0
    # was also what the caller believed for a (a bug would compare only revisions).
    with pytest.raises(ApprovalConflict):
        lock_approval(store, PID, loc.id, concept_version_id=version.id,
                     reference_ids=[confirmed.id], context_token=snap_b.context_token,
                     expected_revision=0, locked_by="b")
    assert store.get_current_approval(PID, loc.id).id == approval_a.id


def test_concurrent_locks_of_different_content_one_wins_one_conflicts(harness_env):
    store, blobs, _, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    img = _make_image(color="magenta")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")

    snap_a = build_approval_snapshot(store, PID, loc.id, version.id, [])
    snap_b = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id])
    assert snap_a.context_token != snap_b.context_token

    results = {}
    barrier = threading.Barrier(2)

    def attempt(name, snapshot, reference_ids):
        barrier.wait()
        try:
            approval, created = lock_approval(
                store, PID, loc.id, concept_version_id=version.id, reference_ids=reference_ids,
                context_token=snapshot.context_token, expected_revision=0, locked_by=name,
            )
            results[name] = ("ok", approval, created)
        except ApprovalConflict as exc:
            results[name] = ("conflict", exc, None)

    t1 = threading.Thread(target=attempt, args=("a", snap_a, []))
    t2 = threading.Thread(target=attempt, args=("b", snap_b, [confirmed.id]))
    t1.start(); t2.start()
    t1.join(timeout=2); t2.join(timeout=2)

    outcomes = [results["a"][0], results["b"][0]]
    assert sorted(outcomes) == ["conflict", "ok"]

    current = store.get_current_approval(PID, loc.id)
    assert current is not None
    assert current.revision == 1
    winner = "a" if results["a"][0] == "ok" else "b"
    assert current.id == results[winner][1].id


def test_concurrent_locks_of_identical_content_both_succeed_without_conflict(harness_env):
    """Two concurrent requests locking the exact same, already-validated package must
    both succeed (one created=True, one created=False) - the idempotency check inside
    put_approval_if_current, not a race on expected_revision, is what makes this safe."""
    store, blobs, _, settings, loc, _, _ = harness_env
    img = _make_image(color="lime")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")
    snapshot = build_approval_snapshot(store, PID, loc.id, version.id, [])

    results = {}
    barrier = threading.Barrier(2)

    def attempt(name):
        barrier.wait()
        approval, created = lock_approval(
            store, PID, loc.id, concept_version_id=version.id, reference_ids=[],
            context_token=snapshot.context_token, expected_revision=0, locked_by=name,
        )
        results[name] = (approval, created)

    t1 = threading.Thread(target=attempt, args=("a",))
    t2 = threading.Thread(target=attempt, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=2); t2.join(timeout=2)

    assert len(results) == 2
    created_flags = sorted(c for _, c in results.values())
    assert created_flags == [False, True]
    assert results["a"][0].id == results["b"][0].id

    current = store.get_current_approval(PID, loc.id)
    assert current.revision == 1


def test_edit_between_validation_and_commit_does_not_change_stored_snapshot(harness_env, monkeypatch):
    """Simulates a concurrent edit landing after build_approval_snapshot validates the
    token but before Store.put_approval_if_current commits. The stored package must
    reflect exactly what was validated - lock_approval must never re-read/rebuild
    content after checking its token (see lock_approval's docstring for why: this
    system approves the previewed snapshot, not whatever is latest at commit time)."""
    store, blobs, _, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    img = _make_image(color="indigo")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")

    snapshot = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id])
    original_revision = snapshot.references[0].revision
    original_guidance = snapshot.references[0].guidance

    real_put = store.put_approval_if_current

    def edit_then_put(project_id, location_id, approval, expected_revision):
        current = store.notes[(project_id, confirmed.id)]
        review_note(store, project_id, confirmed.id, "confirmed", reviewer="someone-else",
                   guidance="a totally different guidance, added mid-flight",
                   expected_revision=current.revision, expected_status=current.status)
        return real_put(project_id, location_id, approval, expected_revision)

    monkeypatch.setattr(store, "put_approval_if_current", edit_then_put)

    approval, created = lock_approval(store, PID, loc.id, concept_version_id=version.id,
                                      reference_ids=[confirmed.id], context_token=snapshot.context_token,
                                      expected_revision=0, locked_by="director")
    assert created is True
    assert approval.references[0].revision == original_revision
    assert approval.references[0].guidance == original_guidance
    assert approval.references[0].guidance != "a totally different guidance, added mid-flight"

    # The live note really did change - proving this isn't stale-by-accident, but a
    # genuine post-validation edit the commit boundary does not re-check.
    live = store.notes[(PID, confirmed.id)]
    assert live.guidance == "a totally different guidance, added mid-flight"
    assert live.revision == original_revision + 1

    # ...and that drift is now visible after the fact via staleness, never hidden.
    reasons = approval_staleness_reasons(store, PID, approval)
    assert any("changed since this was approved" in r for r in reasons)


def test_context_token_is_canonical_and_excludes_timestamps(harness_env):
    store, blobs, _, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    other_src = Source(id="d" * 64, filename="mill3.jpg", mime_type="image/jpeg", kind="image",
                       doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'d' * 64}.jpg",
                       status="digested", source_purpose="reference")
    store.put_source(PID, other_src)
    second_ref = Note(kind="reference_image", body="Second angle.", owner_id=loc.id, author="user",
                      status="proposed", provenance=[Provenance(source_id=other_src.id)])
    store.put_notes(PID, [second_ref])
    second_ref = _confirm(store, second_ref)

    img = _make_image(color="salmon")
    version, _ = upload_concept_version(store, blobs, settings, PID, loc.id, img, "v.jpg")

    snap1 = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id, second_ref.id])

    # Touching a brief note's updated_at without changing revision/status must not
    # move the token - only (id, revision, status) are hashed, never timestamps.
    brief_note = next(n for n in store.notes.values()
                      if n.owner_id == loc.id and n.kind == "description")
    store.notes[(PID, brief_note.id)] = brief_note.touch()
    assert store.notes[(PID, brief_note.id)].updated_at != brief_note.updated_at   # the touch did something
    snap2 = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id, second_ref.id])
    assert snap2.context_token == snap1.context_token

    # Reference id ordering must not matter - the token sorts internally.
    snap_ba = build_approval_snapshot(store, PID, loc.id, version.id, [second_ref.id, confirmed.id])
    assert snap_ba.context_token == snap1.context_token

    # A genuine content change (new brief note) must still move the token.
    store.put_notes(PID, [Note(kind="tone", body="Golden hour, dust motes.", owner_id=loc.id, author="user")])
    snap3 = build_approval_snapshot(store, PID, loc.id, version.id, [confirmed.id, second_ref.id])
    assert snap3.context_token != snap1.context_token


# ==========================================================================================
# Issue 1/3 supporting: locking never mutates notes
# ==========================================================================================

def test_locking_does_not_confirm_notes_or_approve_references(harness_env):
    """Locking only reads and snapshots; it must never itself change a Note's
    status/revision, and reviewing a note independently afterward must still work
    normally (its own review state is unaffected by having been in a snapshot)."""
    store, blobs, images, _, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    img = _make_image(color="olive")
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg", content=img).json()["id"]

    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })
    assert lock.status_code == 201

    stored_note = store.notes[(PID, confirmed.id)]
    assert stored_note.status == "confirmed"          # unchanged by locking (already was)
    assert stored_note.revision == confirmed.revision

    for n in store.notes.values():
        if n.id != confirmed.id and n.owner_id == loc.id and n.kind != "reference_image":
            assert n.status == "proposed"            # brief notes untouched too

    # Independent review of a different note still works exactly as before
    other_note_id = next(n.id for n in store.notes.values()
                         if n.owner_id == loc.id and n.kind == "description")
    r_review = client.post(f"/projects/{PID}/notes/{other_note_id}/review", json={
        "decision": "confirmed", "by": "director",
        "expectedRevision": 1, "expectedStatus": "proposed",
    })
    assert r_review.status_code == 200
    assert r_review.json()["status"] == "confirmed"


# ==========================================================================================
# Issue 3: concept/source cross-purpose behavior
# ==========================================================================================

def test_concept_source_excluded_from_ingestion(harness_env):
    store, blobs, images, settings, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="black")
    client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg", content=img)

    sid = hashlib.sha256(img).hexdigest()
    src = store.get_source(PID, sid)
    assert src.source_purpose == "concept"
    assert src.is_ingest_eligible is False

    r_srcs = client.get(f"/projects/{PID}/sources")
    assert sid not in [s["id"] for s in r_srcs.json()]

    r_ing = client.post(f"/projects/{PID}/ingest", json={"sourceId": sid})
    assert r_ing.status_code == 422
    assert "concept-art" in r_ing.json()["detail"]


def test_concept_upload_then_sources_upload_becomes_ingest_eligible(harness_env):
    """Concept upload -> identical bytes uploaded through Sources: the source becomes
    visible in GET /sources and ingest-eligible, with ingestion state initialized
    correctly (status reset to 'uploaded', digest_version cleared) - the exact same
    promotion path a reference-purpose source gets, applied generically."""
    store, blobs, images, settings, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="fuchsia")
    sid = hashlib.sha256(img).hexdigest()

    r_concept = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=concept.jpg", content=img)
    assert r_concept.status_code == 201
    version_id = r_concept.json()["id"]

    before = store.get_source(PID, sid)
    assert before.source_purpose == "concept"
    assert before.is_ingest_eligible is False

    r_sources = client.post(f"/projects/{PID}/sources?filename=script_page.jpg", content=img)
    assert r_sources.status_code == 201
    assert r_sources.json()["created"] is False   # bytes were not new

    after = store.get_source(PID, sid)
    assert after.source_purpose == "both"
    assert after.is_ingest_eligible is True
    assert after.status == "uploaded"          # ingestion state initialized correctly
    assert after.digest_version is None

    r_srcs = client.get(f"/projects/{PID}/sources")
    assert sid in [s["id"] for s in r_srcs.json()]
    r_ing = client.post(f"/projects/{PID}/ingest", json={"sourceId": sid})
    assert r_ing.status_code == 201

    # The concept version itself is completely unaffected by the promotion.
    assert store.get_concept_version(PID, version_id) is not None
    listing = client.get(f"/projects/{PID}/locations/{loc.id}/concepts").json()
    assert [v["id"] for v in listing["versions"]] == [version_id]


def test_ingest_upload_then_concept_upload_preserves_processing_state(harness_env):
    """Ingestion input -> concept upload: the existing source's processing
    state/provenance (status, digest_version, filename, storage_path) must be
    preserved untouched - concept upload never overwrites an already-registered
    ingest-purpose source."""
    store, blobs, images, settings, loc, _, _ = harness_env
    img = _make_image(color="chartreuse")
    sid = hashlib.sha256(img).hexdigest()

    ingest_ctx = IngestCtx(store=store, blobs=blobs, llm=None, settings=settings)
    src, created = register_file(ingest_ctx, PID, img, "episode_1.jpg")
    assert created is True
    assert src.source_purpose == "ingest"

    # Simulate ingestion having already run and digested it.
    digested = src.touch(status="digested", digest_version="digest-v9")
    store.put_source(PID, digested)

    client = make_client(store, blobs, images)
    r_concept = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=concept.jpg", content=img)
    assert r_concept.status_code == 201

    after = store.get_source(PID, sid)
    assert after.source_purpose == "ingest"        # unchanged - never demoted or reset
    assert after.status == "digested"              # processing state preserved
    assert after.digest_version == "digest-v9"      # provenance preserved
    assert after.filename == "episode_1.jpg"        # original filename preserved
    assert after.is_ingest_eligible is True

    r_srcs = client.get(f"/projects/{PID}/sources")
    assert sid in [s["id"] for s in r_srcs.json()]


def test_reference_and_concept_upload_share_bytes_without_ingest_eligibility(harness_env):
    """Reference upload <-> concept upload, both orders: separate location records
    (a reference Note here, a ConceptVersion there) sharing the same underlying Source
    by content hash, with source_purpose never becoming ingest-eligible merely from
    combining the two."""
    store, blobs, images, settings, loc, other_loc, _ = harness_env
    img = _make_image(color="turquoise")
    sid = hashlib.sha256(img).hexdigest()

    # Order 1: reference first, then concept (different location, same bytes).
    ref_note, ref_created = upload_reference_image(store, blobs, settings, PID, loc.id, img, "ref.jpg")
    assert ref_created is True
    version, version_created = upload_concept_version(store, blobs, settings, PID, other_loc.id, img, "concept.jpg")
    assert version_created is True

    src = store.get_source(PID, sid)
    assert src.source_purpose in ("reference", "concept")   # whichever wrote first wins; either is non-ingest
    assert src.is_ingest_eligible is False

    assert version.source_id == sid
    assert ref_note.provenance[0].source_id == sid
    assert ref_note.owner_id == loc.id
    assert version.location_id == other_loc.id

    client = make_client(store, blobs, images)
    r_srcs = client.get(f"/projects/{PID}/sources")
    assert sid not in [s["id"] for s in r_srcs.json()]


def test_concept_then_reference_upload_share_bytes_without_ingest_eligibility(harness_env):
    """Same as above, reverse order: concept first, then reference."""
    store, blobs, images, settings, loc, other_loc, _ = harness_env
    img = _make_image(color="crimson")
    sid = hashlib.sha256(img).hexdigest()

    version, version_created = upload_concept_version(store, blobs, settings, PID, loc.id, img, "concept.jpg")
    assert version_created is True
    ref_note, ref_created = upload_reference_image(store, blobs, settings, PID, other_loc.id, img, "ref.jpg")
    assert ref_created is True

    src = store.get_source(PID, sid)
    assert src.is_ingest_eligible is False
    assert version.source_id == sid
    assert ref_note.provenance[0].source_id == sid


# ==========================================================================================
# Exclusion from reference-pipeline cleanup
# ==========================================================================================

def test_concept_version_excluded_from_reference_listing_and_pipeline_cleanup(harness_env):
    """A concept version must never appear as a reference card, and the references
    pipeline's own stale-note replacement (which only ever touches Notes) must leave
    concept versions/sources completely alone."""
    store, blobs, images, settings, loc, _, _ = harness_env
    client = make_client(store, blobs, images)
    img = _make_image(color="white")
    version_id = client.post(
        f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg", content=img,
    ).json()["id"]

    r_refs = client.get(f"/projects/{PID}/locations/{loc.id}/references")
    assert version_id not in [r["id"] for r in r_refs.json()["references"]]

    from harness.memory.schemas import VocabularyOut

    class DummyLLM:
        model_id = "test-model"
        def generate(self, *, system, parts, schema, fast=False, thinking_level=None):
            if schema is VocabularyOut:
                return VocabularyOut(terms=[])
            return None

    ref_ctx = RefCtx(store=store, blobs=blobs, llm=DummyLLM(), images=images, settings=settings)
    suggest_references(ref_ctx, PID, loc.id, per_term=1, max_images=1, dry_run=False, bypass_verification=True)

    # Concept version and its source survive untouched
    assert store.get_concept_version(PID, version_id) is not None
    sid = hashlib.sha256(img).hexdigest()
    assert store.get_source(PID, sid) is not None
    listing = client.get(f"/projects/{PID}/locations/{loc.id}/concepts").json()
    assert [v["id"] for v in listing["versions"]] == [version_id]


# ==========================================================================================
# Complete scene coverage (item 2) - reproduces the "Anshul's House" report: a location
# linked to scenes 4, 7, 10 where only scene 7 has a scene-conditional note.
# ==========================================================================================

@pytest.fixture
def house_env():
    """Mirrors the real Anshul's House shape: one location, three linked scenes, only
    the middle one carrying a scene-conditional note. The other two must still show up
    in scene coverage with notes=[], not be silently absent."""
    store = MemoryStore()
    store.put_project(Project(id=PID, name="House Test Film"))
    house = Location(name="Anshul's House", status="confirmed", author="agent")
    store.put_entities(PID, [house])

    scene4 = Scene(name="4. INT. ANSHUL'S HOUSE - NIGHT", number="4", location_ids=[house.id], author="agent")
    scene7 = Scene(name="7. INT./EXT. PANDIT'S DREAM - NIGHT", number="7", location_ids=[house.id], author="agent")
    scene10 = Scene(name="10. EXT. ANSHUL'S HOUSE / DEVGRAM LANES - MORNING", number="10",
                    location_ids=[house.id], author="agent")
    store.put_entities(PID, [scene4, scene7, scene10])

    house_desc = Note(kind="description", body="A warm, modest wooden home. Old beams, worn furniture.",
                      owner_id=house.id, author="user")
    # Reproduces the reported "photograph/reflection" note: same kind, same
    # unconditional applicability as the general house description above - nothing in
    # the stored data distinguishes "standing fact" from "scene-specific visual cue".
    photo_note = Note(
        kind="description",
        body="Family photographs decorate the space, specifically one of a younger man "
             "beside another man whose face is completely obscured by reflected light.",
        owner_id=house.id, author="user",
    )
    scene7_note = Note(kind="description", body="The dream distorts the house's geometry.",
                       owner_id=house.id, author="user",
                       applicability={"scene_id": scene7.id, "include_descendants": False})
    store.put_notes(PID, [house_desc, photo_note, scene7_note])

    blobs = MemoryBlobs()
    settings = Settings(gcp_project="test-gcp", bucket="test-bucket")
    images = MemoryImages({}, {}, {})
    return store, blobs, images, settings, house, scene4, scene7, scene10, house_desc, photo_note, scene7_note


def test_complete_scene_coverage_includes_note_less_scenes(house_env):
    store, blobs, images, settings, house, scene4, scene7, scene10, *_ = house_env
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{house.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]

    prev = client.get(f"/projects/{PID}/locations/{house.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": []}).json()
    scenes = {s["sceneId"]: s for s in prev["sceneRequirements"]}
    assert set(scenes) == {scene4.id, scene7.id, scene10.id}
    assert scenes[scene4.id]["notes"] == []
    assert scenes[scene4.id]["number"] == "4"
    assert scenes[scene10.id]["notes"] == []
    assert scenes[scene10.id]["number"] == "10"
    assert len(scenes[scene7.id]["notes"]) == 1
    assert scenes[scene7.id]["heading"] == "7. INT./EXT. PANDIT'S DREAM - NIGHT"

    lock = client.post(f"/projects/{PID}/locations/{house.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [], "contextToken": prev["contextToken"], "expectedRevision": 0,
    })
    assert lock.status_code == 201
    approval = lock.json()["approval"]
    assert approval["sceneCoverageComplete"] is True
    locked_scenes = {s["sceneId"] for s in approval["sceneRequirements"]}
    assert locked_scenes == {scene4.id, scene7.id, scene10.id}

    state = client.get(f"/projects/{PID}/locations/{house.id}/approval").json()
    assert state["isStale"] is False
    assert {s["sceneId"] for s in state["approval"]["sceneRequirements"]} == {scene4.id, scene7.id, scene10.id}


# ==========================================================================================
# Conditional-note separation / sectioning (item 1)
# ==========================================================================================

def test_notes_are_sectioned_by_kind_and_applicability_without_duplication(house_env):
    store, blobs, images, settings, house, scene4, scene7, scene10, house_desc, photo_note, scene7_note = house_env
    snapshot = build_approval_snapshot(store, PID, house.id,
                                       upload_concept_version(store, blobs, settings, PID, house.id,
                                                             _make_image(), "v.jpg")[0].id, [])
    core_ids = {n.id for n in snapshot.brief_notes if n.kind in ("description", "tone")}
    physical_ids = {n.id for n in snapshot.brief_notes if n.kind == "constraint"}
    scene_note_ids = {n.id for s in snapshot.scene_requirements for n in s.notes}

    assert core_ids == {house_desc.id, photo_note.id}
    assert physical_ids == set()
    assert scene_note_ids == {scene7_note.id}
    # No note appears in both the unconditional brief and a scene bucket.
    assert core_ids.isdisjoint(scene_note_ids)


def test_ambiguous_description_note_is_preserved_not_dropped_or_reclassified(house_env):
    """The reported photograph/reflection note: structurally identical (kind, no
    scene_id) to the plain house-description note. It must land in coreNotes intact,
    with citations/id/revision preserved, and the section must carry an explicit
    caveat rather than the system silently guessing a scene/camera classification."""
    store, blobs, images, settings, house, scene4, scene7, scene10, house_desc, photo_note, scene7_note = house_env
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{house.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{house.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": []}).json()

    core_by_id = {n["id"]: n for n in prev["coreNotes"]}
    assert photo_note.id in core_by_id
    assert core_by_id[photo_note.id]["body"] == photo_note.body
    assert core_by_id[photo_note.id]["revision"] == photo_note.revision
    assert core_by_id[photo_note.id]["status"] == "proposed"
    assert prev["coreNotesCaveat"]
    assert "scene-specific" in prev["coreNotesCaveat"] or "camera" in prev["coreNotesCaveat"]
    # Not fabricated into a scene-specific requirement anywhere.
    assert all(photo_note.id not in [n["id"] for n in s["notes"]] for s in prev["sceneRequirements"])


# ==========================================================================================
# Context token covers the scene roster (item 2)
# ==========================================================================================

def test_context_token_changes_with_scene_relinking_and_heading_changes(house_env):
    store, blobs, images, settings, house, scene4, scene7, scene10, *_ = house_env
    version, _ = upload_concept_version(store, blobs, settings, PID, house.id, _make_image(), "v.jpg")
    baseline = build_approval_snapshot(store, PID, house.id, version.id, [])

    # Renaming a scene's heading moves the token even though no notes changed.
    renamed = scene4.touch(name="4. INT. ANSHUL'S HOUSE - LATER THAT NIGHT")
    store.put_entities(PID, [renamed])
    after_rename = build_approval_snapshot(store, PID, house.id, version.id, [])
    assert after_rename.context_token != baseline.context_token

    # Restore, then link a brand-new scene with zero notes - still must move the token.
    store.put_entities(PID, [scene4])
    restored = build_approval_snapshot(store, PID, house.id, version.id, [])
    assert restored.context_token == baseline.context_token
    scene99 = Scene(name="99. EXT. NEW SCENE - DAY", number="99", location_ids=[house.id], author="agent")
    store.put_entities(PID, [scene99])
    after_new_scene = build_approval_snapshot(store, PID, house.id, version.id, [])
    assert after_new_scene.context_token != baseline.context_token
    assert any(s.scene_id == scene99.id for s in after_new_scene.scene_requirements)


# ==========================================================================================
# Backward compatibility: old (pre-full-roster) approvals stay readable (item 5)
# ==========================================================================================

def test_legacy_approval_is_readable_without_backfilling_missing_scene_coverage(house_env):
    """Directly constructs an approval in the OLD shape: snapshot_schema_version
    omitted (defaults to 1), and brief_conditional containing only scene 7 (the
    scenes-with-notes-only behavior the old code had) with no number/heading. GET
    .../approval must never crash, must never silently backfill scenes 4/10 from
    current retrieval, and must clearly flag the gap."""
    store, blobs, images, settings, house, scene4, scene7, scene10, house_desc, photo_note, scene7_note = house_env
    version, _ = upload_concept_version(store, blobs, settings, PID, house.id, _make_image(), "legacy.jpg")

    legacy = ConceptApproval(
        location_id=house.id, concept_version_id=version.id, concept_source_id=version.source_id,
        concept_filename=version.filename,
        brief_notes=[house_desc, photo_note],
        brief_conditional=[ConditionalNoteSnapshot(scene_id=scene7.id, label="7 · dream", notes=[scene7_note])],
        brief_inherited=[], brief_ancestors=[], superseded_sources=[], references=[],
        context_token="legacy-token-not-recomputable",
        # snapshot_schema_version omitted -> defaults to 1 (legacy)
    )
    assert legacy.snapshot_schema_version == 1
    store.approvals[(PID, legacy.id)] = legacy
    store.current_approval[(PID, house.id)] = legacy.id

    client = make_client(store, blobs, images)
    r = client.get(f"/projects/{PID}/locations/{house.id}/approval")
    assert r.status_code == 200
    body = r.json()
    assert body["approval"]["id"] == legacy.id
    # Old data returned exactly as stored - not backfilled with scenes 4/10.
    assert {s["sceneId"] for s in body["approval"]["sceneRequirements"]} == {scene7.id}
    assert body["approval"]["sceneCoverageComplete"] is False
    assert body["isStale"] is True
    assert any("predates full scene-roster coverage" in r for r in body["staleReasons"])
    # Legacy status doesn't suppress unrelated, still-meaningful diagnostics.
    assert not any("newly linked scene" in r for r in body["staleReasons"])


def test_current_schema_constant_is_stamped_on_fresh_locks(house_env):
    store, blobs, images, settings, house, *_ = house_env
    version, _ = upload_concept_version(store, blobs, settings, PID, house.id, _make_image(), "v.jpg")
    snapshot = build_approval_snapshot(store, PID, house.id, version.id, [])
    approval, _ = lock_approval(store, PID, house.id, concept_version_id=version.id, reference_ids=[],
                                context_token=snapshot.context_token, expected_revision=0, locked_by="director")
    assert approval.snapshot_schema_version == CURRENT_SNAPSHOT_SCHEMA_VERSION
    assert approval_staleness_reasons(store, PID, approval) == []


# ==========================================================================================
# Promote a reference to a concept candidate (item 3)
# ==========================================================================================

def test_promote_proposed_reference_creates_candidate_with_provenance(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    assert ref_note.status == "proposed"
    client = make_client(store, blobs, images)

    r = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/from-reference",
                    json={"referenceId": ref_note.id})
    assert r.status_code == 201
    body = r.json()
    assert body["created"] is True
    assert body["promotedFromNoteId"] == ref_note.id
    assert body["promotedFromNoteRevision"] == ref_note.revision

    version = store.get_concept_version(PID, body["id"])
    assert version.source_id == ref_note.provenance[0].source_id
    assert version.promoted_from_note_id == ref_note.id

    # The reference itself is completely untouched - no confirmation, no revision bump.
    stored_ref = store.notes[(PID, ref_note.id)]
    assert stored_ref.status == "proposed"
    assert stored_ref.revision == ref_note.revision


def test_promote_confirmed_reference_is_permitted(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    version, created = promote_reference_to_concept(store, PID, loc.id, confirmed.id)
    assert created is True
    assert version.promoted_from_note_id == confirmed.id


def test_promote_rejected_reference_is_refused_with_clear_message(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    rejected = ref_note.touch(status="rejected", review_reason="not_useful")
    store.put_note_if_current(PID, rejected, expected_revision=ref_note.revision, expected_status="proposed")
    client = make_client(store, blobs, images)

    r = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/from-reference",
                    json={"referenceId": ref_note.id})
    assert r.status_code == 400
    assert "rejected" in r.json()["detail"]
    assert "cannot be promoted" in r.json()["detail"]


def test_promote_unrelated_reference_is_refused(harness_env):
    store, blobs, images, settings, loc, other_loc, _ = harness_env
    other_src = Source(id="e" * 64, filename="river.jpg", mime_type="image/jpeg", kind="image",
                       doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'e' * 64}.jpg",
                       status="digested", source_purpose="reference")
    store.put_source(PID, other_src)
    other_ref = Note(kind="reference_image", body="River view.", owner_id=other_loc.id, author="user",
                     status="confirmed", provenance=[Provenance(source_id=other_src.id)])
    store.put_notes(PID, [other_ref])

    with pytest.raises(ValueError, match="not a reference applicable"):
        promote_reference_to_concept(store, PID, loc.id, other_ref.id)


def test_promote_retry_is_idempotent(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    v1, created1 = promote_reference_to_concept(store, PID, loc.id, ref_note.id)
    assert created1 is True
    v2, created2 = promote_reference_to_concept(store, PID, loc.id, ref_note.id)
    assert created2 is False
    assert v2.id == v1.id
    assert v2.promoted_from_note_id == ref_note.id


def test_promote_and_raw_upload_of_same_bytes_never_lose_first_recorded_provenance(harness_env):
    """Whichever creation attempt wins first - a promotion or a raw upload - its
    provenance stands; a later, different-purpose attempt on the same (location,
    bytes) must never rewrite it."""
    store, blobs, images, settings, loc, _, _ = harness_env
    raw_bytes = _make_image(color="sienna")
    real_ref, _ = upload_reference_image(store, blobs, settings, PID, loc.id, raw_bytes, "real.jpg")

    promoted, created = promote_reference_to_concept(store, PID, loc.id, real_ref.id)
    assert created is True
    assert promoted.promoted_from_note_id == real_ref.id

    # A raw upload of the identical bytes resolves to the SAME version id and does not
    # overwrite the promotion provenance already recorded.
    reuploaded, created2 = upload_concept_version(store, blobs, settings, PID, loc.id, raw_bytes, "again.jpg")
    assert created2 is False
    assert reuploaded.id == promoted.id
    assert reuploaded.promoted_from_note_id == real_ref.id


def test_concurrent_promotion_of_the_same_reference_is_safe(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    results = {}
    barrier = threading.Barrier(2)

    def attempt(name):
        barrier.wait()
        results[name] = promote_reference_to_concept(store, PID, loc.id, ref_note.id)

    t1 = threading.Thread(target=attempt, args=("a",))
    t2 = threading.Thread(target=attempt, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=2); t2.join(timeout=2)

    (version_a, created_a), (version_b, created_b) = results["a"], results["b"]
    assert version_a.id == version_b.id
    assert sorted([created_a, created_b]) == [False, True]
    assert len(store.list_concept_versions(PID, loc.id)) == 1


def test_promotion_preserves_existing_approval(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    confirmed = _confirm(store, ref_note)
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image(color="peru")).json()["id"]
    prev = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                      params={"conceptVersionId": v1, "referenceIds": [confirmed.id]}).json()
    client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [confirmed.id], "contextToken": prev["contextToken"],
        "expectedRevision": 0,
    })

    r = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/from-reference",
                    json={"referenceId": confirmed.id})
    assert r.status_code == 201
    promoted_id = r.json()["id"]
    assert promoted_id != v1

    approval = client.get(f"/projects/{PID}/locations/{loc.id}/approval").json()
    assert approval["approval"]["conceptVersionId"] == v1
    assert approval["revision"] == 1
    listing = client.get(f"/projects/{PID}/locations/{loc.id}/concepts").json()
    ids = {v["id"]: v["approved"] for v in listing["versions"]}
    assert ids[v1] is True
    assert ids[promoted_id] is False


# ==========================================================================================
# Depiction label (item 4)
# ==========================================================================================

def test_depiction_label_included_in_preview_lock_and_token(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]

    no_label = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                          params={"conceptVersionId": v1, "referenceIds": []}).json()
    with_label = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                            params={"conceptVersionId": v1, "referenceIds": [],
                                    "depictionLabel": "whole-house exterior"}).json()
    assert no_label["depictionLabel"] is None
    assert with_label["depictionLabel"] == "whole-house exterior"
    assert no_label["contextToken"] != with_label["contextToken"]

    lock = client.post(f"/projects/{PID}/locations/{loc.id}/approval/lock", json={
        "conceptVersionId": v1, "referenceIds": [], "contextToken": with_label["contextToken"],
        "expectedRevision": 0, "depictionLabel": "whole-house exterior",
    })
    assert lock.status_code == 201
    assert lock.json()["approval"]["depictionLabel"] == "whole-house exterior"

    # Never changes the version's own owning location.
    version = store.get_concept_version(PID, v1)
    assert version.location_id == loc.id


def test_depiction_label_length_is_bounded(harness_env):
    store, blobs, images, settings, loc, _, ref_note = harness_env
    client = make_client(store, blobs, images)
    v1 = client.post(f"/projects/{PID}/locations/{loc.id}/concepts/upload?filename=v1.jpg",
                     content=_make_image()).json()["id"]
    r = client.get(f"/projects/{PID}/locations/{loc.id}/approval/preview",
                   params={"conceptVersionId": v1, "referenceIds": [], "depictionLabel": "x" * 500})
    assert r.status_code == 400


def test_depiction_label_absent_on_old_approvals_is_unambiguous_none(house_env):
    """Unlike scene coverage, an absent depiction_label on old data needs no version
    flag - None has always correctly meant "no label was given"."""
    store, blobs, images, settings, house, *_ = house_env
    version, _ = upload_concept_version(store, blobs, settings, PID, house.id, _make_image(), "v.jpg")
    legacy = ConceptApproval(
        location_id=house.id, concept_version_id=version.id, concept_source_id=version.source_id,
        concept_filename=version.filename, brief_notes=[], brief_conditional=[], brief_inherited=[],
        brief_ancestors=[], superseded_sources=[], references=[], context_token="t",
    )
    assert legacy.depiction_label is None
    store.approvals[(PID, legacy.id)] = legacy
    store.current_approval[(PID, house.id)] = legacy.id
    client = make_client(store, blobs, images)
    body = client.get(f"/projects/{PID}/locations/{house.id}/approval").json()
    assert body["approval"]["depictionLabel"] is None
