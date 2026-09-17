"""Tests for harness.api - the thin HTTP layer over harness.memory.

Every test uses MemoryStore/MemoryBlobs/MemoryImages and a fake LLM (mirroring the
`world`/`make_ctx` pattern in tests/test_references.py). No real GCP or Gemini call is
made anywhere in this file.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import RetryError
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from harness.api.jobs import InMemoryJobStore
from harness.api.main import create_app, run_startup_checks
from harness.api.settings import ApiSettings
from harness.memory.config import Settings
from harness.memory.curate import review_note
from harness.memory.ingest import LOCK_STALE_AFTER_S
from harness.memory.models import (
    Applicability, Containment, Location, Note, Project, Provenance, Scene, Source,
)
from harness.memory.ports import MemoryBlobs, MemoryImages, MemoryStore
from harness.memory.schemas import ClassifyOut, NotesOut, OutNote

PID = "prj_studio"


def utcnow():
    return datetime.now(timezone.utc)


# --- fixtures ----------------------------------------------------------------------------

@pytest.fixture
def world():
    """Devgram (parent) -> Tree Temple (child, confirmed containment).

    Tree Temple has: one description note, one constraint note, an owned confirmed
    reference (with guidance), an owned proposed reference, an owned rejected reference.
    Devgram has one confirmed reference with include_descendants=True, so it should show
    up as "inherited" on Tree Temple. A Scene is linked to Tree Temple.
    """
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))

    devgram = Location(name="Devgram", status="confirmed", author="agent")
    temple = Location(name="Tree Temple", status="confirmed", author="agent",
                      containment=Containment(parent_id=devgram.id, status="confirmed"))
    scene = Scene(name="EXT. TREE TEMPLE - DAY", number="12", location_ids=[temple.id], author="agent")
    store.put_entities(PID, [devgram, temple, scene])

    store.put_notes(PID, [
        Note(kind="description", body="A stone shrine beneath an old tree.",
            owner_id=temple.id, author="user"),
        Note(kind="constraint", body="Keep the entrance readable.", owner_id=temple.id, author="user"),
    ])

    blobs = MemoryBlobs()
    src_confirmed = Source(id="a" * 64, filename="temple.jpg", mime_type="image/jpeg", kind="image",
                           doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'a' * 64}.jpg",
                           status="digested", origin_url="https://commons.wikimedia.org/wiki/File:Temple.jpg",
                           license="CC BY-SA 4.0", attribution="A. Photographer")
    src_proposed = Source(id="b" * 64, filename="terrain.jpg", mime_type="image/jpeg", kind="image",
                          doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'b' * 64}.jpg",
                          status="digested")
    src_inherited = Source(id="c" * 64, filename="village.jpg", mime_type="image/jpeg", kind="image",
                           doc_type="reference", size_bytes=3, storage_path=f"gs://b/{'c' * 64}.jpg",
                           status="digested")
    for src in (src_confirmed, src_proposed, src_inherited):
        store.put_source(PID, src)
        blobs.put(src.storage_path, f"bytes-{src.id}".encode(), src.mime_type)

    ref_confirmed = Note(kind="reference_image", body="Timber & stone temple — layered walls.",
                         owner_id=temple.id, author="agent", status="confirmed", group="Architecture",
                         direction="Timber & stone temple", direction_rationale="Matches the brief.",
                         guidance="canopy shape, not the bridge",
                         provenance=[Provenance(source_id=src_confirmed.id,
                                                url="https://commons.wikimedia.org/wiki/File:Temple.jpg",
                                                title="Temple.jpg")])
    ref_proposed = Note(kind="reference_image", body="Open terraced slopes.", owner_id=temple.id,
                        author="agent", status="proposed", group="Terrain",
                        provenance=[Provenance(source_id=src_proposed.id)])
    ref_rejected = Note(kind="reference_image", body="Unrelated market stall.", owner_id=temple.id,
                        author="agent", status="rejected", review_reason="not_useful", group="Place",
                        provenance=[Provenance(source_id=src_confirmed.id)])
    ref_inherited = Note(kind="reference_image", body="Prosperous village centre.", owner_id=devgram.id,
                         author="agent", status="confirmed", group="Place",
                         applicability=Applicability(include_descendants=True),
                         provenance=[Provenance(source_id=src_inherited.id)])
    store.put_notes(PID, [ref_confirmed, ref_proposed, ref_rejected, ref_inherited])

    images = MemoryImages({}, {}, {})
    return store, blobs, images, devgram, temple, scene, {
        "confirmed": ref_confirmed, "proposed": ref_proposed, "rejected": ref_rejected,
        "inherited": ref_inherited,
    }


def make_client(store, blobs, images, *, llm_factory=None, job_store=None,
                max_concurrent_jobs=2, max_upload_bytes=None,
                ingest_lock_stale_after_s=None, ingest_lock_renew_interval_s=None) -> TestClient:
    overrides = {}
    if max_upload_bytes:
        overrides["max_upload_bytes"] = max_upload_bytes
    if ingest_lock_stale_after_s is not None:
        overrides["ingest_lock_stale_after_s"] = ingest_lock_stale_after_s
    if ingest_lock_renew_interval_s is not None:
        overrides["ingest_lock_renew_interval_s"] = ingest_lock_renew_interval_s
    app = create_app(
        settings=Settings(gcp_project="t", bucket="b"),
        api_settings=ApiSettings(allowed_hosts=("testserver", "127.0.0.1", "localhost"),
                                 max_concurrent_jobs=max_concurrent_jobs, **overrides),
        store=store, blobs=blobs, images=images, job_store=job_store, llm_factory=llm_factory,
    )
    return TestClient(app)


# --- projects -----------------------------------------------------------------------------

def test_list_and_get_project(world):
    store, blobs, images, *_ = world
    with make_client(store, blobs, images) as client:
        r = client.get("/projects")
        assert r.status_code == 200
        assert r.json() == [{"id": PID, "name": "Dehleez"}]

        r = client.get(f"/projects/{PID}")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Dehleez"
        assert body["locationCount"] == 2 and body["sceneCount"] == 1
        assert body["noteCount"] == 6  # 2 brief notes + 4 reference notes

        assert client.get("/projects/nope").status_code == 404


# --- locations ------------------------------------------------------------------------

def test_list_locations_is_lightweight_and_correct(world):
    store, blobs, images, devgram, temple, scene, _ = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/locations")
        assert r.status_code == 200
        by_name = {row["name"]: row for row in r.json()}
        assert by_name["Devgram"]["parentName"] is None
        assert by_name["Tree Temple"]["parentName"] == "Devgram"
        assert by_name["Tree Temple"]["sceneNumbers"] == ["12"]
        # 2 brief notes + confirmed + proposed + rejected(excluded, status=="rejected") on temple
        assert by_name["Tree Temple"]["noteCount"] == 4


def test_get_location_detail_groups_notes_and_scenes(world):
    store, blobs, images, devgram, temple, scene, _ = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/locations/{temple.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Tree Temple"
        assert body["ancestors"] == ["Devgram"]
        assert [s["number"] for s in body["scenes"]] == ["12"]
        assert len(body["descriptionNotes"]) == 1
        assert len(body["constraintNotes"]) == 1
        assert body["descriptionNotes"][0]["status"] == "proposed"


def test_get_location_detail_404_for_unknown_id(world):
    store, blobs, images, *_ = world
    with make_client(store, blobs, images) as client:
        assert client.get(f"/projects/{PID}/locations/nope").status_code == 404


def test_get_location_detail_400_for_a_scene(world):
    store, blobs, images, devgram, temple, scene, _ = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/locations/{scene.id}")
        assert r.status_code == 400


# --- references: owned vs. inherited, status, include_rejected ----------------------------

def test_references_default_excludes_rejected_and_marks_inherited(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/locations/{temple.id}/references")
        assert r.status_code == 200
        by_id = {row["id"]: row for row in r.json()["references"]}
        assert set(by_id) == {refs["confirmed"].id, refs["proposed"].id, refs["inherited"].id}

        confirmed = by_id[refs["confirmed"].id]
        assert confirmed["owned"] is True and confirmed["inheritedFrom"] is None
        assert confirmed["status"] == "confirmed" and confirmed["selected"] is True
        assert confirmed["category"] == "Architecture" and confirmed["facet"] == "Architecture"
        assert confirmed["direction"] == "Timber & stone temple"
        assert confirmed["guidance"] == "canopy shape, not the bridge"
        assert confirmed["credit"] == "A. Photographer · CC BY-SA 4.0"
        assert confirmed["image"] == f"/projects/{PID}/references/{refs['confirmed'].id}/image"

        proposed = by_id[refs["proposed"].id]
        assert proposed["status"] == "proposed" and proposed["selected"] is False

        inherited = by_id[refs["inherited"].id]
        assert inherited["owned"] is False and inherited["inheritedFrom"] == "Devgram"


def test_references_include_rejected(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/locations/{temple.id}/references?include_rejected=true")
        ids = {row["id"] for row in r.json()["references"]}
        assert refs["rejected"].id in ids


# --- image proxy ------------------------------------------------------------------------

def test_reference_image_streams_bytes(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/references/{refs['confirmed'].id}/image")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == f"bytes-{'a' * 64}".encode()


def test_reference_image_404_cross_project(world):
    store, blobs, images, devgram, temple, scene, refs = world
    store.put_project(Project(id="prj_other", name="Other"))
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/prj_other/references/{refs['confirmed'].id}/image")
        assert r.status_code == 404


def test_reference_image_404_when_note_has_no_resolvable_source(world):
    store, blobs, images, devgram, temple, scene, refs = world
    # a note with only a url provenance (no source_id) cannot be resolved to bytes
    orphan = Note(kind="reference_image", body="No stored source.", owner_id=temple.id,
                 author="agent", status="proposed",
                 provenance=[Provenance(url="https://commons.wikimedia.org/wiki/File:X.jpg")])
    store.put_notes(PID, [orphan])
    with make_client(store, blobs, images) as client:
        r = client.get(f"/projects/{PID}/references/{orphan.id}/image")
        assert r.status_code == 404


# --- review: confirm/reject, guidance, and optimistic concurrency -------------------------

def test_review_confirm_and_reject(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/notes/{refs['proposed'].id}/review", json={
            "decision": "confirmed", "by": "jagan",
            "expectedRevision": refs["proposed"].revision, "expectedStatus": "proposed",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "confirmed" and r.json()["reviewedBy"] == "jagan"

        r = client.post(f"/projects/{PID}/notes/{refs['confirmed'].id}/review", json={
            "decision": "rejected", "reason": "not_useful", "by": "jagan",
            "expectedRevision": refs["confirmed"].revision, "expectedStatus": "confirmed",
        })
        assert r.status_code == 200 and r.json()["status"] == "rejected"


def test_review_rejection_without_reason_is_400(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/notes/{refs['proposed'].id}/review", json={
            "decision": "rejected",
            "expectedRevision": refs["proposed"].revision, "expectedStatus": "proposed",
        })
        assert r.status_code == 400


def test_review_unknown_note_is_404(world):
    store, blobs, images, *_ = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/notes/note_nope/review", json={
            "decision": "confirmed", "expectedRevision": 1, "expectedStatus": "proposed",
        })
        assert r.status_code == 404


def test_review_requires_json_content_type(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/notes/{refs['proposed'].id}/review",
                        content=b"decision=confirmed", headers={"content-type": "text/plain"})
        assert r.status_code == 415


def test_review_conflict_when_status_changed_without_a_revision_bump(world):
    """The scenario correction 4 names: two tabs, one confirms (no guidance -> no
    revision bump), the other's stale (revision, status) pair must still be rejected."""
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        note = refs["proposed"]
        r1 = client.post(f"/projects/{PID}/notes/{note.id}/review", json={
            "decision": "confirmed", "by": "tab-a",
            "expectedRevision": note.revision, "expectedStatus": "proposed",
        })
        assert r1.status_code == 200
        assert r1.json()["revision"] == note.revision  # no guidance -> no revision bump

        # tab B read the same stale (revision, status) before tab A's write landed
        r2 = client.post(f"/projects/{PID}/notes/{note.id}/review", json={
            "decision": "rejected", "reason": "not_useful", "by": "tab-b",
            "expectedRevision": note.revision, "expectedStatus": "proposed",
        })
        assert r2.status_code == 409

        # the note reflects tab A's decision, not tab B's
        current = store.list_notes(PID)
        current_note = next(n for n in current if n.id == note.id)
        assert current_note.status == "confirmed" and current_note.reviewed_by == "tab-a"


def test_review_conflict_with_stale_revision(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        note = refs["proposed"]
        # bump revision via a guidance edit
        assert review_note(store, PID, note.id, "confirmed", reviewer="x",
                           guidance="g").note.revision == note.revision + 1
        r = client.post(f"/projects/{PID}/notes/{note.id}/review", json={
            "decision": "rejected", "reason": "other",
            "expectedRevision": note.revision, "expectedStatus": "proposed",   # stale
        })
        assert r.status_code == 409


# --- jobs: trigger, dedup, empty-success, failure, timeout ---------------------------------

class _Handler:
    """A minimal fake LLM that can be swapped between "blocks until released",
    "raises", and "produces an empty (but successful) vocabulary result" behaviors."""

    def __init__(self, fn):
        self.model_id = "fake"
        self._fn = fn

    def generate(self, *, system, parts, schema, fast=False):
        return self._fn(schema, system, parts)


def test_job_trigger_dedups_while_running_then_reports_failure(world):
    from harness.memory.schemas import VocabularyOut

    store, blobs, images, devgram, temple, scene, refs = world
    started = threading.Event()
    release = threading.Event()

    def fn(schema, system, parts):
        if schema is VocabularyOut:
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("stop early - test only checks dedup while running")
        raise AssertionError("should not reach captioning")

    with make_client(store, blobs, images, llm_factory=lambda: _Handler(fn)) as client:
        r1 = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r1.status_code == 201
        job_id = r1.json()["id"]
        assert started.wait(timeout=5), "background job never started"

        r2 = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r2.status_code == 200
        assert r2.json()["id"] == job_id  # same job returned, not a second run

        status = client.get(f"/projects/{PID}/jobs/{job_id}").json()
        assert status["status"] == "running"

        release.set()
        status = _poll_until_terminal(client, job_id)
        assert status["status"] == "failed"
        assert "stop early" in status["error"]

        # the run slot was released, so a fresh trigger is accepted (not dedup'd forever)
        r3 = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r3.status_code == 201 and r3.json()["id"] != job_id


def test_job_succeeds_with_empty_result_when_nothing_verifies(world):
    """suggest_references returns normally (not an exception) when no vocabulary term
    survives verification - correction 9: that is a succeeded run with an empty result
    and warnings, not a failure."""
    from harness.memory.schemas import OutTerm, VocabularyOut

    store, blobs, images, devgram, temple, scene, refs = world

    def fn(schema, system, parts):
        if schema is VocabularyOut:
            return VocabularyOut(script_phrases=[], terms=[OutTerm(term="nonexistent thing", kind="other")])
        raise AssertionError("should not reach retrieval/captioning")

    with make_client(store, blobs, images, llm_factory=lambda: _Handler(fn)) as client:
        r = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r.status_code == 201
        status = _poll_until_terminal(client, r.json()["id"])
        assert status["status"] == "succeeded"
        assert status["references"] is not None  # still returns the (unchanged) current list
        assert any("verification" in w for w in status["warnings"])


def test_blocking_worker_keeps_lock_and_concurrency_slot_until_it_actually_finishes(world):
    """Regression test for the bug a timeout-based design would reintroduce: neither the
    per-location lock nor the process-wide concurrency slot may be released - and no
    terminal status may be reported - before the worker thread genuinely returns.

    JobRunner has no execution timeout specifically because asyncio.to_thread's thread
    cannot be force-cancelled; a timeout would let wait_for's TimeoutError release the
    lock/slot and report "failed" while that uncancellable thread could still be mid-write.
    This test proves the actual (timeout-free) behavior holds even when a run blocks for
    a long time: same-location dedup holds throughout (the lock), a different location's
    run is accepted but stays queued rather than running (the concurrency slot, which is
    process-wide, not per-location - max_concurrent_jobs=1 here), and only the run's real
    completion frees either.
    """
    from harness.memory.schemas import VocabularyOut

    store, blobs, images, devgram, temple, scene, refs = world

    # one dedicated (started, release) pair per run, handed out in call order - avoids
    # any race from clearing and reusing a single pair while a second run might already
    # be mid-flight (asyncio.to_thread's thread cannot be paused to make that safe)
    phases = [{"started": threading.Event(), "release": threading.Event()} for _ in range(2)]
    next_phase = iter(phases)

    def llm_factory():
        phase = next(next_phase)

        def fn(schema, system, parts):
            if schema is VocabularyOut:
                phase["started"].set()
                phase["release"].wait(timeout=10)
                raise RuntimeError("released")
            raise AssertionError("unreachable")

        return _Handler(fn)

    with make_client(store, blobs, images, llm_factory=llm_factory, max_concurrent_jobs=1) as client:
        r1 = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r1.status_code == 201
        job1_id = r1.json()["id"]
        assert phases[0]["started"].wait(timeout=5), "job1 never started"

        # same location, while job1 is running: dedup'd to the same job (the LOCK holds)
        r_dup = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "references"})
        assert r_dup.status_code == 200 and r_dup.json()["id"] == job1_id

        # a different location: its own lock is free, so a job IS created, but the
        # process-wide concurrency SLOT is held by job1, so it must not start running
        r2 = client.post(f"/projects/{PID}/locations/{devgram.id}/jobs", json={"kind": "references"})
        assert r2.status_code == 201
        job2_id = r2.json()["id"]
        time.sleep(0.1)  # let job2's task get scheduled; it must block on the semaphore
        assert client.get(f"/projects/{PID}/jobs/{job2_id}").json()["status"] == "queued"
        assert not phases[1]["started"].is_set(), "job2 touched the LLM before acquiring the slot"
        assert client.get(f"/projects/{PID}/jobs/{job1_id}").json()["status"] == "running"

        phases[0]["release"].set()  # only now does job1's thread actually return
        status1 = _poll_until_terminal(client, job1_id)
        assert status1["status"] == "failed" and "released" in status1["error"]

        # job1's completion is what frees the slot - job2 gets to start only now
        assert phases[1]["started"].wait(timeout=5), \
            "job2 never started - the slot was not freed on job1's real completion"
        assert client.get(f"/projects/{PID}/jobs/{job2_id}").json()["status"] == "running"
        phases[1]["release"].set()
        status2 = _poll_until_terminal(client, job2_id)
        assert status2["status"] == "failed"
        # ("lock/slot freed for a genuinely new run after completion" is covered by
        # test_job_trigger_dedups_while_running_then_reports_failure's own r3 check;
        # this test's job is proving the slot/lock hold *during* the block, above)


def test_job_concept_kind_is_501(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "concept"})
        assert r.status_code == 501


def test_job_unknown_job_id_is_404(world):
    store, blobs, images, *_ = world
    with make_client(store, blobs, images) as client:
        assert client.get(f"/projects/{PID}/jobs/job_nope").status_code == 404


def _poll_until_terminal(client, job_id, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    status = {"status": "queued"}
    while time.monotonic() < deadline:
        status = client.get(f"/projects/{PID}/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time: {status}")


# --- job store recovery (correction 3) and lock ownership (correction 2) ------------------

def test_recover_orphaned_jobs_clears_both_queued_and_running():
    jobs, locks = {}, {}
    process_a = InMemoryJobStore(jobs, locks)
    queued_job, _ = process_a.create_job(PID, "loc_a", "references")
    running_job, _ = process_a.create_job(PID, "loc_b", "references")
    process_a.update_job(PID, running_job.id, status="running", started_at=utcnow())
    # queued_job is left at "queued" - never even started

    process_b = InMemoryJobStore(jobs, locks)  # "restart": a new process attaches
    recovered = process_b.recover_orphaned_jobs()
    assert {j.id for j in recovered} == {queued_job.id, running_job.id}
    for j in recovered:
        assert j.status == "failed" and j.error == "interrupted: server restarted"

    # locks were dropped, so both locations accept a fresh run immediately
    _, created_a = process_b.create_job(PID, "loc_a", "references")
    _, created_b = process_b.create_job(PID, "loc_b", "references")
    assert created_a and created_b


def test_create_job_dedups_and_frees_the_slot_on_completion():
    store = InMemoryJobStore()
    job1, created1 = store.create_job(PID, "loc_a", "references")
    job2, created2 = store.create_job(PID, "loc_a", "references")
    assert created1 and not created2 and job1.id == job2.id

    store.update_job(PID, job1.id, status="succeeded")
    store.release_lock_if_owner(PID, "loc_a", "references", job1.id)
    job3, created3 = store.create_job(PID, "loc_a", "references")
    assert created3 and job3.id != job1.id


def test_release_lock_if_owner_ignores_a_stale_caller():
    """Correction 2: release a lock only if it still belongs to that job - a late
    finally-block from an already-superseded run must not evict a newer run's lock."""
    store = InMemoryJobStore()
    job1, _ = store.create_job(PID, "loc_a", "references")
    store.update_job(PID, job1.id, status="failed")
    job2, created2 = store.create_job(PID, "loc_a", "references")
    assert created2

    store.release_lock_if_owner(PID, "loc_a", "references", job1.id)  # stale; must no-op

    job3, created3 = store.create_job(PID, "loc_a", "references")
    assert not created3 and job3.id == job2.id


# --- host/CORS boundary (correction 5) -----------------------------------------------------

def test_untrusted_host_header_is_rejected(world):
    store, blobs, images, *_ = world
    app = create_app(settings=Settings(gcp_project="t", bucket="b"),
                     api_settings=ApiSettings(allowed_hosts=("only-this-host",)),
                     store=store, blobs=blobs, images=images)
    with TestClient(app, base_url="http://only-this-host") as trusted, \
         TestClient(app) as untrusted:  # TestClient defaults to Host: testserver
        assert trusted.get("/projects").status_code == 200
        assert untrusted.get("/projects").status_code == 400


# --- Origin enforcement on mutations (correction: CORS alone does not reject a request) ---

def test_mutation_rejects_a_disallowed_origin(world):
    """The core claim being tested: a request whose Host is fine (passes
    TrustedHostMiddleware) and whose Content-Type is fine (passes require_json) must
    still be rejected server-side if its Origin doesn't match - proving something
    besides CORSMiddleware (which only annotates responses, never blocks a request) is
    doing the rejecting."""
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/notes/{refs['proposed'].id}/review",
                        json={"decision": "confirmed", "expectedRevision": refs["proposed"].revision,
                              "expectedStatus": "proposed"},
                        headers={"Origin": "http://evil.example"})
        assert r.status_code == 403

        r = client.post(f"/projects/{PID}/locations/{temple.id}/jobs",
                        json={"kind": "concept"}, headers={"Origin": "http://evil.example"})
        assert r.status_code == 403  # rejected before the route even runs (would be 501 otherwise)

        # the note was not touched by the rejected review attempt
        current = next(n for n in store.list_notes(PID) if n.id == refs["proposed"].id)
        assert current.status == "proposed"


def test_mutation_allows_the_configured_frontend_origin(world):
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:  # default allowed_origins includes this
        r = client.post(f"/projects/{PID}/notes/{refs['proposed'].id}/review",
                        json={"decision": "confirmed", "expectedRevision": refs["proposed"].revision,
                              "expectedStatus": "proposed"},
                        headers={"Origin": "http://localhost:3000"})
        assert r.status_code == 200


def test_mutation_without_an_origin_header_is_allowed(world):
    """Non-browser clients (curl, local testing) never send Origin at all; the check
    only rejects a *present but disallowed* Origin, not its absence - see
    deps.require_allowed_origin."""
    store, blobs, images, devgram, temple, scene, refs = world
    with make_client(store, blobs, images) as client:
        r = client.post(f"/projects/{PID}/locations/{temple.id}/jobs", json={"kind": "concept"})
        assert r.status_code == 501  # reached the route (and its own concept-kind rejection)


# --- startup diagnostics: bounded recovery, no real GCP/Firestore (correction: the real
# incident was recover_orphaned_jobs() hanging on an ADC reauth failure for ~300s) --------

def test_run_startup_checks_succeeds_quickly_on_the_happy_path():
    exits = []
    recovered = asyncio.run(run_startup_checks(InMemoryJobStore(), MemoryStore(), timeout_s=5.0, exiter=exits.append))
    assert recovered == [] and exits == []


def test_run_startup_checks_bounds_a_hanging_call_and_exits_instead_of_waiting_it_out():
    """The actual incident: recover_orphaned_jobs's Firestore call hung (ADC needed
    reauthentication) for far longer than any reasonable startup should take, and the
    server sat at "Waiting for application startup." forever. This proves
    run_startup_checks calls exiter(1) at timeout_s, regardless of how long the
    underlying call actually takes - not after waiting it out.

    Measures the time to the exiter(1) call itself, not how long asyncio.run(...) takes
    to return: in production exiter is os._exit, which terminates the process
    immediately and skips asyncio.run's own cleanup entirely (see run_startup_checks's
    docstring - that cleanup is exactly the ThreadPoolExecutor-join hang os._exit exists
    to avoid). Here, substituting a non-terminating fake exiter to keep the test process
    alive means asyncio.run's cleanup phase *does* run and waits for the fake-hung
    thread - that's a test-harness artifact of not really exiting, not something this
    test is trying to measure.
    """
    class HangingJobStore:
        def recover_orphaned_jobs(self):
            time.sleep(2.0)  # stands in for the observed ~300s real-world hang
            return []

    t0 = time.monotonic()
    exit_calls = []

    def fake_exiter(code):
        exit_calls.append((code, time.monotonic() - t0))

    asyncio.run(run_startup_checks(HangingJobStore(), MemoryStore(), timeout_s=0.2, exiter=fake_exiter))

    assert len(exit_calls) == 1
    code, elapsed_to_exit = exit_calls[0]
    assert code == 1  # what a real process does: os._exit(1), no graceful wait
    assert elapsed_to_exit < 1.0, f"exiter was called after {elapsed_to_exit:.2f}s - not bounded to timeout_s"


def test_run_startup_checks_translates_a_refresh_error_without_forcing_an_exit():
    """A credential that fails fast (no hang - e.g. no ADC file at all, or a refresh
    that errors immediately) doesn't need exiter: the call already returned, a plain
    raise is enough, and the message must name the actual fix."""
    class BrokenCredsJobStore:
        def recover_orphaned_jobs(self):
            raise RefreshError("Reauthentication is needed. Please run "
                               "`gcloud auth application-default login` to reauthenticate.")

    exits = []
    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        asyncio.run(run_startup_checks(BrokenCredsJobStore(), MemoryStore(), timeout_s=5.0, exiter=exits.append))
    assert exits == []


def test_run_startup_checks_translates_missing_default_credentials():
    class NoCredsJobStore:
        def recover_orphaned_jobs(self):
            raise DefaultCredentialsError("Could not automatically determine credentials.")

    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        asyncio.run(run_startup_checks(NoCredsJobStore(), MemoryStore(), timeout_s=5.0))


def test_run_startup_checks_unwraps_a_retry_error_to_name_the_real_cause():
    """google-api-core wraps a persistently-failing call in RetryError after its own
    retry budget - the message must surface the wrapped cause (here, the same
    RefreshError), not just "RetryError", so the operator sees the actual fix."""
    class RetryExhaustedJobStore:
        def recover_orphaned_jobs(self):
            raise RetryError("Deadline of 300.0s exceeded", RefreshError("Reauthentication is needed."))

    with pytest.raises(RuntimeError, match="RefreshError"):
        asyncio.run(run_startup_checks(RetryExhaustedJobStore(), MemoryStore(), timeout_s=5.0))


def test_app_startup_fails_loudly_instead_of_serving_requests_on_broken_credentials():
    """Integration-level: do not suppress the error or let the app come up and accept
    requests as if healthy. A lifespan startup failure must propagate and prevent
    TestClient (standing in for uvicorn) from ever entering the running state."""
    class BrokenCredsJobStore:
        def recover_orphaned_jobs(self):
            raise RefreshError("Reauthentication is needed.")

    store, blobs, images = MemoryStore(), MemoryBlobs(), MemoryImages({}, {}, {})
    store.put_project(Project(id=PID, name="Dehleez"))
    app = create_app(settings=Settings(gcp_project="t", bucket="b"),
                     api_settings=ApiSettings(allowed_hosts=("testserver",)),
                     store=store, blobs=blobs, images=images, job_store=BrokenCredsJobStore())

    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        with TestClient(app):
            pytest.fail("must not reach a running state on a broken startup credential")


# ==========================================================================================
# Ingestion: project create -> upload -> ingest -> poll
# ==========================================================================================

IPID = "prj_fresh"


def make_ingest_client(store, blobs, *, llm_factory=None, job_store=None,
                       max_concurrent_jobs=2, max_upload_bytes=None,
                       ingest_lock_stale_after_s=None, ingest_lock_renew_interval_s=None) -> TestClient:
    return make_client(store, blobs, MemoryImages({}, {}, {}), llm_factory=llm_factory,
                       job_store=job_store, max_concurrent_jobs=max_concurrent_jobs,
                       max_upload_bytes=max_upload_bytes,
                       ingest_lock_stale_after_s=ingest_lock_stale_after_s,
                       ingest_lock_renew_interval_s=ingest_lock_renew_interval_s)


def _ingest_handler(fail_filenames=frozenset()):
    """A fake ingest LLM: classifies everything as "notes" and extracts one project-wide
    description note per file, except files named in fail_filenames, which raise at the
    classify step (simulating a real per-source failure ingest_source is built to
    isolate - "the source records the failure; the batch carries on")."""
    def fn(schema, system, parts):
        if schema is ClassifyOut:
            named = parts[0].text if parts and hasattr(parts[0], "text") else ""
            if any(bad in named for bad in fail_filenames):
                raise RuntimeError(f"simulated extraction failure for {named}")
            return ClassifyOut(doc_type="notes")
        if schema is NotesOut:
            return NotesOut(notes=[OutNote(kind="description", body="A note from this file.",
                                           project_wide=True)])
        raise AssertionError(schema)
    return fn


@pytest.fixture
def ingest_world():
    store = MemoryStore()
    store.put_project(Project(id=IPID, name="Dehleez"))
    return store, MemoryBlobs()


def _upload(client, name: str, body: bytes | None = None):
    # content defaults to something that varies by name - register_file dedupes by
    # content hash, not filename, so two calls with the literal same default body would
    # collide into a single Source no matter what name each claims.
    body = body if body is not None else f"INT. ROOM - DAY\nNotes from {name}.\n".encode()
    return client.post(f"/projects/{IPID}/sources", params={"filename": name}, content=body)


# --- project creation ---------------------------------------------------------------------

def test_create_project_is_simple_and_leaves_others_untouched(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        r = client.post("/projects", json={"name": "New Film"})
        assert r.status_code == 201
        new_id = r.json()["id"]
        assert new_id.startswith("prj_") and new_id != IPID

        assert client.get(f"/projects/{new_id}").json()["name"] == "New Film"
        # the existing project (and its data) is untouched
        assert client.get(f"/projects/{IPID}").json()["name"] == "Dehleez"


# --- upload ---------------------------------------------------------------------------------

def test_upload_new_file_and_duplicate_resubmission(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        body = b"INT. ROOM - DAY\nSome notes.\n"
        r1 = _upload(client, "notes.txt", body)
        assert r1.status_code == 201
        first = r1.json()
        assert first["created"] is True and first["status"] == "uploaded"
        assert first["mimeType"] == "text/plain"

        r2 = _upload(client, "notes.txt", body)  # identical bytes
        assert r2.status_code == 201
        second = r2.json()
        assert second["created"] is False and second["id"] == first["id"]
        assert len(store.list_sources(IPID)) == 1  # no duplicate Source


def test_upload_unsupported_mime_is_stored_not_rejected(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        r = _upload(client, "movie.mp4", b"\x00\x00\x00\x18ftypmp42")
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "unsupported" and body["error"]
        assert len(store.list_sources(IPID)) == 1  # still stored, per register_file


def test_upload_over_the_limit_is_rejected_and_stores_nothing(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs, max_upload_bytes=16) as client:
        r = _upload(client, "big.txt", b"x" * 1000)
        assert r.status_code == 413
        assert store.list_sources(IPID) == []


def test_upload_rejects_disallowed_origin_but_ignores_content_type(ingest_world):
    """Correction 1: keep Origin validation on this route; JSON-only validation must
    NOT apply here (the body is raw file bytes, not JSON)."""
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        r = client.post(f"/projects/{IPID}/sources", params={"filename": "x.txt"},
                        content=b"hello", headers={"Origin": "http://evil.example"})
        assert r.status_code == 403

        r = client.post(f"/projects/{IPID}/sources", params={"filename": "x.txt"},
                        content=b"hello", headers={"Content-Type": "text/plain"})
        assert r.status_code == 201  # not 415 - this route has no JSON requirement


# --- GET /projects/{id}/sources: authoritative uploaded-source listing ---------------------

def test_list_sources_unknown_project_is_404(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        assert client.get("/projects/prj_nope/sources").status_code == 404


def test_list_sources_empty_project_returns_empty_list(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        r = client.get(f"/projects/{IPID}/sources")
        assert r.status_code == 200
        assert r.json() == []


def test_list_sources_is_scoped_to_its_own_project(ingest_world):
    store, blobs = ingest_world
    other_pid = "prj_other_sources"
    store.put_project(Project(id=other_pid, name="Other"))
    with make_ingest_client(store, blobs) as client:
        mine = _upload(client, "mine.txt").json()
        r_other = client.post(f"/projects/{other_pid}/sources", params={"filename": "theirs.txt"},
                              content=b"INT. ELSEWHERE - DAY\nNot this project's file.\n")
        assert r_other.status_code == 201

        r = client.get(f"/projects/{IPID}/sources")
        assert r.status_code == 200
        ids = {row["id"] for row in r.json()}
        assert ids == {mine["id"]}  # the other project's source is not leaked in here

        r2 = client.get(f"/projects/{other_pid}/sources")
        assert {row["id"] for row in r2.json()} == {r_other.json()["id"]}


def test_list_sources_reports_representative_persisted_statuses_and_errors(ingest_world):
    """Statuses and errors come straight from the persisted Source, not recomputed or
    invented - covers uploaded, unsupported (with its register_file-authored error),
    digested, and failed (with an ingest_source-authored error)."""
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        uploaded = _upload(client, "uploaded.txt").json()
        unsupported = _upload(client, "movie.mp4", b"\x00\x00\x00\x18ftypmp42").json()

    # digested/failed are post-ingestion states; constructing them directly keeps this
    # test focused on the read endpoint rather than re-running a full ingest.
    digested_src = Source(id="d" * 64, filename="digested.pdf", mime_type="application/pdf",
                          kind="document", size_bytes=100, storage_path=f"gs://b/{'d' * 64}.pdf",
                          status="digested", doc_type="script")
    failed_src = Source(id="f" * 64, filename="failed.pdf", mime_type="application/pdf",
                        kind="document", size_bytes=50, storage_path=f"gs://b/{'f' * 64}.pdf",
                        status="failed", error="OutputTruncated: roster pass truncated")
    store.put_source(IPID, digested_src)
    store.put_source(IPID, failed_src)

    with make_ingest_client(store, blobs) as client:
        by_id = {row["id"]: row for row in client.get(f"/projects/{IPID}/sources").json()}

    assert by_id[uploaded["id"]]["status"] == "uploaded"
    assert by_id[uploaded["id"]]["error"] is None

    assert by_id[unsupported["id"]]["status"] == "unsupported"
    assert by_id[unsupported["id"]]["error"]  # register_file's own "no v0 handler for ..."

    assert by_id[digested_src.id]["status"] == "digested"
    assert by_id[digested_src.id]["error"] is None
    assert by_id[digested_src.id]["mimeType"] == "application/pdf"
    assert by_id[digested_src.id]["sizeBytes"] == 100

    assert by_id[failed_src.id]["status"] == "failed"
    assert by_id[failed_src.id]["error"] == "OutputTruncated: roster pass truncated"

    assert "created" not in by_id[uploaded["id"]]  # list rows have no upload-only field
    assert len(by_id) == 4


def test_list_sources_excludes_generated_reference_images(ingest_world):
    """The core scoping requirement: a references-pipeline-fetched image
    (references._store_image's shape - doc_type="reference", origin_url set) must not
    appear in the uploaded-sources listing, even though it lives in the same
    store.list_sources(project_id) collection as real uploads."""
    store, blobs = ingest_world
    with make_ingest_client(store, blobs) as client:
        uploaded = _upload(client, "script.txt").json()

    reference_image_src = Source(
        id="e" * 64, filename="Temple.jpg", mime_type="image/jpeg", kind="image",
        size_bytes=900, storage_path=f"gs://b/{'e' * 64}.jpg", status="digested",
        doc_type="reference",  # a genuinely-uploaded lookbook could ALSO get this -
                                # origin_url below is what actually distinguishes it
        origin_url="https://commons.wikimedia.org/wiki/File:Temple.jpg",
        license="CC BY-SA 4.0", attribution="A. Photographer",
    )
    store.put_source(IPID, reference_image_src)

    with make_ingest_client(store, blobs) as client:
        r = client.get(f"/projects/{IPID}/sources")
        assert r.status_code == 200
        ids = [row["id"] for row in r.json()]
        assert ids == [uploaded["id"]]  # the reference image is excluded, not merely hidden last
        # confirm it's genuinely present in the underlying store (so this is proving
        # the endpoint's own filtering, not an empty fixture)
        assert reference_image_src.id in {s.id for s in store.list_sources(IPID)}


def test_list_sources_deterministic_ordering(ingest_world):
    store, blobs = ingest_world
    early = Source(id="1" * 64, filename="z-early.txt", mime_type="text/plain", kind="text",
                   size_bytes=1, storage_path=f"gs://b/{'1' * 64}.txt", status="uploaded",
                   created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = Source(id="2" * 64, filename="a-late.txt", mime_type="text/plain", kind="text",
                  size_bytes=1, storage_path=f"gs://b/{'2' * 64}.txt", status="uploaded",
                  created_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    # inserted out of chronological order - the response must still come back sorted by
    # created_at (then filename, then id), not store/dict insertion order
    store.put_source(IPID, late)
    store.put_source(IPID, early)

    with make_ingest_client(store, blobs) as client:
        r = client.get(f"/projects/{IPID}/sources")
        assert [row["id"] for row in r.json()] == [early.id, late.id]


# --- ingest trigger + polling ---------------------------------------------------------------

def _poll_ingest_until_terminal(client, job_id, *, project_id=IPID, timeout=5.0):
    deadline = time.monotonic() + timeout
    status = {"status": "queued"}
    while time.monotonic() < deadline:
        status = client.get(f"/projects/{project_id}/jobs/{job_id}").json()
        if status.get("status") in ("succeeded", "failed"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time: {status}")


def test_ingest_all_sources_succeed_reports_complete(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler())) as client:
        id_a = _upload(client, "a.txt").json()["id"]
        id_b = _upload(client, "b.txt").json()["id"]

        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 201
        status = _poll_ingest_until_terminal(client, r.json()["id"])

        assert status["status"] == "succeeded"
        assert status["outcome"] == "complete"
        by_id = {s["sourceId"]: s for s in status["sources"]}
        assert by_id[id_a]["status"] == "digested" and by_id[id_b]["status"] == "digested"
        assert by_id[id_a]["notesWritten"] >= 1


def test_ingest_no_sources_is_a_no_op(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler())) as client:
        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 201
        status = _poll_ingest_until_terminal(client, r.json()["id"])
        assert status["status"] == "succeeded"
        assert status["outcome"] == "no_op"
        assert status["sources"] == []


def test_ingest_all_sources_fail_reports_failed_not_succeeded(ingest_world):
    """Correction 5: if every attempted source fails, the job itself must be "failed" -
    the UI must not present this as successful ingestion or auto-advance to locations."""
    store, blobs = ingest_world
    handler = _ingest_handler(fail_filenames={"a.txt", "b.txt"})
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(handler)) as client:
        _upload(client, "a.txt")
        _upload(client, "b.txt")

        r = client.post(f"/projects/{IPID}/ingest", json={})
        status = _poll_ingest_until_terminal(client, r.json()["id"])

        assert status["status"] == "failed"
        assert status["outcome"] == "failed"
        assert "2 source(s) failed" in status["error"]
        assert all(s["status"] == "failed" and s["error"] for s in status["sources"])


def test_ingest_partial_failure_reports_partial_and_succeeds(ingest_world):
    store, blobs = ingest_world
    handler = _ingest_handler(fail_filenames={"bad.txt"})
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(handler)) as client:
        _upload(client, "good.txt")
        _upload(client, "bad.txt")

        r = client.post(f"/projects/{IPID}/ingest", json={})
        status = _poll_ingest_until_terminal(client, r.json()["id"])

        assert status["status"] == "succeeded"  # a real, if partial, success
        assert status["outcome"] == "partial"
        statuses = {s["status"] for s in status["sources"]}
        assert statuses == {"digested", "failed"}


def test_ingest_specific_source_id_must_belong_to_the_project(ingest_world):
    store, blobs = ingest_world
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler())) as client:
        r = client.post(f"/projects/{IPID}/ingest", json={"sourceId": "not-a-real-source"})
        assert r.status_code == 404


# --- dedup vs. conflict (correction 4) -------------------------------------------------------

def test_ingest_equivalent_repeat_trigger_dedupes_to_the_same_job(ingest_world):
    store, blobs = ingest_world
    started = threading.Event()
    release = threading.Event()

    def fn(schema, system, parts):
        if schema is ClassifyOut:
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("released")
        raise AssertionError(schema)

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(fn)) as client:
        _upload(client, "a.txt")
        r1 = client.post(f"/projects/{IPID}/ingest", json={})
        assert r1.status_code == 201
        job_id = r1.json()["id"]
        assert started.wait(timeout=5)

        r2 = client.post(f"/projects/{IPID}/ingest", json={})  # same params: sourceId=None, force=False
        assert r2.status_code == 200
        assert r2.json()["id"] == job_id

        release.set()
        _poll_ingest_until_terminal(client, job_id)


def test_ingest_conflicting_repeat_trigger_is_409_not_silently_dropped(ingest_world):
    """Correction 4: a request with different sourceId/force must not silently return
    the unrelated active job as though its own options were honored."""
    store, blobs = ingest_world
    started = threading.Event()
    release = threading.Event()

    def fn(schema, system, parts):
        if schema is ClassifyOut:
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("released")
        raise AssertionError(schema)

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(fn)) as client:
        _upload(client, "a.txt")
        r1 = client.post(f"/projects/{IPID}/ingest", json={"force": False})
        assert r1.status_code == 201
        assert started.wait(timeout=5)

        r2 = client.post(f"/projects/{IPID}/ingest", json={"force": True})  # different!
        assert r2.status_code == 409

        release.set()
        _poll_ingest_until_terminal(client, r1.json()["id"])


def test_ingest_external_cli_lock_is_409_and_does_not_create_a_job(ingest_world):
    """Simulates a concurrent `harness-memory drop`/`ingest` from the CLI: it holds the
    same domain lock the API would use, via a path the API has no record of."""
    store, blobs = ingest_world
    token = store.acquire_lock(IPID, "cli", LOCK_STALE_AFTER_S)
    assert token is not None

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler())) as client:
        _upload(client, "a.txt")
        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 409
        assert "wasn't started via this API" in r.json()["detail"] or "CLI" in r.json()["detail"]

    store.release_lock(IPID, token)
    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler())) as client:
        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 201  # free again once the external lock is released


# --- scheduling failure releases the lock (correction 2, correction 8) -----------------------

def test_scheduling_failure_releases_the_lock_immediately(ingest_world):
    """If job bookkeeping fails after the domain lock was already won, the lock must be
    released right away, not left until LOCK_STALE_AFTER_S (1 hour) elapses."""
    store, blobs = ingest_world

    class BrokenPutJobStore(InMemoryJobStore):
        def put_new_job(self, job):
            raise RuntimeError("simulated bookkeeping failure")

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(_ingest_handler()),
                            job_store=BrokenPutJobStore()) as client:
        _upload(client, "a.txt")
        with pytest.raises(RuntimeError, match="simulated bookkeeping failure"):
            client.post(f"/projects/{IPID}/ingest", json={})

    # the lock was released despite the failure - a fresh acquire succeeds immediately
    token = store.acquire_lock(IPID, "probe", LOCK_STALE_AFTER_S)
    assert token is not None
    store.release_lock(IPID, token)


# --- restart lock recovery: token-specific, never a CLI-owned lock (correction 2) ------------

def test_restart_recovery_releases_only_the_orphaned_api_owned_lock():
    jobs, locks = {}, {}
    store = MemoryStore()
    store.put_project(Project(id=IPID, name="Dehleez"))

    # simulate an API-triggered ingest that "crashed": the job is left running, and the
    # domain lock is genuinely held with that job's own token (no finally ever ran).
    process_a = InMemoryJobStore(jobs, locks)
    from harness.api.jobs import Job
    token = store.acquire_lock(IPID, "api:job_crashed", LOCK_STALE_AFTER_S)
    crashed = Job(id="job_crashed", project_id=IPID, location_id=IPID, kind="ingest",
                  status="running", lock_token=token, source_ids=["s1"])
    process_a.put_new_job(crashed)

    process_b = InMemoryJobStore(jobs, locks)  # "restart": a new process attaches
    recovered = asyncio.run(run_startup_checks(process_b, store, timeout_s=5.0))

    assert len(recovered) == 1 and recovered[0].id == "job_crashed"
    assert process_b.get_job(IPID, "job_crashed").status == "failed"
    # the domain lock was actually released - a fresh acquire succeeds
    fresh_token = store.acquire_lock(IPID, "probe", LOCK_STALE_AFTER_S)
    assert fresh_token is not None
    store.release_lock(IPID, fresh_token)


def test_restart_recovery_never_releases_a_cli_owned_lock():
    jobs, locks = {}, {}
    store = MemoryStore()
    store.put_project(Project(id=IPID, name="Dehleez"))

    # a CLI ingest is genuinely, legitimately still running - no API Job exists for it
    cli_token = store.acquire_lock(IPID, "cli", LOCK_STALE_AFTER_S)
    assert cli_token is not None

    # meanwhile some *other*, unrelated API job (references, on a made-up location) was
    # left running by the same crash and must still be recovered as usual
    process_a = InMemoryJobStore(jobs, locks)
    process_a.create_job(IPID, "loc_x", "references")

    process_b = InMemoryJobStore(jobs, locks)
    recovered = asyncio.run(run_startup_checks(process_b, store, timeout_s=5.0))

    assert len(recovered) == 1 and recovered[0].kind == "references"
    # the CLI's lock is completely untouched: acquiring again still fails
    assert store.acquire_lock(IPID, "someone-else", LOCK_STALE_AFTER_S) is None
    store.release_lock(IPID, cli_token)


# --- lock staleness / renewal (correction 3) --------------------------------------------------

def test_a_stale_lock_can_be_reclaimed_but_a_renewed_one_cannot():
    store = MemoryStore()
    pid = "prj_lock_test"
    token = store.acquire_lock(pid, "holder-a", stale_after_s=0.15)
    assert token is not None
    assert store.acquire_lock(pid, "holder-b", stale_after_s=0.15) is None  # still fresh

    time.sleep(0.2)
    stolen = store.acquire_lock(pid, "holder-b", stale_after_s=0.15)
    assert stolen is not None and stolen != token  # correctly reclaimed once truly stale
    store.release_lock(pid, stolen)


def test_renew_lock_prevents_a_still_running_holder_from_losing_its_lock():
    """The underlying primitive the periodic renewal loops (JobRunner._run_ingest's and
    cli.py's LockRenewer, tested below) are built on: a single explicit renew_lock call
    resets the staleness clock so a genuinely still-working holder isn't reclaimed."""
    store = MemoryStore()
    pid = "prj_lock_test2"
    token = store.acquire_lock(pid, "holder-a", stale_after_s=0.15)
    assert token is not None

    time.sleep(0.1)
    assert store.renew_lock(pid, token) is True  # "still working" heartbeat
    time.sleep(0.1)  # 0.2s since acquire, but only 0.1s since renewal - still fresh

    assert store.acquire_lock(pid, "holder-b", stale_after_s=0.15) is None  # not stolen
    store.release_lock(pid, token)


def test_lock_tokens_are_unique_per_acquisition_not_derived_from_holder():
    """Guards the bug the token-based restart recovery depends on not existing: two
    different acquisitions by the same holder string must never produce the same
    token, or a stale caller's remembered token could match a different, later,
    currently-live lock."""
    store = MemoryStore()
    pid = "prj_lock_test3"
    token1 = store.acquire_lock(pid, "cli", LOCK_STALE_AFTER_S)
    store.release_lock(pid, token1)
    token2 = store.acquire_lock(pid, "cli", LOCK_STALE_AFTER_S)
    assert token1 != token2
    store.release_lock(pid, token2)


# --- periodic renewal covers the FULL lock-held period, not just gaps between sources ------
#
# Regression tests for the gap flagged in review: renewing only between sources leaves
# two periods unprotected - waiting for the concurrency semaphore, and a single source
# that itself runs long. Both use a short simulated expiry (ingest_lock_stale_after_s)
# well inside a single test's real wall-clock budget, with renewal several times faster
# than that expiry, and prove a *competing* lock acquisition still fails throughout.

def test_periodic_renewal_survives_a_single_long_running_source(ingest_world):
    """The source-processing gap: without periodic (only between-source) renewal, a
    source that itself blocks past the staleness window would have its lock stolen
    mid-processing. Here the window is simulated short (0.15s) and the one uploaded
    source blocks for well past it - a competing acquire_lock must still fail the
    whole time, proving the lock was kept alive *during*, not just around, that source.
    """
    store, blobs = ingest_world
    blocked = threading.Event()
    release = threading.Event()

    def fn(schema, system, parts):
        if schema is ClassifyOut:
            blocked.set()
            release.wait(timeout=5)
            raise RuntimeError("released")
        raise AssertionError(schema)

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(fn),
                            ingest_lock_stale_after_s=0.15,
                            ingest_lock_renew_interval_s=0.03) as client:
        _upload(client, "a.txt")
        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 201
        job_id = r.json()["id"]
        assert blocked.wait(timeout=5), "the source never started"

        time.sleep(0.3)  # 2x the simulated 0.15s staleness window, source still blocked

        # a competing acquisition (standing in for a concurrent CLI drop/ingest) must
        # still be refused - if renewal only happened between sources, this would have
        # succeeded by now, since no source has finished yet to trigger a renewal.
        assert store.acquire_lock(IPID, "competitor", 0.15) is None

        release.set()
        status = _poll_ingest_until_terminal(client, job_id)
        # the source's own result (released just now) is whatever it is; what matters
        # here is that the lock was never lost, which the assertion above already proved
        assert status["status"] in ("succeeded", "failed")


def test_periodic_renewal_survives_waiting_for_the_concurrency_semaphore(ingest_world):
    """The queued-job gap: job B's own domain lock (on its own project) is acquired the
    moment it's triggered, but with max_concurrent_jobs=1 and job A already running, B
    sits queued behind the semaphore before it ever touches a source. A competing
    acquisition on B's project must still fail throughout that wait - proving renewal
    covers "queued," not only "actively processing."
    """
    store, blobs = ingest_world
    ipid2 = "prj_fresh2"
    store.put_project(Project(id=ipid2, name="Second Film"))

    a_blocked = threading.Event()
    a_release = threading.Event()

    def fn(schema, system, parts):
        if schema is ClassifyOut:
            a_blocked.set()
            a_release.wait(timeout=5)
            raise RuntimeError("released")
        raise AssertionError(schema)

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(fn),
                            max_concurrent_jobs=1, ingest_lock_stale_after_s=0.15,
                            ingest_lock_renew_interval_s=0.03) as client:
        _upload(client, "a.txt")
        r_a = client.post(f"/projects/{IPID}/ingest", json={})
        assert r_a.status_code == 201
        assert a_blocked.wait(timeout=5), "job A never started"

        # job B: different project, so its own acquire_lock succeeds immediately: it is
        # NOT blocked on IPID's lock, only on the shared process-wide semaphore.
        r_b = client.post(f"/projects/{ipid2}/sources", params={"filename": "b.txt"},
                          content=b"INT. ROOM - DAY\nNotes for the second film.\n")
        assert r_b.status_code == 201
        r_b_ingest = client.post(f"/projects/{ipid2}/ingest", json={})
        assert r_b_ingest.status_code == 201
        job_b_id = r_b_ingest.json()["id"]

        time.sleep(0.1)  # let B's task actually start and acquire the lock+renewer
        assert client.get(f"/projects/{ipid2}/jobs/{job_b_id}").json()["status"] == "queued"

        time.sleep(0.3)  # 2x the simulated 0.15s staleness window, B still queued

        assert client.get(f"/projects/{ipid2}/jobs/{job_b_id}").json()["status"] == "queued"
        # the competing check that matters: B's own lock, on B's own project, must still
        # be held even though B has never started processing anything.
        assert store.acquire_lock(ipid2, "competitor", 0.15) is None

        a_release.set()
        _poll_ingest_until_terminal(client, r_a.json()["id"])
        _poll_ingest_until_terminal(client, job_b_id, project_id=ipid2)


# --- ownership loss stops further work rather than continuing silently --------------------

def test_lost_ownership_stops_before_the_next_source_and_fails_the_job(ingest_world):
    """Defines the contract asked for: if renewal discovers ownership is gone,
    ingestion must not silently continue as though nothing happened. a.txt's write,
    already in flight when the lock is stolen, cannot be undone (no way to
    force-cancel a thread) and is allowed to finish - but b.txt must never even start,
    and the job must end up "failed" naming lost ownership, not "succeeded" as if
    nothing happened.

    Uses the job_store directly (not just HTTP) to read the job's own lock_token and
    steal it deterministically mid-run, rather than racing against renewal timing.
    """
    store, blobs = ingest_world
    job_store = InMemoryJobStore()
    a_blocked = threading.Event()
    a_release = threading.Event()
    b_started = threading.Event()  # must never be set

    def fn(schema, system, parts):
        if schema is ClassifyOut:
            named = parts[0].text if parts and hasattr(parts[0], "text") else ""
            if "a.txt" in named:
                a_blocked.set()
                a_release.wait(timeout=5)
                return ClassifyOut(doc_type="notes")
            b_started.set()
            return ClassifyOut(doc_type="notes")
        if schema is NotesOut:
            return NotesOut(notes=[OutNote(kind="description", body="note", project_wide=True)])
        raise AssertionError(schema)

    with make_ingest_client(store, blobs, llm_factory=lambda: _Handler(fn), job_store=job_store,
                            ingest_lock_stale_after_s=0.15,
                            ingest_lock_renew_interval_s=0.03) as client:
        _upload(client, "a.txt")
        _upload(client, "b.txt")
        r = client.post(f"/projects/{IPID}/ingest", json={})
        assert r.status_code == 201
        job_id = r.json()["id"]
        assert a_blocked.wait(timeout=5), "a.txt never started"

        # steal the lock out from under the running job while a.txt is still in flight
        job = job_store.get_job(IPID, job_id)
        store.release_lock(IPID, job.lock_token)
        thief_token = store.acquire_lock(IPID, "thief", 0.15)
        assert thief_token is not None

        time.sleep(0.15)  # several renewal intervals (0.03s) - plenty of time to notice
        a_release.set()

        status = _poll_ingest_until_terminal(client, job_id, timeout=5)

        assert status["status"] == "failed"
        assert status["outcome"] == "failed"
        assert "lost exclusive ownership" in status["error"]
        assert not b_started.is_set(), "b.txt must never start once ownership was lost"
        assert len(status["sources"]) <= 1  # a.txt's own in-flight result, if it got one

        store.release_lock(IPID, thief_token)
