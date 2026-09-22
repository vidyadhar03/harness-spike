"""Base-location concept generation, tested with a fake ImageProvider (no Luma, no network).

The workflow only ever sees imagegen.ImageProvider; nothing here imports the Luma adapter."""
from __future__ import annotations

import asyncio
import dataclasses
import io
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image
from starlette.testclient import TestClient

from harness.api.main import create_app
from harness.api.settings import ApiSettings
from harness.memory.concept_generation import (
    ConceptGenerationService, GenerationConfig, IdempotencyConflict, JobStateConflict,
    StaleGenerationContext,
)
from harness.memory.concepts import build_approval_snapshot, lock_approval, upload_concept_version
from harness.memory.config import Settings
from harness.memory.imagegen import (
    DownloadedImage, ImageCapabilities, OutputTooLarge, OutputUnavailable, ProviderJob, ProviderTransientError,
    SubmissionOutcomeUnknown, SubmissionRejected, UnsupportedRequest,
)
from harness.memory.models import Applicability, Containment, Location, Note, Project, Scene
from harness.memory.ports import MemoryBlobs, MemoryImages, MemoryStore
from harness.memory.references import upload_reference_image

PID = "prj_gen"
CAPS = ImageCapabilities(
    models=frozenset({"m1", "m2"}), max_references=3, aspect_ratios=frozenset({"1:1", "16:9"}),
    reference_mime_types=frozenset({"image/png", "image/jpeg"}), max_reference_bytes=10_000_000,
    max_request_bytes=20_000_000, max_prompt_chars=6000, output_formats=frozenset({"png"}))


def png(color="red", size=(24, 24)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return b.getvalue()


class ProcessDied(BaseException):
    """Simulates the process dying mid-call (not an Exception, so nothing can 'handle' it)."""


class FakeProvider:
    name = "fake"
    account_scope = "acct-1"
    capabilities = CAPS

    def __init__(self):
        self.submits: list = []
        self.submit_hook = None                 # callable(req) -> id | raises
        self.get_script: list = []              # ProviderJob | Exception, consumed in order
        self.default_get = None
        self.gets = 0
        self.downloads: list = []
        self.download_hook = None
        self.output_bytes = png("blue")
        self._n = 0

    def submit(self, req):
        self.submits.append(req)
        if self.submit_hook:
            return self.submit_hook(req)
        self._n += 1
        return f"gen-{self._n}"

    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    reported_kind, reported_model = "image", "m1"

    def done(self, gid, url="https://cdn.example/out.png?sig=abc"):
        return ProviderJob(gid, "completed", (url,), kind=self.reported_kind, model=self.reported_model,
                           created_at=self.created_at)

    def get(self, gid):
        self.gets += 1
        if getattr(self, "get_hook", None):
            return self.get_hook(gid)
        item = self.get_script.pop(0) if self.get_script else (self.default_get or self.done(gid))
        if isinstance(item, BaseException):
            raise item
        return item

    def download(self, url):
        self.downloads.append(url)
        if self.download_hook:
            return self.download_hook(url)
        return DownloadedImage(self.output_bytes, "image/png")


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


@pytest.fixture
def env():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Gen Test"))
    village = Location(name="Devgram", status="confirmed", author="agent")
    house = Location(name="Anshul's House", status="confirmed", author="agent",
                     containment=Containment(parent_id=village.id, status="confirmed"))
    store.put_entities(PID, [village, house])
    s4 = Scene(name="4. INT. ANSHUL'S HOUSE - NIGHT", number="4", location_ids=[house.id], author="agent")
    s7 = Scene(name="7. DREAM", number="7", location_ids=[house.id], author="agent")
    store.put_entities(PID, [s4, s7])
    desc = Note(kind="description", body="A warm, modest wooden home with old exposed beams.", owner_id=house.id, author="user")
    cons = Note(kind="constraint", body="Keep the heater visible near the beams.", owner_id=house.id, author="user", status="confirmed")
    tone = Note(kind="tone", body="Quiet, lived-in, slightly melancholic.", owner_id=house.id, author="user")
    inherited = Note(kind="description", body="A village of timber houses in the Himalayan foothills.",
                     owner_id=village.id, author="user", applicability=Applicability(include_descendants=True))
    scene_note = Note(kind="description", body="The dream flood tilts the house toward the water.",
                      owner_id=house.id, author="user", applicability=Applicability(scene_id=s7.id))
    store.put_notes(PID, [desc, cons, tone, inherited, scene_note])
    blobs = MemoryBlobs()
    settings = Settings(gcp_project="t", bucket="b")
    refs = {}
    for name, color in (("a", "green"), ("b", "orange"), ("c", "purple")):
        note, _ = upload_reference_image(store, blobs, settings, PID, house.id, png(color), f"{name}.png")
        refs[name] = note
    for name, guide in (("a", "the canopy shape"), ("b", "the warm wood texture")):
        n = refs[name]
        confirmed = n.touch(status="confirmed", guidance=guide, revision=n.revision + 1,
                            reviewed_revision=n.revision + 1)
        store.put_note_if_current(PID, confirmed, n.revision, n.status)
        refs[name] = confirmed
    provider, clock = FakeProvider(), Clock()
    svc = ConceptGenerationService(store, blobs, settings, provider,
                                   config=GenerationConfig(default_model="m1", poll_interval_s=1, max_wait_s=5, lease_s=30),
                                   sleep=clock.sleep, clock=clock.now)
    ns = type("E", (), dict(store=store, blobs=blobs, settings=settings, provider=provider, svc=svc, clock=clock,
                            house=house, village=village, s4=s4, s7=s7, refs=refs, desc=desc, cons=cons,
                            tone=tone, inherited=inherited, scene_note=scene_note))
    return ns


def submit(e, key="k1", refs=("a",), svc=None, **kw):
    svc = svc or e.svc
    ids = [e.refs[r].id for r in refs]
    prepared = svc.prepare(PID, e.house.id, reference_ids=ids, **{k: v for k, v in kw.items() if k != "token"})
    return svc.submit(PID, e.house.id, reference_ids=ids, context_token=kw.get("token", prepared.context_token),
                      idempotency_key=key, **{k: v for k, v in kw.items() if k != "token"})


# --- context preparation ---------------------------------------------------------------------

def test_standing_and_inherited_included_scene_notes_excluded(env):
    p = env.svc.prepare(PID, env.house.id, reference_ids=[])
    ids = {n["id"] for n in p.snapshot["standingNotes"]}
    assert ids == {env.desc.id, env.cons.id, env.tone.id}
    assert [n["id"] for n in p.snapshot["inheritedNotes"]] == [env.inherited.id]
    assert p.snapshot["inheritedNotes"][0]["ancestorName"] == "Devgram"
    assert "flood" not in p.prompt and env.scene_note.id not in str(p.snapshot)
    # statuses/kinds/revisions preserved, nothing confirmed on the way
    by_id = {n["id"]: n for n in p.snapshot["standingNotes"]}
    assert by_id[env.desc.id]["status"] == "proposed" and by_id[env.cons.id]["status"] == "confirmed"
    assert by_id[env.tone.id]["kind"] == "tone" and by_id[env.desc.id]["revision"] == 1
    assert env.store.notes[(PID, env.desc.id)].status == "proposed"
    # separated, labelled sections
    for heading in ("Location facts:", "Physical constraints", "Tone and atmosphere:", "Context inherited from Devgram"):
        assert heading in p.prompt
    assert "Anshul's House" in p.prompt


def test_depiction_label_and_direction_in_prompt(env):
    p = env.svc.prepare(PID, env.house.id, reference_ids=[], depiction_label="whole-house exterior",
                        direction="  overcast morning light ")
    assert "View / depiction: whole-house exterior" in p.prompt and "overcast morning light" in p.prompt
    assert p.snapshot["direction"] == "overcast morning light"


def test_prompt_is_deterministic_and_token_tracks_content(env):
    a = env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id])
    b = env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id])
    assert (a.prompt, a.context_token) == (b.prompt, b.context_token)
    env.store.put_notes(PID, [Note(kind="tone", body="Hushed.", owner_id=env.house.id, author="user")])
    assert env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id]).context_token != a.context_token


def test_reference_selection_order_guidance_and_correspondence(env):
    ids = [env.refs["b"].id, env.refs["a"].id]
    p = env.svc.prepare(PID, env.house.id, reference_ids=ids)
    assert [r["noteId"] for r in p.snapshot["references"]] == ids
    assert [r["position"] for r in p.snapshot["references"]] == [1, 2]
    assert [r.data for r in p.request.references] == [png("orange"), png("green")]   # image_ref order == positions
    assert p.prompt.index("Reference image 1") < p.prompt.index("Reference image 2")
    assert "borrow: the warm wood texture" in p.prompt.split("Reference image 2")[0]
    assert "borrow: the canopy shape" in p.prompt.split("Reference image 2")[1]
    flipped = env.svc.prepare(PID, env.house.id, reference_ids=ids[::-1])
    assert flipped.context_token != p.context_token                 # order is part of the input
    assert env.svc.prepare(PID, env.house.id, reference_ids=ids + ids).context_token == p.context_token  # dedupe


def test_reference_without_guidance_is_labelled_not_invented(env):
    n = env.refs["a"]
    env.store.put_note_if_current(PID, n.touch(guidance=None), n.revision, n.status)
    p = env.svc.prepare(PID, env.house.id, reference_ids=[n.id])
    assert "no specific guidance was given" in p.prompt


@pytest.mark.parametrize("which,msg", [("c", "not confirmed"), ("zzz", "not a reference applicable")])
def test_unconfirmed_or_unrelated_references_rejected(env, which, msg):
    rid = env.refs["c"].id if which == "c" else "note_nope"
    with pytest.raises(ValueError, match=msg):
        env.svc.prepare(PID, env.house.id, reference_ids=[rid])


def test_rejected_reference_and_other_locations_reference_rejected(env):
    a = env.refs["a"]
    env.store.put_note_if_current(PID, a.touch(status="rejected", review_reason="not_useful"), a.revision, a.status)
    with pytest.raises(ValueError, match="rejected"):
        env.svc.prepare(PID, env.house.id, reference_ids=[a.id])
    other = Location(name="River", status="confirmed", author="agent")
    env.store.put_entities(PID, [other])
    n, _ = upload_reference_image(env.store, env.blobs, env.settings, PID, other.id, png("pink"), "r.png")
    env.store.put_note_if_current(PID, n.touch(status="confirmed"), n.revision, n.status)
    with pytest.raises(ValueError, match="not a reference applicable"):
        env.svc.prepare(PID, env.house.id, reference_ids=[n.id])


def test_capability_rejections(env):
    for kw, msg in (({"model": "ray-3.2"}, "not supported for image generation"),
                    ({"aspect_ratio": "4:3"}, "aspect ratio"), ({"output_format": "gif"}, "output format")):
        with pytest.raises(UnsupportedRequest, match=msg):
            env.svc.prepare(PID, env.house.id, reference_ids=[], **kw)


def test_more_references_than_capability_rejected(env):
    extra = []
    for i in range(2):
        n, _ = upload_reference_image(env.store, env.blobs, env.settings, PID, env.house.id, png(f"#0{i}0{i}ff"), f"x{i}.png")
        env.store.put_note_if_current(PID, n.touch(status="confirmed"), n.revision, n.status)
        extra.append(n.id)
    with pytest.raises(UnsupportedRequest, match="at most 3"):
        env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id, env.refs["b"].id, *extra])


def test_empty_inputs_rejected(env):
    e2 = Location(name="Bare", status="confirmed", author="agent")
    env.store.put_entities(PID, [e2])
    with pytest.raises(ValueError, match="nothing to generate from"):
        env.svc.prepare(PID, e2.id, reference_ids=[])


def test_scene_id_rejected(env):
    with pytest.raises(ValueError, match="scene"):
        env.svc.prepare(PID, env.s4.id, reference_ids=[])


# --- preview freshness / snapshot preservation -----------------------------------------------

def test_submit_rejects_stale_preview_and_persists_exact_snapshot(env):
    p = env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id], model="m2", aspect_ratio="16:9")
    job, created = env.svc.submit(PID, env.house.id, reference_ids=[env.refs["a"].id], model="m2",
                                  aspect_ratio="16:9", context_token=p.context_token, idempotency_key="k")
    assert created and job.state == "queued" and env.provider.submits == []      # nothing sent yet
    assert job.input_snapshot == p.snapshot and job.prompt == p.prompt and job.context_token == p.context_token
    assert (job.provider, job.model, job.prompt_version, job.workflow_version) == ("fake", "m2", "base-location-v1", "1")
    # later edits never alter the stored snapshot ...
    env.store.put_notes(PID, [Note(kind="tone", body="Changed later.", owner_id=env.house.id, author="user")])
    assert env.store.get_generation_job(PID, job.id).input_snapshot == p.snapshot
    # ... and a stale preview cannot create a new job
    with pytest.raises(StaleGenerationContext):
        env.svc.submit(PID, env.house.id, reference_ids=[env.refs["a"].id], model="m2", aspect_ratio="16:9",
                       context_token=p.context_token, idempotency_key="k2")
    assert len(env.store.list_generation_jobs(PID)) == 1


def test_worker_sends_the_validated_prompt_and_reference_bytes_unchanged(env):
    job, _ = submit(env, refs=("b", "a"))
    env.svc.run(PID, job.id)
    req = env.provider.submits[0]
    assert req.prompt == job.prompt and req.model == "m1" and req.aspect_ratio is None
    assert [r.data for r in req.references] == [png("orange"), png("green")]


# --- idempotency ---------------------------------------------------------------------------------

def test_same_key_same_payload_returns_same_job_different_payload_conflicts(env):
    j1, c1 = submit(env, key="same")
    j2, c2 = submit(env, key="same")
    assert (c1, c2, j1.id == j2.id) == (True, False, True)
    with pytest.raises(IdempotencyConflict):
        submit(env, key="same", direction="different")
    assert len(env.store.list_generation_jobs(PID)) == 1
    # a genuine retry still resolves even if inputs have since changed (it is a retry, not new work)
    p = env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id])
    env.store.put_notes(PID, [Note(kind="tone", body="Changed.", owner_id=env.house.id, author="user")])
    j3, c3 = env.svc.submit(PID, env.house.id, reference_ids=[env.refs["a"].id], context_token=p.context_token,
                            idempotency_key="same")
    assert (c3, j3.id) == (False, j1.id)


def test_idempotency_scoped_per_project(env):
    assert (env.svc.job_id_for("p1", "k") != env.svc.job_id_for("p2", "k")
            and env.svc.job_id_for("p1", "k") == env.svc.job_id_for("p1", "k"))


def test_duplicate_submission_never_creates_two_provider_generations(env):
    j, _ = submit(env, key="dup")
    env.svc.run(PID, j.id)
    j2, _ = submit(env, key="dup")
    env.svc.run(PID, j2.id)          # terminal job: not claimable
    assert len(env.provider.submits) == 1 and j2.id == j.id


# --- atomic ownership / concurrency -----------------------------------------------------------

def test_concurrent_workers_submit_once(env):
    j, _ = submit(env)
    started = threading.Event()

    def slow_submit(req):
        started.set()
        time.sleep(0.15)
        return "gen-only"

    env.provider.submit_hook = slow_submit
    real = ConceptGenerationService(env.store, env.blobs, env.settings, env.provider,
                                    config=GenerationConfig(default_model="m1", poll_interval_s=0.01, max_wait_s=5, lease_s=30))
    out = []
    ts = [threading.Thread(target=lambda: out.append(real.run(PID, j.id))) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert len(env.provider.submits) == 1
    final = env.store.get_generation_job(PID, j.id)
    assert final.state == "succeeded" and final.submit_attempts == 1 and final.provider_generation_id == "gen-only"


def test_expired_lease_can_be_reclaimed_but_live_lease_cannot(env):
    j, _ = submit(env)
    env.provider.get_script = [ProcessDied()]
    with pytest.raises(ProcessDied):
        env.svc.run(PID, j.id, owner="w1")
    stuck = env.store.get_generation_job(PID, j.id)
    assert stuck.state == "submitted" and stuck.lease_owner == "w1"
    assert env.svc.run(PID, j.id, owner="w2").lease_owner == "w1"      # live lease: not stolen
    assert not env.svc.can_resume(stuck)
    env.clock.sleep(31)                                                # lease lapses
    assert env.svc.can_resume(stuck)
    assert env.svc.run(PID, j.id, owner="w2").state == "succeeded"
    assert len(env.provider.submits) == 1


# --- ambiguous submission -------------------------------------------------------------------------

def test_unknown_submission_outcome_is_never_auto_resubmitted(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("timeout after send"))
    out = env.svc.run(PID, j.id)
    assert out.state == "submission_unknown" and out.provider_generation_id is None and out.submit_attempts == 1
    for _ in range(3):
        env.svc.run(PID, j.id)
    assert env.svc.recover() == []
    assert len(env.provider.submits) == 1 and not env.svc.can_resume(out)


def test_process_death_mid_submit_becomes_unknown_after_restart(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(ProcessDied())
    with pytest.raises(ProcessDied):
        env.svc.run(PID, j.id)
    assert env.store.get_generation_job(PID, j.id).state == "submitting"
    fresh = ConceptGenerationService(env.store, env.blobs, env.settings, env.provider, config=env.svc.config,
                                     sleep=env.clock.sleep, clock=env.clock.now)
    assert fresh.recover() == []            # the dead worker's lease is still live: recovery must not touch it
    assert env.store.get_generation_job(PID, j.id).state == "submitting"
    env.clock.sleep(31)                     # ... until it lapses
    assert fresh.recover() == []            # then it becomes unknown, and nothing is auto-run
    after = env.store.get_generation_job(PID, j.id)
    assert after.state == "submission_unknown" and after.lease_owner is None
    fresh.run(PID, j.id)
    assert len(env.provider.submits) == 1


def test_resubmit_requires_explicit_acknowledgement(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("x"))
    env.svc.run(PID, j.id)
    with pytest.raises(ValueError, match="acknowledge"):
        env.svc.resolve(PID, j.id, action="resubmit")
    with pytest.raises(ValueError, match="attach_provider_id or resubmit"):
        env.svc.resolve(PID, j.id, action="cancel")
    assert env.store.get_generation_job(PID, j.id).state == "submission_unknown"


def test_attach_existing_provider_job_completes_without_new_submission(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("x"))
    env.svc.run(PID, j.id)
    with pytest.raises(ValueError, match="providerGenerationId"):
        env.svc.resolve(PID, j.id, action="attach_provider_id")
    r = env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-found")
    assert r.state == "submitted"
    assert env.svc.run(PID, j.id).state == "succeeded"
    assert len(env.provider.submits) == 1


def test_explicit_resubmit_is_the_only_way_to_a_second_submission(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("x"))
    env.svc.run(PID, j.id)
    env.provider.submit_hook = None
    assert env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True).state == "queued"
    out = env.svc.run(PID, j.id)
    assert out.state == "succeeded" and len(env.provider.submits) == 2 and out.submit_attempts == 2
    with pytest.raises(JobStateConflict):
        env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True)


def test_definite_rejection_fails_without_pretending_unknown(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionRejected("rate limit", code="rate_limited"))
    out = env.svc.run(PID, j.id)
    assert (out.state, out.failure_stage, out.failure_code) == ("failed", "submit", "rate_limited")
    assert env.svc.recover() == []
    assert env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True).state == "queued"


# --- restart recovery / polling / provider failure ---------------------------------------------------

def test_restart_with_known_provider_id_resumes_without_resubmitting(env):
    j, _ = submit(env)
    env.provider.get_script = [ProcessDied()]              # dies while polling, provider id already persisted
    with pytest.raises(ProcessDied):
        env.svc.run(PID, j.id)
    assert env.store.get_generation_job(PID, j.id).provider_generation_id == "gen-1"
    fresh = ConceptGenerationService(env.store, env.blobs, env.settings, env.provider, config=env.svc.config,
                                     sleep=env.clock.sleep, clock=env.clock.now)
    assert fresh.recover() == []            # lease of the dead worker not yet lapsed: hands off
    env.clock.sleep(31)
    assert fresh.recover() == [(PID, j.id)]
    out = fresh.run(PID, j.id)
    assert out.state == "succeeded" and len(env.provider.submits) == 1 and out.candidate_ids


def test_polling_deadline_is_not_failure_and_resume_continues_same_provider_job(env):
    j, _ = submit(env)
    env.provider.default_get = ProviderJob("gen-1", "processing")
    out = env.svc.run(PID, j.id)
    assert out.state == "poll_deadline_exceeded" and out.failure_stage is None and out.provider_generation_id == "gen-1"
    assert "may still be running" in out.error and env.svc.can_resume(out)
    env.provider.default_get = None                        # provider finished meanwhile
    out = env.svc.run(PID, j.id)
    assert out.state == "succeeded" and len(env.provider.submits) == 1


def test_consecutive_read_errors_stop_polling_without_concluding_failure(env):
    j, _ = submit(env)
    env.provider.get_script = [ProviderTransientError("503")] * 5
    out = env.svc.run(PID, j.id)
    assert out.state == "poll_deadline_exceeded" and "unaffected" in out.error and out.failure_stage is None
    assert env.svc.run(PID, j.id).state == "succeeded"


def test_transient_read_error_then_success(env):
    j, _ = submit(env)
    env.provider.get_script = [ProviderTransientError("x"), ProviderJob("gen-1", "processing")]
    assert env.svc.run(PID, j.id).state == "succeeded"


def test_provider_failure_is_terminal_and_not_resubmitted(env):
    j, _ = submit(env)
    env.provider.default_get = ProviderJob("gen-1", "failed", failure_code="content_moderated", failure_reason="flagged")
    out = env.svc.run(PID, j.id)
    assert (out.state, out.failure_stage, out.failure_code, out.error) == ("failed", "provider", "content_moderated", "flagged")
    assert env.svc.run(PID, j.id).state == "failed" and len(env.provider.submits) == 1 and out.candidate_ids == []


# --- import ---------------------------------------------------------------------------------------------

def test_expired_output_url_recovers_via_fresh_url_from_same_provider_job(env):
    j, _ = submit(env)
    env.provider.get_script = [ProviderJob("gen-1", "completed", ("https://cdn/old.png?sig=1",)),
                               ProviderJob("gen-1", "completed", ("https://cdn/old.png?sig=1",))]
    env.provider.download_hook = lambda url: (_ for _ in ()).throw(OutputUnavailable("expired"))
    out = env.svc.run(PID, j.id)
    assert out.state == "import_failed" and out.provider_completed_at and out.candidate_ids == []
    env.provider.download_hook = None
    env.provider.default_get = ProviderJob("gen-1", "completed", ("https://cdn/new.png?sig=2",))
    out = env.svc.run(PID, j.id)
    assert out.state == "succeeded" and len(env.provider.submits) == 1
    assert env.provider.downloads[-1].endswith("sig=2")


def test_import_success_links_candidate_with_provenance_and_stays_out_of_ingest(env):
    j, _ = submit(env)
    out = env.svc.run(PID, j.id)
    assert out.state == "succeeded" and len(out.candidate_ids) == 1 and out.reused_candidate_ids == []
    v = env.store.get_concept_version(PID, out.candidate_ids[0])
    assert (v.author, v.generation_job_id, v.location_id) == ("agent", j.id, env.house.id)
    src = env.store.get_source(PID, v.source_id)
    assert src.source_purpose == "concept" and not src.is_ingest_eligible
    assert env.blobs.get(src.storage_path) == env.provider.output_bytes
    # not a reference note, not an ingestible source
    assert all(n.kind != "reference_image" or n.id in {r.id for r in env.refs.values()}
               for n in env.store.list_notes(PID))


def test_import_retry_is_exactly_once_and_not_miscounted_as_reuse(env):
    j, _ = submit(env)
    first = env.svc.run(PID, j.id)
    # simulate a crash after the candidate was created but before the job recorded it
    env.store.update_generation_job(PID, j.id, lambda x: True, lambda x: x.model_copy(update={
        "state": "generated", "candidate_ids": [], "reused_candidate_ids": [], "finished_at": None}))
    again = env.svc.run(PID, j.id)
    assert again.state == "succeeded" and again.candidate_ids == first.candidate_ids
    assert again.reused_candidate_ids == [] and len(env.store.list_concept_versions(PID, env.house.id)) == 1
    assert len(env.provider.submits) == 1


def test_identical_output_reuses_candidate_but_keeps_each_jobs_history(env):
    j1, _ = submit(env, key="one")
    o1 = env.svc.run(PID, j1.id)
    j2, _ = submit(env, key="two")
    o2 = env.svc.run(PID, j2.id)
    assert o2.candidate_ids == o1.candidate_ids and o2.reused_candidate_ids == o1.candidate_ids
    v = env.store.get_concept_version(PID, o1.candidate_ids[0])
    assert v.generation_job_id == j1.id                                   # provenance not falsely rewritten
    assert len(env.store.list_concept_versions(PID, env.house.id)) == 1
    assert len(env.provider.submits) == 2                                 # two real generations, both recorded


def test_generated_bytes_matching_an_uploaded_candidate_keep_the_uploads_provenance(env):
    up, _ = upload_concept_version(env.store, env.blobs, env.settings, PID, env.house.id, env.provider.output_bytes, "mine.png")
    j, _ = submit(env)
    out = env.svc.run(PID, j.id)
    assert out.candidate_ids == [up.id] and out.reused_candidate_ids == [up.id]
    v = env.store.get_concept_version(PID, up.id)
    assert (v.author, v.generation_job_id) == ("user", None)


def test_invalid_output_bytes_are_an_import_failure_not_a_provider_failure(env):
    j, _ = submit(env)
    env.provider.output_bytes = b"definitely not an image"
    out = env.svc.run(PID, j.id)
    assert (out.state, out.failure_stage, out.failure_code) == ("import_failed", "import", "output_invalid_image")
    assert out.provider_generation_id == "gen-1" and out.error.startswith("Import failed (the provider generated")
    assert "generation failed" not in out.error.lower()
    assert len(env.provider.submits) == 1 and env.store.list_concept_versions(PID, env.house.id) == []


def test_download_network_failure_is_resumable(env):
    j, _ = submit(env)
    env.provider.download_hook = lambda url: (_ for _ in ()).throw(ProviderTransientError("reset"))
    assert env.svc.run(PID, j.id).state == "import_failed"
    env.provider.download_hook = None
    assert env.svc.run(PID, j.id).state == "succeeded"


# --- generation never approves / confirms / ingests ------------------------------------------------------

def test_generation_never_touches_approval_or_note_review(env):
    up, _ = upload_concept_version(env.store, env.blobs, env.settings, PID, env.house.id, png("gray"), "u.png")
    snap = build_approval_snapshot(env.store, PID, env.house.id, up.id, [])
    approval, _ = lock_approval(env.store, PID, env.house.id, concept_version_id=up.id, reference_ids=[],
                                context_token=snap.context_token, expected_revision=0, locked_by="d")
    before = {k: n.model_copy(deep=True) for k, n in env.store.notes.items()}
    j, _ = submit(env)
    out = env.svc.run(PID, j.id)
    assert out.state == "succeeded"
    cur = env.store.get_current_approval(PID, env.house.id)
    assert (cur.id, cur.revision, cur.concept_version_id) == (approval.id, 1, up.id)
    assert env.store.notes == before                                     # no note confirmed/edited/created
    src_ids = {s.id for s in env.store.list_sources(PID) if s.is_ingest_eligible}
    assert env.store.get_concept_version(PID, out.candidate_ids[0]).source_id not in src_ids


# --- HTTP ----------------------------------------------------------------------------------------------------

def client_for(e, provider="default", run=False):
    app = create_app(
        settings=Settings(gcp_project="t", bucket="b"),
        api_settings=ApiSettings(allowed_hosts=("testserver",), generation_default_model="m1"),
        store=e.store, blobs=e.blobs, images=MemoryImages({}, {}, {}),
        image_provider=e.provider if provider == "default" else None, run_generation_jobs=run)
    if provider == "default":
        app.state.generation_service = e.svc      # share the fake clock/config
        app.state.generation_dispatcher = None
    return TestClient(app, base_url="http://testserver")


def test_http_preview_submit_inspect_flow(env):
    c = client_for(env)
    base = f"/projects/{PID}/locations/{env.house.id}/concept-generation"
    opts = c.get(f"/projects/{PID}/concept-generation/options").json()
    assert (opts["configured"], opts["provider"], opts["defaultModel"], opts["maxReferences"]) == (True, "fake", "m1", 3)
    assert opts["promptMaxChars"] == 6000 and opts["maxRequestBytes"] == 20_000_000

    body = {"referenceIds": [env.refs["b"].id, env.refs["a"].id], "depictionLabel": "whole-house exterior",
            "direction": "morning light", "aspectRatio": "16:9"}
    pv = c.post(f"{base}/preview", json=body)
    assert pv.status_code == 200
    pv = pv.json()
    assert pv["model"] == "m1" and pv["promptVersion"] == "base-location-v1" and "morning light" in pv["prompt"]
    assert [r["position"] for r in pv["references"]] == [1, 2]
    assert pv["references"][0]["image"] == f"/projects/{PID}/references/{env.refs['b'].id}/image"
    assert pv["snapshot"]["settings"]["aspectRatio"] == "16:9"

    r = c.post(f"{base}/jobs", json={**body, "contextToken": pv["contextToken"], "idempotencyKey": "ui-1"})
    assert r.status_code == 201 and r.json()["created"] is True
    job = r.json()["job"]
    assert job["state"] == "queued" and job["inputSnapshot"] == pv["snapshot"] and job["prompt"] == pv["prompt"]
    assert env.provider.submits == []                                     # submit persisted, worker not yet run

    r2 = c.post(f"{base}/jobs", json={**body, "contextToken": pv["contextToken"], "idempotencyKey": "ui-1"})
    assert r2.status_code == 200 and r2.json()["job"]["id"] == job["id"] and r2.json()["created"] is False
    assert c.post(f"{base}/jobs", json={**body, "direction": "other", "contextToken": pv["contextToken"],
                                        "idempotencyKey": "ui-1"}).status_code == 409

    env.svc.run(PID, job["id"])
    done = c.get(f"/projects/{PID}/concept-generation/jobs/{job['id']}").json()
    assert done["state"] == "succeeded" and done["terminal"] and not done["needsAttention"]
    cand = done["candidates"][0]
    assert cand["reused"] is False and c.get(cand["image"]).status_code == 200
    listing = c.get(f"{base}/jobs").json()
    assert [j["id"] for j in listing] == [job["id"]]
    cv = c.get(f"/projects/{PID}/locations/{env.house.id}/concepts").json()
    assert cv["approvedVersionId"] is None and cv["versions"][0]["generationJobId"] == job["id"]
    assert cv["versions"][0]["author"] == "agent" and cv["versions"][0]["approved"] is False


def test_http_stale_preview_409_and_validation_400_and_404(env):
    c = client_for(env)
    base = f"/projects/{PID}/locations/{env.house.id}/concept-generation"
    pv = c.post(f"{base}/preview", json={"referenceIds": []}).json()
    env.store.put_notes(PID, [Note(kind="tone", body="New.", owner_id=env.house.id, author="user")])
    assert c.post(f"{base}/jobs", json={"referenceIds": [], "contextToken": pv["contextToken"]}).status_code == 409
    assert c.post(f"{base}/preview", json={"referenceIds": [], "model": "ray-3.2"}).status_code == 400
    assert c.post(f"{base}/preview", json={"referenceIds": [env.refs["c"].id]}).status_code == 400
    assert c.post(f"/projects/{PID}/locations/nope/concept-generation/preview", json={}).status_code == 404
    assert c.get(f"/projects/{PID}/concept-generation/jobs/gjob_none").status_code == 404
    assert c.post(f"{base}/preview", json={}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_http_not_configured_disables_actions_but_keeps_history(env, monkeypatch):
    monkeypatch.delenv("LUMA_AGENTS_API_KEY", raising=False)
    j, _ = submit(env)
    env.svc.run(PID, j.id)                                                # history created while configured
    c = client_for(env, provider=None)
    base = f"/projects/{PID}/locations/{env.house.id}/concept-generation"
    for r in (c.post(f"{base}/preview", json={}), c.post(f"{base}/jobs", json={"contextToken": "x"}),
              c.post(f"/projects/{PID}/concept-generation/jobs/{j.id}/resume"),
              c.post(f"/projects/{PID}/concept-generation/jobs/{j.id}/resolve", json={"action": "resubmit"})):
        assert r.status_code == 503 and "LUMA_AGENTS_API_KEY" in r.json()["detail"]
    opts = c.get(f"/projects/{PID}/concept-generation/options").json()
    assert opts["configured"] is False and "LUMA_AGENTS_API_KEY" in opts["configurationError"] and opts["models"] == []
    got = c.get(f"/projects/{PID}/concept-generation/jobs/{j.id}")
    assert got.status_code == 200 and got.json()["state"] == "succeeded" and got.json()["resumable"] is False
    assert [x["id"] for x in c.get(f"{base}/jobs").json()] == [j.id]
    cand = got.json()["candidates"][0]
    assert c.get(cand["image"]).status_code == 200
    assert c.get(f"/projects/{PID}/locations/{env.house.id}/concepts").json()["versions"][0]["id"] == cand["id"]


def test_http_resume_and_resolve(env):
    c = client_for(env)
    j, _ = submit(env)
    env.provider.default_get = ProviderJob("gen-1", "processing")
    env.svc.run(PID, j.id)
    got = c.get(f"/projects/{PID}/concept-generation/jobs/{j.id}").json()
    assert got["state"] == "poll_deadline_exceeded" and got["resumable"] and got["failureStage"] is None
    assert c.post(f"/projects/{PID}/concept-generation/jobs/{j.id}/resume").status_code == 202
    env.provider.default_get = None
    env.svc.run(PID, j.id)
    assert c.post(f"/projects/{PID}/concept-generation/jobs/{j.id}/resume").status_code == 409     # already succeeded

    j2, _ = submit(env, key="unk")
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("x"))
    env.svc.run(PID, j2.id)
    assert c.get(f"/projects/{PID}/concept-generation/jobs/{j2.id}").json()["needsAttention"] is True
    url = f"/projects/{PID}/concept-generation/jobs/{j2.id}/resolve"
    assert c.post(url, json={"action": "resubmit"}).status_code == 400
    assert c.post(url, json={"action": "attach_provider_id", "providerGenerationId": "gen-x"}).status_code == 200
    assert c.post(url, json={"action": "attach_provider_id", "providerGenerationId": "gen-x"}).status_code == 409


def test_http_lifespan_recovers_and_resumes_jobs(env):
    j, _ = submit(env)
    env.provider.get_script = [ProcessDied()]
    with pytest.raises(ProcessDied):
        env.svc.run(PID, j.id)
    stuck2, _ = submit(env, key="dies-in-submit")
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(ProcessDied())
    with pytest.raises(ProcessDied):
        env.svc.run(PID, stuck2.id)
    env.provider.submit_hook = None
    app = create_app(settings=Settings(gcp_project="t", bucket="b"),
                     api_settings=ApiSettings(allowed_hosts=("testserver",), generation_poll_interval_s=0.01),
                     store=env.store, blobs=env.blobs, images=MemoryImages({}, {}, {}),
                     image_provider=env.provider, run_generation_jobs=True)
    with TestClient(app, base_url="http://testserver"):
        deadline = time.time() + 5
        while time.time() < deadline and env.store.get_generation_job(PID, j.id).state != "succeeded":
            time.sleep(0.02)
        assert env.store.get_generation_job(PID, j.id).state == "succeeded"
        assert env.store.get_generation_job(PID, stuck2.id).state == "submission_unknown"
    assert len(env.provider.submits) == 2         # one per original submit; recovery added none


# =====================================================================================================
# Hardening pass
# =====================================================================================================

def real_svc(env, provider="same", **cfg):
    """A second service sharing the SAME store - stands in for another dispatcher/process. Real clock."""
    base = dict(default_model="m1", poll_interval_s=0.01, max_wait_s=5, lease_s=0.3)
    base.update(cfg)
    return ConceptGenerationService(env.store, env.blobs, env.settings,
                                    env.provider if provider == "same" else provider, config=GenerationConfig(**base))


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


# -- 2. worker ownership: startup / a second dispatcher must not disturb a live lease ------------------

def test_second_dispatcher_startup_does_not_invalidate_a_working_workers_lease(env):
    a = real_svc(env)
    j, _ = submit(env, svc=a)
    env.provider.get_hook = lambda gid: (time.sleep(1.0), env.provider.done(gid))[1]     # long provider call
    t = threading.Thread(target=lambda: a.run(PID, j.id))
    t.start()
    assert wait_for(lambda: env.store.get_generation_job(PID, j.id).state == "submitted")
    owner = env.store.get_generation_job(PID, j.id).lease_owner
    b = real_svc(env)                                        # "startup" of another server sharing the store
    for _ in range(6):                                       # spans > 3x the 0.3s lease: only heartbeats keep it alive
        assert b.recover() == []
        cur = env.store.get_generation_job(PID, j.id)
        assert cur.lease_owner == owner and cur.state == "submitted"
        assert b.run(PID, j.id).lease_owner == owner          # cannot steal it either
        time.sleep(0.15)
    t.join(timeout=10)
    final = env.store.get_generation_job(PID, j.id)
    assert final.state == "succeeded" and len(env.provider.submits) == 1


def test_second_startup_during_a_long_submit_never_marks_it_unknown(env):
    a = real_svc(env)
    j, _ = submit(env, svc=a)
    env.provider.submit_hook = lambda req: (time.sleep(0.9), "gen-slow")[1]
    t = threading.Thread(target=lambda: a.run(PID, j.id))
    t.start()
    assert wait_for(lambda: env.store.get_generation_job(PID, j.id).state == "submitting")
    b = real_svc(env)
    for _ in range(5):
        assert b.recover() == [] and env.store.get_generation_job(PID, j.id).state == "submitting"
        time.sleep(0.15)
    t.join(timeout=10)
    final = env.store.get_generation_job(PID, j.id)
    assert final.state == "succeeded" and final.provider_generation_id == "gen-slow" and len(env.provider.submits) == 1
    assert [x["outcome"] for x in final.attempt_history] == ["accepted"]


def test_recovery_only_takes_unowned_or_lapsed_jobs(env):
    live, _ = submit(env, key="live")
    lapsed, _ = submit(env, key="lapsed")
    fresh, _ = submit(env, key="fresh")
    now = datetime.now(timezone.utc)
    for job, expires in ((live, now + timedelta(seconds=60)), (lapsed, now - timedelta(seconds=1))):
        env.store.update_generation_job(PID, job.id, lambda x: True, lambda x, e=expires: x.model_copy(update={
            "state": "submitted", "provider_generation_id": "gen-x", "lease_owner": "dead-or-alive",
            "lease_expires_at": e}))
    got = real_svc(env).recover()
    assert (PID, live.id) not in got and (PID, lapsed.id) in got and (PID, fresh.id) in got
    assert env.store.get_generation_job(PID, live.id).lease_owner == "dead-or-alive"      # untouched
    assert env.store.get_generation_job(PID, lapsed.id).lease_owner is None               # stale lease cleared


def test_lost_lease_fences_the_old_worker_out(env):
    j, _ = submit(env)

    def steal(gid):
        env.store.update_generation_job(PID, j.id, lambda x: True, lambda x: x.model_copy(update={
            "lease_owner": "someone-else", "lease_expires_at": datetime.now(timezone.utc) + timedelta(seconds=60)}))
        return env.provider.done(gid)

    env.provider.get_hook = steal
    out = env.svc.run(PID, j.id)
    assert out.lease_owner == "someone-else" and out.state == "submitted" and out.candidate_ids == []
    assert env.store.list_concept_versions(PID, env.house.id) == []


def test_late_provider_id_is_adopted_after_recovery_recorded_the_attempt_unknown(env):
    j, _ = submit(env)

    def submit_then_recovery_interferes(req):
        env.store.update_generation_job(PID, j.id, lambda x: True, lambda x: x.model_copy(update={
            "state": "submission_unknown", "lease_owner": None, "lease_expires_at": None,
            "attempt_history": [{"attempt": 1, "outcome": "unknown", "detail": "recorded by recovery"}]}))
        return "gen-late"

    env.provider.submit_hook = submit_then_recovery_interferes
    out = env.svc.run(PID, j.id)
    assert out.state == "submitted" and out.provider_generation_id == "gen-late" and out.lease_owner is None
    assert [a["outcome"] for a in out.attempt_history] == ["unknown", "accepted"]
    assert "already been recorded as unknown" in out.attempt_history[1]["detail"]
    env.provider.submit_hook = None
    assert env.svc.run(PID, j.id).state == "succeeded" and len(env.provider.submits) == 1


def test_two_dispatchers_sharing_a_store_run_a_job_once(env):
    from harness.api.generation import GenerationDispatcher
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (time.sleep(0.1), "gen-once")[1]

    async def go():
        d1 = GenerationDispatcher(real_svc(env), max_concurrent=2)
        d2 = GenerationDispatcher(real_svc(env), max_concurrent=2)
        d1.schedule(PID, j.id)
        d2.schedule(PID, j.id)
        await asyncio.gather(*d1._tasks, *d2._tasks)

    asyncio.run(go())
    assert len(env.provider.submits) == 1
    assert env.store.get_generation_job(PID, j.id).state == "succeeded"


def test_sweeper_adopts_a_dead_workers_job_after_its_lease_lapses(env):
    from harness.api.generation import GenerationDispatcher
    j, _ = submit(env)
    env.store.update_generation_job(PID, j.id, lambda x: True, lambda x: x.model_copy(update={
        "state": "submitted", "provider_generation_id": "gen-dead", "lease_owner": "dead",
        "lease_expires_at": datetime.now(timezone.utc) + timedelta(seconds=0.4)}))

    async def go():
        d = GenerationDispatcher(real_svc(env), max_concurrent=1)
        d.start_sweeper(0.05)
        try:
            for _ in range(200):
                await asyncio.sleep(0.02)
                if env.store.get_generation_job(PID, j.id).state == "succeeded":
                    return True
            return False
        finally:
            await d.stop()

    assert asyncio.run(go())
    assert env.provider.submits == []                               # adopted, never resubmitted


def test_submission_unknown_is_left_alone_by_every_recovery_path(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("x"))
    env.svc.run(PID, j.id)
    other = real_svc(env)
    for _ in range(3):
        assert other.recover() == []
        assert other.run(PID, j.id).state == "submission_unknown"
    assert len(env.provider.submits) == 1


# -- 3. availability vs history ----------------------------------------------------------------------------

def test_service_without_provider_reads_history_and_repairs_bookkeeping_only(env):
    done, _ = submit(env, key="done")
    env.svc.run(PID, done.id)
    stuck, _ = submit(env, key="stuck")
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(ProcessDied())
    with pytest.raises(ProcessDied):
        env.svc.run(PID, stuck.id)
    waiting, _ = submit(env, key="waiting")
    env.store.update_generation_job(PID, waiting.id, lambda x: True, lambda x: x.model_copy(update={
        "state": "submitted", "provider_generation_id": "gen-w"}))
    bare = real_svc(env, provider=None)
    from harness.memory.concept_generation import ProviderNotConfigured
    assert not bare.configured
    assert bare.get(PID, done.id).state == "succeeded" and len(bare.list(PID, env.house.id)) == 3
    for call in (lambda: bare.prepare(PID, env.house.id, reference_ids=[]),
                 lambda: bare.run(PID, waiting.id),
                 lambda: bare.resolve(PID, stuck.id, action="resubmit", acknowledge_no_provider_job=True)):
        with pytest.raises(ProviderNotConfigured):
            call()
    assert not bare.can_resume(bare.get(PID, waiting.id))
    time.sleep(0.4)                                                  # the dead worker's lease lapses
    assert bare.recover() == []                                      # nothing to resume without a provider ...
    assert bare.get(PID, stuck.id).state == "submission_unknown"     # ... but the ambiguity is still recorded
    assert bare.get(PID, waiting.id).state == "submitted"            # and nothing was disturbed or resubmitted


def test_app_starts_without_provider_and_serves_history(env, monkeypatch):
    monkeypatch.delenv("LUMA_AGENTS_API_KEY", raising=False)
    j, _ = submit(env)
    env.svc.run(PID, j.id)
    app = create_app(settings=Settings(gcp_project="t", bucket="b"), api_settings=ApiSettings(allowed_hosts=("testserver",)),
                     store=env.store, blobs=env.blobs, images=MemoryImages({}, {}, {}))
    with TestClient(app, base_url="http://testserver") as c:          # lifespan (incl. recovery) must not fail
        assert c.get(f"/projects/{PID}/concept-generation/jobs/{j.id}").json()["state"] == "succeeded"
        assert c.get(f"/projects/{PID}/concept-generation/options").json()["configured"] is False


# -- 4. explicit resolution --------------------------------------------------------------------------------

def unknown_job(env, key="k"):
    j, _ = submit(env, key=key)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionOutcomeUnknown("timeout after send"))
    env.svc.run(PID, j.id)
    env.provider.submit_hook = None
    return j


def test_attach_verifies_kind_model_and_creation_window(env):
    j = unknown_job(env)
    attach = lambda gid="gen-9": env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id=gid)
    env.provider.reported_kind = "image_edit"
    with pytest.raises(ValueError, match="not an image generation"):
        attach()
    env.provider.reported_kind = None
    with pytest.raises(ValueError, match="what kind of generation"):
        attach()
    env.provider.reported_kind, env.provider.reported_model = "image", "other-model"
    with pytest.raises(ValueError, match="requested 'm1'"):
        attach()
    env.provider.reported_model = None
    with pytest.raises(ValueError, match="did not report the model"):
        attach()
    env.provider.reported_model = "m1"
    env.provider.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)          # long before the submission
    with pytest.raises(ValueError, match="before this job's submission began"):
        attach()
    env.provider.created_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="future"):
        attach()
    assert env.store.get_generation_job(PID, j.id).state == "submission_unknown"   # nothing changed on refusal


def test_attach_rejects_an_id_already_associated_with_another_job_in_any_project(env):
    first, _ = submit(env, key="first")
    env.svc.run(PID, first.id)                                                       # provider id gen-1
    j = unknown_job(env, key="second")
    from harness.memory.concept_generation import JobStateConflict
    with pytest.raises(JobStateConflict, match=first.id):
        env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-1")
    # ... including a job in a different project
    env.store.put_project(Project(id="prj_other", name="Other"))
    twin = env.store.get_generation_job(PID, first.id).model_copy(update={
        "id": "gjob_other", "project_id": "prj_other", "provider_generation_id": "gen-elsewhere"})
    env.store.put_generation_job_if_absent("prj_other", twin)
    with pytest.raises(JobStateConflict, match="gjob_other"):
        env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-elsewhere")


def test_attach_rejects_when_the_configured_provider_changed(env):
    j = unknown_job(env)
    env.provider.name = "another"
    from harness.memory.concept_generation import JobStateConflict
    try:
        with pytest.raises(JobStateConflict, match="provider 'fake'"):
            env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-9")
    finally:
        env.provider.name = "fake"


def test_attach_records_audit_with_what_was_and_was_not_verified(env):
    j = unknown_job(env)
    out = env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-found", by="director")
    r = out.resolutions[-1]
    assert (r["action"], r["by"], r["previousState"], r["providerGenerationId"]) == (
        "attach_provider_id", "director", "submission_unknown", "gen-found")
    assert r["previousError"] and "timeout after send" in r["previousError"]
    assert r["verified"]["kind"] == "image" and r["verified"]["model"] == "m1" and r["verified"]["notAssociatedWithAnotherJob"]
    assert any("prompt" in u for u in r["unverifiable"]) and any("reference images" in u for u in r["unverifiable"])
    assert [a["outcome"] for a in out.attempt_history] == ["unknown"]                   # ambiguity still on record
    done = env.svc.run(PID, j.id)
    assert done.state == "succeeded" and done.resolutions == out.resolutions and len(env.provider.submits) == 1


def test_attach_without_creation_time_is_allowed_but_flagged_unverifiable(env):
    j = unknown_job(env)
    env.provider.created_at = None
    out = env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-9")
    assert "the creation time" in out.resolutions[-1]["unverifiable"] and out.resolutions[-1]["verified"]["createdAt"] is None


def test_acknowledged_resubmit_never_erases_the_ambiguous_attempt(env):
    j = unknown_job(env)
    q = env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True, by="director")
    assert q.state == "queued" and q.error is None
    assert [a["outcome"] for a in q.attempt_history] == ["unknown"]                     # survives the reset
    r = q.resolutions[-1]
    assert r["acknowledgedDuplicateChargeRisk"] is True and r["previousState"] == "submission_unknown"
    assert "second, separately charged" in r["warning"] and "timeout after send" in r["previousError"]
    done = env.svc.run(PID, j.id)
    assert done.state == "succeeded" and done.submit_attempts == 2
    assert [a["outcome"] for a in done.attempt_history] == ["unknown", "accepted"]
    assert [a["attempt"] for a in done.attempt_history] == [1, 2]
    assert done.resolutions[-1]["acknowledgedDuplicateChargeRisk"] is True             # still there after success


def test_resubmit_after_a_definite_rejection_records_no_charge_risk_and_keeps_history(env):
    j, _ = submit(env)
    env.provider.submit_hook = lambda req: (_ for _ in ()).throw(SubmissionRejected("nope", code="http_422"))
    failed = env.svc.run(PID, j.id)
    assert [a["outcome"] for a in failed.attempt_history] == ["rejected"]
    env.provider.submit_hook = None
    q = env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True)
    assert "definitively rejected" in q.resolutions[-1]["warning"] and [a["outcome"] for a in q.attempt_history] == ["rejected"]


def test_http_resolve_records_by_and_conflicts(env):
    c = client_for(env)
    j = unknown_job(env)
    url = f"/projects/{PID}/concept-generation/jobs/{j.id}/resolve"
    r = c.post(url, json={"action": "attach_provider_id", "providerGenerationId": "gen-h", "by": "jagan"})
    assert r.status_code == 200 and r.json()["resolutions"][0]["by"] == "jagan"
    assert r.json()["attemptHistory"][0]["outcome"] == "unknown"
    j2 = unknown_job(env, key="k2")
    assert c.post(f"/projects/{PID}/concept-generation/jobs/{j2.id}/resolve",
                  json={"action": "attach_provider_id", "providerGenerationId": "gen-h"}).status_code == 409
    env.provider.reported_kind = "video"
    assert c.post(f"/projects/{PID}/concept-generation/jobs/{j2.id}/resolve",
                  json={"action": "attach_provider_id", "providerGenerationId": "gen-z"}).status_code == 400


# -- 5. output limits ---------------------------------------------------------------------------------------

def test_oversized_output_is_an_import_failure_with_provider_id_kept_and_no_regeneration(env):
    tight = real_svc(env, max_output_pixels=100)                     # our import bound; the fake image is 24x24=576 px
    j, _ = submit(env, svc=tight)
    out = tight.run(PID, j.id)
    assert (out.state, out.failure_stage, out.failure_code) == ("import_failed", "import", "output_too_many_pixels")
    assert out.provider_generation_id == "gen-1" and "limit is 100 px" in out.error
    assert out.error.startswith("Import failed (the provider generated the image successfully)")
    assert "provider" not in (out.failure_stage or "provider")[:0] and out.failure_stage == "import"
    assert len(env.provider.submits) == 1 and out.provider_completed_at is not None and out.candidate_ids == []
    assert tight.can_resume(out) and tight.recover() == []          # user-driven; never auto-regenerated or auto-retried
    again = tight.run(PID, j.id)                                     # same config: same specific failure, still no regeneration
    assert again.state == "import_failed" and again.failure_code == "output_too_many_pixels" and len(env.provider.submits) == 1


def test_import_retry_after_a_configuration_change_succeeds_from_the_same_provider_job(env):
    tight = real_svc(env, max_output_pixels=100)
    j, _ = submit(env, svc=tight)
    assert tight.run(PID, j.id).state == "import_failed"
    roomy = real_svc(env, max_output_pixels=1_000_000)               # e.g. HARNESS_GENERATION_MAX_OUTPUT_PIXELS raised, restarted
    out = roomy.run(PID, j.id)
    assert out.state == "succeeded" and out.failure_code is None and out.failure_stage is None and out.error is None
    assert len(out.candidate_ids) == 1 and len(env.provider.submits) == 1 and out.submit_attempts == 1


def test_output_bounds_are_still_bounded_and_specific(env):
    with pytest.raises(ValueError):
        GenerationConfig(max_output_pixels=0)
    with pytest.raises(ValueError):
        GenerationConfig(max_output_pixels=10**9)
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, format="GIF")
    for output, mime, code in ((gif.getvalue(), "image/gif", "output_unsupported_format"),
                               (gif.getvalue(), "image/png", "output_invalid_image"),      # mislabelled bytes
                               (b"junk", "image/png", "output_invalid_image")):
        e_job, _ = submit(env, key=f"{code}-{mime}-{len(output)}")
        env.provider.download_hook = lambda url, o=output, m=mime: DownloadedImage(o, m)
        out = env.svc.run(PID, e_job.id)
        assert (out.state, out.failure_code) == ("import_failed", code) and out.provider_generation_id


def test_download_too_large_and_missing_output_are_import_failures(env):
    j, _ = submit(env, key="big")
    env.provider.download_hook = lambda url: (_ for _ in ()).throw(OutputTooLarge("output exceeds the bound"))
    out = env.svc.run(PID, j.id)
    assert (out.state, out.failure_code) == ("import_failed", "output_too_large_bytes")
    env.provider.download_hook = None
    j2, _ = submit(env, key="missing")
    env.provider.default_get = ProviderJob("gen-2", "completed", ())
    out = env.svc.run(PID, j2.id)
    assert (out.state, out.failure_code) == ("import_failed", "output_missing")
    assert len(env.provider.submits) == 2


def test_prompt_over_the_provider_limit_is_rejected_before_any_job_exists(env):
    for i in range(4):
        env.store.put_notes(PID, [Note(kind="description", body=f"{i}" + "x" * 1900, owner_id=env.house.id, author="user")])
    with pytest.raises(UnsupportedRequest, match="characters"):
        env.svc.prepare(PID, env.house.id, reference_ids=[])
    assert env.store.list_generation_jobs(PID) == []


def test_request_size_limit_is_enforced_at_preparation_and_labelled_ours(env):
    env.provider.capabilities = dataclasses.replace(CAPS, max_request_bytes=1500)
    try:
        with pytest.raises(UnsupportedRequest, match="operational limit"):
            env.svc.prepare(PID, env.house.id, reference_ids=[env.refs["a"].id])
    finally:
        env.provider.capabilities = CAPS


def test_import_failed_http_shape(env):
    tight = real_svc(env, max_output_pixels=100)
    j, _ = submit(env, svc=tight)
    tight.run(PID, j.id)
    c = client_for(env)
    app_svc = c.app.state.generation_service
    app_svc.config = tight.config           # the app's own bound is what an operator would change
    got = c.get(f"/projects/{PID}/concept-generation/jobs/{j.id}").json()
    assert (got["state"], got["terminal"], got["resumable"], got["failureStage"], got["failureCode"]) == (
        "import_failed", False, True, "import", "output_too_many_pixels")
    assert got["providerGenerationId"] == "gen-1" and got["candidates"] == []


# --- atomic provider-generation-id claims ---------------------------------------------------------

def two_jobs(env):
    a, _ = submit(env, key="ka")
    b, _ = submit(env, key="kb")
    return a, b


def claim(env, job, gid="gen-x", scope="acct-1", project=PID):
    return env.store.claim_provider_generation(
        project, job.id, provider="fake", scope=scope, generation_id=gid,
        guard=lambda j: j.provider_generation_id is None,
        apply=lambda j: j.model_copy(update={"provider_generation_id": gid, "state": "submitted"}))


def test_store_claim_is_idempotent_for_the_same_job_and_conflicts_for_another(env):
    from harness.memory.ports import ProviderIdConflict
    a, b = two_jobs(env)
    first = claim(env, a)
    again = claim(env, a)                                   # retry after an ambiguous store error
    assert first.provider_generation_id == again.provider_generation_id == "gen-x"
    assert len(env.store.get_generation_job(PID, a.id).attempt_history) == 0    # no double-apply
    with pytest.raises(ProviderIdConflict) as e:
        claim(env, b)
    assert e.value.job_id == a.id
    assert env.store.get_generation_job(PID, b.id).provider_generation_id is None   # nothing written
    assert claim(env, b, scope="acct-2").provider_generation_id == "gen-x"          # other account: distinct id space


def test_store_claim_guard_failure_writes_no_claim(env):
    a, b = two_jobs(env)
    assert env.store.claim_provider_generation(PID, a.id, provider="fake", scope="acct-1", generation_id="gen-g",
                                               guard=lambda j: False, apply=lambda j: j) is None
    assert claim(env, b, gid="gen-g").provider_generation_id == "gen-g"             # still free


def test_store_claim_spans_projects(env):
    from harness.memory.ports import ProviderIdConflict
    a, _ = two_jobs(env)
    env.store.put_project(Project(id="prj_other", name="Other"))
    twin = a.model_copy(update={"id": "gjob_twin", "project_id": "prj_other"})
    env.store.put_generation_job_if_absent("prj_other", twin)
    claim(env, a)
    with pytest.raises(ProviderIdConflict):
        claim(env, twin, project="prj_other")


def test_concurrent_store_claims_yield_exactly_one_winner(env):
    from harness.memory.ports import ProviderIdConflict
    jobs = [submit(env, key=f"k{i}")[0] for i in range(8)]
    barrier, results = threading.Barrier(len(jobs)), []

    def go(j):
        barrier.wait()
        try:
            results.append(("won", claim(env, j, gid="gen-race").id))
        except ProviderIdConflict:
            results.append(("lost", j.id))
    ts = [threading.Thread(target=go, args=(j,)) for j in jobs]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(r[0] for r in results) == ["lost"] * 7 + ["won"]
    holders = [j for j in env.store.list_generation_jobs(PID) if j.provider_generation_id == "gen-race"]
    assert len(holders) == 1


def test_concurrent_manual_attach_of_one_id_to_two_jobs_has_one_winner(env):
    a, b = unknown_job(env, key="ka"), unknown_job(env, key="kb")
    barrier = threading.Barrier(2)
    inner = env.provider.done

    def get_hook(gid):                     # both pass the provider check, then race for the claim
        barrier.wait(timeout=5)
        return inner(gid)
    env.provider.get_hook = get_hook
    outcomes = {}

    def attach(j):
        try:
            outcomes[j.id] = env.svc.resolve(PID, j.id, action="attach_provider_id", provider_generation_id="gen-same")
        except JobStateConflict as exc:
            outcomes[j.id] = exc
    ts = [threading.Thread(target=attach, args=(j,)) for j in (a, b)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    won = [j for j in (a, b) if not isinstance(outcomes[j.id], Exception)]
    lost = [j for j in (a, b) if isinstance(outcomes[j.id], Exception)]
    assert len(won) == 1 and len(lost) == 1 and "already associated" in str(outcomes[lost[0].id])
    loser = env.store.get_generation_job(PID, lost[0].id)
    assert loser.state == "submission_unknown" and loser.provider_generation_id is None and loser.resolutions == []


def test_submission_result_already_claimed_by_another_job_is_not_adopted(env):
    a, b = two_jobs(env)
    env.provider.submit_hook = lambda req: "gen-dup"
    env.svc.run(PID, a.id)
    out = env.svc.run(PID, b.id)                                   # provider hands back the same id
    assert out.state == "submission_unknown" and out.provider_generation_id is None and out.lease_owner is None
    assert "already associated" in out.error
    assert out.attempt_history[-1]["outcome"] == "accepted_not_adopted"
    assert out.attempt_history[-1]["providerGenerationId"] == "gen-dup"
    assert env.store.get_generation_job(PID, a.id).provider_generation_id == "gen-dup"


def test_late_response_after_acknowledged_resubmission_never_overwrites_it(env):
    j, _ = submit(env)
    stale = {}

    def slow_first_attempt(req):
        # while attempt 1 is in flight: recovery gives up on it, and a human acknowledges a resubmit
        env.store.update_generation_job(PID, j.id, lambda x: True, lambda x: x.model_copy(update={
            "state": "submission_unknown", "lease_owner": None, "lease_expires_at": None,
            "attempt_history": [{"attempt": 1, "outcome": "unknown", "detail": "recorded by recovery"}]}))
        stale["job"] = env.store.get_generation_job(PID, j.id)
        env.svc.resolve(PID, j.id, action="resubmit", acknowledge_no_provider_job=True)
        return "gen-late"                                          # attempt 1's answer finally arrives

    env.provider.submit_hook = slow_first_attempt
    out = env.svc.run(PID, j.id)
    assert out.state == "queued" and out.provider_generation_id is None       # resubmission untouched
    assert [r["action"] for r in out.resolutions] == ["resubmit"]
    kinds = [(a["attempt"], a["outcome"], a.get("providerGenerationId")) for a in out.attempt_history]
    assert kinds == [(1, "unknown", None), (1, "accepted_not_adopted", "gen-late")]
    assert ("fake", "acct-1", "gen-late") not in env.store.provider_claims

    # attempt 2 proceeds normally, and a straggler from attempt 1 arriving mid-flight is still refused
    def second(req):
        with pytest.raises(Exception):
            env.svc._persist_provider_id(stale["job"], "old-owner", "gen-late")
        return "gen-2"
    env.provider.submit_hook = second
    done = env.svc.run(PID, j.id)
    assert done.provider_generation_id == "gen-2" and done.state == "succeeded"
    assert done.attempt_history[0]["outcome"] == "unknown" and done.resolutions[0]["acknowledgedDuplicateChargeRisk"]
    assert env.store.get_generation_job(PID, j.id).provider_generation_id == "gen-2"


def test_late_adoption_after_recovery_claims_the_id_and_blocks_a_second_job(env):
    a, b = two_jobs(env)

    def recovery_interferes(req):
        env.store.update_generation_job(PID, a.id, lambda x: True, lambda x: x.model_copy(update={
            "state": "submission_unknown", "lease_owner": None, "lease_expires_at": None}))
        return "gen-late"
    env.provider.submit_hook = recovery_interferes
    assert env.svc.run(PID, a.id).state == "submitted"
    assert env.store.provider_claims[("fake", "acct-1", "gen-late")] == (PID, a.id)
    env.provider.submit_hook = lambda req: "gen-late"
    assert env.svc.run(PID, b.id).state == "submission_unknown"
