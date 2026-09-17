"""Minimal async job execution for the references pipeline.

No queue: single-process localhost, a background asyncio task per job, bounded
concurrency, Firestore-backed so a job's outcome survives a server restart even
though nothing resumes execution after one (see JobStore.recover_orphaned_jobs).

Job/lock documents are API-owned bookkeeping, deliberately kept out of
harness.memory.models - they are not part of the memory domain schema.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Protocol
from uuid import uuid4

from harness.memory.config import Settings
from harness.memory.ingest import Ctx as IngestCtx
from harness.memory.ingest import LOCK_RENEW_INTERVAL_S, LOCK_STALE_AFTER_S, LockRenewer, ingest_source
from harness.memory.ports import LLM, Blobs, Images, Store
from harness.memory.references import RefCtx, suggest_references

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobKind = Literal["references", "concept", "ingest"]
IngestOutcome = Literal["complete", "partial", "failed", "no_op"]
_ACTIVE = ("queued", "running")
_INTERRUPTED = "interrupted: server restarted"
_INGEST_SUCCESS_STATUSES = ("digested", "skipped")
_INGEST_TERMINAL_STATUSES = ("digested", "skipped", "failed")


class IngestTriggerConflict(Exception):
    """Raised by JobRunner.trigger_ingest when the project's ingest slot is not
    available to this request. Two distinct cases, both surfaced as 409 by the route:

    - a *different* API-tracked ingest request is already active for this project
      (different sourceId/force - see JobRunner.trigger_ingest and correction 4: two
      equivalent requests dedupe to the same job, but non-equivalent ones must not
      silently share one as though both were honored).
    - the underlying domain lock (ingest.ingest_lock's `store.acquire_lock`) is held by
      something the API has no record of - almost always a concurrent `harness-memory
      drop`/`ingest` from the CLI on the same project.
    """
    def __init__(self, message: str, *, active_job: "Job | None" = None):
        super().__init__(message)
        self.active_job = active_job


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_job_id() -> str:
    return f"job_{uuid4().hex[:12]}"


def _default_llm(settings: Settings) -> LLM:
    from harness.memory.gcp import GeminiLLM

    return GeminiLLM(settings)


@dataclass
class Job:
    id: str
    project_id: str
    location_id: str
    kind: JobKind
    status: JobStatus = "queued"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # ingest-only fields, all additive/optional so existing persisted "references" Job
    # documents (field set: id/project_id/location_id/kind/status/error/warnings/
    # created_at/updated_at/started_at/finished_at) still round-trip unchanged - no
    # rename of location_id, no schema migration (see JobRunner.trigger_ingest and
    # _run_ingest for how each is used).
    lock_token: str | None = None               # the domain ingest_lock token this job
                                                  # holds, for exact-match release only -
                                                  # never a blanket "clear the lock"
    requested_source_id: str | None = None       # exactly what the caller asked for
    requested_force: bool = False                # (None means "all eligible sources") -
                                                  # used to tell equivalent repeat
                                                  # triggers from conflicting ones apart
    source_ids: list[str] = field(default_factory=list)   # resolved once at trigger time;
                                                  # later uploads never change this job's
                                                  # workload retroactively
    outcome: IngestOutcome | None = None
    source_reports: list[dict] = field(default_factory=list)  # one dict per IngestReport,
                                                  # dataclasses.asdict(report)


class JobStore(Protocol):
    def create_job(self, project_id: str, location_id: str, kind: JobKind) -> tuple[Job, bool]:
        """Atomically claim the (kind, location_id) run slot.

        Returns (job, True) for a freshly created queued job, or (job, False) with the
        already-active job when one is already queued/running for that (kind, location_id) -
        the caller should not start a second run. Both cases return 200-able state; the
        route layer decides the HTTP status.
        """
        ...

    def get_job(self, project_id: str, job_id: str) -> Job | None: ...

    def update_job(self, project_id: str, job_id: str, **changes) -> None: ...

    def release_lock_if_owner(self, project_id: str, location_id: str, kind: JobKind, job_id: str) -> None:
        """Release the run slot only if it still points at this job id - a stale caller
        (e.g. a timed-out run that raced a fresh trigger) must not release someone else's lock."""
        ...

    def put_new_job(self, job: Job) -> None:
        """Persists an already-constructed Job with no locking of its own - used only by
        JobRunner.trigger_ingest, whose actual exclusivity comes from the domain's
        ingest_lock (store.acquire_lock), not from this store. Never call this for a
        "references" job; create_job's (kind, location_id) claim is what protects those.
        """
        ...

    def get_active_project_job(self, project_id: str, kind: JobKind) -> Job | None:
        """The one queued/running job of this kind for this project, if any - project-
        scoped (not (kind, location_id)-scoped like create_job's lock), for kinds like
        "ingest" that apply to a whole project rather than one location."""
        ...

    def recover_orphaned_jobs(self) -> list[Job]:
        """Mark every job still queued/running as failed and drop every lock.

        Called once at process startup. Single-process by design (see ApiSettings): if
        this process is only now starting, nothing legitimately holds a lock or is
        running a job yet, so every queued/running job left over IS the previous
        process's unfinished work, regardless of age - unlike a distributed worker
        pool, there is no "someone else is still running it" case to protect against.
        """
        ...


# --- in-memory (tests, and a reference implementation of the contract) ------------------

class InMemoryJobStore:
    """Two instances sharing the same dicts simulate 'a new process attaches to what the
    old one persisted' for recovery tests, without needing a real Firestore emulator."""

    def __init__(self, jobs: dict[tuple[str, str], Job] | None = None,
                locks: dict[tuple[str, str, str], str] | None = None):
        self._jobs: dict[tuple[str, str], Job] = jobs if jobs is not None else {}
        self._locks: dict[tuple[str, str, str], str] = locks if locks is not None else {}

    def create_job(self, project_id, location_id, kind):
        lock_key = (project_id, kind, location_id)
        held_id = self._locks.get(lock_key)
        if held_id is not None:
            held = self._jobs.get((project_id, held_id))
            if held is not None and held.status in _ACTIVE:
                return held, False
        job = Job(id=new_job_id(), project_id=project_id, location_id=location_id, kind=kind)
        self._jobs[(project_id, job.id)] = job
        self._locks[lock_key] = job.id
        return job, True

    def get_job(self, project_id, job_id):
        return self._jobs.get((project_id, job_id))

    def update_job(self, project_id, job_id, **changes):
        job = self._jobs[(project_id, job_id)]
        for k, v in changes.items():
            setattr(job, k, v)
        job.updated_at = utcnow()

    def release_lock_if_owner(self, project_id, location_id, kind, job_id):
        lock_key = (project_id, kind, location_id)
        if self._locks.get(lock_key) == job_id:
            del self._locks[lock_key]

    def put_new_job(self, job):
        self._jobs[(job.project_id, job.id)] = job

    def get_active_project_job(self, project_id, kind):
        for job in self._jobs.values():
            if job.project_id == project_id and job.kind == kind and job.status in _ACTIVE:
                return job
        return None

    def recover_orphaned_jobs(self):
        recovered = []
        for job in self._jobs.values():
            if job.status in _ACTIVE:
                job.status, job.error, job.finished_at = "failed", _INTERRUPTED, utcnow()
                job.updated_at = job.finished_at
                recovered.append(job)
        self._locks.clear()
        return recovered


# --- Firestore-backed (real deployments) ------------------------------------------------

class FirestoreJobStore:
    """Own Firestore client, independent of FirestoreStore's - see factory.build_firestore_client.

    NOT exercised against real or emulated Firestore by this project's automated test
    suite - tests/test_api.py drives JobRunner and every route exclusively through
    InMemoryJobStore, which mirrors this class's contract (same method signatures, same
    claim/release/recover semantics) but is a plain in-process dict, not a network call
    or a transaction. That gives confidence the *contract* (JobRunner's orchestration,
    the routes' error handling) is correct, but it does NOT verify that
    create_job/release_lock_if_owner/recover_orphaned_jobs's actual Firestore
    transactions and collection_group queries behave as written - concurrent-transaction
    retry behavior, collection_group query indexing requirements, and the "in" filter on
    status are all untested here. Run this class against FIRESTORE_EMULATOR_HOST (or a
    real project) at least once - e.g. exercise create_job from two concurrent requests
    and confirm only one wins, and confirm recover_orphaned_jobs actually needs a
    composite index for the collection_group + where query before relying on it in
    anything beyond local/manual use.
    """

    def __init__(self, client):
        self._db = client

    def _jobs_col(self, project_id: str):
        return self._db.collection("projects").document(project_id).collection("jobs")

    def _locks_col(self, project_id: str):
        return self._db.collection("projects").document(project_id).collection("job_locks")

    def create_job(self, project_id, location_id, kind):
        from google.cloud import firestore

        lock_ref = self._locks_col(project_id).document(f"{kind}:{location_id}")
        jobs_col = self._jobs_col(project_id)
        transaction = self._db.transaction()

        @firestore.transactional
        def _claim(tx):
            lock_snap = lock_ref.get(transaction=tx)
            if lock_snap.exists:
                held_id = lock_snap.to_dict().get("job_id")
                held_snap = jobs_col.document(held_id).get(transaction=tx) if held_id else None
                if held_snap is not None and held_snap.exists and held_snap.to_dict().get("status") in _ACTIVE:
                    return _job_from_dict(held_snap.to_dict()), False
            job = Job(id=new_job_id(), project_id=project_id, location_id=location_id, kind=kind)
            tx.set(lock_ref, {"job_id": job.id, "updated_at": job.created_at})
            tx.set(jobs_col.document(job.id), _job_to_dict(job))
            return job, True

        return _claim(transaction)

    def get_job(self, project_id, job_id):
        snap = self._jobs_col(project_id).document(job_id).get()
        return _job_from_dict(snap.to_dict()) if snap.exists else None

    def update_job(self, project_id, job_id, **changes):
        self._jobs_col(project_id).document(job_id).set({**changes, "updated_at": utcnow()}, merge=True)

    def release_lock_if_owner(self, project_id, location_id, kind, job_id):
        from google.cloud import firestore

        lock_ref = self._locks_col(project_id).document(f"{kind}:{location_id}")
        transaction = self._db.transaction()

        @firestore.transactional
        def _release(tx):
            snap = lock_ref.get(transaction=tx)
            if snap.exists and snap.to_dict().get("job_id") == job_id:
                tx.delete(lock_ref)

        _release(transaction)

    def put_new_job(self, job):
        self._jobs_col(job.project_id).document(job.id).set(_job_to_dict(job))

    def get_active_project_job(self, project_id, kind):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = (self._jobs_col(project_id)
             .where(filter=FieldFilter("kind", "==", kind))
             .where(filter=FieldFilter("status", "in", list(_ACTIVE)))
             .limit(1))
        for snap in q.stream():
            return _job_from_dict(snap.to_dict())
        return None

    def recover_orphaned_jobs(self):
        """Runs synchronously on a worker thread (see main.lifespan) - logging here is
        what lets an operator see which of the two queries below is the one stuck, since
        the caller's own timeout only knows the whole call didn't return in time.
        """
        from google.cloud.firestore_v1.base_query import FieldFilter

        log.info("recover_orphaned_jobs: querying collection_group('jobs') for queued/running...")
        recovered = []
        q = self._db.collection_group("jobs").where(filter=FieldFilter("status", "in", list(_ACTIVE)))
        for snap in q.stream():
            job = _job_from_dict(snap.to_dict())
            finished_at = utcnow()
            snap.reference.set({"status": "failed", "error": _INTERRUPTED,
                                "finished_at": finished_at, "updated_at": finished_at}, merge=True)
            job.status, job.error, job.finished_at = "failed", _INTERRUPTED, finished_at
            recovered.append(job)
        log.info("recover_orphaned_jobs: %d job(s) marked failed; clearing collection_group('job_locks')...",
                 len(recovered))
        for snap in self._db.collection_group("job_locks").stream():
            snap.reference.delete()
        log.info("recover_orphaned_jobs: done")
        return recovered


_JOB_FIELDS = (
    "id", "project_id", "location_id", "kind", "status", "error", "warnings",
    "created_at", "updated_at", "started_at", "finished_at",
    # additive ingest fields - see Job's docstring comment on why these are optional
    "lock_token", "requested_source_id", "requested_force", "source_ids", "outcome",
    "source_reports",
)


def _job_to_dict(job: Job) -> dict:
    return {k: getattr(job, k) for k in _JOB_FIELDS}


def _job_from_dict(d: dict) -> Job:
    # a "references" Job document written before the ingest fields existed simply
    # lacks those keys; Job's own field defaults (None/False/[]) fill them in exactly
    # as if the job had been created with today's code - no migration needed.
    return Job(**{k: d[k] for k in _JOB_FIELDS if k in d})


# --- runner ------------------------------------------------------------------------------

class JobRunner:
    """Dispatches references-pipeline runs off the event loop, bounded and per-run-isolated.

    A fresh GeminiLLM is built per run rather than sharing one: GeminiLLM.accumulated_usage
    and .http_retries are mutable instance counters read at the end of a run
    (references._collect_token_usage) - sharing one instance across concurrent runs would
    let their token/retry counts cross-contaminate each other's ReferenceReport. Store,
    Blobs and Images adapters carry no such per-run counters (WikimediaImages._categories
    is a pure memoization cache, safe to share) and are reused as-is.

    Deliberately has no execution timeout. asyncio.to_thread runs suggest_references on a
    real OS thread that Python cannot force-cancel; wrapping the await in
    asyncio.wait_for(timeout=...) would only cancel *waiting* for it; the thread keeps
    running underneath, unbounded and untracked, while wait_for's TimeoutError would let
    this method proceed straight to marking the job "failed" and releasing its lock and
    concurrency slot - both while the abandoned thread could still be mid-write inside
    suggest_references's atomic _write/replace_notes step. That both permits a second
    concurrent run for the same location (two pipeline runs writing notes at once, which
    replace_notes's per-note precondition does not fully guard against) and reports a
    false "failed" for a run that might still succeed. There is no safe way to bound this
    from inside the process, so we don't try: a run either completes or the process is
    restarted (which correctly recovers it - see recover_orphaned_jobs). The frontend's
    own poll loop already gives up waiting after ~12 minutes independent of this.
    """

    def __init__(self, *, store: Store, blobs: Blobs, images: Images, settings: Settings,
                job_store: JobStore, max_concurrent_jobs: int,
                llm_factory: Callable[[], LLM] | None = None,
                lock_stale_after_s: float = LOCK_STALE_AFTER_S,
                lock_renew_interval_s: float = LOCK_RENEW_INTERVAL_S):
        self._store, self._blobs, self._images, self._settings = store, blobs, images, settings
        self.job_store = job_store
        self._sema = asyncio.Semaphore(max_concurrent_jobs)
        # injectable so tests never construct a real GeminiLLM (no live Gemini calls);
        # the real entrypoint leaves this unset and gets a fresh GeminiLLM per run - see
        # the class docstring for why it must be fresh rather than shared.
        self._llm_factory = llm_factory or (lambda: _default_llm(settings))
        # overridable so tests can use a short simulated expiry instead of the real
        # 1-hour/15-minute defaults - see the ingest section's regression tests.
        self._lock_stale_after_s = lock_stale_after_s
        self._lock_renew_interval_s = lock_renew_interval_s

    async def trigger(self, project_id: str, location_id: str, kind: JobKind) -> tuple[Job, bool]:
        job, created = self.job_store.create_job(project_id, location_id, kind)
        if created and kind == "references":
            asyncio.create_task(self._run_references(job))
        return job, created

    async def _run_references(self, job: Job) -> None:
        async with self._sema:
            self.job_store.update_job(job.project_id, job.id, status="running", started_at=utcnow())
            llm = self._llm_factory()
            ref_ctx = RefCtx(self._store, self._blobs, llm, self._images, self._settings)
            try:
                # awaited to actual completion, not wrapped in a timeout - see the class
                # docstring for why: the lock and the semaphore slot above must not be
                # released before the thread genuinely finishes writing (or fails to).
                report = await asyncio.to_thread(suggest_references, ref_ctx, job.project_id, job.location_id)
            except Exception as exc:
                log.exception("references job %s (project %s, location %s) failed",
                              job.id, job.project_id, job.location_id)
                self.job_store.update_job(job.project_id, job.id, status="failed",
                                          error=str(exc)[:2000], finished_at=utcnow())
            else:
                # suggest_references returns normally even when it found nothing to write
                # (e.g. no terms survived verification) - that is a succeeded run with an
                # empty result and report.warnings explaining why, not a failure.
                self.job_store.update_job(job.project_id, job.id, status="succeeded",
                                          warnings=list(report.warnings), finished_at=utcnow())
            finally:
                self.job_store.release_lock_if_owner(job.project_id, job.location_id, job.kind, job.id)

    # --- ingest --------------------------------------------------------------------------
    #
    # Deliberately does not go through create_job/job_locks the way references jobs do:
    # that lock is a Firestore collection the API invented and fully owns. Ingestion can
    # *also* be started from the CLI (harness-memory drop/ingest) on the same project,
    # completely outside the API's knowledge - if the API used its own separate lock for
    # dedup, the CLI and the API could each believe they're the only writer and run
    # ingest_source concurrently on the same project, which is exactly the corruption
    # ingest.ingest_lock exists to prevent. So the real exclusivity gate here is always
    # ctx.store.acquire_lock (harness.memory.ingest.LOCK_STALE_AFTER_S,
    # ingest.ingest_lock's own two primitives) - the same lock the CLI takes. This
    # JobStore is only ever consulted (get_active_project_job) for the idempotent-trigger
    # UX on top of that, never as the actual gate.

    async def trigger_ingest(self, project_id: str, *, source_id: str | None,
                             force: bool) -> tuple[Job, bool]:
        """Raises IngestTriggerConflict (→ 409) instead of returning when this request
        can't be honored right now - see the class's "ingest" section docstring and
        IngestTriggerConflict's docstring for the two distinct cases.
        """
        active = self.job_store.get_active_project_job(project_id, "ingest")
        if active is not None:
            if active.requested_source_id == source_id and active.requested_force == force:
                return active, False  # equivalent request - idempotent trigger, like references
            raise IngestTriggerConflict(
                f"a different ingest is already running for this project "
                f"(sourceId={active.requested_source_id!r}, force={active.requested_force}); "
                "wait for it to finish or match its parameters",
                active_job=active,
            )

        source_ids = await self._resolve_ingest_source_ids(project_id, source_id)
        job_id = new_job_id()
        token = await asyncio.to_thread(self._store.acquire_lock, project_id, f"api:{job_id}",
                                        self._lock_stale_after_s)
        if token is None:
            raise IngestTriggerConflict(
                "an ingest is already running for this project that wasn't started via "
                "this API (most likely `harness-memory drop`/`ingest` from the CLI); "
                "wait for it to finish, or check `harness-memory status`"
            )

        job = Job(id=job_id, project_id=project_id, location_id=project_id, kind="ingest",
                  lock_token=token, requested_source_id=source_id, requested_force=force,
                  source_ids=source_ids)
        try:
            await asyncio.to_thread(self.job_store.put_new_job, job)
        except Exception:
            # job bookkeeping failed after we already won the real lock - release it
            # immediately rather than leaving the project locked until the 1-hour
            # staleness window (or a restart) clears it.
            await asyncio.to_thread(self._store.release_lock, project_id, token)
            raise
        try:
            asyncio.create_task(self._run_ingest(job))
        except Exception:
            await asyncio.to_thread(self._store.release_lock, project_id, token)
            await asyncio.to_thread(self.job_store.update_job, project_id, job.id,
                                    status="failed", error="failed to schedule the ingest run",
                                    finished_at=utcnow())
            raise
        return job, True

    async def _resolve_ingest_source_ids(self, project_id: str, source_id: str | None) -> list[str]:
        """Captured once, at trigger time - the background run always processes exactly
        this list, so an upload made after triggering never changes an in-flight or
        already-finished job's reported workload. "all eligible" reuses
        Store.list_sources as-is and lets ingest_source's own existing skip/unsupported
        rules decide what actually happens to each - no eligibility logic is
        reimplemented here.
        """
        if source_id is not None:
            return [source_id]
        sources = await asyncio.to_thread(self._store.list_sources, project_id)
        # Exclude reference-only sources from "ingest all" runs; they must not enter the
        # ingestion pipeline. Explicit sourceId requests are rejected at the route level
        # before reaching here, so this filter covers only the all-eligible path.
        return [s.id for s in sources if s.is_ingest_eligible]

    async def _run_ingest(self, job: Job) -> None:
        # Started here, before the semaphore wait - not inside it, and not only between
        # sources - so the lock is kept alive for the run's ENTIRE held duration: queued
        # behind another job, and mid-processing of a single long source, not just the
        # gaps between sources. See LockRenewer's docstring for exactly what it does and
        # does not guarantee.
        renewer = LockRenewer(self._store, job.project_id, job.lock_token,
                              interval_s=self._lock_renew_interval_s)
        await asyncio.to_thread(renewer.start)
        try:
            async with self._sema:
                if renewer.lost.is_set():
                    # lost ownership while merely queued, before any source was even
                    # attempted - starting now would be running with no exclusive lock.
                    log.error("ingest job %s (project %s): lock lost while queued; "
                             "no source was attempted", job.id, job.project_id)
                    self.job_store.update_job(
                        job.project_id, job.id, status="failed", outcome="failed",
                        error="lost exclusive ownership of the project's ingest lock while "
                             "queued (renewal failed) - stopped before any source was attempted",
                        finished_at=utcnow())
                    return

                self.job_store.update_job(job.project_id, job.id, status="running", started_at=utcnow())
                llm = self._llm_factory()
                ctx = IngestCtx(self._store, self._blobs, llm, self._settings)
                reports: list[dict] = []
                lost = False
                try:
                    for source_id in job.source_ids:
                        if renewer.lost.is_set():
                            lost = True
                            log.error("ingest job %s (project %s): lock lost; stopping "
                                     "before source %s", job.id, job.project_id, source_id)
                            break
                        # ingest_source never raises - a per-source failure is recorded
                        # on the returned report ("the source records the failure; the
                        # batch carries on") - so this loop only stops early on
                        # something outside that contract (a real bug), or lost
                        # ownership detected above/below.
                        report = await asyncio.to_thread(ingest_source, ctx, job.project_id,
                                                         source_id, force=job.requested_force)
                        reports.append(_report_to_dict(report))
                        if renewer.lost.is_set():
                            # ownership could have been lost WHILE that source was
                            # processing; its write already happened and cannot be
                            # undone (the same can't-force-cancel-a-thread limit as
                            # having no execution timeout) - but no FURTHER source
                            # starts without exclusive ownership.
                            lost = True
                            log.error("ingest job %s (project %s): lock lost during/after "
                                     "source %s; stopping before any further source",
                                     job.id, job.project_id, source_id)
                            break
                except Exception as exc:
                    log.exception("ingest job %s (project %s) failed", job.id, job.project_id)
                    self.job_store.update_job(job.project_id, job.id, status="failed",
                                              error=str(exc)[:2000], finished_at=utcnow(),
                                              source_reports=reports, outcome="failed")
                else:
                    if lost:
                        self.job_store.update_job(
                            job.project_id, job.id, status="failed", outcome="failed",
                            error="lost exclusive ownership of the project's ingest lock "
                                 "partway through (renewal failed) - stopped before continuing "
                                 "without it; results for sources already processed are below",
                            finished_at=utcnow(), source_reports=reports)
                    else:
                        outcome, warnings = _ingest_outcome(reports)
                        # every attempted source failing is a real failure, not a success
                        # with an empty/partial result (unlike the references job's
                        # empty-result case) - the frontend must not present this as
                        # succeeded or advance to locations on the strength of it.
                        status = "failed" if outcome == "failed" else "succeeded"
                        error = (f"all {len(reports)} source(s) failed to ingest; see the "
                                "per-source results for detail" if outcome == "failed" else None)
                        self.job_store.update_job(job.project_id, job.id, status=status, error=error,
                                                  warnings=warnings, finished_at=utcnow(),
                                                  source_reports=reports, outcome=outcome)
        finally:
            # stop renewing before releasing - a renewal firing after release could
            # otherwise recreate a lock nobody owns.
            await asyncio.to_thread(renewer.stop)
            # exact-token release, same as ingest_lock's own try/finally - never a
            # blanket clear, so a lock some other run already legitimately holds (e.g.
            # reclaimed after this one lost ownership) is never touched.
            await asyncio.to_thread(self._store.release_lock, job.project_id, job.lock_token)


def _report_to_dict(report) -> dict:
    return dataclasses.asdict(report)


def _ingest_outcome(reports: list[dict]) -> tuple[IngestOutcome, list[str]]:
    attempted = [r for r in reports if r["status"] in _INGEST_TERMINAL_STATUSES]
    succeeded = [r for r in attempted if r["status"] in _INGEST_SUCCESS_STATUSES]
    failed = [r for r in attempted if r["status"] == "failed"]
    warnings = [w for r in reports for w in r.get("warnings", [])]
    if not attempted:
        outcome: IngestOutcome = "no_op"
    elif not failed:
        outcome = "complete"
    elif not succeeded:
        outcome = "failed"
    else:
        outcome = "partial"
    return outcome, warnings
