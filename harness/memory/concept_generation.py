"""Base-location concept candidate generation: context snapshot, versioned prompt, durable job,
worker, and candidate import - on top of a provider-neutral imagegen.ImageProvider.

This module builds no vendor wire payloads (that is the adapter's job) and the adapter knows
nothing of locations/notes (see luma.py). Swap the injected provider and nothing here changes.

Scope: a BASE-LOCATION depiction only. Scene-scoped notes (dream events, temporary damage,
scene-specific lighting) are never part of the inputs - the snapshot reads only the location's
standing notes and inherited standing notes (both already unconditional by retrieval's own
rules); they stay in the existing brief/approval features. A generated image is only a
ConceptVersion candidate: nothing here approves, confirms, or touches any approval or note.

State machine (ConceptGenerationJob.state)::

    queued --claim (persisted BEFORE the call)--> submitting --id--> submitted --poll--> generated --import--> succeeded
       submitting --definite rejection--> failed(stage=submit)
       submitting --ambiguous outcome / process death--> submission_unknown   (never auto-resubmitted)
       submitted  --provider says failed--> failed(stage=provider)
       submitted  --local deadline / read errors--> poll_deadline_exceeded    (resumable; NOT a provider failure)
       generated  --download/expired URL/etc--> import_failed                 (resumable from the existing provider job)
       submission_unknown | failed(submit) --explicit human resolve--> submitted | queued

Recovery guarantee: the job row is durable before any provider call and `submitting` is persisted
before the POST. After a restart, jobs with a known provider id are resumed by polling/importing
that same provider job - never by generating again. The one window that cannot be closed
locally (Luma documents no idempotency key): the provider accepts the POST but the process dies
before the id is persisted. That job is left `submission_unknown`; a human must attach the
provider id or explicitly resubmit, accepting that an orphaned paid generation may exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable
from uuid import uuid4

from .concepts import (
    _brief_from_pack, _concept_version_id, _reference_notes_for_location,
    _validate_depiction_label, register_concept_bytes,
)
from .config import Settings
from .imagegen import (
    ImageGenerationError, ImageInput, ImageProvider, ImageRequest, OutputTooLarge, OutputUnavailable,
    ProviderTransientError, SubmissionOutcomeUnknown, SubmissionRejected, UnsupportedRequest,
)
from .files import MAX_REFERENCE_PIXELS, ImageValidationError
from .models import ConceptGenerationJob, ConceptVersion, Location, Note, utcnow
from .ports import Blobs, ProviderIdConflict, Store
from .retrieval import _Graph, get_context, resolve_scope

log = logging.getLogger(__name__)

WORKFLOW = "base_location_concept"
WORKFLOW_VERSION = "1"
PROMPT_VERSION = "base-location-v1"
MAX_DIRECTION_LENGTH = 1000
_KIND_RANK = {"description": 0, "constraint": 1, "tone": 2}
_RESUMABLE = ("poll_deadline_exceeded", "import_failed", "submitted", "generated")
_AUTO_RESUME = ("queued", "submitted", "generated")


class StaleGenerationContext(RuntimeError):
    """The inputs recomputed now no longer match the preview the caller saw (-> 409)."""


class IdempotencyConflict(RuntimeError):
    """Same idempotency key, different payload (-> 409)."""


class JobStateConflict(RuntimeError):
    """The requested action is not valid for the job's current state / ownership (-> 409)."""


class _LeaseLost(Exception):
    pass


CONFIG_ERROR = ("image generation is not configured: set LUMA_AGENTS_API_KEY and restart the server. "
                "Existing jobs and saved candidates remain available.")


class ProviderNotConfigured(RuntimeError):
    """A provider-dependent action was requested but no image provider is configured (-> 503).
    Reading job history and saved candidates never needs a provider and never raises this."""
    def __init__(self):
        super().__init__(CONFIG_ERROR)


MAX_OUTPUT_PIXELS_CEILING = 64_000_000     # hard upper bound for any configured output pixel limit


@dataclass(frozen=True)
class GenerationConfig:
    default_model: str = "uni-1"
    poll_interval_s: float = 3.0
    max_wait_s: float = 600.0          # local waiting budget per run - NOT a provider timeout
    lease_s: float = 120.0             # a worker's ownership lapses this long after its last renewal
    max_consecutive_poll_errors: int = 5
    # Bound on the pixel count of an IMPORTED generated image (still bounded - never unlimited). The
    # default equals the upload bound. Raising it (config) and then resuming an import_failed job is
    # the supported way to import an output that was rejected for size.
    max_output_pixels: int = MAX_REFERENCE_PIXELS

    def __post_init__(self):
        if not 0 < self.max_output_pixels <= MAX_OUTPUT_PIXELS_CEILING:
            raise ValueError(f"max_output_pixels must be between 1 and {MAX_OUTPUT_PIXELS_CEILING}")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


# --- snapshot, prompt, token -------------------------------------------------------------

def _note_entry(n: Note, sources: dict[str, str]) -> dict:
    return {
        "id": n.id, "kind": n.kind, "body": n.body, "status": n.status, "revision": n.revision,
        "citations": [{"sourceId": p.source_id, "filename": sources.get(p.source_id or ""),
                       "page": p.page, "quote": p.quote, "url": p.url, "title": p.title}
                      for p in n.provenance],
    }


def _sorted(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda n: (_KIND_RANK.get(n.kind, 9), n.id))


def render_prompt(snap: dict) -> str:
    """Pure function of the snapshot (prompt version PROMPT_VERSION). Facts, constraints, tone,
    inherited context and per-reference guidance stay in separate labelled sections; reference N
    in the text is image_ref[N-1] in the request."""
    lines = [
        "Create one coherent concept image of a base location for a film. Show the location itself in "
        "its standing, everyday state, as a clear depiction a 3D artist could later interpret when "
        "blocking out the space. Do not depict story events, characters in action, temporary damage, "
        "or scene-specific lighting, and do not invent measurements or dimensions.",
        "",
        f"Location: {snap['location']['name']}",
    ]
    if snap["depictionLabel"]:
        lines.append(f"View / depiction: {snap['depictionLabel']}")

    def section(title: str, notes: list[dict]):
        if notes:
            lines.extend(["", title, *[f"- {' '.join(n['body'].split())}" for n in notes]])

    own = snap["standingNotes"]
    section("Location facts:", [n for n in own if n["kind"] == "description"])
    section("Physical constraints (must be respected):", [n for n in own if n["kind"] == "constraint"])
    section("Tone and atmosphere:", [n for n in own if n["kind"] == "tone"])
    by_ancestor: dict[str, list[dict]] = {}
    for n in snap["inheritedNotes"]:
        by_ancestor.setdefault(n["ancestorName"], []).append(n)
    for name, notes in by_ancestor.items():
        section(f"Context inherited from {name} (applies within it):", notes)
    if snap["references"]:
        lines.extend(["", "Reference images (attached in this order; reference image N below is the Nth attached image):"])
        for r in snap["references"]:
            label = f"Reference image {r['position']}" + (f" ({r['direction']})" if r["direction"] else "")
            borrow = (f"borrow: {' '.join(r['guidance'].split())}" if r["guidance"]
                      else "general visual reference for the location's look; no specific guidance was given")
            lines.append(f"- {label} - {borrow}")
    if snap["direction"]:
        lines.extend(["", "Additional direction from the user:", snap["direction"]])
    return "\n".join(lines) + "\n"


@dataclass
class Prepared:
    snapshot: dict
    prompt: str
    context_token: str
    request: ImageRequest             # includes reference bytes; never persisted or logged
    location_id: str


# --- service ---------------------------------------------------------------------------------

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


_IMPORT_CODES = {"too_many_pixels": "output_too_many_pixels", "unsupported_format": "output_unsupported_format",
                 "animated": "output_unsupported_format", "corrupt": "output_invalid_image",
                 "format_mismatch": "output_invalid_image"}


class _Heartbeat:
    """Keeps a worker's lease alive for as long as it is working - including inside long, blocking
    provider calls (submit with inline images, downloads, imports) that cannot renew it themselves.
    If a renewal is refused (someone else owns the job now) it records that and stops; every write
    the worker makes is separately fenced on lease ownership, so a lost lease can never overwrite
    the new owner's state."""

    def __init__(self, svc: "ConceptGenerationService", project_id: str, job_id: str, owner: str):
        self._svc, self._pid, self._jid, self._owner = svc, project_id, job_id, owner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"lease-{job_id}")
        self.lost = False

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self):
        interval = max(self._svc.config.lease_s / 3.0, 0.01)
        while not self._stop.wait(interval):
            try:
                out = self._svc.store.update_generation_job(
                    self._pid, self._jid, lambda j: j.lease_owner == self._owner,
                    lambda j: j.model_copy(update={"lease_expires_at": self._svc._lease_until()}))
            except Exception:
                log.warning("lease renewal for %s failed; will retry", self._jid)
                continue
            if out is None:                # released (normal end of run) or taken over
                self.lost = True
                return


class ConceptGenerationService:
    def __init__(self, store: Store, blobs: Blobs, settings: Settings, provider: ImageProvider | None, *,
                 config: GenerationConfig = GenerationConfig(),
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], datetime] = utcnow):
        self.store, self.blobs, self.settings, self.provider = store, blobs, settings, provider
        self.config, self._sleep, self._now = config, sleep, clock

    @property
    def configured(self) -> bool:
        return self.provider is not None

    def _require_provider(self) -> ImageProvider:
        if self.provider is None:
            raise ProviderNotConfigured()
        return self.provider

    # -- preparation ----------------------------------------------------------------------
    def _location(self, project_id: str, location_ref: str) -> Location:
        graph = _Graph(self.store.list_entities(project_id))
        scope_id = resolve_scope(self.store, project_id, location_ref, graph)
        entity = graph.by_id.get(scope_id)
        if not isinstance(entity, Location):
            raise ValueError(f"{location_ref!r} is a scene; base-location generation is per location")
        return entity

    def _load_reference(self, project_id: str, source_id: str) -> tuple[ImageInput, str]:
        src = self.store.get_source(project_id, source_id)
        if src is None or src.kind != "image":
            raise ValueError(f"reference asset {source_id[:12]}... is not a stored image")
        data = self.blobs.get(src.storage_path)
        if hashlib.sha256(data).hexdigest() != source_id:
            raise ValueError(f"reference asset {source_id[:12]}... no longer matches its stored identity")
        return ImageInput(data=data, mime_type=src.mime_type), src.mime_type

    def prepare(self, project_id: str, location_ref: str, *, reference_ids: list[str],
                depiction_label: str | None = None, direction: str | None = None,
                model: str | None = None, aspect_ratio: str | None = None,
                output_format: str | None = None) -> Prepared:
        """Deterministic and read-only. Raises LookupError/ValueError (-> 404/400), ProviderNotConfigured."""
        provider = self._require_provider()
        loc = self._location(project_id, location_ref)
        label = _validate_depiction_label(depiction_label)
        direction = (direction or "").strip() or None
        if direction and len(direction) > MAX_DIRECTION_LENGTH:
            raise ValueError(f"direction is {len(direction)} characters; the limit is {MAX_DIRECTION_LENGTH}")

        graph = _Graph(self.store.list_entities(project_id))
        pack = get_context(self.store, project_id, loc.id, include_proposed=True)
        # Standing (owned, unconditional) + inherited standing notes only: get_context puts
        # scene-scoped notes in pack.conditional, which is deliberately never read here.
        own, _scene_requirements_ignored, inherited = _brief_from_pack(pack)

        available = _reference_notes_for_location(self.store, project_id, loc.id, graph, require_confirmed=False)
        refs, inputs, seen = [], [], set()
        for rid in reference_ids:
            if rid in seen:
                continue
            seen.add(rid)
            note = available.get(rid)
            if note is None:
                raise ValueError(f"{rid!r} is not a reference applicable to location {loc.id!r}")
            if note.status != "confirmed":
                raise ValueError(f"{rid!r} is {note.status}, not confirmed; only confirmed references "
                                 "may be used for generation")
            prov = note.provenance[0] if note.provenance else None
            if prov is None or prov.source_id is None or prov.page is not None:
                raise ValueError(f"{rid!r} is not a whole-image reference and cannot be sent as an image")
            image, mime = self._load_reference(project_id, prov.source_id)
            inputs.append(image)
            refs.append({"position": len(refs) + 1, "noteId": note.id, "revision": note.revision,
                         "status": note.status, "sourceId": prov.source_id, "mimeType": mime,
                         "caption": note.body, "direction": note.direction,
                         "guidance": note.guidance})

        if not (own or inherited or refs or direction):
            raise ValueError("nothing to generate from: the location has no standing notes and no "
                             "references or direction were given")

        model = model or self.config.default_model
        snapshot = {
            "workflow": WORKFLOW, "workflowVersion": WORKFLOW_VERSION, "promptVersion": PROMPT_VERSION,
            "location": {"id": loc.id, "name": loc.name, "aliases": list(loc.aliases)},
            "depictionLabel": label, "direction": direction,
            "standingNotes": [_note_entry(n, pack.sources) for n in _sorted(own)],
            "inheritedNotes": [
                {"ancestorId": g.entity_id, "ancestorName": g.name, **_note_entry(n, pack.sources)}
                for g in inherited for n in _sorted(g.notes)],
            "references": refs,
            "settings": {"provider": provider.name, "model": model,
                         "aspectRatio": aspect_ratio, "outputFormat": output_format},
        }
        prompt = render_prompt(snapshot)
        request = ImageRequest(prompt=prompt, model=model, references=tuple(inputs),
                               aspect_ratio=aspect_ratio, output_format=output_format)
        provider.capabilities.validate(request)          # UnsupportedRequest (a ValueError)
        return Prepared(snapshot=snapshot, prompt=prompt, context_token=_sha(snapshot),
                        request=request, location_id=loc.id)

    # -- submission (durable record first) --------------------------------------------------
    @staticmethod
    def job_id_for(project_id: str, key: str) -> str:
        return "gjob_" + hashlib.sha256(f"{project_id}\x00{WORKFLOW}\x00{key}".encode()).hexdigest()[:16]

    def submit(self, project_id: str, location_ref: str, *, reference_ids: list[str], context_token: str,
               depiction_label: str | None = None, direction: str | None = None, model: str | None = None,
               aspect_ratio: str | None = None, output_format: str | None = None,
               idempotency_key: str | None = None) -> tuple[ConceptGenerationJob, bool]:
        """Creates (or returns) the durable job; does NOT call the provider - run() does.
        Returns (job, created)."""
        provider = self._require_provider()
        loc = self._location(project_id, location_ref)
        key = idempotency_key or f"auto-{uuid4().hex}"
        job_id = self.job_id_for(project_id, key)
        fingerprint = _sha({
            "location": loc.id, "references": list(dict.fromkeys(reference_ids)),
            "depictionLabel": _validate_depiction_label(depiction_label),
            "direction": (direction or "").strip() or None,
            "model": model or self.config.default_model, "aspectRatio": aspect_ratio,
            "outputFormat": output_format, "contextToken": context_token, "provider": provider.name,
        })

        def _same_or_conflict(existing: ConceptGenerationJob):
            if existing.request_fingerprint != fingerprint:
                raise IdempotencyConflict(
                    f"idempotency key {key!r} was already used for a different generation request")
            return existing, False

        existing = self.store.get_generation_job(project_id, job_id)
        if existing is not None:
            return _same_or_conflict(existing)          # before any staleness check: a retry is a retry

        prepared = self.prepare(project_id, loc.id, reference_ids=reference_ids, depiction_label=depiction_label,
                                direction=direction, model=model, aspect_ratio=aspect_ratio,
                                output_format=output_format)
        if prepared.context_token != context_token:
            raise StaleGenerationContext(
                "the inputs changed since this preview was generated (brief, references, or settings); "
                "fetch a new preview and submit that")
        job = ConceptGenerationJob(
            id=job_id, project_id=project_id, location_id=loc.id, workflow=WORKFLOW,
            workflow_version=WORKFLOW_VERSION, prompt_version=PROMPT_VERSION,
            provider=provider.name, model=prepared.request.model, idempotency_key=key,
            request_fingerprint=fingerprint, context_token=prepared.context_token,
            input_snapshot=prepared.snapshot, prompt=prepared.prompt,
        )
        stored, created = self.store.put_generation_job_if_absent(project_id, job)
        return (stored, True) if created else _same_or_conflict(stored)

    # History reads: deliberately provider-free, so they keep working when nothing is configured.
    def get(self, project_id: str, job_id: str) -> ConceptGenerationJob | None:
        return self.store.get_generation_job(project_id, job_id)

    def list(self, project_id: str, location_ref: str | None = None) -> list[ConceptGenerationJob]:
        loc_id = self._location(project_id, location_ref).id if location_ref else None
        return self.store.list_generation_jobs(project_id, loc_id)

    # -- ownership ---------------------------------------------------------------------------
    def _lease_until(self) -> datetime:
        return self._now() + timedelta(seconds=self.config.lease_s)

    @staticmethod
    def _free(j: ConceptGenerationJob, now: datetime) -> bool:
        """Unowned, or the owner's lease has lapsed. A live lease is another worker's and is never
        touched by claims or recovery."""
        return j.lease_owner is None or (j.lease_expires_at is not None and j.lease_expires_at <= now)

    def _claim(self, project_id: str, job_id: str, owner: str) -> ConceptGenerationJob | None:
        now = self._now()
        into = {"queued": "submitting", "submitted": "submitted", "poll_deadline_exceeded": "submitted",
                "generated": "generated", "import_failed": "generated"}

        def guard(j):
            return j.state in into and self._free(j, now)

        def apply(j):
            extra = ({"submit_attempts": j.submit_attempts + 1, "submit_started_at": now}
                     if j.state == "queued" else {})
            return j.model_copy(update={"state": into[j.state], "lease_owner": owner,
                                        "lease_expires_at": self._lease_until(), "error": None, **extra})

        return self.store.update_generation_job(project_id, job_id, guard, apply)

    def _update(self, job: ConceptGenerationJob, owner: str, *, release: bool = False,
                attempt: dict | None = None, **changes) -> ConceptGenerationJob:
        extra = {"lease_owner": None, "lease_expires_at": None} if release else {"lease_expires_at": self._lease_until()}

        def apply(j):
            upd = {**changes, **extra}
            if attempt is not None:
                upd["attempt_history"] = [*j.attempt_history, attempt]
            return j.model_copy(update=upd)

        out = self.store.update_generation_job(job.project_id, job.id, lambda j: j.lease_owner == owner, apply)
        if out is None:
            raise _LeaseLost(job.id)
        return out

    def _attempt(self, job: ConceptGenerationJob, outcome: str, detail: str, gid: str | None = None) -> dict:
        return {"attempt": job.submit_attempts, "startedAt": _iso(job.submit_started_at),
                "endedAt": _iso(self._now()), "outcome": outcome, "detail": detail[:300],
                "providerGenerationId": gid}

    # -- worker ----------------------------------------------------------------------------------
    def run(self, project_id: str, job_id: str, *, owner: str | None = None) -> ConceptGenerationJob | None:
        """Drives one job as far as it can go. Safe to call concurrently (even from another process
        sharing the store): only the caller that wins the atomic claim proceeds, and while it works a
        heartbeat keeps its lease alive so no other worker or startup recovery can take the job.
        Raises ProviderNotConfigured when no provider is configured."""
        self._require_provider()
        owner = owner or f"w_{uuid4().hex[:10]}"
        job = self._claim(project_id, job_id, owner)
        if job is None:
            return self.store.get_generation_job(project_id, job_id)
        try:
            with _Heartbeat(self, project_id, job_id, owner):
                if job.state == "submitting":
                    job = self._submit(job, owner)
                if job.state == "submitted":
                    job = self._poll(job, owner)
                if job.state == "generated":
                    job = self._import(job, owner)
            return job
        except _LeaseLost:
            return self.store.get_generation_job(project_id, job_id)
        except Exception as exc:          # a bug must not strand the lease or masquerade as provider failure
            log.exception("generation job %s crashed in worker", job_id)
            try:
                cur = self.store.get_generation_job(project_id, job_id)
                if cur is not None and cur.lease_owner == owner:
                    # Unknown-effect state: with no provider id we cannot claim nothing was sent.
                    unknown = cur.state == "submitting" and not cur.provider_generation_id
                    state = "submission_unknown" if unknown else (
                        "import_failed" if cur.state == "generated" else "poll_deadline_exceeded")
                    self._update(cur, owner, release=True, state=state, error=f"worker error: {type(exc).__name__}",
                                 attempt=self._attempt(cur, "unknown", f"worker error: {type(exc).__name__}")
                                 if unknown else None)
            except Exception:
                log.exception("could not record worker error for %s", job_id)
            return self.store.get_generation_job(project_id, job_id)

    def _references_for(self, job: ConceptGenerationJob) -> tuple[ImageInput, ...]:
        return tuple(self._load_reference(job.project_id, r["sourceId"])[0]
                     for r in job.input_snapshot["references"])

    def _submit(self, job, owner):
        provider = self._require_provider()
        try:
            s = job.input_snapshot["settings"]
            req = ImageRequest(prompt=job.prompt, model=job.model, references=self._references_for(job),
                               aspect_ratio=s.get("aspectRatio"), output_format=s.get("outputFormat"))
        except (ValueError, KeyError, OSError) as exc:      # nothing was sent
            return self._update(job, owner, release=True, state="failed", failure_stage="submit",
                                failure_code="reference_unavailable", error=str(exc)[:500],
                                finished_at=self._now(),
                                attempt=self._attempt(job, "rejected", f"reference unavailable: {exc}"))
        try:
            gid = provider.submit(req)
        except (SubmissionRejected, UnsupportedRequest) as exc:
            return self._update(job, owner, release=True, state="failed", failure_stage="submit",
                                failure_code=getattr(exc, "code", "unsupported"), error=str(exc)[:500],
                                finished_at=self._now(), attempt=self._attempt(job, "rejected", str(exc)))
        except SubmissionOutcomeUnknown as exc:
            return self._update(job, owner, release=True, state="submission_unknown", error=str(exc)[:500],
                                attempt=self._attempt(job, "unknown", str(exc)))
        except Exception as exc:
            # Anything else after a possible send is ambiguous - never guess "not sent".
            log.exception("unexpected error submitting job %s", job.id)
            msg = f"unexpected {type(exc).__name__} during submission"
            return self._update(job, owner, release=True, state="submission_unknown", error=msg,
                                attempt=self._attempt(job, "unknown", msg))
        finally:
            req = None      # drop reference bytes promptly
        return self._persist_provider_id(job, owner, gid)

    def _scope(self) -> str:
        return getattr(self.provider, "account_scope", None) or "default"

    def _persist_provider_id(self, job, owner, gid):
        """The provider id is the most valuable thing we hold: retry persisting it, and accept it
        even if our lease lapsed and a recovery already recorded this very attempt as unknown -
        adopting a KNOWN id is always safe, whereas losing it is not.

        The id is recorded through Store.claim_provider_generation, which atomically makes
        (provider, account scope, id) belong to this job - so no other job, in any project, can hold
        it - and, in the same step, checks this response still belongs to the ATTEMPT that produced
        it: the job must still be in that attempt's state (owned by us, or recorded unknown by
        recovery). A response that arrives after an acknowledged resubmission (state queued, or a
        newer attempt) fails that check and is only noted in the audit trail; it never overwrites the
        newer attempt."""
        entry = self._attempt(job, "accepted", "the provider accepted the request", gid)
        late = {**entry, "detail": "the provider accepted the request; its id arrived after this attempt "
                                   "had already been recorded as unknown"}
        attempt_no = job.submit_attempts

        def guard(j):
            return j.provider_generation_id is None and j.submit_attempts == attempt_no and (
                j.lease_owner == owner or (j.state == "submission_unknown" and j.lease_owner is None))

        def apply(j):
            mine = j.lease_owner == owner
            upd = {"state": "submitted", "provider_generation_id": gid, "submitted_at": self._now(),
                   "error": None, "attempt_history": [*j.attempt_history, entry if mine else late],
                   "lease_expires_at": self._lease_until() if mine else None}
            if not mine:
                upd["lease_owner"] = None
            return j.model_copy(update=upd)

        for attempt in range(3):
            try:
                out = self.store.claim_provider_generation(
                    job.project_id, job.id, provider=job.provider, scope=self._scope(),
                    generation_id=gid, guard=guard, apply=apply)
            except ProviderIdConflict as exc:
                log.error("provider returned generation %s for job %s but it is already associated with job %s",
                          gid, job.id, exc.job_id)
                msg = (f"the provider returned generation {gid!r}, which is already associated with job "
                       f"{exc.job_id!r}; not adopted, so this attempt's outcome is unknown")
                try:      # we still hold the lease: park the job for a human, keeping the audit entry
                    return self._update(job, owner, release=True, state="submission_unknown", error=msg[:500],
                                        attempt={**entry, "outcome": "accepted_not_adopted", "detail": msg[:300]})
                except _LeaseLost:
                    self._record_unadopted(job, attempt_no, gid, msg)
                    raise
            except Exception:
                if attempt == 2:
                    # Log the (non-secret) provider id so an operator can attach it via resolve.
                    log.error("provider accepted job %s as generation %s but persisting the id failed",
                              job.id, gid)
                    raise
                self._sleep(0.2 * (attempt + 1))
                continue
            if out is None:
                log.error("provider accepted job %s as generation %s but the job no longer accepts it", job.id, gid)
                self._record_unadopted(job, attempt_no, gid, "the provider accepted this attempt after the job had "
                                       "moved on (resolved or resubmitted); its generation was not adopted")
                raise _LeaseLost(job.id)
            if out.lease_owner != owner:       # id adopted after recovery; another worker resumes it
                raise _LeaseLost(job.id)
            return out

    def _record_unadopted(self, job, attempt_no, gid, detail):
        """Audit-only: notes, against the ORIGINAL attempt, a provider generation we could not adopt.
        Never touches state, ownership or provider_generation_id. Best effort and idempotent."""
        entry = {"attempt": attempt_no, "startedAt": _iso(job.submit_started_at), "endedAt": _iso(self._now()),
                 "outcome": "accepted_not_adopted", "detail": detail[:300], "providerGenerationId": gid}
        try:
            self.store.update_generation_job(
                job.project_id, job.id,
                lambda j: not any(a.get("providerGenerationId") == gid for a in j.attempt_history),
                lambda j: j.model_copy(update={"attempt_history": [*j.attempt_history, entry]}))
        except Exception:
            log.exception("could not audit unadopted provider generation for %s", job.id)

    def _poll(self, job, owner):
        provider = self._require_provider()
        deadline = self._now() + timedelta(seconds=self.config.max_wait_s)
        errors = 0
        while True:
            try:
                pj = provider.get(job.provider_generation_id)
                errors = 0
            except ProviderTransientError as exc:
                errors += 1
                if errors >= self.config.max_consecutive_poll_errors:
                    return self._update(job, owner, release=True, state="poll_deadline_exceeded",
                                        error=f"stopped polling after {errors} consecutive read errors "
                                              f"({exc}); the provider job is unaffected")
                pj = None
            except ImageGenerationError as exc:
                return self._update(job, owner, release=True, state="poll_deadline_exceeded",
                                    error=f"could not read the provider job ({exc}); nothing was concluded "
                                          "about its outcome")
            if pj is not None:
                if pj.state == "completed":
                    return self._update(job, owner, state="generated", provider_completed_at=self._now())
                if pj.state == "failed":
                    return self._update(job, owner, release=True, state="failed", failure_stage="provider",
                                        failure_code=pj.failure_code, error=pj.failure_reason,
                                        finished_at=self._now())
            if self._now() >= deadline:
                return self._update(job, owner, release=True, state="poll_deadline_exceeded",
                                    error=f"stopped waiting after {self.config.max_wait_s:.0f}s; the provider "
                                          "job may still be running - resume to keep waiting")
            self._update(job, owner)         # renew
            self._sleep(self.config.poll_interval_s)

    def _import(self, job, owner):
        """The provider has (or claims to have) produced output; this step only fetches and stores it.
        Every failure here keeps the provider id and lands in import_failed with a specific code - it
        is NEVER reported as a provider failure and NEVER triggers a new generation. A later resume
        re-reads the same provider job (fresh URLs) and retries the import."""
        provider = self._require_provider()
        prefix = "Import failed (the provider generated the image successfully): "

        def fail(code, msg):
            return self._update(job, owner, release=True, state="import_failed", failure_stage="import",
                                failure_code=code, error=(prefix + msg)[:600])

        try:
            pj = provider.get(job.provider_generation_id)        # fresh output URLs
        except ImageGenerationError as exc:
            return fail("provider_read_failed", f"could not re-read the provider job: {exc}")
        if pj.state == "failed":
            return self._update(job, owner, release=True, state="failed", failure_stage="provider",
                                failure_code=pj.failure_code, error=pj.failure_reason, finished_at=self._now())
        if pj.state != "completed":
            return self._update(job, owner, release=True, state="poll_deadline_exceeded",
                                error="the provider job is not complete yet")
        if not pj.output_urls:
            return fail("output_missing", "the provider reports completion but returned no output")

        candidate_ids, reused = [], []
        for i, url in enumerate(pj.output_urls, start=1):
            try:
                img = provider.download(url)
            except OutputUnavailable as exc:
                return fail("output_url_unavailable", str(exc)[:400])
            except OutputTooLarge as exc:
                return fail("output_too_large_bytes", str(exc)[:400])
            except ImageGenerationError as exc:
                return fail("output_download_failed", str(exc)[:400])
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(img.mime_type, "img")
            try:
                src = register_concept_bytes(self.store, self.blobs, self.settings, job.project_id,
                                             img.data, f"{job.id}-{i}.{ext}",
                                             max_pixels=self.config.max_output_pixels)
            except ImageValidationError as exc:
                return fail(_IMPORT_CODES.get(exc.code, "output_invalid_image"), str(exc)[:400])
            except ValueError as exc:
                return fail("output_invalid_image", str(exc)[:400])
            version = ConceptVersion(
                id=_concept_version_id(job.location_id, src.id), location_id=job.location_id,
                source_id=src.id, filename=f"{job.id}-{i}.{ext}", author="agent", generation_job_id=job.id)
            stored, created = self.store.put_concept_version_if_absent(job.project_id, version)
            if stored.id not in candidate_ids:
                candidate_ids.append(stored.id)
                # existing candidate created by an earlier attempt of THIS job is not a "reuse"
                if not created and stored.generation_job_id != job.id:
                    reused.append(stored.id)
        return self._update(job, owner, release=True, state="succeeded", candidate_ids=candidate_ids,
                            reused_candidate_ids=reused, error=None, failure_stage=None, failure_code=None,
                            finished_at=self._now())

    # -- restart recovery and human actions ------------------------------------------------------
    def recover(self) -> list[tuple[str, str]]:
        """Startup (and periodic) recovery. Touches ONLY jobs that are unowned or whose owner's lease
        has lapsed - a live lease belongs to a working worker (possibly another dispatcher or process
        sharing this store) and is never invalidated. Jobs stuck in `submitting` become
        `submission_unknown` (never resubmitted); returns the (project_id, job_id) pairs that are safe
        to run. Without a configured provider it still repairs that bookkeeping but returns nothing to
        run - resuming needs the provider."""
        to_run = []
        now = self._now()
        for project in self.store.list_projects():
            for job in self.store.list_generation_jobs(project.id):
                if not self._free(job, now):
                    continue
                if job.state == "submitting":
                    def apply(j, _job=job):
                        return j.model_copy(update={
                            "state": "submission_unknown", "lease_owner": None, "lease_expires_at": None,
                            "error": "the worker stopped while the request was being submitted; the provider "
                                     "may have accepted it",
                            "attempt_history": [*j.attempt_history, {
                                "attempt": j.submit_attempts, "startedAt": _iso(j.submit_started_at),
                                "endedAt": _iso(now), "outcome": "unknown", "providerGenerationId": None,
                                "detail": "the worker stopped mid-submission (recorded by recovery)"}]})
                    self.store.update_generation_job(
                        project.id, job.id,
                        lambda j: j.state == "submitting" and j.provider_generation_id is None and self._free(j, now),
                        apply)
                elif job.state in _AUTO_RESUME:
                    if self.provider is None:
                        continue
                    if job.lease_owner is not None:      # stale lease from a dead worker
                        self.store.update_generation_job(
                            project.id, job.id, lambda j: j.state in _AUTO_RESUME and self._free(j, now),
                            lambda j: j.model_copy(update={"lease_owner": None, "lease_expires_at": None}))
                    to_run.append((project.id, job.id))
        return to_run

    def can_resume(self, job: ConceptGenerationJob) -> bool:
        return self.configured and job.state in _RESUMABLE and self._free(job, self._now())

    def _other_job_with_provider_id(self, gid: str, provider: str, exclude: str) -> ConceptGenerationJob | None:
        for project in self.store.list_projects():
            for j in self.store.list_generation_jobs(project.id):
                if j.id != exclude and j.provider == provider and j.provider_generation_id == gid:
                    return j
        return None

    def resolve(self, project_id: str, job_id: str, *, action: str, provider_generation_id: str | None = None,
                acknowledge_no_provider_job: bool = False, by: str = "user") -> ConceptGenerationJob:
        """Human resolution of an ambiguous/failed submission - the ONLY paths that can adopt a provider
        job or lead to a second paid submission. Every call appends to job.resolutions; nothing is
        ever erased (the ambiguous attempt stays in attempt_history)."""
        provider = self._require_provider()
        job = self.store.get_generation_job(project_id, job_id)
        if job is None:
            raise LookupError(f"no generation job {job_id!r}")
        now = self._now()
        if action == "attach_provider_id":
            if job.state != "submission_unknown":
                raise JobStateConflict("attach_provider_id is only valid for a submission_unknown job")
            if not provider_generation_id:
                raise ValueError("providerGenerationId is required")
            if job.provider != provider.name:
                raise JobStateConflict(f"this job was created for provider {job.provider!r}, but {provider.name!r} "
                                       "is configured now")
            # Early, friendly check that also covers records written before claims existed; the atomic
            # claim below is the authority.
            other = self._other_job_with_provider_id(provider_generation_id, provider.name, job.id)
            if other is not None:
                raise JobStateConflict(f"provider generation {provider_generation_id!r} is already associated "
                                       f"with job {other.id!r}")
            try:
                pj = provider.get(provider_generation_id)
            except ImageGenerationError as exc:
                raise ValueError(f"could not confirm that provider job exists: {exc}") from None
            if pj.kind is None:
                raise ValueError("the provider did not report what kind of generation this is, so it cannot "
                                 "be verified as an image generation")
            if pj.kind != "image":
                raise ValueError(f"that provider job is a {pj.kind!r} generation, not an image generation")
            if pj.model is None:
                raise ValueError("the provider did not report the model, so it cannot be checked against this job")
            if pj.model != job.model:
                raise ValueError(f"that provider job used model {pj.model!r}; this job requested {job.model!r}")
            unverifiable = ["the prompt", "the reference images", "the aspect ratio / output format",
                            "that this generation was created by this server rather than another client"]
            window_start = (job.submit_started_at or job.created_at) - timedelta(seconds=60)
            if pj.created_at is None:
                unverifiable.append("the creation time")
            else:
                if pj.created_at < window_start:
                    raise ValueError("that provider job was created before this job's submission began, so it "
                                     "cannot be its result")
                if pj.created_at > now + timedelta(seconds=60):
                    raise ValueError("that provider job's creation time is in the future")
            entry = {"at": _iso(now), "action": "attach_provider_id", "by": by, "previousState": job.state,
                     "previousError": job.error, "providerGenerationId": provider_generation_id,
                     "verified": {"exists": True, "kind": pj.kind, "model": pj.model,
                                  "providerState": pj.state, "createdAt": _iso(pj.created_at),
                                  "notAssociatedWithAnotherJob": True},
                     "unverifiable": unverifiable}
            attach_guard = lambda j: j.state == "submission_unknown" and j.provider_generation_id is None
            attach_apply = lambda j: j.model_copy(update={
                "state": "submitted", "provider_generation_id": provider_generation_id, "submitted_at": now,
                "error": None, "resolutions": [*j.resolutions, entry]})
            try:      # atomic: the id is claimed for this job and recorded in one step
                out = self.store.claim_provider_generation(
                    project_id, job_id, provider=provider.name, scope=self._scope(),
                    generation_id=provider_generation_id, guard=attach_guard, apply=attach_apply)
            except ProviderIdConflict as exc:
                raise JobStateConflict(f"provider generation {provider_generation_id!r} is already associated "
                                       f"with job {exc.job_id!r}") from None
            if out is None:
                raise JobStateConflict("the job changed state; re-read it and retry")
            return out
        elif action == "resubmit":
            if not acknowledge_no_provider_job:
                raise ValueError("resubmit requires acknowledgeNoProviderJob=true: confirm no generation "
                                 "exists at the provider for this job, or accept a possible duplicate charge")
            eligible = lambda j: ((j.state == "submission_unknown" or
                                   (j.state == "failed" and j.failure_stage == "submit"))
                                  and not j.provider_generation_id)
            if not eligible(job):
                raise JobStateConflict("resubmit is only valid for a submission_unknown job or a job whose "
                                       "submission was definitively rejected")
            ambiguous = job.state == "submission_unknown"
            entry = {"at": _iso(now), "action": "resubmit", "by": by, "previousState": job.state,
                     "previousError": job.error, "previousAttempts": job.submit_attempts,
                     "acknowledgedDuplicateChargeRisk": True,
                     "warning": ("The earlier submission's outcome was unknown: the provider may have accepted "
                                 "it, so this resubmission can produce a second, separately charged "
                                 "generation.") if ambiguous else
                                "The earlier submission was definitively rejected; no generation was created."}
            guard = eligible
            # attempt_history / earlier resolutions are kept; only the live status fields reset
            apply = lambda j: j.model_copy(update={
                "state": "queued", "error": None, "failure_stage": None, "failure_code": None,
                "finished_at": None, "resolutions": [*j.resolutions, entry]})
        else:
            raise ValueError("action must be attach_provider_id or resubmit")
        out = self.store.update_generation_job(project_id, job_id, guard, apply)
        if out is None:
            raise JobStateConflict("the job changed state; re-read it and retry")
        return out
