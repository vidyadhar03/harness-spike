"""FastAPI app factory + `harness-api` console-script entrypoint.

See API_CONTRACT.md for the endpoint contract and README.md for exact startup commands.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.api_core.exceptions import RetryError
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from harness.memory.config import Settings
from harness.memory.factory import build_blobs, build_firestore_client, build_images, build_store
from harness.memory.concept_generation import ConceptGenerationService, GenerationConfig
from harness.memory.imagegen import ImageProvider
from harness.memory.ports import LLM, Blobs, Images, Store

from .generation import GenerationDispatcher
from .jobs import FirestoreJobStore, InMemoryJobStore, Job, JobRunner, JobStore
from .routes import concept_generation as concept_generation_routes
from .routes import concepts as concepts_routes
from .routes import images as images_routes
from .routes import ingest as ingest_routes
from .routes import jobs as jobs_routes
from .routes import locations as locations_routes
from .routes import projects as projects_routes
from .routes import references as references_routes
from .routes import sources as sources_routes
from .settings import ApiSettings

log = logging.getLogger(__name__)

_REAUTH_HELP = "Run `gcloud auth application-default login`, then restart harness-api."


async def _bounded_startup_call(description: str, fn: Callable[[], object], timeout_s: float,
                                exiter: Callable[[int], None]) -> object:
    """Runs one blocking startup call off the event loop, bounded so a broken Firestore
    credential fails promptly and clearly instead of hanging - see run_startup_checks,
    which is the only caller and carries the full rationale.
    """
    log.info("startup: %s (bounded to %.0fs)...", description, timeout_s)
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.error(
            "startup: %s did not respond within %.0fs (elapsed %.1fs). This almost always "
            "means the ADC credential needs reauthentication - the underlying client can "
            "retry silently for minutes rather than failing fast. %s",
            description, timeout_s, time.monotonic() - t0, _REAUTH_HELP,
        )
        exiter(1)
        return None  # only reached if exiter didn't actually terminate the process (tests)
    except (RefreshError, DefaultCredentialsError) as exc:
        # failed fast here (no hang, thread already finished) - a normal raise is
        # enough, no need for exiter.
        log.error("startup: %s - Firestore credentials are invalid (%s: %s). %s",
                  description, type(exc).__name__, exc, _REAUTH_HELP)
        raise RuntimeError(f"Firestore credentials are invalid or expired. {_REAUTH_HELP}") from exc
    except RetryError as exc:
        cause = exc.cause if isinstance(exc.cause, BaseException) else exc
        log.error("startup: %s - Firestore call failed after its internal retries (%s: %s). %s",
                  description, type(cause).__name__, cause, _REAUTH_HELP)
        raise RuntimeError(
            f"Firestore startup check failed after retries ({type(cause).__name__}: {cause}). "
            f"If this is a credentials problem: {_REAUTH_HELP}"
        ) from exc
    else:
        log.info("startup: %s completed in %.1fs", description, time.monotonic() - t0)
        return result


async def run_startup_checks(job_store: JobStore, store: Store, timeout_s: float, *,
                             exiter: Callable[[int], None] = lambda code: os._exit(code)) -> list[Job]:
    """Recovers orphaned jobs (JobStore.recover_orphaned_jobs) and, for any of them that
    were "ingest" jobs holding the domain's ingest_lock, releases that lock too - a
    process crash never runs the background task's own `finally: release_lock`, so
    without this the project would stay locked until ingest_lock's 1-hour staleness
    window elapses on its own, even though nothing is actually running.

    Each blocking call is bounded (see _bounded_startup_call) so a broken Firestore
    credential fails startup promptly and clearly instead of leaving
    "Waiting for application startup." on screen indefinitely - confirmed by direct
    testing against the real local GCP project that an ADC credential needing
    reauthentication hangs inside google-auth's own token-refresh logic for minutes
    (observed: ~300s), *before* any Firestore RPC-level timeout/retry parameter ever
    takes effect, so a per-call `timeout=` on the Firestore call itself does not help;
    only wrapping the whole call in asyncio.wait_for does.

    Release is by the job's own persisted lock_token, never a blanket clear - a CLI
    ingest running concurrently (or one that grabbed the lock after this process died
    and before it restarted) holds a *different* token, and store.release_lock already
    refuses to delete a lock whose current token doesn't match the one it's asked to
    release. "Never release CLI-owned locks" falls out of that existing check for free.
    """
    recovered = await _bounded_startup_call(
        "recovering orphaned jobs/locks", job_store.recover_orphaned_jobs, timeout_s, exiter,
    ) or []
    if recovered:
        log.warning("recovered %d job(s) left queued/running by a previous process", len(recovered))

    orphaned_ingest_locks = [j for j in recovered if j.kind == "ingest" and j.lock_token]
    for job in orphaned_ingest_locks:
        await _bounded_startup_call(
            f"releasing orphaned ingest lock for project {job.project_id}",
            lambda j=job: store.release_lock(j.project_id, j.lock_token),
            timeout_s, exiter,
        )

    return recovered


def create_app(*, settings: Settings | None = None, api_settings: ApiSettings | None = None,
              store: Store | None = None, blobs: Blobs | None = None, images: Images | None = None,
              job_store: JobStore | None = None, llm_factory: Callable[[], LLM] | None = None,
              image_provider: ImageProvider | None = None, run_generation_jobs: bool = True) -> FastAPI:
    """Builds a fully-wired app.

    Tests pass store=MemoryStore(), blobs=MemoryBlobs(), images=MemoryImages(...), a fake
    Settings, and a fake llm_factory; job_store is left unset (defaults to InMemoryJobStore).
    No real GCP or Gemini call happens anywhere in that path. Only run() below (the real
    `harness-api` process) leaves everything unset except job_store, and gets the real
    GCP-backed adapters plus a fresh real GeminiLLM per job run.
    """
    settings = settings or Settings.from_env()
    api_settings = api_settings or ApiSettings.from_env()
    store = store or build_store(settings)
    blobs = blobs or build_blobs(settings)
    images = images or build_images()
    job_store = job_store or InMemoryJobStore()
    if image_provider is None:
        from harness.memory.luma import from_env as luma_from_env
        image_provider = luma_from_env(max_request_bytes=api_settings.generation_max_request_bytes)
    # The service always exists so job history and saved candidates stay readable; only actions that
    # need the provider (preview/submit/resume/resolve/recovery) require one - see ProviderNotConfigured.
    generation_service = ConceptGenerationService(
        store, blobs, settings, image_provider,
        config=GenerationConfig(default_model=api_settings.generation_default_model,
                                poll_interval_s=api_settings.generation_poll_interval_s,
                                max_wait_s=api_settings.generation_max_wait_s,
                                lease_s=api_settings.generation_lease_s,
                                max_output_pixels=api_settings.generation_max_output_pixels))
    generation_dispatcher = None

    job_runner = JobRunner(store=store, blobs=blobs, images=images, settings=settings,
                           job_store=job_store, max_concurrent_jobs=api_settings.max_concurrent_jobs,
                           llm_factory=llm_factory,
                           lock_stale_after_s=api_settings.ingest_lock_stale_after_s,
                           lock_renew_interval_s=api_settings.ingest_lock_renew_interval_s)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # single-process by design (see ApiSettings docstring): anything still
        # queued/running belongs to a process that is no longer here. Bounded and
        # logged - see run_startup_checks - so a broken credential fails fast and
        # clearly instead of leaving "Waiting for application startup." on screen
        # indefinitely.
        await run_startup_checks(job_store, store, api_settings.startup_timeout_s)
        # Recovery only touches unowned / expired-lease jobs (never a live worker's) and never
        # resubmits; see ConceptGenerationService.recover. Runs even without a provider so that
        # bookkeeping is repaired - it just has nothing to resume then.
        for pid, jid in await _bounded_startup_call(
                "recovering concept generation jobs", generation_service.recover,
                api_settings.startup_timeout_s, lambda code: os._exit(code)) or []:
            if generation_dispatcher is not None:
                generation_dispatcher.schedule(pid, jid)
        if generation_dispatcher is not None:
            generation_dispatcher.start_sweeper(api_settings.generation_sweep_interval_s)
        try:
            yield
        finally:
            if generation_dispatcher is not None:
                await generation_dispatcher.stop()

    app = FastAPI(title="video-harness API", lifespan=lifespan)
    app.state.settings = settings
    app.state.api_settings = api_settings
    app.state.store = store
    app.state.blobs = blobs
    app.state.images = images
    app.state.job_runner = job_runner
    app.state.generation_service = generation_service
    if image_provider is not None:
        generation_dispatcher = GenerationDispatcher(
            generation_service, max_concurrent=api_settings.max_concurrent_generations,
            enabled=run_generation_jobs)
    app.state.generation_dispatcher = generation_dispatcher

    # TrustedHostMiddleware guards the server itself against a forged Host header on a
    # direct (non-browser) request; CORSMiddleware only ever constrains what a *browser*
    # will let a page read back. Neither is authentication - see ApiSettings and
    # API_CONTRACT.md for the localhost-only access-boundary rationale.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(api_settings.allowed_hosts))
    app.add_middleware(CORSMiddleware, allow_origins=list(api_settings.allowed_origins),
                       allow_credentials=False, allow_methods=["GET", "POST"],
                       allow_headers=["content-type"])

    app.include_router(projects_routes.router)
    app.include_router(sources_routes.router)
    app.include_router(ingest_routes.router)
    app.include_router(locations_routes.router)
    app.include_router(references_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(images_routes.router)
    app.include_router(concepts_routes.router)
    app.include_router(concept_generation_routes.router)
    return app


def run() -> None:
    """Console-script entrypoint: `harness-api`.

    Binds to 127.0.0.1 by default. Do not change the bind host to 0.0.0.0 (or put this
    behind a public reverse proxy) without first adding real authentication - see
    API_CONTRACT.md's access-boundary section.
    """
    import uvicorn

    # without this, run_startup_checks's log.info/log.error calls (and everything else
    # under the "harness" logger tree) have no configured handler and are silently
    # dropped - uvicorn's own "Waiting for application startup." messages come from
    # uvicorn's separately-configured loggers, not these, so their presence does not
    # imply ours are visible too. Matches harness.memory.cli.main's own basicConfig call.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env()
    api_settings = ApiSettings.from_env()
    job_store = FirestoreJobStore(build_firestore_client(settings))
    app = create_app(settings=settings, api_settings=api_settings, job_store=job_store)
    uvicorn.run(app, host=api_settings.host, port=api_settings.port)


if __name__ == "__main__":
    run()
