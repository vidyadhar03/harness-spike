from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from harness.memory.config import Settings
from harness.memory.models import Project
from harness.memory.ports import Blobs, Store

from .jobs import JobRunner
from .settings import ApiSettings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_blobs(request: Request) -> Blobs:
    return request.app.state.blobs


def get_job_runner(request: Request) -> JobRunner:
    return request.app.state.job_runner


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_api_settings(request: Request) -> ApiSettings:
    return request.app.state.api_settings


async def existing_project(project_id: str, request: Request) -> Project:
    """Every nested route depends on this: a bad project_id 404s before anything
    else runs, and no route needs to remember to check project scoping by hand."""
    store: Store = request.app.state.store
    project = await run_in_threadpool(store.get_project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


async def require_json(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


async def require_allowed_origin(request: Request) -> None:
    """Actively rejects a mutation whose Origin header isn't in allowed_origins.

    CORSMiddleware alone does not do this: for a non-preflight request it lets the
    downstream app run regardless of Origin and only adds (or omits)
    Access-Control-Allow-Origin on the *response* - the browser enforces CORS by
    discarding that response client-side, but the mutation has already executed
    server-side by then. This dependency is the actual server-side rejection, checked
    before the route body runs. TrustedHostMiddleware does not cover this either: Host
    reflects the request's target (this server), not Origin, the page that issued the
    request - a malicious page in the user's own browser can freely set Host correctly
    while its Origin is itself.

    A request with no Origin header at all (curl, the operator testing locally, any
    non-browser client) is let through - Origin only exists on browser-issued
    cross-origin requests, and non-browser access is bounded by bind address and
    TrustedHostMiddleware instead (see README's access-boundary section).
    """
    origin = request.headers.get("origin")
    if origin is None:
        return
    allowed = request.app.state.api_settings.allowed_origins
    if origin not in allowed:
        raise HTTPException(status_code=403, detail="origin not allowed")
