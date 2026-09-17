from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from harness.memory.config import Settings
from harness.memory.ingest import Ctx as IngestCtx
from harness.memory.ingest import register_file
from harness.memory.models import Project
from harness.memory.ports import Blobs, Store

from ..deps import existing_project, get_api_settings, get_blobs, get_settings, get_store, require_allowed_origin
from ..dto import SourceOut, SourceSummaryOut
from ..mapping import source_out, uploaded_sources
from ..settings import ApiSettings

router = APIRouter(tags=["sources"])


@router.get("/projects/{project_id}/sources", response_model=list[SourceSummaryOut])
async def list_sources(project_id: str, store: Store = Depends(get_store),
                       _project: Project = Depends(existing_project)) -> list[SourceSummaryOut]:
    """Authoritative uploaded-source listing for the Sources screen, e.g. after a
    reload - see mapping.uploaded_sources for the exact scope (uploaded ingestion
    inputs only, never a references-pipeline-fetched image) and ordering rule.
    """
    return await run_in_threadpool(uploaded_sources, store, project_id)


async def _read_body_bounded(request: Request, max_bytes: int) -> bytes:
    """Reads the raw request body in chunks, aborting as soon as the running total
    exceeds max_bytes - enforced while receiving, not after the fact.

    This route deliberately does not use multipart/form-data or FastAPI's
    UploadFile/File(...): File(...) is resolved as a route *dependency*, which means
    Starlette/python-multipart would already have fully parsed (and possibly spooled to
    disk) the entire body before this function - or any code of ours - ever runs, making
    any size check here always too late. A raw body sidesteps that entirely: the
    frontend sends the file's own bytes directly (`fetch(url, {method: "POST", body:
    file})`), naming it via the `filename` query parameter instead of a form field. One
    file per request; the frontend loops client-side for more than one (see
    API_CONTRACT.md) - both because it keeps this bound simple and because per-file
    upload progress/errors are then independent instead of one request succeeding or
    failing as a batch.
    """
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413,
                                detail=f"upload exceeds the {max_bytes}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _upload_ctx(store: Store, blobs: Blobs, settings: Settings) -> IngestCtx:
    # register_file only ever touches ctx.store/ctx.blobs/ctx.settings - never ctx.llm
    # (that's only read once actual ingestion, ingest_source, runs) - so llm=None here
    # is safe, not a shortcut around something that matters for this call.
    return IngestCtx(store=store, blobs=blobs, llm=None, settings=settings)  # type: ignore[arg-type]


@router.post("/projects/{project_id}/sources", response_model=SourceOut, status_code=201,
            dependencies=[Depends(require_allowed_origin)])
async def upload_source(project_id: str, request: Request,
                        filename: str = Query(..., min_length=1, max_length=255),
                        store: Store = Depends(get_store), blobs: Blobs = Depends(get_blobs),
                        settings: Settings = Depends(get_settings),
                        api_settings: ApiSettings = Depends(get_api_settings),
                        _project: Project = Depends(existing_project)) -> SourceOut:
    """Registers one file (harness.memory.ingest.register_file, unchanged) - does not
    trigger ingestion. Deduplicates by content (sha256) exactly as register_file
    already does; a re-upload of identical bytes is a safe no-op (created: false), not
    reimplemented here. An unsupported mime type is still stored, not rejected -
    matching register_file's existing "never rejects a file" behavior; ingestion later
    reports it as such rather than this endpoint pre-guessing.

    No Content-Type / JSON validation here (this route's body is the file's own raw
    bytes, not JSON) - only the Origin check applies, same as every other mutation.
    """
    data = await _read_body_bounded(request, api_settings.max_upload_bytes)
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    ctx = _upload_ctx(store, blobs, settings)
    src, created = await run_in_threadpool(register_file, ctx, project_id, data, filename)
    return source_out(src, created=created)
