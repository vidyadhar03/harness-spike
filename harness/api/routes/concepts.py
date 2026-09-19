from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from harness.memory.concepts import (
    StaleApprovalContext, build_approval_snapshot, lock_approval, promote_reference_to_concept,
    upload_concept_version,
)
from harness.memory.config import Settings
from harness.memory.models import Location, Project
from harness.memory.ports import ApprovalConflict, Blobs, Store
from harness.memory.retrieval import _Graph, resolve_scope

from ..deps import (existing_project, get_api_settings, get_blobs, get_settings, get_store,
                    require_allowed_origin, require_json)
from ..dto import (
    ApprovalLockRequest, ApprovalLockResult, ApprovalPreviewOut, ApprovalStateOut,
    ConceptUploadOut, ConceptVersionListOut, PromoteReferenceRequest,
)
from ..mapping import approval_package_out, approval_preview_out, approval_state, concept_upload_out, concept_version_list
from ..settings import ApiSettings

router = APIRouter(tags=["concepts"])


def _resolve_location(store: Store, project_id: str, location_id: str) -> str:
    """Resolve location_id to a live entity id and confirm it is a Location, not a
    Scene - concept art and its approval are per location only, mirroring the
    references/upload route's own inline check."""
    graph = _Graph(store.list_entities(project_id))
    resolved_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(resolved_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; concept art is per location")
    return resolved_id


# --- concept versions -------------------------------------------------------------------

@router.get("/projects/{project_id}/locations/{location_id}/concepts", response_model=ConceptVersionListOut)
async def list_concepts(project_id: str, location_id: str, store: Store = Depends(get_store),
                        _project: Project = Depends(existing_project)) -> ConceptVersionListOut:
    try:
        return await run_in_threadpool(concept_version_list, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/{project_id}/locations/{location_id}/concepts/upload",
            response_model=ConceptUploadOut, status_code=201,
            dependencies=[Depends(require_allowed_origin)])
async def upload_concept(
    project_id: str, location_id: str, request: Request,
    filename: str = Query(..., description="Original filename of the uploaded concept image"),
    store: Store = Depends(get_store),
    blobs: Blobs = Depends(get_blobs),
    settings: Settings = Depends(get_settings),
    api_settings: ApiSettings = Depends(get_api_settings),
    _project: Project = Depends(existing_project),
) -> ConceptUploadOut:
    """Upload a finished concept-art image as a new version for a location.

    Raw body (not multipart), one image per request - same convention as
    references/upload. Accepted: JPEG, PNG, WebP. Never approves anything, and never
    touches an existing approval either way - see concepts.upload_concept_version.
    """
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > api_settings.max_upload_bytes:
            raise HTTPException(status_code=413,
                                detail=f"upload exceeds the {api_settings.max_upload_bytes}-byte limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="request body is empty")

    try:
        resolved_id = await run_in_threadpool(_resolve_location, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        version, created = await run_in_threadpool(
            upload_concept_version, store, blobs, settings, project_id, resolved_id, data, filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await run_in_threadpool(concept_upload_out, store, project_id, version, created=created)


@router.post("/projects/{project_id}/locations/{location_id}/concepts/from-reference",
            response_model=ConceptUploadOut, status_code=201,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def promote_reference(
    project_id: str, location_id: str, body: PromoteReferenceRequest,
    store: Store = Depends(get_store),
    _project: Project = Depends(existing_project),
) -> ConceptUploadOut:
    """Create a concept-candidate version from an existing, applicable reference image -
    no bytes are downloaded or re-uploaded (see concepts.promote_reference_to_concept).

    Permits a proposed or confirmed reference; a rejected one is refused with a clear
    400. Never confirms/mutates the reference note, and never touches any existing
    approval. This is a distinct action from selecting a *confirmed* reference as
    supporting evidence for an approval (POST .../approval/lock's referenceIds) - the
    same reference note may play both roles, or neither, independently.
    """
    try:
        version, created = await run_in_threadpool(
            promote_reference_to_concept, store, project_id, location_id, body.reference_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await run_in_threadpool(concept_upload_out, store, project_id, version, created=created)


@router.get("/projects/{project_id}/concepts/{concept_version_id}/image")
async def get_concept_image(project_id: str, concept_version_id: str, store: Store = Depends(get_store),
                            blobs: Blobs = Depends(get_blobs),
                            _project: Project = Depends(existing_project)) -> Response:
    """Streams the bytes server-side, same rationale as GET .../references/{id}/image
    (no signing key available under the documented local dev ADC credential)."""
    version = await run_in_threadpool(store.get_concept_version, project_id, concept_version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="concept version not found")
    src = await run_in_threadpool(store.get_source, project_id, version.source_id)
    if src is None:
        raise HTTPException(status_code=404, detail="image unavailable")
    try:
        data = await run_in_threadpool(blobs.get, src.storage_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="image unavailable") from exc
    return Response(content=data, media_type=src.mime_type,
                    headers={"Cache-Control": "private, max-age=86400"})


# --- concept approval ---------------------------------------------------------------------

@router.get("/projects/{project_id}/locations/{location_id}/approval", response_model=ApprovalStateOut)
async def get_approval(project_id: str, location_id: str, store: Store = Depends(get_store),
                       _project: Project = Depends(existing_project)) -> ApprovalStateOut:
    try:
        return await run_in_threadpool(approval_state, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/projects/{project_id}/locations/{location_id}/approval/preview", response_model=ApprovalPreviewOut)
async def preview_approval_route(
    project_id: str, location_id: str,
    concept_version_id: str = Query(..., alias="conceptVersionId"),
    reference_ids: list[str] = Query(default=[], alias="referenceIds"),
    depiction_label: str | None = Query(default=None, alias="depictionLabel"),
    store: Store = Depends(get_store),
    _project: Project = Depends(existing_project),
) -> ApprovalPreviewOut:
    """Read-only: exactly what POST .../approval/lock would capture right now for this
    candidate + explicit reference selection + depiction label, including the
    contextToken to echo back on that call. Persists nothing."""
    try:
        snapshot = await run_in_threadpool(
            build_approval_snapshot, store, project_id, location_id, concept_version_id,
            reference_ids, depiction_label,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await run_in_threadpool(approval_preview_out, store, project_id, snapshot)


@router.post("/projects/{project_id}/locations/{location_id}/approval/lock",
            response_model=ApprovalLockResult,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def lock_approval_route(project_id: str, location_id: str, body: ApprovalLockRequest, response: Response,
                              store: Store = Depends(get_store),
                              _project: Project = Depends(existing_project)) -> ApprovalLockResult:
    """Explicitly locks a concept version + the brief/context + explicitly selected
    references as the approved visual direction for this location.

    contextToken and expectedRevision are both required - see ApprovalLockRequest and
    concepts.lock_approval. A 409 here always means "re-fetch GET .../approval and/or
    .../approval/preview and retry", never a silent approval of something newer.
    """
    try:
        approval, created = await run_in_threadpool(
            lock_approval, store, project_id, location_id,
            concept_version_id=body.concept_version_id, reference_ids=body.reference_ids,
            context_token=body.context_token, expected_revision=body.expected_revision,
            locked_by=body.by, depiction_label=body.depiction_label,
        )
    except StaleApprovalContext as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ApprovalConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response.status_code = 201 if created else 200
    package = await run_in_threadpool(approval_package_out, store, project_id, approval)
    return ApprovalLockResult(created=created, approval=package)
