from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from harness.memory.config import Settings
from harness.memory.curate import NoteSuccessorSpec, correct_note, review_note
from harness.memory.models import Location, Project
from harness.memory.ports import Blobs, NoteReviewConflict, Store
from harness.memory.references import upload_reference_image
from harness.memory.retrieval import _Graph, resolve_scope

from ..deps import (existing_project, get_api_settings, get_blobs, get_settings,
                    get_store, require_allowed_origin, require_json)
from ..dto import (
    CorrectNoteRequest, NoteCorrectionResultOut, ReferenceListOut, ReferenceUploadOut,
    ReviewRequest, ReviewResultOut,
)
from ..mapping import note_correction_result_out, reference_list
from ..settings import ApiSettings

router = APIRouter(tags=["references"])


@router.get("/projects/{project_id}/locations/{location_id}/references", response_model=ReferenceListOut)
async def get_references(project_id: str, location_id: str, include_rejected: bool = False,
                         store: Store = Depends(get_store),
                         _project: Project = Depends(existing_project)) -> ReferenceListOut:
    try:
        return await run_in_threadpool(reference_list, store, project_id, location_id,
                                       include_rejected=include_rejected)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/{project_id}/notes/{note_id}/review", response_model=ReviewResultOut,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def review(project_id: str, note_id: str, body: ReviewRequest, store: Store = Depends(get_store),
                 _project: Project = Depends(existing_project)) -> ReviewResultOut:
    """Confirm or reject one note (any kind - reference_image reviews go through here too).

    Always goes through curate.review_note; this route never writes a Note directly, so
    it inherits review_note's business rules (guidance only on confirmed reference_image
    notes, a reject needs a reason, etc.) without duplicating them. expectedRevision and
    expectedStatus are both required and enforced atomically by the store - a stale pair
    means someone else (another tab, or a pipeline re-run) already changed this note;
    that is a 409, not a silent overwrite.
    """
    try:
        result = await run_in_threadpool(
            review_note, store, project_id, note_id, body.decision,
            reviewer=body.by, reason=body.reason, duplicate_of=body.duplicate_of,
            guidance=body.guidance, expected_revision=body.expected_revision,
            expected_status=body.expected_status,
        )
    except NoteReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    n = result.note
    return ReviewResultOut(id=n.id, status=n.status, revision=n.revision,
                           reviewed_by=n.reviewed_by, reviewed_at=n.reviewed_at, guidance=n.guidance,
                           review_reason=n.review_reason)


@router.post("/projects/{project_id}/notes/{note_id}/correct", response_model=NoteCorrectionResultOut,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def correct(project_id: str, note_id: str, body: CorrectNoteRequest, store: Store = Depends(get_store),
                  _project: Project = Depends(existing_project)) -> NoteCorrectionResultOut:
    """Correct a description/constraint/tone note's text, kind, or applicability - or
    split it into a standing note and a scene-specific one - without ever editing it in
    place. See curate.correct_note for the full contract (concurrency, provenance,
    review-state, retrieval, re-ingestion, and approval-staleness behavior).

    One successor in the body is a plain correction; two is a split. Ownership
    (owner_id) is never settable here - both/all successors always inherit the
    original note's owner. expectedRevision/expectedStatus are both required, checked
    atomically against the note being corrected - a stale pair is a 409, and nothing
    is written (the caller's draft, the successors it already built, is not lost;
    re-fetch the note and retry with the current values).
    """
    try:
        result = await run_in_threadpool(
            correct_note, store, project_id, note_id,
            [NoteSuccessorSpec(kind=s.kind, body=s.body, scene_id=s.scene_id,
                              include_descendants=s.include_descendants)
             for s in body.successors],
            reviewer=body.by, expected_revision=body.expected_revision,
            expected_status=body.expected_status,
        )
    except NoteReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await run_in_threadpool(note_correction_result_out, store, project_id, result)


@router.post("/projects/{project_id}/locations/{location_id}/references/upload",
             response_model=ReferenceUploadOut, status_code=201,
             dependencies=[Depends(require_allowed_origin)])
async def upload_reference(
    project_id: str, location_id: str, request: Request,
    filename: str = Query(..., description="Original filename of the uploaded image"),
    store: Store = Depends(get_store),
    blobs: Blobs = Depends(get_blobs),
    settings: Settings = Depends(get_settings),
    api_settings: ApiSettings = Depends(get_api_settings),
    _project: Project = Depends(existing_project),
) -> ReferenceUploadOut:
    """Upload a user-provided image as a reference for a specific location.

    Raw body (not multipart). The filename is supplied as a query parameter. One image
    per request; the frontend loops for batches, matching the existing sources upload
    convention. Accepted: JPEG, PNG, WebP. Max size: HARNESS_API_MAX_UPLOAD_BYTES.

    Does NOT call any LLM. No caption, direction, attribution, or licensing is generated.
    The note begins in "proposed" status and must be confirmed or rejected via the
    standard review endpoint before it influences any visual concept.

    created: true  -> a new attachment note was written for this location.
    created: false -> this image was already attached to this location; the existing note
                      (with its current review state) is returned unchanged.
    """
    # ---- read body, bounded ---------------------------------------------------
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > api_settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds the {api_settings.max_upload_bytes}-byte limit",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="request body is empty")

    # ---- validate location ownership -----------------------------------------
    # resolve_scope raises LookupError if the ref is ambiguous/missing; ValueError
    # if the resolved entity is a Scene (references are per Location, not per Scene).
    try:
        resolved_id = await run_in_threadpool(resolve_scope, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Confirm the resolved entity is a Location, not a Scene (resolve_scope accepts both).
    def _check_is_location():
        graph = _Graph(store.list_entities(project_id))
        entity = graph.by_id.get(resolved_id)
        if not isinstance(entity, Location):
            raise ValueError(f"{location_id!r} is a scene; reference uploads are per location")

    try:
        await run_in_threadpool(_check_is_location)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---- domain call ---------------------------------------------------------
    try:
        note, created = await run_in_threadpool(
            upload_reference_image,
            store, blobs, settings, project_id, resolved_id, data, filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_path = f"/projects/{project_id}/references/{note.id}/image"
    return ReferenceUploadOut(
        note_id=note.id,
        image_path=image_path,
        created=created,
        status=note.status,
        revision=note.revision,
    )
