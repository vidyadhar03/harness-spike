from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from starlette.concurrency import run_in_threadpool

from harness.memory.models import Location, Project
from harness.memory.ports import Store
from harness.memory.retrieval import resolve_scope

from ..deps import existing_project, get_job_runner, get_store, require_allowed_origin, require_json
from ..dto import JobCreateRequest, JobCreateResponse, JobStatusOut
from ..jobs import JobRunner
from ..mapping import ingest_source_result_out, reference_list

router = APIRouter(tags=["jobs"])


@router.post("/projects/{project_id}/locations/{location_id}/jobs", response_model=JobCreateResponse,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def create_job(project_id: str, location_id: str, body: JobCreateRequest, response: Response,
                     runner: JobRunner = Depends(get_job_runner), store: Store = Depends(get_store),
                     _project: Project = Depends(existing_project)) -> JobCreateResponse:
    if body.kind == "concept":
        # no concept/image-generation pipeline exists in the harness - explicit 501,
        # not a silently-accepted job that can never succeed.
        raise HTTPException(status_code=501,
                            detail="concept generation is not implemented by the harness yet")
    try:
        scope_id = await run_in_threadpool(resolve_scope, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entity = await run_in_threadpool(store.get_entity, project_id, scope_id)
    if not isinstance(entity, Location):
        raise HTTPException(status_code=404, detail="location not found in project")

    job, created = await runner.trigger(project_id, entity.id, "references")
    # idempotent trigger: a repeat POST while a run is already active returns the same
    # job id with 200, not a second concurrent run (matches the frontend's existing
    # reload-and-reconnect-to-a-pending-job behavior)
    response.status_code = 201 if created else 200
    return JobCreateResponse(id=job.id)


@router.get("/projects/{project_id}/jobs/{job_id}", response_model=JobStatusOut)
async def get_job_status(project_id: str, job_id: str, runner: JobRunner = Depends(get_job_runner),
                         store: Store = Depends(get_store),
                         _project: Project = Depends(existing_project)) -> JobStatusOut:
    job = await run_in_threadpool(runner.job_store.get_job, project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    references = None
    sources = None
    if job.status == "succeeded" and job.kind == "references":
        # the authoritative current list, not a delta - the frontend should replace its
        # state with this rather than merge, see API_CONTRACT.md
        result = await run_in_threadpool(reference_list, store, project_id, job.location_id,
                                         include_rejected=False)
        references = result.references
    elif job.kind == "ingest" and job.status in ("succeeded", "failed"):
        # unlike references, per-source detail is worth showing even when the overall
        # job status is "failed" (every attempted source failed) - that's exactly what
        # explains the failure, not something to hide alongside it.
        sources = [ingest_source_result_out(r) for r in job.source_reports]
    return JobStatusOut(status=job.status, error=job.error, warnings=job.warnings,
                        references=references, outcome=job.outcome, sources=sources)
