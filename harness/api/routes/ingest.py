from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from starlette.concurrency import run_in_threadpool

from harness.memory.models import Project
from harness.memory.ports import Store

from ..deps import existing_project, get_job_runner, get_store, require_allowed_origin, require_json
from ..dto import IngestRequest, JobCreateResponse
from ..jobs import IngestTriggerConflict, JobRunner

router = APIRouter(tags=["ingest"])


@router.post("/projects/{project_id}/ingest", response_model=JobCreateResponse,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def start_ingest(project_id: str, body: IngestRequest, response: Response,
                       runner: JobRunner = Depends(get_job_runner), store: Store = Depends(get_store),
                       _project: Project = Depends(existing_project)) -> JobCreateResponse:
    """Starts (or reconnects to) a background ingest run - see JobRunner.trigger_ingest
    for exactly how this shares harness.memory.ingest.ingest_lock with the CLI instead
    of inventing a second, competing lock.
    """
    if body.source_id is not None:
        # correction: a sourceId must belong to this project before it can become part
        # of a job's fixed workload - 404, not a job that would silently process nothing.
        src = await run_in_threadpool(store.get_source, project_id, body.source_id)
        if src is None:
            raise HTTPException(status_code=404, detail="source not found in project")
        if not src.is_ingest_eligible:
            # effective_purpose is "reference" or "concept" here (the only two
            # non-ingest-eligible values) - label accordingly so the message stays
            # accurate for a concept-art source instead of misdescribing it.
            label = "concept-art image" if src.effective_purpose == "concept" else "reference image"
            raise HTTPException(
                status_code=422,
                detail=(
                    f"this source is stored as a {label} and cannot be used as "
                    "an ingestion input; it does not appear in GET /sources"
                ),
            )

    try:
        job, created = await runner.trigger_ingest(project_id, source_id=body.source_id, force=body.force)
    except IngestTriggerConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response.status_code = 201 if created else 200
    return JobCreateResponse(id=job.id)
