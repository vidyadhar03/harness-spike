from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from harness.memory.concept_generation import (
    CONFIG_ERROR, ConceptGenerationService, IdempotencyConflict, JobStateConflict, ProviderNotConfigured,
    StaleGenerationContext,
)
from harness.memory.models import Project

from ..deps import existing_project, require_allowed_origin, require_json
from ..dto import (
    GenerationJobOut, GenerationOptionsOut, GenerationPreviewOut, GenerationPreviewRequest,
    GenerationResolveRequest, GenerationSubmitRequest, GenerationSubmitResultOut,
)
from ..generation import GenerationDispatcher
from ..mapping import generation_job_out, generation_preview_out

router = APIRouter(tags=["concept-generation"])


def get_generation_service(request: Request) -> ConceptGenerationService:
    """History/inspection: always available (needs no provider)."""
    return request.app.state.generation_service


def get_configured_service(svc: ConceptGenerationService = Depends(get_generation_service)) -> ConceptGenerationService:
    """Anything that talks to the provider: 503 with a clear configuration error when there is none."""
    if not svc.configured:
        raise HTTPException(status_code=503, detail=CONFIG_ERROR)
    return svc


def get_dispatcher(request: Request) -> GenerationDispatcher | None:
    return request.app.state.generation_dispatcher


def _job_out(svc: ConceptGenerationService, job) -> GenerationJobOut:
    return generation_job_out(job, resumable=svc.can_resume(job))


async def _guard(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (StaleGenerationContext, IdempotencyConflict, JobStateConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:       # includes imagegen.UnsupportedRequest
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/projects/{project_id}/concept-generation/options", response_model=GenerationOptionsOut)
async def generation_options(svc: ConceptGenerationService = Depends(get_generation_service),
                             _project: Project = Depends(existing_project)) -> GenerationOptionsOut:
    if not svc.configured:
        return GenerationOptionsOut(
            configured=False, configuration_error=CONFIG_ERROR, provider=None, default_model=None, models=[],
            aspect_ratios=[], output_formats=[], max_references=None, prompt_max_chars=None,
            max_request_bytes=None, max_output_pixels=svc.config.max_output_pixels)
    c = svc.provider.capabilities
    return GenerationOptionsOut(
        configured=True, configuration_error=None, provider=svc.provider.name,
        default_model=svc.config.default_model, models=sorted(c.models),
        aspect_ratios=sorted(c.aspect_ratios), output_formats=sorted(c.output_formats),
        max_references=c.max_references, prompt_max_chars=c.max_prompt_chars,
        max_request_bytes=c.max_request_bytes, max_output_pixels=svc.config.max_output_pixels)


@router.post("/projects/{project_id}/locations/{location_id}/concept-generation/preview",
             response_model=GenerationPreviewOut,
             dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def preview_generation(project_id: str, location_id: str, body: GenerationPreviewRequest,
                             svc: ConceptGenerationService = Depends(get_configured_service),
                             _project: Project = Depends(existing_project)) -> GenerationPreviewOut:
    """Read-only and free: builds the exact prompt/snapshot/token a submit would use. No provider call."""
    prepared = await _guard(
        svc.prepare, project_id, location_id, reference_ids=body.reference_ids,
        depiction_label=body.depiction_label, direction=body.direction, model=body.model,
        aspect_ratio=body.aspect_ratio, output_format=body.output_format)
    return generation_preview_out(project_id, prepared.location_id, prepared, svc.provider.name)


@router.post("/projects/{project_id}/locations/{location_id}/concept-generation/jobs",
             response_model=GenerationSubmitResultOut, status_code=201,
             dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def submit_generation(project_id: str, location_id: str, body: GenerationSubmitRequest,
                            response: Response,
                            svc: ConceptGenerationService = Depends(get_configured_service),
                            dispatcher: GenerationDispatcher | None = Depends(get_dispatcher),
                            _project: Project = Depends(existing_project)) -> GenerationSubmitResultOut:
    """Persists the job (no provider call yet), then starts it in the background. 201 = new job,
    200 = same idempotency key + payload (the existing job). 409 = stale preview / key reused with a
    different payload."""
    job, created = await _guard(
        svc.submit, project_id, location_id, reference_ids=body.reference_ids, context_token=body.context_token,
        depiction_label=body.depiction_label, direction=body.direction, model=body.model,
        aspect_ratio=body.aspect_ratio, output_format=body.output_format, idempotency_key=body.idempotency_key)
    if dispatcher is not None and job.state == "queued":
        dispatcher.schedule(project_id, job.id)
    response.status_code = 201 if created else 200
    return GenerationSubmitResultOut(created=created, job=_job_out(svc, job))


@router.get("/projects/{project_id}/locations/{location_id}/concept-generation/jobs",
            response_model=list[GenerationJobOut])
async def list_generation_jobs(project_id: str, location_id: str,
                               svc: ConceptGenerationService = Depends(get_generation_service),
                               _project: Project = Depends(existing_project)) -> list[GenerationJobOut]:
    jobs = await _guard(svc.list, project_id, location_id)
    return [_job_out(svc, j) for j in jobs]


@router.get("/projects/{project_id}/concept-generation/jobs/{job_id}", response_model=GenerationJobOut)
async def get_generation_job(project_id: str, job_id: str,
                             svc: ConceptGenerationService = Depends(get_generation_service),
                             _project: Project = Depends(existing_project)) -> GenerationJobOut:
    job = await run_in_threadpool(svc.get, project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="generation job not found")
    return _job_out(svc, job)


@router.post("/projects/{project_id}/concept-generation/jobs/{job_id}/resume",
             response_model=GenerationJobOut, status_code=202,
             dependencies=[Depends(require_allowed_origin)])
async def resume_generation_job(project_id: str, job_id: str,
                                svc: ConceptGenerationService = Depends(get_configured_service),
                                dispatcher: GenerationDispatcher | None = Depends(get_dispatcher),
                                _project: Project = Depends(existing_project)) -> GenerationJobOut:
    """Continue an existing provider job (poll and/or import). Never submits a new generation."""
    job = await run_in_threadpool(svc.get, project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="generation job not found")
    if not svc.can_resume(job):
        raise HTTPException(status_code=409, detail=f"job is {job.state} and cannot be resumed right now")
    if dispatcher is not None:
        dispatcher.schedule(project_id, job_id)
    return _job_out(svc, job)


@router.post("/projects/{project_id}/concept-generation/jobs/{job_id}/resolve",
             response_model=GenerationJobOut,
             dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def resolve_generation_job(project_id: str, job_id: str, body: GenerationResolveRequest,
                                 svc: ConceptGenerationService = Depends(get_configured_service),
                                 dispatcher: GenerationDispatcher | None = Depends(get_dispatcher),
                                 _project: Project = Depends(existing_project)) -> GenerationJobOut:
    """Human resolution of an ambiguous submission (submission_unknown) or a rejected one."""
    job = await _guard(svc.resolve, project_id, job_id, action=body.action,
                       provider_generation_id=body.provider_generation_id,
                       acknowledge_no_provider_job=body.acknowledge_no_provider_job, by=body.by)
    if dispatcher is not None:
        dispatcher.schedule(project_id, job_id)
    return _job_out(svc, job)
