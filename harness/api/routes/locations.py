from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from harness.memory.models import Project
from harness.memory.ports import Store

from ..deps import existing_project, get_store
from ..dto import LocationDetail, LocationSummary
from ..mapping import location_detail, location_summaries

router = APIRouter(tags=["locations"])


@router.get("/projects/{project_id}/locations", response_model=list[LocationSummary])
async def list_locations(project_id: str, store: Store = Depends(get_store),
                         _project: Project = Depends(existing_project)) -> list[LocationSummary]:
    return await run_in_threadpool(location_summaries, store, project_id)


@router.get("/projects/{project_id}/locations/{location_id}", response_model=LocationDetail)
async def get_location(project_id: str, location_id: str, store: Store = Depends(get_store),
                       _project: Project = Depends(existing_project)) -> LocationDetail:
    try:
        return await run_in_threadpool(location_detail, store, project_id, location_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
