from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from harness.memory.models import Location, Project, Scene
from harness.memory.ports import Store

from ..deps import existing_project, get_store, require_allowed_origin, require_json
from ..dto import ProjectCreateRequest, ProjectDetail, ProjectSummary

router = APIRouter(tags=["projects"])


@router.get("/projects", response_model=list[ProjectSummary])
async def list_projects(store: Store = Depends(get_store)) -> list[ProjectSummary]:
    projects = await run_in_threadpool(store.list_projects)
    return [ProjectSummary(id=p.id, name=p.name) for p in projects]


@router.post("/projects", response_model=ProjectSummary, status_code=201,
            dependencies=[Depends(require_allowed_origin), Depends(require_json)])
async def create_project(body: ProjectCreateRequest, store: Store = Depends(get_store)) -> ProjectSummary:
    """Fresh-project creation only - identical to the CLI's `project create`. No
    migration or compatibility handling for existing projects; this never touches one."""
    project = Project(name=body.name)
    await run_in_threadpool(store.put_project, project)
    return ProjectSummary(id=project.id, name=project.name)


@router.get("/projects/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str, store: Store = Depends(get_store),
                      project: Project = Depends(existing_project)) -> ProjectDetail:
    sources = await run_in_threadpool(store.list_sources, project_id)
    entities = await run_in_threadpool(store.list_entities, project_id)
    notes = await run_in_threadpool(store.list_notes, project_id)
    return ProjectDetail(
        id=project.id, name=project.name, source_count=len(sources), note_count=len(notes),
        location_count=sum(isinstance(e, Location) for e in entities),
        scene_count=sum(isinstance(e, Scene) for e in entities),
    )
