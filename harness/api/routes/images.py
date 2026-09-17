from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from starlette.concurrency import run_in_threadpool

from harness.memory.models import Project
from harness.memory.ports import Blobs, Store
from harness.memory.retrieval import _reference

from ..deps import existing_project, get_blobs, get_store
from ..mapping import find_note

router = APIRouter(tags=["images"])


@router.get("/projects/{project_id}/references/{note_id}/image")
async def get_reference_image(project_id: str, note_id: str, store: Store = Depends(get_store),
                              blobs: Blobs = Depends(get_blobs),
                              _project: Project = Depends(existing_project)) -> Response:
    """Streams the bytes server-side rather than issuing a signed URL: the documented
    local dev credential (`gcloud auth application-default login`, end-user ADC) has no
    private key to sign with, so GCSBlobs.get (already implemented, works under any ADC)
    is the delivery path that actually works without provisioning a signing-capable
    service account. See API_CONTRACT.md.
    """
    note = await run_in_threadpool(find_note, store, project_id, note_id)
    if note is None or note.kind != "reference_image":
        raise HTTPException(status_code=404, detail="reference not found")

    source_ids = [p.source_id for p in note.provenance if p.source_id]
    sources = await run_in_threadpool(store.get_sources, project_id, source_ids)
    ref = _reference(note, sources)
    if ref is None:
        raise HTTPException(status_code=404, detail="image unavailable")

    try:
        data = await run_in_threadpool(blobs.get, ref.uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="image unavailable") from exc

    src = sources.get(note.provenance[0].source_id) if note.provenance else None
    # a rendered PDF page (ref.page is set) is always the PNG the ingest pipeline
    # rendered it to; anything else keeps the original source's own mime type.
    mime = "image/png" if ref.page is not None or src is None else src.mime_type
    # content is sha256-addressed and immutable once digested, so caching aggressively
    # is safe; "private" because there is no CDN/shared-cache boundary in Phase 1.
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=86400"})
