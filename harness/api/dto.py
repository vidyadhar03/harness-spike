"""Response/request shapes for the frontend. Kept separate from harness.memory.schemas,
which are the LLM structured-output schemas for a different audience entirely.

All models serialize as camelCase (matching lib/model.ts) via CamelModel; construct them
with snake_case keyword args from Python and FastAPI handles the rename on the wire.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


# --- projects -----------------------------------------------------------------------

class ProjectSummary(CamelModel):
    id: str
    name: str


class ProjectDetail(CamelModel):
    id: str
    name: str
    source_count: int
    note_count: int
    location_count: int
    scene_count: int


class ProjectCreateRequest(CamelModel):
    name: str = Field(min_length=1)


# --- sources / ingest -------------------------------------------------------------------

class SourceOut(CamelModel):
    """Response for one uploaded file - direct mapping of the Source register_file
    returns, plus whether this upload was new or an already-registered duplicate."""
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    status: Literal["uploaded", "digesting", "digested", "failed", "unsupported"]
    created: bool
    error: str | None = None


class SourceSummaryOut(CamelModel):
    """One row for the Sources/onboarding screen - GET /projects/{id}/sources.

    Scope: uploaded ingestion inputs only (register_file) - never a references-pipeline-
    fetched image (references._store_image). See mapping.uploaded_sources for the exact
    exclusion rule: Source.origin_url is not None marks the latter; register_file never
    sets that field, so it's a reliable signal (the only two places a Source is ever
    constructed in the domain are those two functions).
    """
    id: str                    # sha256 of the bytes - stable, matches Source.id
    filename: str
    mime_type: str
    size_bytes: int             # Source.size_bytes is a required field - always present
    status: Literal["uploaded", "digesting", "digested", "failed", "unsupported"]
    # The Source's own last-recorded failure (register_file's "no v0 handler for
    # <mime>", or ingest_source's caught exception message) - persisted fact, not
    # derived or guessed. This is NOT the same field as, and does not replace, an
    # ingest JOB's error from GET .../jobs/{jobId}: a source can show error=null here
    # even while its most recent ingest job failed for a reason that never touched this
    # source at all (e.g. a different source in the same batch, or a lock/scheduling
    # problem). Check both where relevant, don't conflate them.
    error: str | None = None


class IngestRequest(CamelModel):
    # None means "every eligible source in the project" (mirrors `harness-memory ingest
    # $PID` with no --source). force defaults False - no UI for forced re-ingestion yet.
    source_id: str | None = None
    force: bool = False


class IngestSourceResultOut(CamelModel):
    """Direct mapping of one IngestReport - see harness.memory.ingest.IngestReport."""
    source_id: str
    filename: str
    status: str
    doc_type: str | None = None
    entities_created: int = 0
    notes_written: int = 0
    notes_replaced: int = 0
    notes_reused: int = 0
    warnings: list[str] = []
    error: str | None = None


# --- locations ------------------------------------------------------------------------

class LocationSummary(CamelModel):
    id: str
    name: str
    aliases: list[str]
    status: str
    parent_name: str | None
    scene_numbers: list[str]
    note_count: int


class CitationOut(CamelModel):
    source_id: str | None
    filename: str | None
    page: int | None
    quote: str | None
    url: str | None
    title: str | None


class NoteOut(CamelModel):
    id: str
    kind: str
    body: str
    status: Literal["proposed", "confirmed", "rejected"]
    revision: int
    citations: list[CitationOut]


class SceneRefOut(CamelModel):
    id: str
    number: str | None
    name: str


class LocationDetail(CamelModel):
    id: str
    name: str
    aliases: list[str]
    status: str
    ancestors: list[str]
    scenes: list[SceneRefOut]
    description_notes: list[NoteOut]
    constraint_notes: list[NoteOut]
    tone_notes: list[NoteOut]
    superseded_sources: list[str]


# --- references -----------------------------------------------------------------------

class ReferenceOut(CamelModel):
    id: str
    title: str
    image: str                          # root-relative path; caller resolves against apiBase
    category: str                       # the raw backend facet: Place | Terrain | Architecture | Material
    facet: str | None
    reason: str
    direction: str | None
    direction_rationale: str | None
    source: str | None
    credit: str | None
    license: str | None
    attribution: str | None
    selected: bool                      # == (status == "confirmed"), kept for existing UI code
    status: Literal["proposed", "confirmed", "rejected"]
    guidance: str
    owned: bool                         # False when inherited from a confirmed ancestor location
    inherited_from: str | None          # ancestor location name, when owned is False
    revision: int


class ReferenceListOut(CamelModel):
    location_id: str
    references: list[ReferenceOut]


class ReferenceUploadOut(CamelModel):
    """Response from POST .../references/upload.

    note_id   - stable ID of the attachment note; use for review and image preview routes.
    image_path - root-relative path to the preview image: /projects/{p}/references/{note_id}/image
    created   - True if a NEW attachment note was written for this location; False if the same
                image bytes were already attached to this location (the existing note, with its
                current status/revision/guidance, is returned unchanged). This field refers to
                the location attachment, not to whether the image bytes were new.
    status    - current review status of the note ("proposed" on first attach).
    revision  - current revision counter of the note.
    """
    note_id: str
    image_path: str
    created: bool
    status: Literal["proposed", "confirmed", "rejected"]
    revision: int


# --- review ---------------------------------------------------------------------------

class ReviewRequest(CamelModel):
    decision: Literal["confirmed", "rejected"]
    reason: Literal["false", "wrong_scope", "duplicate", "not_useful", "other"] | None = None
    duplicate_of: str | None = None
    guidance: str | None = None
    by: str = "user"
    # both required: the concurrency guard checks (revision, status) together, because a
    # plain confirm/reject with no guidance edit never bumps revision by itself - see
    # harness.memory.ports.NoteReviewConflict
    expected_revision: int
    expected_status: Literal["proposed", "confirmed", "rejected"]


class ReviewResultOut(CamelModel):
    id: str
    status: str
    revision: int
    reviewed_by: str | None
    reviewed_at: datetime | None
    guidance: str | None


# --- jobs -----------------------------------------------------------------------------

class JobCreateRequest(CamelModel):
    kind: Literal["references", "concept"]


class JobCreateResponse(CamelModel):
    id: str


class JobStatusOut(CamelModel):
    status: Literal["queued", "running", "succeeded", "failed"]
    error: str | None = None
    warnings: list[str] = []
    references: list[ReferenceOut] | None = None        # kind == "references"
    # kind == "ingest": outcome distinguishes complete/partial/failed/no_op even though
    # status only has succeeded/failed - see JobRunner._ingest_outcome. sources carries
    # per-file detail whenever there's anything to show, including on status=="failed"
    # (unlike references, an all-failed ingest still has per-source detail worth seeing).
    outcome: Literal["complete", "partial", "failed", "no_op"] | None = None
    sources: list[IngestSourceResultOut] | None = None
