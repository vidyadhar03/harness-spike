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
    # Applicability, exposed so a client can tell a standing note from a scene-scoped
    # one without a separate lookup - e.g. to show which notes POST .../correct could
    # move to a specific scene, or confirm a split landed where intended.
    scene_id: str | None = None
    include_descendants: bool = False


class SceneRefOut(CamelModel):
    id: str
    number: str | None
    name: str


class ScopedNoteOut(NoteOut):
    """A NoteOut plus what a client needs to decide whether/how to offer "Correct this
    note" - see POST .../notes/{noteId}/correct."""
    owner_id: str
    owned: bool                         # False = inherited from an ancestor (read-only here)
    inherited_from: str | None = None   # ancestor location name, when owned is False
    editable: bool                      # True iff owned and a kind correct_note accepts


class LocationSceneRequirementOut(CamelModel):
    """One scene's scene-scoped notes, exactly as retrieval.get_context surfaces them.

    linked=True entries are the location's authoritative linked-scene roster (one per
    linked scene, notes=[] when none were extracted - a real fact, not an omission).
    linked=False entries are conditional notes whose sceneId is NOT in that roster -
    they are not linked scenes and must not be presented as such (number is null and
    heading is whatever retrieval could resolve, or the raw scene id).
    """
    scene_id: str
    number: str | None
    heading: str | None
    linked: bool
    notes: list[ScopedNoteOut]


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
    scene_requirements: list[LocationSceneRequirementOut] = []
    out_of_roster_scene_requirements: list[LocationSceneRequirementOut] = []
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
    # "corrected" here means this exact note was replaced by POST .../correct below -
    # see NoteCorrectionResultOut.newNotes for what replaced it. Null for an ordinary
    # confirm/reject, and for anything reviewed before this field existed.
    review_reason: str | None = None


# --- note correction / splitting -------------------------------------------------------

class NoteCorrectionSuccessor(CamelModel):
    kind: Literal["description", "constraint", "tone"]
    body: str = Field(min_length=1, max_length=2000)
    # None (the default) means "standing" (applies to the whole location, not one
    # scene) - matching Applicability.scene_id's own meaning. Set it to scope this
    # successor to one scene instead. Never inferred from the note's wording.
    scene_id: str | None = None
    include_descendants: bool = False


class CorrectNoteRequest(CamelModel):
    """One successor = a plain correction (fix wording/kind, or move between standing
    and a specific scene). Two = a split - e.g. one standing successor plus one
    scene-specific successor from the same original note. Ownership (owner_id) is not
    settable here - it is always inherited from the note being corrected."""
    successors: list[NoteCorrectionSuccessor] = Field(min_length=1, max_length=2)
    # both required, same rationale as ReviewRequest's - a stale pair means someone
    # else already reviewed or corrected this note since you read it: 409, re-fetch.
    expected_revision: int
    expected_status: Literal["proposed", "confirmed", "rejected"]
    by: str = "user"


class NoteCorrectionResultOut(CamelModel):
    original: ReviewResultOut     # now status="rejected", reviewReason="corrected"
    new_notes: list[NoteOut]      # in the same order as the request's successors


# --- concept versions -------------------------------------------------------------------

class ConceptVersionOut(CamelModel):
    id: str
    image: str                          # root-relative path; caller resolves against apiBase
    filename: str
    created_at: datetime
    author: Literal["agent", "user"]
    approved: bool                      # True iff this is the location's currently approved version
    # Both null for a raw upload. Both set iff this version was created via
    # POST .../concepts/from-reference - see ConceptVersionOut docs below and
    # harness.memory.concepts.promote_reference_to_concept.
    promoted_from_note_id: str | None = None
    promoted_from_note_revision: int | None = None


class ConceptVersionListOut(CamelModel):
    location_id: str
    approved_version_id: str | None
    versions: list[ConceptVersionOut]


class ConceptUploadOut(CamelModel):
    """Response from POST .../concepts/upload or POST .../concepts/from-reference.

    created - True if a NEW version was written; False if these exact bytes were
    already registered as a version for this location (the existing version is
    returned unchanged, including its original promotedFromNoteId if any - a second
    promotion attempt, or a raw upload, that resolves to the same existing record
    never overwrites its recorded provenance). Never implies anything about approval
    either way.
    """
    id: str
    image: str
    filename: str
    created: bool
    approved: bool
    promoted_from_note_id: str | None = None
    promoted_from_note_revision: int | None = None


class PromoteReferenceRequest(CamelModel):
    reference_id: str


# --- concept approval --------------------------------------------------------------------

class SceneRequirementOut(CamelModel):
    """One entry per scene in the location's linked-scene roster - always present
    regardless of whether the scene has any requirements (see ApprovalPackageOut's
    sceneCoverageComplete: an empty notes list here is only a confirmed fact when
    that is true)."""
    scene_id: str
    number: str | None
    heading: str | None
    notes: list[NoteOut]


class ApprovalInheritedOut(CamelModel):
    entity_id: str
    name: str
    notes: list[NoteOut]


class ApprovedReferenceOut(CamelModel):
    note_id: str
    revision: int
    status: Literal["proposed", "confirmed", "rejected"]
    guidance: str | None
    direction: str | None
    caption: str
    image: str                          # root-relative path to the reference's own image


class ApprovalPackageOut(CamelModel):
    id: str
    location_id: str
    revision: int
    concept_version_id: str
    concept_image: str
    concept_filename: str
    # Free-text, user-supplied ("whole-house exterior", "bedroom interior - top
    # view") - never inferred, never validated against any room/location relationship,
    # never a claim of spatial accuracy. Null if none was given (always null on an
    # approval locked before this field existed - a true "none was given", not a gap).
    depiction_label: str | None
    # kind in ("description", "tone"), unconditional (no scene_id) - the location's
    # own standing character/facts.
    core_notes: list[NoteOut]
    # Always present alongside core_notes; see harness.memory.concepts.CORE_NOTES_CAVEAT
    # for exactly what it does and does not claim.
    core_notes_caveat: str
    # kind == "constraint", unconditional - the existing structured slot for physical/
    # standing requirements, reused as "set-dressing requirements" as-is.
    physical_notes: list[NoteOut]
    # One entry per linked scene (see SceneRequirementOut) - any kind, scoped to that
    # scene via applicability.scene_id.
    scene_requirements: list[SceneRequirementOut]
    # False iff this approval predates full-roster coverage (locked before this slice) -
    # when False, an empty notes list on a scene_requirements entry, or a scene simply
    # absent from the list, must NOT be read as "no requirements"; see staleReasons on
    # GET .../approval, which explains this same gap for an already-stale package.
    scene_coverage_complete: bool
    brief_inherited: list[ApprovalInheritedOut]
    brief_ancestors: list[str]
    superseded_sources: list[str]
    references: list[ApprovedReferenceOut]
    context_token: str
    locked_by: str | None
    locked_at: datetime


class ApprovalStateOut(CamelModel):
    """GET .../approval - also the retrieval boundary a later, separate "Send to 3D
    blockout" action would read from; this endpoint never marks anything as sent."""
    location_id: str
    revision: int                       # 0 when approval is None; feeds the next lock's expectedRevision
    approval: ApprovalPackageOut | None
    is_stale: bool                      # recomputed live; False whenever approval is None
    stale_reasons: list[str] = []        # human-readable; empty whenever is_stale is False


class ApprovalPreviewOut(CamelModel):
    """GET .../approval/preview - exactly what a lock right now would capture, for a
    given candidate + explicit reference selection + depiction label. Persists
    nothing, and is always computed against complete, current scene coverage (there is
    no legacy/partial state for a preview - that only exists for an already-persisted
    ApprovalPackageOut)."""
    location_id: str
    concept_version_id: str
    concept_image: str
    concept_filename: str
    depiction_label: str | None
    core_notes: list[NoteOut]
    core_notes_caveat: str
    physical_notes: list[NoteOut]
    scene_requirements: list[SceneRequirementOut]
    brief_inherited: list[ApprovalInheritedOut]
    brief_ancestors: list[str]
    superseded_sources: list[str]
    references: list[ApprovedReferenceOut]
    context_token: str


class ApprovalLockRequest(CamelModel):
    concept_version_id: str
    reference_ids: list[str] = []
    # Small free-text description of what the concept image depicts - see
    # ApprovalPackageOut.depictionLabel. Optional; omit or send null/empty for none.
    depiction_label: str | None = None
    # both required, same rationale as ReviewRequest's expectedRevision/expectedStatus:
    # contextToken guards against locking material that changed since it was previewed;
    # expectedRevision guards against a race with another concurrent lock. Always echo
    # back exactly what GET .../approval/preview and GET .../approval last returned.
    context_token: str
    expected_revision: int
    by: str = "user"                    # free-text label; not an authenticated identity


class ApprovalLockResult(CamelModel):
    created: bool                       # False when this exact package was already current (idempotent retry)
    approval: ApprovalPackageOut


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
