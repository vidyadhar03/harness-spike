"""MotionX harness - schema v0.5: memory dump, context retrieval, real-world references.

Firestore
  projects/{project_id}                  Project
    sources/{sha256}                     Source
    entities/{entity_id}                 Location | Scene
    notes/{note_id}                      Note

Storage
  projects/{project_id}/sources/{sha256}/original.{ext}
  projects/{project_id}/sources/{sha256}/derived/text.md
  projects/{project_id}/sources/{sha256}/derived/pages/{n:04d}.png
  projects/{project_id}/sources/{sha256}/extractions/{extraction_id}/candidates.json

Rules
- Bytes live in Storage. Firestore holds metadata, identity, and notes.
- Entities carry identity and relationships only. Every fact is a Note.
- One note, one owner. `owner_id` is what the note is about; `applicability` says where
  and when it applies; `mentions` are discovery links that never confer applicability.
- A note's reviewed assertion is (body, owner_id, applicability). Changing any of those
  needs a new note that supersedes the old one - reviewed rows are never edited in place.
- Every extraction archives its validated candidates before anything is replaced, so a
  later merge or rescope can never erase what a source originally said.
- Anything re-derivable from the original file stays out until a step needs it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.5"
PROJECT_SCOPE = "project"  # owner id for production-wide rules: period, grammar, soundscape


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Strict(BaseModel):
    # Greenfield, no legacy reads: forbid extras on reads and writes.
    model_config = ConfigDict(extra="forbid")


class Doc(Strict):
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self, **changes):
        """Copy with changes and a fresh updated_at. Every write goes through this."""
        return self.model_copy(update={**changes, "updated_at": utcnow()})


Author = Literal["agent", "user"]
ReviewStatus = Literal["proposed", "confirmed", "rejected"]


# --- Project -----------------------------------------------------------------

class Project(Doc):
    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION


# --- Source: one per dropped file ---------------------------------------------

SourceKind = Literal["document", "image", "text", "table", "audio", "video", "other"]
DocType = Literal["script", "lookbook", "recce", "research", "schedule", "notes", "reference", "concept", "unknown"]
SourceStatus = Literal["uploaded", "digesting", "digested", "failed", "unsupported"]


class Derived(Strict):
    text_path: str | None = None        # .../derived/text.md
    pages_prefix: str | None = None     # .../derived/pages/
    page_count: int | None = Field(default=None, ge=0)


class Source(Doc):
    """Immutable bytes, mutable processing metadata.

    `id` identifies the bytes; `document_id` identifies the logical document across
    revisions, so Draft 3 of a screenplay supersedes Draft 2 rather than arriving as an
    unrelated file. Knowing a source is superseded does NOT tell you whether a note from
    it is still true - that is a human reconciliation.
    """
    id: str = Field(pattern=r"^[0-9a-f]{64}$")  # sha256 of bytes; re-drops are no-ops
    filename: str
    mime_type: str
    kind: SourceKind
    size_bytes: int = Field(ge=0)
    storage_path: str                   # gs:// path, never a signed URL
    status: SourceStatus = "uploaded"
    doc_type: DocType | None = None     # set by classify
    derived: Derived = Field(default_factory=Derived)
    digest_version: str | None = None   # prompt + code version that digested it
    extraction_id: str | None = None    # the archived run behind the current notes
    error: str | None = None

    document_id: str | None = None      # logical document; set when a revision chain exists
    revision_label: str | None = None   # "Draft 3", "v1.4"
    supersedes_source_id: str | None = None
    superseded_by_source_id: str | None = None

    origin_url: str | None = None       # where a fetched file came from (external references)
    license: str | None = None          # e.g. "CC BY-SA 4.0"; required for fetched images
    attribution: str | None = None      # author/credit line to reproduce with the image

    # Explicit upload intent.  None means "legacy record" - use effective_purpose below.
    # "ingest"    : uploaded for screenplay/notes ingestion; appears in GET /sources.
    # "reference" : stored only for visual reference; excluded from ingestion paths.
    # "both"      : was reference-only, then explicitly promoted via POST /sources.
    # "concept"   : stored only for a location's uploaded concept-art version (see
    #               ConceptVersion); excluded from ingestion paths exactly like
    #               "reference". promote_source_to_ingest treats it exactly like a
    #               "reference"-purpose source too: if the same bytes are later
    #               explicitly uploaded via POST /sources, it is promoted to "both"
    #               (ingest-eligible) the same way - there is no special-cased
    #               exclusion for "concept" there, on purpose.
    #
    # IMPORTANT: source_purpose alone does NOT tell you whether a source has location
    # reference note attachments - that association lives entirely in Note.owner_id.
    # An "ingest"-purpose source can have reference notes; a "both"-purpose source has
    # reference notes AND is eligible for ingestion. Do not infer reference usage from
    # this field; query notes_for_owners instead. Similarly, whether a source backs a
    # concept version lives entirely in ConceptVersion.source_id, never here.
    source_purpose: Literal["ingest", "reference", "both", "concept"] | None = None

    @field_validator("storage_path")
    @classmethod
    def _no_signed_urls(cls, v: str) -> str:
        if not v.startswith("gs://") or "?" in v:
            raise ValueError("storage_path must be a plain gs:// path")
        return v

    @property
    def superseded(self) -> bool:
        return self.superseded_by_source_id is not None

    @property
    def effective_purpose(self) -> Literal["ingest", "reference", "both", "concept"]:
        """Backward-compatible purpose resolution.

        Honors an explicit source_purpose when set - this is the only path that can
        ever produce "concept": no legacy record predates that value, so a None
        source_purpose (see below) never resolves to it. For legacy records written
        before this field existed, retains the origin_url-based distinction:
        - Wikimedia-fetched images always have origin_url set  -> 'reference'
        - Screenplay/notes uploads via register_file never do  -> 'ingest'
        No bulk migration required; existing persisted documents continue to work.
        """
        if self.source_purpose is not None:
            return self.source_purpose
        return "reference" if self.origin_url is not None else "ingest"

    @property
    def is_ingest_eligible(self) -> bool:
        """Single authoritative check used by all five ingestion paths.

        True for 'ingest' and 'both'; False for 'reference' and 'concept'. Callers
        must use this property, not compare effective_purpose strings directly, so
        the set of eligible values stays consistent if new purpose values are added.
        """
        return self.effective_purpose in ("ingest", "both")


# --- Entities: what notes are about --------------------------------------------

EntityStatus = Literal["proposed", "confirmed", "rejected", "merged"]


class RetrievalOrigin(Strict):
    term: str
    query: str
    region: str | None = None
    via: str = ""


class Provenance(Strict):
    """One source occurrence supporting a note.

    Keeps what the occurrence itself said, not only where it was found, so a later edit
    or merge cannot erase the original extracted meaning.
    """
    id: str = Field(default_factory=lambda: new_id("occ"))
    source_id: str | None = None
    url: str | None = None
    title: str | None = None                          # page or file title, for citations
    page: int | None = Field(default=None, ge=1)       # 1-based; None for images/text
    quote: str | None = Field(default=None, max_length=300)
    extraction_id: str | None = None                   # archived run that produced it
    extracted_body: str | None = Field(default=None, max_length=2000)
    extracted_owner_id: str | None = None
    link_status: ReviewStatus = "proposed"             # does this occurrence support the note?
    retrieval_origins: list[RetrievalOrigin] = Field(default_factory=list)
    depicted_place: str | None = None

    @model_validator(mode="after")
    def _anchored(self):
        if not self.source_id and not self.url:
            raise ValueError("provenance needs a source_id or a url")
        return self


class Containment(Strict):
    """Physical containment in the story world, and nothing else.

    Not a geographic region, not a reference grouping, not a real venue standing in for a
    fictional place, not proximity. Only confirmed containment inherits notes downward.
    """
    parent_id: str
    status: ReviewStatus = "proposed"
    provenance: list[Provenance] = Field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class EntityBase(Doc):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)  # "the well", "Devgram well"
    status: EntityStatus = "proposed"
    author: Author
    merged_into: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def _merge_target(self):
        if (self.status == "merged") != (self.merged_into is not None):
            raise ValueError("merged_into is required iff status == 'merged'")
        return self


class Location(EntityBase):
    type: Literal["location"] = "location"
    id: str = Field(default_factory=lambda: new_id("loc"))
    containment: Containment | None = None
    venue_id: str | None = None         # a real place standing in for this story location

    @model_validator(mode="after")
    def _no_self_parent(self):
        if self.containment is not None and self.containment.parent_id == self.id:
            raise ValueError("a location cannot contain itself")
        return self


class Scene(EntityBase):
    type: Literal["scene"] = "scene"
    id: str = Field(default_factory=lambda: new_id("scn"))
    number: str | None = None           # as written in one draft: "12", "12A"
    location_ids: list[str] = Field(default_factory=list)
    source_id: str | None = None        # the draft this numbering came from


Entity = Annotated[Union[Location, Scene], Field(discriminator="type")]


# --- Notes: the unit of memory ------------------------------------------------

NoteKind = Literal["description", "constraint", "reference_image", "tone", "vocabulary"]
RejectReason = Literal["false", "wrong_scope", "duplicate", "not_useful", "other", "corrected"]


class NoteOrigin(Strict):
    """Which step wrote this note, so a re-run can reconcile against its own proposals."""
    digest_version: str
    producer: str = "ingest"            # "ingest" | "references" | "user"
    source_id: str | None = None        # ingest: the file digested
    scope: str | None = None            # references: the entity the run was for


class Applicability(Strict):
    """Where and when a note applies, separate from what it is about.

    include_descendants defaults to False: "Devgram has 300 houses" must not become a
    fact about the market square inside it. scene_id marks a fact true only during one
    scene, which stays visibly conditional and never becomes permanent geometry.
    """
    include_descendants: bool = False
    scene_id: str | None = None


class Note(Doc):
    id: str = Field(default_factory=lambda: new_id("note"))
    kind: NoteKind
    body: str = Field(min_length=1, max_length=2000)   # markdown
    owner_id: str                                      # one entity id, or PROJECT_SCOPE
    applicability: Applicability = Field(default_factory=Applicability)
    mentions: list[str] = Field(default_factory=list)  # discovery links only, never applied
    provenance: list[Provenance] = Field(default_factory=list)
    status: ReviewStatus = "proposed"
    author: Author
    group: str | None = None            # optional cluster label, e.g. a visual direction
    origin: NoteOrigin | None = None
    guidance: str | None = Field(default=None, max_length=1000)
    direction: str | None = None
    direction_rationale: str | None = None

    revision: int = Field(default=1, ge=1)
    reviewed_revision: int | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_reason: RejectReason | None = None
    duplicate_of: str | None = None     # the surviving note, when rejected as a duplicate
    supersedes_note_id: str | None = None   # set on a NEW note: the note it replaces

    # Set on the OLD note when curate.correct_note replaces it with one or two new
    # notes (a correction, or a standing/scene-specific split) - the reciprocal of
    # supersedes_note_id above, mirroring Source.superseded_by_source_id's existing
    # revision-chain pattern. A note with entries here is never edited in place and
    # never resurrected by re-ingestion (see curate.correct_note's docstring); it stays
    # exactly as it was for history, just excluded from retrieval like any other
    # rejected note.
    superseded_by_note_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules(self):
        if self.author == "agent" and not self.provenance and self.kind != "vocabulary":
            raise ValueError("agent notes require provenance")
        if self.kind == "reference_image" and len(self.provenance) != 1:
            raise ValueError("reference_image notes point at exactly one image or page")
        if self.owner_id == PROJECT_SCOPE and self.applicability.include_descendants:
            raise ValueError("project notes already apply everywhere")
        if self.duplicate_of is not None and self.review_reason != "duplicate":
            raise ValueError("duplicate_of requires review_reason == 'duplicate'")
        if self.superseded_by_note_ids and self.review_reason != "corrected":
            raise ValueError("superseded_by_note_ids requires review_reason == 'corrected'")
        if self.status == "proposed" and self.reviewed_revision is not None:
            raise ValueError("a proposed note has no review")
        return self

    @property
    def assertion(self) -> tuple:
        """What a reviewer signed off on. Changing this needs a new note, not an edit."""
        return (" ".join(self.body.lower().split()), self.owner_id,
                self.applicability.include_descendants, self.applicability.scene_id)

    @property
    def review_is_current(self) -> bool:
        return self.status == "proposed" or self.reviewed_revision == self.revision


# --- Retrieval contract: assembled at read time, never stored ----------------

class ReferenceImage(Strict):
    note_id: str
    uri: str                            # gs:// original image, or rendered PDF page
    caption: str
    status: ReviewStatus
    source_id: str
    page: int | None = None
    group: str | None = None            # authoritative facet: "Place", "Terrain", etc.
    origin_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    guidance: str | None = None
    direction: str | None = None
    direction_rationale: str | None = None
    retrieval_origins: list[RetrievalOrigin] = Field(default_factory=list)
    depicted_place: str | None = None

    @property
    def facet(self) -> str | None:
        return self.group


class InheritedNotes(Strict):
    entity_id: str
    name: str
    notes: list[Note] = Field(default_factory=list)


class ConditionalNotes(Strict):
    """Notes true only during one scene. Never rendered as the place's general state."""
    scene_id: str
    label: str
    notes: list[Note] = Field(default_factory=list)


class ContextPack(Strict):
    scope_id: str                       # entity id after following merges, or PROJECT_SCOPE
    entity: Entity | None               # None when scope_id == PROJECT_SCOPE
    include_proposed: bool
    merged_ids: list[str] = Field(default_factory=list)       # entities merged into this one
    notes: list[Note] = Field(default_factory=list)           # owned, unconditional
    conditional: list[ConditionalNotes] = Field(default_factory=list)
    inherited: list[InheritedNotes] = Field(default_factory=list)
    project_notes: list[Note] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)         # location packs: scenes set here
    locations: list[Location] = Field(default_factory=list)   # scene packs: where it is set
    ancestors: list[Location] = Field(default_factory=list)   # confirmed containment chain
    reference_images: list[ReferenceImage] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)     # source_id -> filename
    superseded_sources: list[str] = Field(default_factory=list)  # filenames, for a warning


# --- concept art: uploaded versions and their locked approval -----------------

class ConceptVersion(Doc):
    """One uploaded concept-art image for a location. Immutable once created - a repeat
    upload of the same bytes to the same location resolves to this same record (see
    concepts.upload_concept_version) rather than creating a new one, and never touches
    any existing ConceptApproval.

    Deliberately not a Note: keeping concept art out of the Note collection is what
    keeps it invisible to note review, retrieval context packs, and the references
    pipeline's stale-note cleanup, with no filtering code needed anywhere.
    """
    id: str = Field(default_factory=lambda: new_id("cvn"))
    location_id: str
    source_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str
    author: Author = "user"   # always "user" today; kept open for a future generated version

    # Set iff this version was created via concepts.promote_reference_to_concept rather
    # than a raw upload - both None means "uploaded directly". Points at the reference
    # note (and its revision at promotion time) whose stored image this version reuses,
    # so provenance survives even though no new bytes were written. Never rewritten -
    # if a later promotion/upload resolves to this same (location_id, source_id) record
    # via put_concept_version_if_absent, the existing values win untouched (see
    # promote_reference_to_concept's docstring).
    promoted_from_note_id: str | None = None
    promoted_from_note_revision: int | None = None


class ConditionalNoteSnapshot(Strict):
    """One linked scene's requirements, captured at approval time - see ConceptApproval.

    One entry per scene in the location's authoritative linked-scene roster
    (retrieval.get_context's pack.scenes), regardless of whether that scene has any
    scene-conditional notes - notes=[] on a roster scene means "no additional
    requirements were extracted for this scene", a real fact, not a gap.

    number/heading are denormalized display fields, added after label; both None on any
    ConceptApproval persisted before that (see ConceptApproval.snapshot_schema_version -
    such a record's own brief_conditional list may ALSO be incomplete as a roster
    (pre-dates iterating pack.scenes instead of only scenes-with-notes), which is what
    snapshot_schema_version actually flags; missing number/heading here is a narrower,
    always-true-for-old-data symptom of that same age, not a separate gap to track.
    """
    scene_id: str
    label: str
    notes: list[Note] = Field(default_factory=list)
    number: str | None = None
    heading: str | None = None


class InheritedNoteSnapshot(Strict):
    """Ancestor-owned brief notes captured at approval time - see ConceptApproval."""
    entity_id: str
    name: str
    notes: list[Note] = Field(default_factory=list)


class ApprovedReference(Strict):
    """One reference note exactly as explicitly selected and seen at approval time.

    This class itself does not require status == "confirmed" - locking records what
    the approver looked at, it does not itself confirm or reject anything, and this
    model has no opinion on what fed it. In practice status is always "confirmed" here
    today because concepts.build_approval_snapshot refuses to select anything else
    (see its docstring) - that is a business rule enforced by the caller, not a
    constraint of this model.
    """
    note_id: str
    revision: int
    status: ReviewStatus
    guidance: str | None = None
    direction: str | None = None
    caption: str


class ConceptApproval(Doc):
    """An immutable, explicit lock of one location's visual direction.

    Every successful lock writes a brand-new record (revision = prior current + 1,
    starting at 1); no ConceptApproval is ever edited or deleted, so history is
    preserved automatically. Which one is "current" for a location is tracked
    separately by the store (see Store.get_current_approval /
    Store.put_approval_if_current) so later brief/reference edits can never rewrite
    an already-locked snapshot.
    """
    id: str = Field(default_factory=lambda: new_id("apr"))
    location_id: str
    revision: int = Field(default=1, ge=1)

    concept_version_id: str
    concept_source_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    concept_filename: str

    brief_notes: list[Note] = Field(default_factory=list)                    # unconditional, owned
    brief_conditional: list[ConditionalNoteSnapshot] = Field(default_factory=list)
    brief_inherited: list[InheritedNoteSnapshot] = Field(default_factory=list)
    brief_ancestors: list[str] = Field(default_factory=list)                 # ancestor names, display only
    superseded_sources: list[str] = Field(default_factory=list)

    references: list[ApprovedReference] = Field(default_factory=list)

    # Free-text, user-supplied description of what the concept image depicts (e.g.
    # "whole-house exterior", "bedroom interior - top view") - see
    # concepts.MAX_DEPICTION_LABEL_LENGTH / _validate_depiction_label. Purely
    # descriptive: never used to infer or change location_id, never validated against
    # any room/location hierarchy, never a claim about spatial/geometric accuracy.
    # None on any approval locked before this field existed, which is a true "no label
    # was given" for that record, not an ambiguous gap - unlike snapshot_schema_version
    # below, no version flag is needed for this one field.
    depiction_label: str | None = None

    # sha256 hex over the exact inputs above (see concepts._context_token) - the
    # concurrency/staleness primitive: a lock request must recompute to the same
    # token as what it was shown, and GET .../approval recomputes it against live
    # state to expose whether the approved package is now stale.
    context_token: str

    # Bumped whenever the *shape/completeness semantics* of what gets captured here
    # changes in a way that makes old data genuinely ambiguous to interpret under the
    # new logic - not for every additive field (most, like depiction_label above, are
    # safely None=absent on old records with no ambiguity). Version 1 (the default,
    # meaning "this field was absent" - true for every record persisted before it
    # existed): brief_conditional only contains scenes that had at least one
    # scene-conditional note, so an old record's scene list may be silently
    # INCOMPLETE relative to the location's actual linked-scene roster - a missing
    # scene here must never be read as "confirmed no requirements". Version 2:
    # brief_conditional always contains one entry per scene in the roster (see
    # ConditionalNoteSnapshot), so an empty notes=[] there is a real fact. Read via
    # concepts.CURRENT_SNAPSHOT_SCHEMA_VERSION; never bump a record after the fact -
    # only a fresh lock stamps the current value.
    snapshot_schema_version: int = 1

    # Free-text label, not an authenticated identity - this system has no login.
    locked_by: str | None = None
    locked_at: datetime = Field(default_factory=utcnow)
