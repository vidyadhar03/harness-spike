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
DocType = Literal["script", "lookbook", "recce", "research", "schedule", "notes", "reference", "unknown"]
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

    @field_validator("storage_path")
    @classmethod
    def _no_signed_urls(cls, v: str) -> str:
        if not v.startswith("gs://") or "?" in v:
            raise ValueError("storage_path must be a plain gs:// path")
        return v

    @property
    def superseded(self) -> bool:
        return self.superseded_by_source_id is not None


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
RejectReason = Literal["false", "wrong_scope", "duplicate", "not_useful", "other"]


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
    supersedes_note_id: str | None = None

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
