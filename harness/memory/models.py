"""MotionX harness - schema v0.3: memory dump, context retrieval, real-world references.

Firestore
  projects/{project_id}                  Project
    sources/{sha256}                     Source
    entities/{entity_id}                 Location | Scene
    notes/{note_id}                      Note

Storage
  projects/{project_id}/sources/{sha256}/original.{ext}
  projects/{project_id}/sources/{sha256}/derived/text.md
  projects/{project_id}/sources/{sha256}/derived/pages/{n:04d}.png

Rules
- Bytes live in Storage. Firestore holds metadata, identity, and notes.
- Entities carry identity and relationships only. Every fact is a Note.
- Anything re-derivable from the original file stays out until a step needs it.
- Changes are additive. Each new component bumps SCHEMA_VERSION.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.4"
PROJECT_SCOPE = "project"  # scope ref for project-wide notes: tone, world rules


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


Author = Literal["agent", "user"]


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
    id: str = Field(pattern=r"^[0-9a-f]{64}$")  # sha256 of bytes; re-drops are no-ops
    origin_url: str | None = None       # where a fetched file came from (external references)
    license: str | None = None          # e.g. "CC BY-SA 4.0"; required for fetched images
    attribution: str | None = None      # author/credit line to reproduce with the image
    filename: str
    mime_type: str
    kind: SourceKind
    size_bytes: int = Field(ge=0)
    storage_path: str                   # gs:// path, never a signed URL
    status: SourceStatus = "uploaded"
    doc_type: DocType | None = None     # set by classify
    derived: Derived = Field(default_factory=Derived)
    digest_version: str | None = None   # prompt/pipeline version that digested it
    error: str | None = None

    @field_validator("storage_path")
    @classmethod
    def _no_signed_urls(cls, v: str) -> str:
        if not v.startswith("gs://") or "?" in v:
            raise ValueError("storage_path must be a plain gs:// path")
        return v


# --- Entities: scope targets for notes ----------------------------------------

EntityStatus = Literal["proposed", "confirmed", "rejected", "merged"]


class EntityBase(Doc):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)  # "the well", "Devgram well"
    status: EntityStatus = "proposed"
    author: Author
    merged_into: str | None = None

    @model_validator(mode="after")
    def _merge_target(self):
        if (self.status == "merged") != (self.merged_into is not None):
            raise ValueError("merged_into is required iff status == 'merged'")
        return self


class Location(EntityBase):
    type: Literal["location"] = "location"
    id: str = Field(default_factory=lambda: new_id("loc"))


class Scene(EntityBase):
    type: Literal["scene"] = "scene"
    id: str = Field(default_factory=lambda: new_id("scn"))
    number: str | None = None           # as written: "12", "12A"
    location_ids: list[str] = Field(default_factory=list)


Entity = Annotated[Union[Location, Scene], Field(discriminator="type")]


# --- Notes: the unit of memory ------------------------------------------------

NoteKind = Literal["description", "constraint", "reference_image", "tone", "vocabulary"]
NoteStatus = Literal["proposed", "confirmed", "rejected"]


class Provenance(Strict):
    """Where a note came from: a source in the dump, an external page, or both."""
    source_id: str | None = None
    url: str | None = None
    title: str | None = None                          # page or file title, for citations
    page: int | None = Field(default=None, ge=1)       # 1-based; None for images/text
    quote: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _anchored(self):
        if not self.source_id and not self.url:
            raise ValueError("provenance needs a source_id or a url")
        return self


class NoteOrigin(Strict):
    """Which step wrote this note, so a re-run can replace its own proposals."""
    digest_version: str
    producer: str = "ingest"            # "ingest" | "references" | ...
    source_id: str | None = None        # ingest: the file digested
    scope: str | None = None            # references: the entity the run was for


class Note(Doc):
    id: str = Field(default_factory=lambda: new_id("note"))
    kind: NoteKind
    body: str = Field(min_length=1, max_length=2000)   # markdown
    scope_refs: list[str] = Field(min_length=1)        # entity ids or PROJECT_SCOPE
    provenance: list[Provenance] = Field(default_factory=list)
    status: NoteStatus = "proposed"
    author: Author
    group: str | None = None            # optional cluster label, e.g. a visual direction
    origin: NoteOrigin | None = None

    @model_validator(mode="after")
    def _rules(self):
        if self.author == "agent" and not self.provenance:
            raise ValueError("agent notes require provenance")
        if self.kind == "reference_image" and len(self.provenance) != 1:
            raise ValueError("reference_image notes point at exactly one image or page")
        return self


# --- Retrieval contract: assembled at read time, never stored ----------------

class ReferenceImage(Strict):
    note_id: str
    uri: str                            # gs:// original image, or rendered PDF page
    caption: str
    status: NoteStatus
    source_id: str
    page: int | None = None
    group: str | None = None
    origin_url: str | None = None
    license: str | None = None
    attribution: str | None = None


class ContextPack(Strict):
    scope_id: str                       # entity id after following merges, or PROJECT_SCOPE
    entity: Entity | None               # None when scope_id == PROJECT_SCOPE
    include_proposed: bool
    merged_ids: list[str] = Field(default_factory=list)       # entities merged into this one
    notes: list[Note] = Field(default_factory=list)           # scoped to the entity or its merged ids
    related_notes: list[Note] = Field(default_factory=list)   # location: its scenes' notes; scene: its locations'
    related_by: dict[str, list[str]] = Field(default_factory=dict)  # related entity id -> its note ids, each note once
    project_notes: list[Note] = Field(default_factory=list)   # project-wide text notes
    scenes: list[Scene] = Field(default_factory=list)         # location packs: scenes set here
    locations: list[Location] = Field(default_factory=list)   # scene packs: where the scene is set
    reference_images: list[ReferenceImage] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)     # source_id -> filename, for citations
