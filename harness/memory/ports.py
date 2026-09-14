"""Interfaces the ingest worker depends on, plus in-memory implementations for tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, Union

from pydantic import BaseModel

from .models import Location, Note, Project, Scene, Source

T = TypeVar("T", bound=BaseModel)
EntityDoc = Union[Location, Scene]


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Blob:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class Uri:
    uri: str        # gs:// path, readable by the model service
    mime_type: str


Part = Union[Text, Blob, Uri]


class OutputTruncated(RuntimeError):
    """The model stopped at its output token limit. The response is partial and must not be used."""


class LLM(Protocol):
    model_id: str   # part of the digest version

    def generate(self, *, system: str, parts: list[Part], schema: type[T], fast: bool = False,thinking_level: str | None = None) -> T: ...


@dataclass(frozen=True)
class TermHit:
    """A page confirming a search term exists and means what the model claimed."""
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class ImageHit:
    title: str
    page_url: str                 # the file's description page, for attribution
    image_url: str                # full-size file
    preview_url: str              # scaled version to store and to show the model
    description: str = ""
    license: str | None = None
    attribution: str | None = None
    width: int | None = None
    height: int | None = None
    mime_type: str = "image/jpeg"


class Images(Protocol):
    """An image archive: term lookup plus image search. Wikimedia by default."""

    def verify_term(self, term: str) -> TermHit | None: ...
    def search_images(self, term: str, limit: int, region: str | None = None,
                      region_title: str | None = None) -> list[ImageHit]:
        """Licensed images for a term. With a region, only images from that region;
        region_title is the page the region term verified against. A term scoped to
        itself (term == region) means the region's own images."""
        ...
    def fetch(self, url: str) -> bytes: ...


class Blobs(Protocol):
    def put(self, uri: str, data: bytes, mime_type: str) -> None: ...
    def get(self, uri: str) -> bytes: ...


class Store(Protocol):
    def get_project(self, project_id: str) -> Project | None: ...
    def put_project(self, project: Project) -> None: ...
    def get_source(self, project_id: str, source_id: str) -> Source | None: ...
    def put_source(self, project_id: str, source: Source) -> None: ...
    def list_sources(self, project_id: str) -> list[Source]: ...
    def list_entities(self, project_id: str) -> list[EntityDoc]: ...
    def get_entity(self, project_id: str, entity_id: str) -> EntityDoc | None: ...
    def put_entities(self, project_id: str, entities: list[EntityDoc]) -> None: ...
    def list_notes(self, project_id: str, *, source_id: str | None = None) -> list[Note]: ...
    def notes_for_owners(self, project_id: str, owner_ids: list[str]) -> list[Note]: ...
    def get_sources(self, project_id: str, source_ids: list[str]) -> dict[str, Source]: ...
    def acquire_lock(self, project_id: str, holder: str, stale_after_s: int) -> str | None: ...
    def release_lock(self, project_id: str, token: str) -> None: ...
    def put_notes(self, project_id: str, notes: list[Note]) -> None: ...
    def delete_notes(self, project_id: str, note_ids: list[str]) -> None: ...


class MemoryImages:
    def __init__(self, terms: dict[str, TermHit], images: dict[str, list[ImageHit]], blobs: dict[str, bytes],
                 scoped: dict[tuple[str, str], list[ImageHit]] | None = None):
        self.terms, self.images, self.blobs = terms, images, blobs
        self.scoped = scoped or {}      # (term, region) -> hits
        self.searched: list[str] = []
        self.queries: list[tuple[str, str | None]] = []

    def verify_term(self, term):
        return self.terms.get(term.lower())

    def search_images(self, term, limit, region=None, region_title=None):
        self.searched.append(term)
        self.queries.append((term, region))
        if region is None:
            return self.images.get(term.lower(), [])[:limit]
        return self.scoped.get((term.lower(), region.lower()), [])[:limit]

    def fetch(self, url):
        if url not in self.blobs:
            raise KeyError(url)
        return self.blobs[url]


class MemoryBlobs:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    def put(self, uri: str, data: bytes, mime_type: str) -> None:
        self.objects[uri] = (data, mime_type)

    def get(self, uri: str) -> bytes:
        return self.objects[uri][0]


class MemoryStore:
    def __init__(self) -> None:
        self.projects: dict[str, Project] = {}
        self.sources: dict[tuple[str, str], Source] = {}
        self.entities: dict[tuple[str, str], EntityDoc] = {}
        self.notes: dict[tuple[str, str], Note] = {}
        self.locks: dict[str, str] = {}

    def get_project(self, project_id):
        return self.projects.get(project_id)

    def put_project(self, project):
        self.projects[project.id] = project.model_copy(deep=True)

    def get_source(self, project_id, source_id):
        s = self.sources.get((project_id, source_id))
        return s.model_copy(deep=True) if s else None

    def put_source(self, project_id, source):
        self.sources[(project_id, source.id)] = source.model_copy(deep=True)

    def list_sources(self, project_id):
        return [s.model_copy(deep=True) for (p, _), s in self.sources.items() if p == project_id]

    def list_entities(self, project_id):
        return [e.model_copy(deep=True) for (p, _), e in self.entities.items() if p == project_id]

    def get_entity(self, project_id, entity_id):
        e = self.entities.get((project_id, entity_id))
        return e.model_copy(deep=True) if e else None

    def put_entities(self, project_id, entities):
        for e in entities:
            self.entities[(project_id, e.id)] = e.model_copy(deep=True)

    def list_notes(self, project_id, *, source_id=None):
        return [
            n.model_copy(deep=True) for (p, _), n in self.notes.items()
            if p == project_id and (source_id is None or (n.origin and n.origin.source_id == source_id))
        ]

    def notes_for_owners(self, project_id, owner_ids):
        owners = set(owner_ids)
        return [n for n in self.list_notes(project_id) if n.owner_id in owners]

    def get_sources(self, project_id, source_ids):
        return {i: s for i in dict.fromkeys(source_ids) if (s := self.get_source(project_id, i))}

    def acquire_lock(self, project_id, holder, stale_after_s):
        current = self.locks.get(project_id)
        if current is not None:
            return None
        token = f"lock_{holder}"
        self.locks[project_id] = token
        return token

    def release_lock(self, project_id, token):
        if self.locks.get(project_id) == token:
            del self.locks[project_id]

    def put_notes(self, project_id, notes):
        for n in notes:
            self.notes[(project_id, n.id)] = n.model_copy(deep=True)

    def delete_notes(self, project_id, note_ids):
        for i in note_ids:
            self.notes.pop((project_id, i), None)
