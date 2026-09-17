"""Interfaces the ingest worker depends on, plus in-memory implementations for tests."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, TypeVar, Union

from pydantic import BaseModel

from .models import Location, Note, Project, Scene, Source, new_id, utcnow

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


class VerificationServiceError(RuntimeError):
    """The verification service (e.g. Wikipedia API) was unreachable; distinct from 'term not found'."""


class ReplacementTooLarge(RuntimeError):
    """The atomic note replacement exceeds the store's single-operation limit.

    The previous reference set is left intact.
    """
    def __init__(self, needed: int, limit: int):
        super().__init__(f"replacement needs {needed} operations but the store limit is {limit}")
        self.needed, self.limit = needed, limit


class NoteReviewConflict(RuntimeError):
    """put_note_if_current found the note at a different (revision, status) than expected.

    Raised instead of silently overwriting a decision another reviewer just made. Revision
    alone is not enough: a plain confirm/reject with no guidance edit never bumps
    Note.revision (see models.Note - revision tracks the reviewed *assertion*, not the
    review action), so two tabs racing a bare confirm vs. reject on the same note would
    both see the same revision and neither would conflict. Status is part of the check
    for exactly that reason - the write does not happen if either differs.
    """
    def __init__(self, note_id: str, expected_revision: int, actual_revision: int | None,
                expected_status: str, actual_status: str | None):
        super().__init__(f"note {note_id} is at (revision={actual_revision!r}, status={actual_status!r}), "
                         f"expected (revision={expected_revision}, status={expected_status!r})")
        self.note_id = note_id
        self.expected_revision, self.actual_revision = expected_revision, actual_revision
        self.expected_status, self.actual_status = expected_status, actual_status


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
    def list_projects(self) -> list[Project]: ...
    def get_source(self, project_id: str, source_id: str) -> Source | None: ...
    def put_source(self, project_id: str, source: Source) -> None: ...
    def put_source_if_absent(self, project_id: str, source: Source) -> tuple[Source, bool]:
        """Write source only if no source with this ID exists.

        Atomic per adapter. Returns (source, True) when written, (existing, False) when
        already present. Never overwrites purpose, status, or provenance from a competing
        writer that reached the store first.
        """
        ...
    def promote_source_to_ingest(self, project_id: str, source_id: str) -> Source:
        """Atomically promote a reference-only source to dual 'both' purpose.

        Reads the current record under the lock/transaction and:
        - Returns it unchanged if already ingest-eligible (is_ingest_eligible is True).
        - Otherwise sets source_purpose='both', resets status='uploaded', clears
          digest_version so ingest_source does not mistake the pipeline's 'digested'
          state for completed ingestion.
        Raises KeyError if source_id not found in project.
        """
        ...
    def put_note_if_absent(self, project_id: str, note: Note) -> tuple[Note, bool]:
        """Write note only if no note with this ID currently exists.

        Atomic per adapter. Returns (note, True) when written, (existing, False) when
        the ID was already present. Never overwrites an existing note - preserves status,
        revision, guidance, and rejection reason unconditionally.
        """
        ...
    def list_sources(self, project_id: str) -> list[Source]: ...
    def list_entities(self, project_id: str) -> list[EntityDoc]: ...
    def get_entity(self, project_id: str, entity_id: str) -> EntityDoc | None: ...
    def put_entities(self, project_id: str, entities: list[EntityDoc]) -> None: ...
    def list_notes(self, project_id: str, *, source_id: str | None = None) -> list[Note]: ...
    def notes_for_owners(self, project_id: str, owner_ids: list[str]) -> list[Note]: ...
    def get_sources(self, project_id: str, source_ids: list[str]) -> dict[str, Source]: ...
    def acquire_lock(self, project_id: str, holder: str, stale_after_s: int) -> str | None: ...
    def release_lock(self, project_id: str, token: str) -> None: ...
    def renew_lock(self, project_id: str, token: str) -> bool:
        """Refreshes the lock's staleness clock so a genuinely still-running holder isn't
        mistaken for an abandoned one and reclaimed mid-run. Returns True iff the given
        token still owned the lock (and was renewed); False if it doesn't - the caller has
        lost the lock (someone else's stale-timeout reclaim already happened) and must
        stop treating itself as the exclusive holder.
        """
        ...
    def put_notes(self, project_id: str, notes: list[Note]) -> None: ...
    def put_note_if_current(self, project_id: str, note: Note, expected_revision: int,
                            expected_status: str) -> None:
        """Write note only if the currently stored note is at (expected_revision, expected_status).

        Atomic per adapter. Raises NoteReviewConflict (writing nothing) if the stored note
        is missing or differs on either field - e.g. reviewed by someone else, or replaced
        by a pipeline re-run, since the caller last read it. Both fields are needed: a
        plain confirm/reject with no guidance edit does not change revision (see
        NoteReviewConflict's docstring).
        """
        ...
    def delete_notes(self, project_id: str, note_ids: list[str]) -> None: ...
    def replace_notes(self, project_id: str, to_delete: list[str], to_put: list[Note],
                      expected_status: dict[str, str] | None = None) -> None:
        """Atomically delete old notes and write new ones.

        If the total operation count exceeds the store's limit, raises
        ReplacementTooLarge without modifying anything.

        expected_status maps note id -> the status the caller last saw.  If a
        note's current status differs (e.g. it was confirmed while the run was
        in progress), that note is silently skipped from deletion.
        """
        ...


class MemoryImages:
    def __init__(self, terms: dict[str, TermHit], images: dict[str, list[ImageHit]], blobs: dict[str, bytes],
                 scoped: dict[tuple[str, str], list[ImageHit]] | None = None):
        self.terms, self.images, self.blobs = terms, images, blobs
        self.scoped = scoped or {}      # (term, region) -> hits
        self.searched: list[str] = []
        self.queries: list[tuple[str, str | None]] = []
        self.search_calls: list[dict] = []

    def verify_term(self, term):
        return self.terms.get(term.lower())

    def search_images(self, term, limit, region=None, region_title=None):
        self.searched.append(term)
        self.queries.append((term, region))
        self.search_calls.append({
            "term": term,
            "limit": limit,
            "region": region,
            "region_title": region_title,
        })
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
        self.lock_taken_at: dict[str, datetime] = {}
        # Guards put_source_if_absent, promote_source_to_ingest, and put_note_if_absent.
        # Existing unconditional writes (put_source, put_notes) remain unlocked because
        # they are not called from concurrent paths today; only the conditional operations
        # that must be atomic across threads need the lock.
        self._sources_lock = threading.Lock()
        self._notes_lock = threading.Lock()

    def get_project(self, project_id):
        return self.projects.get(project_id)

    def put_project(self, project):
        self.projects[project.id] = project.model_copy(deep=True)

    def list_projects(self):
        return [p.model_copy(deep=True) for p in self.projects.values()]

    def get_source(self, project_id, source_id):
        s = self.sources.get((project_id, source_id))
        return s.model_copy(deep=True) if s else None

    def put_source(self, project_id, source):
        with self._sources_lock:
            self.sources[(project_id, source.id)] = source.model_copy(deep=True)

    def put_source_if_absent(self, project_id, source):
        with self._sources_lock:
            key = (project_id, source.id)
            existing = self.sources.get(key)
            if existing is not None:
                return existing.model_copy(deep=True), False
            stored = source.model_copy(deep=True)
            self.sources[key] = stored
            return stored, True

    def promote_source_to_ingest(self, project_id, source_id):
        with self._sources_lock:
            key = (project_id, source_id)
            existing = self.sources.get(key)
            if existing is None:
                raise KeyError(source_id)
            # Re-check eligibility inside the lock - a concurrent promotion or an
            # ingestion run may have already changed the source since the caller read it.
            if existing.is_ingest_eligible:
                return existing.model_copy(deep=True)
            updated = existing.touch(
                source_purpose="both",
                status="uploaded",
                digest_version=None,
            )
            self.sources[key] = updated.model_copy(deep=True)
            return updated.model_copy(deep=True)

    def put_note_if_absent(self, project_id, note):
        with self._notes_lock:
            key = (project_id, note.id)
            existing = self.notes.get(key)
            if existing is not None:
                return existing.model_copy(deep=True), False
            stored = note.model_copy(deep=True)
            self.notes[key] = stored
            return stored, True

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
            taken_at = self.lock_taken_at.get(project_id)
            if taken_at is not None and (utcnow() - taken_at).total_seconds() < stale_after_s:
                return None
        # unique per acquisition, not derived from holder: two different acquisitions by
        # the same holder string (e.g. the CLI's default holder="cli") must never produce
        # the same token, or a caller holding a genuinely stale token could match a
        # different, currently-live lock and release/renew someone else's - mirrors
        # FirestoreStore.acquire_lock's new_id("lock") uniqueness.
        token = new_id("lock")
        self.locks[project_id] = token
        self.lock_taken_at[project_id] = utcnow()
        return token

    def release_lock(self, project_id, token):
        if self.locks.get(project_id) == token:
            del self.locks[project_id]
            self.lock_taken_at.pop(project_id, None)

    def renew_lock(self, project_id, token):
        if self.locks.get(project_id) != token:
            return False
        self.lock_taken_at[project_id] = utcnow()
        return True

    def put_notes(self, project_id, notes):
        with self._notes_lock:
            for n in notes:
                self.notes[(project_id, n.id)] = n.model_copy(deep=True)

    def put_note_if_current(self, project_id, note, expected_revision, expected_status):
        with self._notes_lock:
            current = self.notes.get((project_id, note.id))
            current_rev = current.revision if current is not None else None
            current_status = current.status if current is not None else None
            if current_rev != expected_revision or current_status != expected_status:
                raise NoteReviewConflict(note.id, expected_revision, current_rev, expected_status, current_status)
            self.notes[(project_id, note.id)] = note.model_copy(deep=True)

    def delete_notes(self, project_id, note_ids):
        with self._notes_lock:
            for i in note_ids:
                self.notes.pop((project_id, i), None)

    def replace_notes(self, project_id, to_delete, to_put, expected_status=None):
        total = len(to_delete) + len(to_put)
        if total > 500:
            raise ReplacementTooLarge(total, 500)
        expected_status = expected_status or {}
        with self._notes_lock:
            for nid in to_delete:
                existing = self.notes.get((project_id, nid))
                if existing is not None:
                    exp = expected_status.get(nid)
                    if exp is not None and existing.status != exp:
                        continue   # reviewed during run; skip
                    self.notes.pop((project_id, nid), None)
            for n in to_put:
                self.notes[(project_id, n.id)] = n.model_copy(deep=True)
