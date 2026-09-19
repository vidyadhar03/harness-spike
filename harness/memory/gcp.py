"""Firestore, Cloud Storage, and Gemini (Vertex) adapters."""
from __future__ import annotations

import logging
import time

from pydantic import TypeAdapter

from .config import Settings
from .models import ConceptApproval, ConceptVersion, Entity, Note, Project, Source
from .ports import (
    ApprovalConflict, Blob, EntityDoc, NoteReviewConflict, OutputTruncated, ReplacementTooLarge, Part, T, Text, Uri,
)
from .schemas import gemini_schema, parse_json

log = logging.getLogger(__name__)


def firestore_transactional(db):
    from google.cloud import firestore

    def wrap(fn):
        return firestore.transactional(fn)

    return wrap
_ENTITY = TypeAdapter(Entity)
_BATCH = 400  # Firestore caps a batch at 500 writes
_ANY = 30     # Firestore caps array-contains-any at 30 values


class FirestoreStore:
    def __init__(self, gcp_project: str, database: str = "(default)"):
        from google.cloud import firestore  # honours FIRESTORE_EMULATOR_HOST

        self._db = firestore.Client(project=gcp_project, database=database)

    def _project(self, pid: str):
        return self._db.collection("projects").document(pid)

    def _col(self, pid: str, name: str):
        return self._project(pid).collection(name)

    def _write(self, pid: str, col: str, docs: list[tuple[str, dict | None]]) -> None:
        for i in range(0, len(docs), _BATCH):
            batch = self._db.batch()
            for doc_id, data in docs[i : i + _BATCH]:
                ref = self._col(pid, col).document(doc_id)
                batch.delete(ref) if data is None else batch.set(ref, data)
            batch.commit()

    def get_project(self, project_id):
        snap = self._project(project_id).get()
        return Project.model_validate(snap.to_dict()) if snap.exists else None

    def put_project(self, project):
        self._project(project.id).set(project.model_dump())

    def list_projects(self):
        return [Project.model_validate(d.to_dict()) for d in self._db.collection("projects").stream()]

    def get_source(self, project_id, source_id):
        snap = self._col(project_id, "sources").document(source_id).get()
        return Source.model_validate(snap.to_dict()) if snap.exists else None

    def put_source(self, project_id, source):
        self._col(project_id, "sources").document(source.id).set(source.model_dump())

    def put_source_if_absent(self, project_id, source):
        """Atomic create-if-absent for sources. Uses a Firestore transaction so two
        concurrent writers (e.g. register_file and _store_image for the same sha256)
        cannot both succeed and clobber each other's purpose/status/provenance.
        """
        from .models import Source as _Source
        ref = self._col(project_id, "sources").document(source.id)
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _create(tx):
            snap = ref.get(transaction=tx)
            if snap.exists:
                return _Source.model_validate(snap.to_dict()), False
            tx.set(ref, source.model_dump())
            return source, True

        return _create(transaction)

    def promote_source_to_ingest(self, project_id, source_id):
        """Atomically promote a non-ingest-eligible source ('reference' or 'concept'
        purpose) to dual 'both' purpose - generic over which one it started as.

        Reads the source inside the transaction and returns it unchanged if already
        ingest-eligible - a delayed second promotion after ingestion started must not
        reset the source's processing state. Raises KeyError when not found.
        """
        from .models import Source as _Source, utcnow
        ref = self._col(project_id, "sources").document(source_id)
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _promote(tx):
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise KeyError(source_id)
            src = _Source.model_validate(snap.to_dict())
            if src.is_ingest_eligible:
                # Already ingest-eligible (ingest or both): preserve current state.
                return src
            # Transition from a non-ingest purpose ('reference' or 'concept'): mark as
            # both and reset ingestion state so ingest_source does not skip it as
            # already-digested.
            updated = src.touch(
                source_purpose="both",
                status="uploaded",
                digest_version=None,
            )
            tx.set(ref, updated.model_dump())
            return updated

        return _promote(transaction)

    def put_note_if_absent(self, project_id, note):
        """Atomic create-if-absent for notes. Follows the same transactional pattern as
        put_note_if_current so concurrent uploads to the same location do not overwrite
        a reviewed note written between their reads.
        """
        from .models import Note as _Note
        ref = self._col(project_id, "notes").document(note.id)
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _create(tx):
            snap = ref.get(transaction=tx)
            if snap.exists:
                return _Note.model_validate(snap.to_dict()), False
            tx.set(ref, note.model_dump())
            return note, True

        return _create(transaction)

    def list_sources(self, project_id):
        return [Source.model_validate(d.to_dict()) for d in self._col(project_id, "sources").stream()]

    def list_entities(self, project_id) -> list[EntityDoc]:
        return [_ENTITY.validate_python(d.to_dict()) for d in self._col(project_id, "entities").stream()]

    def get_entity(self, project_id, entity_id):
        snap = self._col(project_id, "entities").document(entity_id).get()
        return _ENTITY.validate_python(snap.to_dict()) if snap.exists else None

    def put_entities(self, project_id, entities):
        self._write(project_id, "entities", [(e.id, e.model_dump()) for e in entities])

    def list_notes(self, project_id, *, source_id=None):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self._col(project_id, "notes")
        if source_id is not None:
            q = q.where(filter=FieldFilter("origin.source_id", "==", source_id))
        return [Note.model_validate(d.to_dict()) for d in q.stream()]

    def notes_for_owners(self, project_id, owner_ids):
        """Owner is a single field, so this is plain equality - batched only because
        Firestore caps an `in` query at 30 values."""
        from google.cloud.firestore_v1.base_query import FieldFilter

        col = self._col(project_id, "notes")
        found: dict = {}
        values = list(dict.fromkeys(owner_ids))
        for i in range(0, len(values), _ANY):
            q = col.where(filter=FieldFilter("owner_id", "in", values[i : i + _ANY]))
            for d in q.stream():
                found.setdefault(d.id, Note.model_validate(d.to_dict()))
        return list(found.values())

    def get_sources(self, project_id, source_ids):
        refs = [self._col(project_id, "sources").document(i) for i in dict.fromkeys(source_ids)]
        if not refs:
            return {}
        return {snap.id: Source.model_validate(snap.to_dict()) for snap in self._db.get_all(refs) if snap.exists}

    def acquire_lock(self, project_id, holder, stale_after_s):
        """One ingest at a time per project. Serial operation is what makes it safe to
        replace proposals without transactional publication; this makes it enforced
        rather than a convention someone forgets in a second terminal."""
        from datetime import timedelta

        from google.api_core import exceptions

        from .models import new_id, utcnow

        ref = self._project(project_id).collection("locks").document("ingest")
        token = new_id("lock")
        now = utcnow()
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _claim(tx):
            snap = ref.get(transaction=tx)
            if snap.exists:
                held = snap.to_dict()
                taken_at = held.get("taken_at")
                if taken_at is not None and now - taken_at < timedelta(seconds=stale_after_s):
                    return None
            tx.set(ref, {"token": token, "holder": holder, "taken_at": now})
            return token

        try:
            return _claim(transaction)
        except exceptions.Aborted:
            return None

    def release_lock(self, project_id, token):
        ref = self._project(project_id).collection("locks").document("ingest")
        snap = ref.get()
        if snap.exists and snap.to_dict().get("token") == token:
            ref.delete()

    def renew_lock(self, project_id, token):
        """Refreshes taken_at so a genuinely still-running holder isn't mistaken for an
        abandoned one and reclaimed mid-run by acquire_lock's staleness check - see
        ingest.py's per-source renewal calls in its drop/ingest loops."""
        from .models import utcnow

        ref = self._project(project_id).collection("locks").document("ingest")
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _renew(tx):
            snap = ref.get(transaction=tx)
            if not snap.exists or snap.to_dict().get("token") != token:
                return False
            tx.update(ref, {"taken_at": utcnow()})
            return True

        return _renew(transaction)

    def put_notes(self, project_id, notes):
        self._write(project_id, "notes", [(n.id, n.model_dump()) for n in notes])

    def put_note_if_current(self, project_id, note, expected_revision, expected_status):
        """Transactional compare-and-swap on (revision, status).

        Guards the two-browser-tabs case: both read the same (revision, status), both
        try to write; the loser's transaction sees the note has moved on and aborts with
        NoteReviewConflict rather than silently clobbering the winner's decision. Status
        is checked alongside revision because a plain confirm/reject with no guidance
        edit never changes revision - see NoteReviewConflict's docstring.

        NOT exercised against real or emulated Firestore by this project's automated
        tests (tests/test_api.py and tests/test_curate.py drive this behavior only
        through MemoryStore.put_note_if_current, an in-process dict equivalent). The
        transaction shape (read-check-write, raising inside the transactional function
        to abort without committing) matches acquire_lock above, the one Firestore
        transaction in this file that predates this method, but that is a design
        precedent, not a verification of this method itself. Run against
        FIRESTORE_EMULATOR_HOST at least once before depending on this in anything
        beyond local/manual use.
        """
        ref = self._col(project_id, "notes").document(note.id)
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _swap(tx):
            snap = ref.get(transaction=tx)
            data = snap.to_dict() if snap.exists else {}
            current_rev = data.get("revision") if snap.exists else None
            current_status = data.get("status") if snap.exists else None
            if current_rev != expected_revision or current_status != expected_status:
                raise NoteReviewConflict(note.id, expected_revision, current_rev, expected_status, current_status)
            tx.set(ref, note.model_dump())

        _swap(transaction)

    def correct_note(self, project_id, original_update, new_notes, expected_revision, expected_status):
        """Same transactional CAS as put_note_if_current, extended to write new_notes'
        documents inside the same transaction - so a correction/split can never be
        observed half-applied (original rejected but no successor, or a successor
        present while the original still looks live). Not exercised against real or
        emulated Firestore by this project's automated tests - see put_note_if_current's
        own docstring for that caveat, which applies identically here.
        """
        ref = self._col(project_id, "notes").document(original_update.id)
        notes_col = self._col(project_id, "notes")
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _correct(tx):
            snap = ref.get(transaction=tx)
            data = snap.to_dict() if snap.exists else {}
            current_rev = data.get("revision") if snap.exists else None
            current_status = data.get("status") if snap.exists else None
            if current_rev != expected_revision or current_status != expected_status:
                raise NoteReviewConflict(original_update.id, expected_revision, current_rev,
                                         expected_status, current_status)
            tx.set(ref, original_update.model_dump())
            for n in new_notes:
                tx.set(notes_col.document(n.id), n.model_dump())

        _correct(transaction)
        return original_update, new_notes

    def delete_notes(self, project_id, note_ids):
        self._write(project_id, "notes", [(i, None) for i in note_ids])

    def put_concept_version_if_absent(self, project_id, version):
        """Atomic create-if-absent, same shape as put_source_if_absent/put_note_if_absent."""
        ref = self._col(project_id, "concept_versions").document(version.id)
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _create(tx):
            snap = ref.get(transaction=tx)
            if snap.exists:
                return ConceptVersion.model_validate(snap.to_dict()), False
            tx.set(ref, version.model_dump())
            return version, True

        return _create(transaction)

    def list_concept_versions(self, project_id, location_id):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self._col(project_id, "concept_versions").where(filter=FieldFilter("location_id", "==", location_id))
        versions = [ConceptVersion.model_validate(d.to_dict()) for d in q.stream()]
        return sorted(versions, key=lambda v: (v.created_at, v.id))

    def get_concept_version(self, project_id, version_id):
        snap = self._col(project_id, "concept_versions").document(version_id).get()
        return ConceptVersion.model_validate(snap.to_dict()) if snap.exists else None

    def _approval_pointer(self, project_id, location_id):
        return self._project(project_id).collection("concept_approval_current").document(location_id)

    def get_current_approval(self, project_id, location_id):
        ptr_snap = self._approval_pointer(project_id, location_id).get()
        if not ptr_snap.exists:
            return None
        aid = ptr_snap.to_dict().get("approval_id")
        snap = self._col(project_id, "concept_approvals").document(aid).get()
        return ConceptApproval.model_validate(snap.to_dict()) if snap.exists else None

    def put_approval_if_current(self, project_id, location_id, approval, expected_revision):
        """Transactional CAS on the location's "current approval" pointer.

        Same read-check-write-inside-transaction shape as put_note_if_current (see its
        docstring for the general pattern this follows) - not exercised against real or
        emulated Firestore by this project's automated tests, which drive this behavior
        only through MemoryStore.put_approval_if_current.
        """
        ptr_ref = self._approval_pointer(project_id, location_id)
        approvals_col = self._col(project_id, "concept_approvals")
        transaction = self._db.transaction()

        @firestore_transactional(self._db)
        def _lock(tx):
            ptr_snap = ptr_ref.get(transaction=tx)
            current = None
            if ptr_snap.exists:
                aid = ptr_snap.to_dict().get("approval_id")
                cur_snap = approvals_col.document(aid).get(transaction=tx)
                if cur_snap.exists:
                    current = ConceptApproval.model_validate(cur_snap.to_dict())
            current_rev = current.revision if current is not None else 0
            if current is not None and current.context_token == approval.context_token:
                return current, False
            if current_rev != expected_revision:
                raise ApprovalConflict(location_id, expected_revision, current_rev)
            stored = approval.model_copy(update={"revision": current_rev + 1})
            tx.set(approvals_col.document(stored.id), stored.model_dump())
            tx.set(ptr_ref, {"approval_id": stored.id, "revision": stored.revision})
            return stored, True

        return _lock(transaction)

    def replace_notes(self, project_id, to_delete, to_put, expected_status=None):
        """Atomic note replacement with precondition checks.

        Re-reads each deletion candidate immediately before the batch write.
        If a note's status no longer matches expected_status, it is skipped
        (it was reviewed during the run and must not be destroyed).
        Delete operations carry an update_time precondition so Firestore
        itself rejects the batch if any document changed between the read
        and the commit (closes the TOCTOU gap).
        Refuses to proceed if the total operation count exceeds the Firestore
        single-batch limit.
        """
        expected_status = expected_status or {}
        col = self._col(project_id, "notes")

        # re-read deletion candidates to check for concurrent reviews
        actual_deletes = []  # list of (nid, snapshot) tuples
        for nid in to_delete:
            snap = col.document(nid).get()
            if not snap.exists:
                continue
            exp = expected_status.get(nid)
            if exp is not None and snap.to_dict().get("status") != exp:
                continue  # reviewed during run; protect it
            actual_deletes.append((nid, snap))

        total = len(actual_deletes) + len(to_put)
        if total > _BATCH:
            raise ReplacementTooLarge(total, _BATCH)
        if total == 0:
            return

        from google.cloud.firestore_v1 import _helpers  # noqa: F401

        batch = self._db.batch()
        for nid, snap in actual_deletes:
            ref = col.document(nid)
            # Precondition: only delete if the document has not been modified
            # since we read it.  If a concurrent review changed the note
            # between our read and this commit, Firestore will fail the batch.
            batch.delete(ref, option=self._db.write_option(last_update_time=snap.update_time))
        for n in to_put:
            batch.set(col.document(n.id), n.model_dump())
        batch.commit()


class GCSBlobs:
    def __init__(self, gcp_project: str):
        from google.cloud import storage  # honours STORAGE_EMULATOR_HOST

        self._client = storage.Client(project=gcp_project)

    def _blob(self, uri: str):
        if not uri.startswith("gs://"):
            raise ValueError(f"not a gs:// uri: {uri}")
        bucket, _, key = uri[5:].partition("/")
        return self._client.bucket(bucket).blob(key)

    def put(self, uri, data, mime_type):
        self._blob(uri).upload_from_string(data, content_type=mime_type)

    def get(self, uri):
        return self._blob(uri).download_as_bytes()


_RETRYABLE = {408, 429, 500, 502, 503, 504}


class GeminiLLM:
    def __init__(self, settings: Settings):
        from google import genai

        self._s = settings
        self._client = genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_location)
        self.model_id = f"{settings.model}+{settings.model_fast}"
        self.accumulated_usage: dict[str, int] = {}   # accumulates across all calls in a run
        self.http_retries: int = 0

    def _accumulate(self, resp) -> None:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            self.accumulated_usage.setdefault("unknown_calls", 0)
            self.accumulated_usage["unknown_calls"] += 1
            return
        for attr in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            val = getattr(meta, attr, None)
            if val is not None:
                self.accumulated_usage[attr] = self.accumulated_usage.get(attr, 0) + val

    def reset_usage(self) -> None:
        self.accumulated_usage.clear()
        self.http_retries = 0

    def generate(self, *, system: str, parts: list[Part], schema: type[T], fast: bool = False,
                 thinking_level: str | None = None) -> T:
        """thinking_level caps reasoning for calls that do not need it. Thinking tokens
        count against max_output_tokens, so a long mechanical task (captioning two dozen
        images) can exhaust the budget on reasoning and truncate before writing an answer."""
        from google.genai import errors, types

        model = self._s.model_fast if fast else self._s.model
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=gemini_schema(schema),
            temperature=self._s.temperature,
            max_output_tokens=self._s.max_output_tokens,
            thinking_config=(types.ThinkingConfig(thinking_level=thinking_level)
                             if thinking_level else None),
        )
        contents = [_to_part(p, types) for p in parts]
        for attempt in range(self._s.max_retries + 1):
            try:
                resp = self._client.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as exc:
                if exc.code in _RETRYABLE and attempt < self._s.max_retries:
                    self.http_retries += 1
                    delay = min(60, 2 ** (attempt + 1))
                    log.warning("%s returned %s; retrying in %ss", model, exc.code, delay)
                    time.sleep(delay)
                    continue
                raise
            self._accumulate(resp)
            try:
                return parse_json(_checked_text(resp, model, self._s.max_output_tokens), schema)
            except OutputTruncated:
                self._accumulate(resp)  # count truncated response tokens too
                raise
        raise RuntimeError("unreachable")


def _checked_text(resp, model: str, max_tokens: int) -> str:
    """Only a clean STOP is usable. Anything else is partial or blocked, and json_repair
    would happily turn a cut-off response into valid-looking, incomplete output."""
    if not resp.candidates:
        feedback = getattr(resp, "prompt_feedback", None)
        raise RuntimeError(f"{model} returned no candidates ({feedback})")
    reason = resp.candidates[0].finish_reason
    name = getattr(reason, "name", str(reason))
    if name == "MAX_TOKENS":
        raise OutputTruncated(f"{model} hit max_output_tokens={max_tokens}")
    if name != "STOP":
        raise RuntimeError(f"{model} finished with {name}")
    if not resp.text:
        raise RuntimeError(f"{model} returned an empty response")
    return resp.text


def _to_part(p: Part, types):
    if isinstance(p, Text):
        return types.Part.from_text(text=p.text)
    if isinstance(p, Blob):
        return types.Part.from_bytes(data=p.data, mime_type=p.mime_type)
    if isinstance(p, Uri):
        return types.Part.from_uri(file_uri=p.uri, mime_type=p.mime_type)
    raise TypeError(p)