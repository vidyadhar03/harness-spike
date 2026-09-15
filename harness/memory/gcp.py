"""Firestore, Cloud Storage, and Gemini (Vertex) adapters."""
from __future__ import annotations

import logging
import time

from pydantic import TypeAdapter

from .config import Settings
from .models import Entity, Note, Project, Source
from .ports import Blob, EntityDoc, OutputTruncated, ReplacementTooLarge, Part, T, Text, Uri
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

    def get_source(self, project_id, source_id):
        snap = self._col(project_id, "sources").document(source_id).get()
        return Source.model_validate(snap.to_dict()) if snap.exists else None

    def put_source(self, project_id, source):
        self._col(project_id, "sources").document(source.id).set(source.model_dump())

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

    def put_notes(self, project_id, notes):
        self._write(project_id, "notes", [(n.id, n.model_dump()) for n in notes])

    def delete_notes(self, project_id, note_ids):
        self._write(project_id, "notes", [(i, None) for i in note_ids])

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