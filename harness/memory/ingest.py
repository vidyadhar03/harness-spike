"""Memory dump ingest: register a dropped file, then digest it into entities and notes.

`ingest_source(ctx, project_id, source_id)` is the unit of work. It is idempotent
(keyed on sha256 + digest version) so it can later become a Cloud Tasks handler as-is.
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .config import Settings
from .chunks import MARKER, MARKER_RULE, NO_PAGES, PdfUnit, SceneRef, TextUnit, Unit, find_heading, pack
from .files import (
    PDF, SUPPORTED_MIME, extension_for, is_visual, kind_for, page_marked, page_ranges,
    pdf_page_texts, pdf_subset, render_pages_png, sniff_mime, text_chunks,
)
from .models import (
    PROJECT_SCOPE, Applicability, Derived, Note, NoteOrigin, Provenance, Source, new_id, utcnow,
)
from .ports import LLM, Blob, Blobs, OutputTruncated, Part, Store, T, Text, Uri
from .resolver import EntityResolver
from .schemas import ClassifyOut, ImageOut, NotesOut, OutReference, OutScene, RosterOut

log = logging.getLogger(__name__)

PIPELINE_VERSION = "ingest-v2"
CODE_FILES = ("ingest.py", "chunks.py", "files.py", "resolver.py", "schemas.py", "models.py")
PROMPTS = Path(__file__).parent / "prompts"


LOCK_STALE_AFTER_S = 3600


@dataclass
class Ctx:
    store: Store
    blobs: Blobs
    llm: LLM
    settings: Settings


@dataclass
class IngestReport:
    source_id: str
    filename: str
    status: str
    doc_type: str | None = None
    entities_created: int = 0
    notes_written: int = 0
    notes_replaced: int = 0
    notes_reused: int = 0
    extraction_id: str | None = None
    warn_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def load_prompt(name: str, shared: bool = True) -> str:
    body = (PROMPTS / f"{name}.md").read_text()
    return f"{(PROMPTS / '_shared.md').read_text()}\n\n{body}" if shared else body


def digest_version(llm: LLM) -> str:
    """Changes whenever a prompt, the model, or the extraction code changes.

    Chunking and parsing decide what the model ever sees, so they belong in the hash:
    a chunk-size change that alters every note must mark sources stale like a prompt edit.
    """
    h = hashlib.sha256()
    for p in sorted(PROMPTS.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    here = Path(__file__).parent
    for name in CODE_FILES:
        f = here / name
        if f.exists():
            h.update(name.encode())
            h.update(f.read_bytes())
    h.update(llm.model_id.encode())
    return f"{PIPELINE_VERSION}-{h.hexdigest()[:8]}"


def source_uri(settings: Settings, project_id: str, source_id: str, name: str) -> str:
    return f"gs://{settings.bucket}/projects/{project_id}/sources/{source_id}/{name}"


# --- register -----------------------------------------------------------------

def register_file(ctx: Ctx, project_id: str, data: bytes, filename: str, *,
                  document_id: str | None = None, revision_label: str | None = None,
                  supersedes: str | None = None) -> tuple[Source, bool]:
    """Store the bytes and create the Source. Never rejects a file. Returns (source, created).

    `document_id` identifies the logical document across drafts. Superseding an earlier
    source marks it, so retrieval can warn that some notes come from an old draft; it
    does NOT transfer any confirmation, which is a human reconciliation.
    """
    sid = hashlib.sha256(data).hexdigest()
    existing = ctx.store.get_source(project_id, sid)
    if existing is not None:
        return existing, False
    mime = sniff_mime(data, filename)
    supported = mime in SUPPORTED_MIME
    uri = source_uri(ctx.settings, project_id, sid, "original" + extension_for(filename, mime))
    ctx.blobs.put(uri, data, mime)  # bytes first, so a Source never points at a missing object
    prior = ctx.store.get_source(project_id, supersedes) if supersedes else None
    if supersedes and prior is None:
        raise KeyError(f"cannot supersede unknown source {supersedes}")
    src = Source(
        id=sid, filename=filename, mime_type=mime, kind=kind_for(mime), size_bytes=len(data),
        storage_path=uri, status="uploaded" if supported else "unsupported",
        error=None if supported else f"no v0 handler for {mime}",
        document_id=document_id or (prior.document_id if prior else None),
        revision_label=revision_label, supersedes_source_id=supersedes,
    )
    ctx.store.put_source(project_id, src)
    if prior is not None:
        ctx.store.put_source(project_id, prior.touch(superseded_by_source_id=sid))
    return src, True


# --- worker ---------------------------------------------------------------------

@contextmanager
def ingest_lock(ctx: Ctx, project_id: str, holder: str = "cli"):
    """Ingest runs one at a time per project. Replacing proposals is only safe while no
    other run or review is writing; this makes that a check rather than a habit."""
    token = ctx.store.acquire_lock(project_id, holder, LOCK_STALE_AFTER_S)
    if token is None:
        raise RuntimeError(f"another ingest is running for {project_id}; wait for it to finish")
    try:
        yield token
    finally:
        ctx.store.release_lock(project_id, token)


def ingest_source(ctx: Ctx, project_id: str, source_id: str, *, force: bool = False) -> IngestReport:
    src = ctx.store.get_source(project_id, source_id)
    if src is None:
        raise KeyError(f"source {source_id} not found in project {project_id}")
    report = IngestReport(source_id=src.id, filename=src.filename, status=src.status, doc_type=src.doc_type)
    version = digest_version(ctx.llm)

    if src.status == "unsupported":
        return report
    if not force and src.status == "digested" and src.digest_version == version:
        report.status = "skipped"
        return report

    extraction_id = new_id("ext")
    report.extraction_id = extraction_id
    src = _update(ctx, project_id, src, status="digesting", error=None)
    try:
        data = ctx.blobs.get(src.storage_path)
        job = _Job(ctx, project_id, src, version, EntityResolver(ctx.store.list_entities(project_id)),
                   report.warnings, extraction_id)
        doc_type, derived, notes = job.run(data)
        report.warnings.extend(job.resolver.apply_parents())
        if job.unowned:
            report.warn_counts["unowned_notes"] = job.unowned
        # archive before anything is replaced: a later merge or rescope must never be able
        # to erase what this source originally said
        _archive(job, doc_type, notes)
        _commit(job, notes, report)
        _update(ctx, project_id, src, status="digested", doc_type=doc_type, derived=derived,
                digest_version=version, extraction_id=extraction_id)
        report.status, report.doc_type = "digested", doc_type
    except Exception as exc:  # the source records the failure; the batch carries on
        log.exception("ingest failed: %s", src.filename)
        report.status, report.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
        _update(ctx, project_id, src, status="failed", error=report.error)
    return report


def _update(ctx: Ctx, project_id: str, src: Source, **changes) -> Source:
    data = src.model_dump()
    data.update(changes, updated_at=utcnow())
    if isinstance(data.get("derived"), BaseModel):
        data["derived"] = data["derived"].model_dump()
    new = Source.model_validate(data)
    ctx.store.put_source(project_id, new)
    return new


def _archive(job: "_Job", doc_type: str, notes: list[Note]) -> None:
    """One write per extraction. Keeps each candidate's own body, owner and applicability,
    which is the material a later evidence split or dispute resolution needs."""
    payload = {
        "extraction_id": job.extraction_id,
        "source_id": job.src.id,
        "filename": job.src.filename,
        "doc_type": doc_type,
        "digest_version": job.version,
        "created_at": utcnow().isoformat(),
        "entities_created": [e.model_dump(mode="json") for e in job.resolver.new],
        "candidates": [n.model_dump(mode="json") for n in notes],
    }
    uri = source_uri(job.settings, job.project_id, job.src.id,
                     f"extractions/{job.extraction_id}/candidates.json")
    job.ctx.blobs.put(uri, json.dumps(payload, indent=2).encode(), "application/json")


def _commit(job: "_Job", notes: list[Note], report: IngestReport) -> None:
    """Entities first, then reconcile this source's notes against what is already there.

    A candidate whose assertion is unchanged keeps its existing id and its human decision:
    a confirmed note stays confirmed, a rejected one is not proposed again. Only unmatched
    machine proposals are replaced. Reviewed notes are never deleted.
    """
    store, pid, sid = job.ctx.store, job.project_id, job.src.id
    if job.resolver.new:
        store.put_entities(pid, job.resolver.new)

    existing = [n for n in store.list_notes(pid, source_id=sid)
                if n.origin is None or n.origin.producer == "ingest"]
    by_assertion: dict[tuple, Note] = {}
    for n in existing:
        by_assertion.setdefault(n.assertion, n)

    write: list[Note] = []
    matched: set[str] = set()
    reused = 0
    for candidate in _merge_duplicates(notes):
        prior = by_assertion.get(candidate.assertion)
        if prior is None:
            write.append(candidate)
            continue
        matched.add(prior.id)
        if prior.status != "proposed":
            reused += 1          # the human decision stands; nothing to write
            continue
        write.append(candidate.model_copy(update={
            "id": prior.id, "created_at": prior.created_at, "revision": prior.revision,
        }))

    stale = [n.id for n in existing
             if n.id not in matched and n.status == "proposed" and n.author == "agent"]
    if stale:
        store.delete_notes(pid, stale)
    if write:
        store.put_notes(pid, write)
    report.entities_created = len(job.resolver.new)
    report.notes_written = len(write)
    report.notes_replaced = len(stale)
    report.notes_reused = reused


def _body_key(n: Note) -> tuple:
    page = n.provenance[0].page if n.kind == "reference_image" and n.provenance else None
    return (n.kind, *n.assertion, page)


def _merge_duplicates(notes: list[Note]) -> list[Note]:
    """The same fact from two chunks becomes one note with both scopes and provenances."""
    merged: dict[tuple, Note] = {}
    for n in notes:
        key = _body_key(n)
        if key not in merged:
            merged[key] = n
            continue
        m = merged[key]
        merged[key] = m.model_copy(update={
            "mentions": list(dict.fromkeys(m.mentions + n.mentions)),
            "provenance": m.provenance if n.kind == "reference_image" else m.provenance + n.provenance,
        })
    return list(merged.values())


class _Job:
    def __init__(self, ctx: Ctx, project_id: str, src: Source, version: str,
                 resolver: EntityResolver, warnings: list[str], extraction_id: str):
        self.ctx, self.project_id, self.src, self.version = ctx, project_id, src, version
        self.resolver, self.warnings = resolver, warnings
        self.settings = ctx.settings
        self.extraction_id = extraction_id
        self.unowned = 0

    def warn(self, msg: str) -> None:
        log.warning("%s: %s", self.src.filename, msg)
        self.warnings.append(msg)

    def generate(self, prompt: str, parts: list[Part], schema: type[T], *, fast: bool = False) -> T:
        return self.ctx.llm.generate(system=load_prompt(prompt, shared=prompt != "classify"),
                                     parts=parts, schema=schema, fast=fast)

    def uri(self, name: str) -> str:
        return source_uri(self.settings, self.project_id, self.src.id, name)

    def context(self, doc_type: str | None, page_rule: str, unit: Unit | None = None) -> Text:
        lines = [f"Source file: {self.src.filename}"]
        if doc_type:
            lines.append(f"Document type: {doc_type}")
        lines += ["", "Known entities:", self.resolver.roster_text()]
        if unit is not None and unit.scenes:
            lines += ["", "Scenes in scope for this excerpt:"]
            lines += [f"- {s.number} | {s.heading}" for s in unit.scenes]
        if unit is not None and unit.continuation:
            c = unit.continuation
            lines += ["", f"This excerpt continues scene {c.number} ({c.heading}) from the previous "
                          "excerpt, so its slugline is not repeated here."]
        lines += ["", f"Page numbers: {page_rule}"]
        return Text("\n".join(lines))

    def whole_file(self, data: bytes) -> Part:
        if len(data) <= self.settings.inline_limit_bytes:
            return Blob(data, self.src.mime_type)
        return Uri(self.src.storage_path, self.src.mime_type)

    def classify(self, sample: list[Part]) -> str:
        return self.generate("classify", [Text(f"Filename: {self.src.filename}"), *sample], ClassifyOut, fast=True).doc_type

    def run(self, data: bytes):
        if self.src.mime_type == PDF:
            return self.pdf(data)
        if self.src.kind == "image":
            return self.image(data)
        if self.src.kind == "text":
            return self.text(data)
        raise ValueError(f"no handler for {self.src.mime_type}")

    # --- handlers ---

    def pdf(self, data: bytes):
        texts = pdf_page_texts(data)
        n = len(texts)
        visual = is_visual(texts)
        derived = Derived(page_count=n)
        if visual:
            sample: list[Part] = [Blob(pdf_subset(data, 0, min(3, n)), PDF)]
        else:
            derived.text_path = self.uri("derived/text.md")
            self.ctx.blobs.put(derived.text_path, page_marked(texts, 0, n).encode(), "text/markdown")
            sample = [Text(page_marked(texts, 0, min(3, n)))]
        doc_type = self.classify(sample)

        if doc_type == "script":
            return doc_type, derived, self.script(data, texts, visual)

        if visual:
            derived.pages_prefix = self.uri("derived/pages/")
            for page_no, png in render_pages_png(data, self.settings.render_scale):
                self.ctx.blobs.put(f"{derived.pages_prefix}{page_no:04d}.png", png, "image/png")
            units: list[Unit] = [PdfUnit(a, b) for a, b in page_ranges(n, self.settings.visual_chunk_pages)]
            return doc_type, derived, self.run_units("visual_pages", doc_type, data, units, n, references=True)
        units = [TextUnit(page_marked(texts, a, b), MARKER_RULE) for a, b in page_ranges(n, self.settings.text_chunk_pages)]
        return doc_type, derived, self.run_units("document_notes", doc_type, data, units, n, references=False)

    def script(self, data: bytes, texts: list[str], visual: bool) -> list[Note]:
        n = len(texts)
        marked = page_marked(texts, 0, n)
        full = self.whole_file(data) if visual else Text(marked)
        rule = "the 1-based page position in the attached PDF." if visual else MARKER_RULE
        try:
            roster = self.generate("script_roster", [self.context("script", rule), full], RosterOut)
        except OutputTruncated as exc:
            raise OutputTruncated(f"roster pass truncated for a {n}-page script; this needs a rolling "
                                  f"roster, which v0 does not have ({exc})") from exc
        for loc in roster.locations:
            self.resolver.location(loc.name, loc.aliases, loc.existing_id, loc.inside)
        for sc in roster.scenes:
            loc_ids = [i for ref in sc.locations if (i := self.resolver.location(ref))]
            self.resolver.scene(sc.number, heading=sc.heading, location_ids=loc_ids, create=True)

        if not roster.scenes:
            self.warn("roster returned no scenes; falling back to page chunks")
            units: list[Unit] = (
                [PdfUnit(a, b) for a, b in page_ranges(n, self.settings.scanned_script_chunk_pages)] if visual
                else [TextUnit(page_marked(texts, a, b), MARKER_RULE) for a, b in page_ranges(n, 10)]
            )
        elif visual:
            units = self.scanned_script_units(roster.scenes, n)
        else:
            units = self.text_script_units(roster.scenes, marked)
        return self.run_units("script_notes", "script", data, units, n, references=False)

    def text_script_units(self, scenes: list[OutScene], marked: str) -> list[Unit]:
        """Locate every slugline in the text, then pack whole scenes up to the char budget."""
        page_offsets = {int(m.group(1)): m.start() for m in MARKER.finditer(marked)}
        starts, last = [], 0
        for sc in scenes:
            off = find_heading(marked, sc.heading, last)
            if off is None:
                off = max(page_offsets.get(sc.start_page or 0, last), last)
                self.warn(f"scene {sc.number}: slugline not found in text; using page {sc.start_page}")
            starts.append(off)
            last = off
        starts[0] = 0  # title page and anything before the first scene travel with it
        ends = starts[1:] + [len(marked)]
        refs = [SceneRef(sc.number, sc.heading) for sc in scenes]
        units: list[Unit] = []
        for group in pack([ends[i] - starts[i] for i in range(len(scenes))], self.settings.script_chunk_chars):
            a, b = starts[group[0]], ends[group[-1]]
            text, shift = _with_marker(marked, a, b)
            offsets = [0] + [starts[i] - a + shift for i in group[1:]]
            units.append(TextUnit(text, MARKER_RULE, [refs[i] for i in group], offsets))
        return units

    def scanned_script_units(self, scenes: list[OutScene], n: int) -> list[Unit]:
        starts, last = [], 0
        for sc in scenes:
            idx = (sc.start_page or 0) - 1
            if not 0 <= idx < n or idx < last:
                self.warn(f"scene {sc.number}: start_page {sc.start_page} invalid or out of order; using page {last + 1}")
                idx = last
            starts.append(idx)
            last = idx
        starts[0] = 0
        ends = [min(n, s + 1) for s in starts[1:]] + [n]  # boundary page belongs to both scenes
        refs = [SceneRef(sc.number, sc.heading) for sc in scenes]
        units: list[Unit] = []
        for group in pack([ends[i] - starts[i] for i in range(len(scenes))], self.settings.scanned_script_chunk_pages):
            units.append(PdfUnit(starts[group[0]], ends[group[-1]], [refs[i] for i in group], [starts[i] for i in group]))
        return units

    def image(self, data: bytes):
        out = self.generate("image", [self.context(None, NO_PAGES), self.whole_file(data)], ImageOut)
        if len(out.references) > 1:
            self.warn(f"{len(out.references)} references for one image; keeping the first")
            out.references = out.references[:1]
        return out.doc_type, Derived(), self.notes_from(out, page_offset=0, page_count=None)

    def text(self, data: bytes):
        text = data.decode("utf-8", errors="replace")
        doc_type = self.classify([Text(text[:6000])])
        if doc_type == "script":
            self.warn("text-format scripts get document notes only; no scene roster in v0")
        units: list[Unit] = [TextUnit(c, NO_PAGES) for c in text_chunks(text, self.settings.text_chunk_chars)]
        return doc_type, Derived(), self.run_units("document_notes", doc_type, data, units, None, references=False)

    # --- running units ---

    def run_units(self, prompt: str, doc_type: str, data: bytes, units: list[Unit],
                  page_count: int | None, *, references: bool) -> list[Note]:
        """Runs each unit; a truncated unit is split and retried, never accepted partially."""
        notes: list[Note] = []
        queue = list(units)
        while queue:
            unit = queue.pop(0)
            try:
                out = self.generate(prompt, [self.context(doc_type, unit.rule, unit), *unit.parts(data)], NotesOut)
            except OutputTruncated as exc:
                halves = unit.split()
                if not halves:
                    raise OutputTruncated(f"{unit.label}: output truncated and cannot be split further ({exc})") from exc
                self.warn(f"{unit.label}: output truncated; split into {len(halves)} and retried")
                queue[0:0] = halves
                continue
            in_scope = {i for s in unit.scenes if (i := self.resolver.scene(s.number))}
            notes += self.notes_from(out, page_offset=unit.offset, page_count=page_count,
                                     references=references, scene_scope=in_scope or None, label=unit.label)
        return notes

    # --- model output -> notes ---

    def notes_from(self, out: NotesOut, *, page_offset: int, page_count: int | None,
                   references: bool = True, scene_scope: set[str] | None = None, label: str = "") -> list[Note]:
        for loc in out.locations:
            self.resolver.location(loc.name, loc.aliases, loc.existing_id, loc.inside)
        notes: list[Note] = []
        out_of_scope = 0
        for n in out.notes:
            owner = self.owner(n.owner, n.project_wide)
            if owner is None:
                self.unowned += 1
                self.warn(f"dropped note with no owner: {n.body[:80]!r}")
                continue
            scene_id = self.resolver.scene(n.only_during_scene) if n.only_during_scene else None
            if n.only_during_scene and scene_id is None:
                self.warn(f"unknown scene {n.only_during_scene!r}; note kept without the condition")
            if scene_scope is not None and scene_id is not None and scene_id not in scene_scope:
                out_of_scope += 1  # a neighbouring scene's excerpt owns this fact
                continue
            applicability = Applicability(
                include_descendants=bool(n.applies_to_places_within) and owner != PROJECT_SCOPE,
                scene_id=scene_id,
            )
            note = self.note(n.kind, n.body, owner, self.page(n.page, page_offset, page_count), n.quote,
                             applicability=applicability, mentions=self.mentions(n.mentions, owner))
            if note:
                notes.append(note)
        if out_of_scope:
            self.warn(f"{label}: dropped {out_of_scope} notes about scenes outside the excerpt's scope")
        if references:
            notes += [x for r in out.references if (x := self.reference(r, page_offset, page_count))]
        elif out.references:
            self.warn(f"ignored {len(out.references)} references from a text-only pass")
        return notes

    def reference(self, r: OutReference, page_offset: int, page_count: int | None) -> Note | None:
        page = self.page(r.page, page_offset, page_count)
        if page_count is not None and page is None:
            self.warn(f"dropped reference without a valid page: {r.caption[:80]!r}")
            return None
        owner = self.owner(r.location, not r.location) or PROJECT_SCOPE
        return self.note("reference_image", r.caption, owner, page, None)

    def owner(self, ref: str, project_wide: bool) -> str | None:
        """One owner per note. A named place wins over the project_wide flag, because a
        model that sets both is describing a place, not a production-wide rule."""
        ref = (ref or "").strip()
        if ref:
            if owner_id := self.resolver.location(ref):
                return owner_id
            self.warn(f"owner {ref!r} is rejected or empty; note dropped")
            return None
        return PROJECT_SCOPE if project_wide else None

    def mentions(self, refs: list[str], owner: str) -> list[str]:
        ids = []
        for ref in refs:
            if (i := self.resolver.location(ref)) and i != owner and i not in ids:
                ids.append(i)
        return ids

    def page(self, page: int | None, offset: int, count: int | None) -> int | None:
        if page is None or count is None:
            return None
        absolute = page + offset
        if 1 <= absolute <= count:
            return absolute
        self.warn(f"page {page} (offset {offset}) outside 1-{count}; provenance page cleared")
        return None

    def note(self, kind: str, body: str, owner: str, page: int | None, quote: str | None,
             applicability: Applicability | None = None, mentions: list[str] | None = None) -> Note | None:
        body = body.strip()
        if not body:
            return None
        if len(body) > 2000:
            self.warn(f"note body truncated from {len(body)} chars")
            body = body[:2000]
        applicability = applicability or Applicability()
        return Note(
            kind=kind, body=body, owner_id=owner, applicability=applicability,
            mentions=mentions or [], author="agent",
            provenance=[Provenance(
                source_id=self.src.id, page=page, quote=(quote or "").strip()[:300] or None,
                extraction_id=self.extraction_id, extracted_body=body, extracted_owner_id=owner,
            )],
            origin=NoteOrigin(producer="ingest", source_id=self.src.id, digest_version=self.version),
        )


def _with_marker(marked: str, a: int, b: int) -> tuple[str, int]:
    """Slice [a, b) of page-marked text, re-headed with its page marker if it starts mid-page."""
    chunk = marked[a:b]
    if MARKER.match(chunk):
        return chunk, 0
    before = list(MARKER.finditer(marked, 0, a))
    prefix = f"<<<PAGE {before[-1].group(1)}>>>\n" if before else ""
    return prefix + chunk, len(prefix)
