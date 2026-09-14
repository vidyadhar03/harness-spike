"""Real-world visual references for a location.

Three passes, each one reviewable on its own:
  1. vocabulary  - turn the location's memory context into archive search terms
  2. verify      - confirm every term exists; unverified terms are dropped, never searched
  3. curate      - retrieve licensed images, then group them into visual directions

Nothing here decides where to shoot. Results land as proposed reference_image notes on
the location, so they go through the same review path as anything ingest wrote.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .files import extension_for
from .ingest import PROMPTS, load_prompt, source_uri
from .models import Location, Note, NoteOrigin, Provenance, Source
from .ports import LLM, Blob, Blobs, ImageHit, Images, Store, T, Text, TermHit
from .retrieval import get_context, render_context_md, resolve_scope
from .schemas import CurateOut, VocabularyOut

log = logging.getLogger(__name__)

PRODUCER = "references"
REFERENCES_VERSION = "references-v1"


@dataclass
class RefCtx:
    store: Store
    blobs: Blobs
    llm: LLM
    images: Images
    settings: Settings


@dataclass
class ReferenceReport:
    scope_id: str
    location: str
    terms_proposed: int = 0
    terms_verified: list[str] = field(default_factory=list)
    terms_dropped: list[str] = field(default_factory=list)
    images_found: int = 0
    images_kept: int = 0
    directions: list[tuple[str, int]] = field(default_factory=list)
    notes_written: int = 0
    notes_replaced: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        log.warning("%s: %s", self.location, msg)
        self.warnings.append(msg)


def references_version(llm: LLM) -> str:
    h = hashlib.sha256()
    for name in ("_shared.md", "vocabulary.md", "curate_references.md"):
        h.update((PROMPTS / name).read_bytes())
    h.update(llm.model_id.encode())
    return f"{REFERENCES_VERSION}-{h.hexdigest()[:8]}"


def suggest_references(ctx: RefCtx, project_id: str, scope_ref: str, *, per_term: int = 6,
                       max_images: int = 32, dry_run: bool = False) -> ReferenceReport:
    scope_id = resolve_scope(ctx.store, project_id, scope_ref)
    entity = ctx.store.get_entity(project_id, scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"references are per location; {scope_ref!r} resolved to a {type(entity).__name__.lower()}")

    pack = get_context(ctx.store, project_id, scope_id)
    context_md = render_context_md(pack)
    report = ReferenceReport(scope_id=scope_id, location=entity.name)
    version = references_version(ctx.llm)

    vocab = _vocabulary(ctx, context_md, report)
    verified = _verify(ctx, vocab, report)
    if not verified:
        report.warn("no search terms survived verification; nothing to retrieve")
        return report

    hits = _retrieve(ctx, verified, per_term, max_images, report)
    if not hits:
        report.warn("no licensed images found for the verified terms")
    curated = _curate(ctx, context_md, hits, report) if hits else ([], {})

    if dry_run:
        return report
    _write(ctx, project_id, scope_id, version, vocab, verified, hits, curated, report)
    return report


# --- passes -------------------------------------------------------------------------

def _vocabulary(ctx: RefCtx, context_md: str, report: ReferenceReport) -> VocabularyOut:
    out = ctx.llm.generate(system=load_prompt("vocabulary"), parts=[Text(context_md)], schema=VocabularyOut)
    seen, terms = set(), []
    for t in out.terms:
        key = " ".join(t.term.lower().split())
        if key and key not in seen:
            seen.add(key)
            terms.append(t)
    out.terms = terms
    report.terms_proposed = len(terms)
    return out


def _verify(ctx: RefCtx, vocab: VocabularyOut, report: ReferenceReport) -> list[tuple[str, TermHit]]:
    """A term the encyclopedia does not recognise is usually invented, and searching it
    returns plausible-looking rubbish, so it is dropped rather than downweighted."""
    verified: list[tuple[str, TermHit]] = []
    for t in vocab.terms:
        hit = ctx.images.verify_term(t.term)
        if hit is None:
            report.terms_dropped.append(t.term)
            continue
        verified.append((t.term, hit))
        report.terms_verified.append(t.term)
    return verified


def _retrieve(ctx: RefCtx, verified: list[tuple[str, TermHit]], per_term: int, max_images: int,
              report: ReferenceReport) -> list[ImageHit]:
    """Round-robin across terms so one prolific term cannot crowd out the rest."""
    per_term_hits = []
    for term, _ in verified:
        found = ctx.images.search_images(term, per_term)
        report.images_found += len(found)
        per_term_hits.append(found)

    seen: set[str] = set()
    out: list[ImageHit] = []
    for rank in range(per_term):
        for found in per_term_hits:
            if rank >= len(found):
                continue
            hit = found[rank]
            key = hit.image_url or hit.page_url
            if not key or key in seen or not hit.preview_url:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) >= max_images:
                return out
    return out


def _curate(ctx: RefCtx, context_md: str, hits: list[ImageHit],
            report: ReferenceReport) -> tuple[list[tuple[str, str, list[int]]], dict[int, str]]:
    parts = [Text(context_md), Text(f"{len(hits)} candidate images follow.")]
    usable: list[int] = []
    for i, hit in enumerate(hits):
        try:
            data = ctx.images.fetch(hit.preview_url)
        except Exception as exc:
            report.warn(f"could not fetch image {i} ({hit.title}): {exc}")
            continue
        usable.append(i)
        parts.append(Text(f"[{i}] {hit.title}" + (f" — {hit.description}" if hit.description else "")))
        parts.append(Blob(data, hit.mime_type))
    if not usable:
        return [], {}

    out = ctx.llm.generate(system=load_prompt("curate_references"), parts=parts, schema=CurateOut)
    captions = {c.index: c.caption.strip() for c in out.captions if c.index in set(usable) and c.caption.strip()}
    directions: list[tuple[str, str, list[int]]] = []
    placed: set[int] = set()
    for d in out.directions:
        members = [i for i in d.images if i in captions and i not in placed]
        if len(members) < 2:
            if members:
                report.warn(f"direction {d.name!r} had fewer than 2 usable images; dropped")
            continue
        placed |= set(members)
        directions.append((d.name.strip() or "Untitled direction", d.why.strip(), members))
    dropped = len(usable) - len(placed)
    if dropped:
        report.warn(f"{dropped} candidate images were not placed in a direction and were dropped")
    report.images_kept = len(placed)
    report.directions = [(name, len(members)) for name, _, members in directions]
    return directions, captions


# --- write-back -----------------------------------------------------------------------

def _store_image(ctx: RefCtx, project_id: str, hit: ImageHit) -> Source | None:
    try:
        data = ctx.images.fetch(hit.preview_url)
    except Exception as exc:
        log.warning("could not fetch %s: %s", hit.preview_url, exc)
        return None
    sid = hashlib.sha256(data).hexdigest()
    existing = ctx.store.get_source(project_id, sid)
    if existing is not None:
        return existing
    name = Path(hit.title).name or "reference"
    uri = source_uri(ctx.settings, project_id, sid, "original" + extension_for(name, hit.mime_type))
    ctx.blobs.put(uri, data, hit.mime_type)
    src = Source(id=sid, filename=name, mime_type=hit.mime_type, kind="image", doc_type="reference",
                 size_bytes=len(data), storage_path=uri, status="digested",
                 origin_url=hit.page_url or hit.image_url, license=hit.license, attribution=hit.attribution)
    ctx.store.put_source(project_id, src)
    return src


def _vocabulary_note(scope_id: str, version: str, vocab: VocabularyOut,
                     verified: list[tuple[str, TermHit]]) -> Note:
    lines = ["Search vocabulary for this location."]
    if vocab.script_phrases:
        lines.append("Script's words: " + "; ".join(p.strip() for p in vocab.script_phrases if p.strip()))
    by_term = {t.term: t for t in vocab.terms}
    lines.append("Verified terms: " + "; ".join(
        f"{term} ({by_term[term].kind})" if term in by_term else term for term, _ in verified))
    return Note(
        kind="vocabulary", body="\n".join(lines)[:2000], owner_id=scope_id, author="agent",
        provenance=[Provenance(url=hit.url, title=hit.title) for _, hit in verified[:8]],
        origin=NoteOrigin(producer=PRODUCER, scope=scope_id, digest_version=version),
    )


def _write(ctx: RefCtx, project_id: str, scope_id: str, version: str, vocab: VocabularyOut,
           verified: list[tuple[str, TermHit]], hits: list[ImageHit],
           curated: tuple[list[tuple[str, str, list[int]]], dict[int, str]], report: ReferenceReport) -> None:
    directions, captions = curated
    notes: list[Note] = [_vocabulary_note(scope_id, version, vocab, verified)]
    for name, why, members in directions:
        group = f"{name} — {why}" if why else name   # the rationale belongs to the group, not each caption
        for i in members:
            src = _store_image(ctx, project_id, hits[i])
            if src is None:
                report.warn(f"image {i} could not be stored; skipped")
                continue
            notes.append(Note(
                kind="reference_image", body=captions[i][:2000], owner_id=scope_id,
                author="agent", group=group[:300],
                provenance=[Provenance(source_id=src.id, url=hits[i].page_url or None,
                                       title=hits[i].title or None)],
                origin=NoteOrigin(producer=PRODUCER, scope=scope_id, digest_version=version),
            ))

    existing = [n for n in ctx.store.notes_for_owners(project_id, [scope_id])
                if n.origin and n.origin.producer == PRODUCER and n.origin.scope == scope_id]
    stale = [n.id for n in existing if n.status == "proposed"]
    reviewed = {_key(n) for n in existing if n.status != "proposed"}
    fresh = [n for n in notes if _key(n) not in reviewed]
    if stale:
        ctx.store.delete_notes(project_id, stale)
    if fresh:
        ctx.store.put_notes(project_id, fresh)
    report.notes_replaced = len(stale)
    report.notes_written = len(fresh)


def _key(n: Note) -> tuple:
    """A reviewed image keeps its verdict across re-runs; its identity is the file, not the caption."""
    src = next((p.source_id for p in n.provenance if p.source_id), None)
    return (n.kind, src) if n.kind == "reference_image" else (n.kind, " ".join(n.body.lower().split()))
