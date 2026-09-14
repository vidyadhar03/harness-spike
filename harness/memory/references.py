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
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .files import extension_for
from .ingest import PROMPTS, load_prompt, source_uri
from .models import Location, Note, NoteOrigin, Provenance, Source
from .ports import LLM, Blob, Blobs, ImageHit, Images, OutputTruncated, Store, T, Text, TermHit
from .retrieval import get_context, render_context_md, resolve_scope
from .schemas import CaptionsOut, CurateOut, VocabularyOut

log = logging.getLogger(__name__)

PRODUCER = "references"
UNSORTED = "Unsorted"
FACET_LABELS = {"place": "Place", "terrain": "Terrain",
                "architecture": "Architecture", "material": "Material"}
REFERENCES_VERSION = "references-v2"
CAPTION_BATCH = 8          # images per captioning call; halved on truncation


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
    images_uncaptioned: int = 0
    kept: list[tuple[str, str]] = field(default_factory=list)       # (title, term and query that found it)
    origins: dict[str, str] = field(default_factory=dict)           # image key -> term and query
    directions: list[tuple[str, int]] = field(default_factory=list)
    facets: Counter = field(default_factory=Counter)
    notes_written: int = 0
    notes_replaced: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        log.warning("%s: %s", self.location, msg)
        self.warnings.append(msg)


def references_version(llm: LLM) -> str:
    h = hashlib.sha256()
    for name in ("_shared.md", "vocabulary.md", "caption_references.md", "group_references.md"):
        h.update((PROMPTS / name).read_bytes())
    h.update(llm.model_id.encode())
    return f"{REFERENCES_VERSION}-{h.hexdigest()[:8]}"


def suggest_references(ctx: RefCtx, project_id: str, scope_ref: str, *, per_term: int = 6,
                       max_images: int = 32, dry_run: bool = False,
                       terms_only: bool = False) -> ReferenceReport:
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

    if terms_only:
        return report      # tuning the vocabulary costs one model call, not dozens of fetches

    hits = _retrieve(ctx, vocab, verified, per_term, max_images, report)
    if not hits:
        report.warn("no licensed images found for the verified terms")
    curated = _curate(ctx, context_md, hits, report) if hits else ([], {}, {})

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
        log.info("verifying term %r", t.term)
        hit = ctx.images.verify_term(t.term)
        if hit is None:
            report.terms_dropped.append(t.term)
            continue
        verified.append((t.term, hit))
        report.terms_verified.append(t.term)
    return verified


def _retrieve(ctx: RefCtx, vocab: VocabularyOut, verified: list[tuple[str, TermHit]], per_term: int,
              max_images: int, report: ReferenceReport) -> list[ImageHit]:
    """Round-robin across terms so one prolific term cannot crowd out the rest.

    A generic term on its own returns the world's most photographed example ("river gorge"
    is the New River Gorge), so a term needing a region is searched inside every region term
    first, and the bare term is only a fallback when no region has anything.
    """
    by_term = {t.term: t for t in vocab.terms}
    regions = [(term, hit) for term, hit in verified if by_term[term].kind == "region"]
    per_term_hits = []
    for term, hit in verified:
        spec = by_term[term]
        if spec.kind == "region":
            found = ctx.images.search_images(term, per_term, region=term, region_title=hit.title)
            via = "its own region"
        elif spec.needs_region and regions:
            scoped, via_regions = [], []
            for region, region_hit in regions:
                r_found = ctx.images.search_images(term, per_term, region=region, region_title=region_hit.title)
                if r_found:
                    scoped.append(r_found)
                    via_regions.append(f"{region} ({len(r_found)})")
            if scoped:
                found = _dedupe(_interleave(scoped))[:per_term]
                via = "in " + ", ".join(via_regions)
            else:
                found = ctx.images.search_images(term, per_term)
                via = "unscoped, no region had results"
        else:
            found = ctx.images.search_images(term, per_term)
            via = "unscoped"
        log.info("%s: %r returned %d licensed image(s) %s", report.location, term, len(found), via)
        if not found:
            report.warn(f"{term!r} returned no licensed images")
        report.images_found += len(found)
        per_term_hits.append(found)
        for h in found:
            report.origins.setdefault(h.image_url or h.page_url, f"{term} ({via})")

    seen: set[str] = set()
    out: list[ImageHit] = []
    for hit in _interleave(per_term_hits):
        key = hit.image_url or hit.page_url
        if not key or key in seen or not hit.preview_url:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= max_images:
            return out
    return out


def _interleave(lists: list[list[ImageHit]]) -> Iterator[ImageHit]:
    for rank in range(max((len(found) for found in lists), default=0)):
        for found in lists:
            if rank < len(found):
                yield found[rank]


def _dedupe(hits: Iterable[ImageHit]) -> list[ImageHit]:
    seen: set[str] = set()
    out = []
    for hit in hits:
        key = hit.image_url or hit.page_url
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out


def caption_parts(images: Images, context_md: str, hits: list[ImageHit], indices: list[int],
                  report: ReferenceReport) -> tuple[list, list[int]]:
    """One caption call's input, and the indices whose images could be fetched.

    Shared with the caption eval, because which images share a call is part of what it measures.
    """
    parts: list = [Text(context_md), Text(f"{len(indices)} candidate images follow.")]
    present: list[int] = []
    for i in indices:
        hit = hits[i]
        try:
            data = images.fetch(hit.preview_url)
        except Exception as exc:
            report.warn(f"could not fetch image {i} ({hit.title}): {exc}")
            continue
        present.append(i)
        parts.append(Text(f"[{i}] {hit.title}" + (f" — {hit.description}" if hit.description else "")))
        parts.append(Blob(data, hit.mime_type))
    return parts, present


def caption_all(ctx: RefCtx, context_md: str, hits: list[ImageHit], report: ReferenceReport
                ) -> tuple[dict[int, str], dict[int, str]]:
    """Caption every candidate in bounded batches, in retrieval order."""
    captions: dict[int, str] = {}
    facets: dict[int, str] = {}
    order = list(range(len(hits)))
    for start in range(0, len(order), CAPTION_BATCH):
        batch_c, batch_f = _caption(ctx, context_md, hits, order[start:start + CAPTION_BATCH], report)
        captions.update(batch_c)
        facets.update(batch_f)
    return captions, facets


def _caption(ctx: RefCtx, context_md: str, hits: list[ImageHit], indices: list[int],
             report: ReferenceReport) -> tuple[dict[int, str], dict[int, str]]:
    """Caption one batch of images, splitting and retrying if the model runs out of budget.

    Thinking tokens share the output budget, so a large batch can truncate before any
    answer is written. Halving is always safe: captioning one image never depends on
    seeing the others.
    """
    parts, present = caption_parts(ctx.images, context_md, hits, indices, report)
    if not present:
        return {}, {}

    try:
        out = ctx.llm.generate(system=load_prompt("caption_references"), parts=parts,
                               schema=CaptionsOut)
    except OutputTruncated as exc:
        if len(present) == 1:
            report.warn(f"image {present[0]} could not be captioned within the output budget; skipped")
            return {}, {}
        mid = len(present) // 2
        report.warn(f"captioning {len(present)} images truncated; split into "
                    f"{mid} and {len(present) - mid} and retried")
        left_c, left_f = _caption(ctx, context_md, hits, present[:mid], report)
        right_c, right_f = _caption(ctx, context_md, hits, present[mid:], report)
        return {**left_c, **right_c}, {**left_f, **right_f}

    keep = set(present)
    captions = {c.index: c.caption.strip() for c in out.captions
                if c.index in keep and c.caption.strip()}
    facets = {c.index: c.facet for c in out.captions if c.index in captions}
    return captions, facets


def _curate(ctx: RefCtx, context_md: str, hits: list[ImageHit], report: ReferenceReport
            ) -> tuple[list[tuple[str, str, list[int]]], dict[int, str], dict[int, str]]:
    """Caption in bounded batches, then group the captions in one text-only call.

    Captioning is per-image work that scales with the candidate count; grouping is one
    small judgement over text. Keeping them apart stops the whole pass from growing with
    the number of images retrieved.
    """
    captions, facets = caption_all(ctx, context_md, hits, report)
    order = list(range(len(hits)))

    uncaptioned = [i for i in order if i not in captions]
    if uncaptioned:
        report.warn(f"{len(uncaptioned)} image(s) judged irrelevant and left uncaptioned: "
                    + ", ".join(hits[i].title[:40] for i in uncaptioned))
    if not captions:
        return [], {}, {}

    listing = "\n".join(f"[{i}] ({facets.get(i, 'place')}) {captions[i]}" for i in sorted(captions))
    out = ctx.llm.generate(system=load_prompt("group_references"),
                           parts=[Text(context_md), Text("Captioned images:\n" + listing)],
                           schema=CurateOut)

    directions: list[tuple[str, str, list[int]]] = []
    placed: set[int] = set()
    for d in out.directions:
        members = [i for i in d.images if i in captions and i not in placed]
        if not members:
            # a named direction with no usable images is the model changing its mind mid-answer
            report.warn(f"direction {d.name.strip() or 'unnamed'!r} named no usable images; dropped")
            continue
        placed |= set(members)
        directions.append((d.name.strip() or "Untitled direction", d.why.strip(), members))

    # a captioned image the model forgot to file is still a usable reference: keep it rather
    # than silently discarding work the director may want
    unplaced = [i for i in captions if i not in placed]
    if unplaced:
        report.warn(f"{len(unplaced)} captioned image(s) were not placed in a direction; "
                    f"kept under {UNSORTED!r}")
        directions.append((UNSORTED, "", unplaced))
        placed |= set(unplaced)

    report.images_kept = len(placed)
    report.kept = [(hits[i].title, report.origins.get(hits[i].image_url or hits[i].page_url, ""))
                   for _, _, members in directions for i in members]
    report.images_uncaptioned = len(uncaptioned)
    report.directions = [(name, len(members)) for name, _, members in directions]
    report.facets = Counter(facets[i] for i in placed if i in facets)
    return directions, captions, facets


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
           curated: tuple[list, dict, dict], report: ReferenceReport) -> None:
    directions, captions, facets = curated
    notes: list[Note] = [_vocabulary_note(scope_id, version, vocab, verified)]
    for name, why, members in directions:
        for i in members:
            # the facet is the heading, because "could stand in for Devgram" and "right
            # stonework, wrong country" are what a reader needs to tell apart first. The
            # direction names the look and rides with the caption, so one direction spanning
            # several facets stays one idea instead of becoming a repeated heading.
            group = FACET_LABELS.get(facets.get(i, "place"), "Place")
            body = f"{name} — {captions[i]}" if name != UNSORTED else captions[i]
            src = _store_image(ctx, project_id, hits[i])
            if src is None:
                report.warn(f"image {i} could not be stored; skipped")
                continue
            notes.append(Note(
                kind="reference_image", body=body[:2000], owner_id=scope_id,
                author="agent", group=group,
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