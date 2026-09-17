"""Real-world visual references for a location.

Four passes, each one reviewable on its own:
  1. vocabulary  - turn the location's memory context into archive search terms
  2. verify      - confirm every term exists; unverified terms are dropped, never searched
  3. retrieve    - fetch licensed images from Commons
  4. curate      - caption in bounded batches, then group into visual directions

Nothing here decides where to shoot. Results land as proposed reference_image notes on
the location, so they go through the same review path as anything ingest wrote.
"""
from __future__ import annotations

import hashlib
import html
import logging
import mimetypes
import re
import tempfile
import time
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .files import ACCEPTED_REFERENCE_MIMES, extension_for, kind_for, sniff_mime, validate_reference_image
from .ingest import PROMPTS, load_prompt, source_uri
from .models import Location, Note, NoteOrigin, Provenance, RetrievalOrigin, Source
from .ports import (
    LLM, Blob, Blobs, ImageHit, Images, OutputTruncated, ReplacementTooLarge,
    Store, T, Text, TermHit, VerificationServiceError,
)
from .retrieval import (
    get_references_context, render_references_context_md, resolve_scope,
)
from .schemas import CaptionsOut, CurateOut, VocabularyOut

log = logging.getLogger(__name__)

PRODUCER = "references"
UNSORTED = "Unsorted"
FACET_LABELS = {"place": "Place", "terrain": "Terrain",
                "architecture": "Architecture", "material": "Material"}
REFERENCES_VERSION = "references-v2"
CAPTION_BATCH = 8          # images per captioning call; halved on truncation

# safe URL schemes for HTML export links
_SAFE_URL_SCHEMES = frozenset({"http", "https"})


@dataclass
class RefCtx:
    store: Store
    blobs: Blobs
    llm: LLM
    images: Images
    settings: Settings


@dataclass
class StageTiming:
    """Elapsed time for one pipeline stage."""
    name: str
    elapsed_s: float
    includes_downloads: bool = False


@dataclass
class ReferenceReport:
    scope_id: str
    location: str
    terms_proposed: int = 0
    terms_verified: list[str] = field(default_factory=list)
    terms_bypassed: list[str] = field(default_factory=list)
    terms_dropped: list[str] = field(default_factory=list)
    verification_failures: list[str] = field(default_factory=list)
    images_found: int = 0
    images_kept: int = 0
    images_uncaptioned: int = 0
    candidates_retrieved: int = 0
    candidates_evaluated: int = 0
    candidates_omitted: int = 0
    failed_downloads: int = 0
    kept: list[tuple[str, str]] = field(default_factory=list)       # (title, term and query that found it)
    origins: dict[str, str] = field(default_factory=dict)           # image key -> term and query
    retrieval_origins_map: dict[str, list[RetrievalOrigin]] = field(default_factory=dict)
    directions: list[tuple[str, int]] = field(default_factory=list)
    facets: Counter = field(default_factory=Counter)
    notes_written: int = 0
    notes_replaced: int = 0
    notes_skipped_reviewed: int = 0
    warnings: list[str] = field(default_factory=list)
    # performance
    stage_timings: list[StageTiming] = field(default_factory=list)
    cache_hits: int = 0
    cache_downloads: int = 0
    model_calls: int = 0
    http_retries: int = 0
    truncation_retries: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        log.warning("%s: %s", self.location, msg)
        self.warnings.append(msg)


def references_version(llm: LLM) -> str:
    h = hashlib.sha256()
    for name in ("_shared.md", "vocabulary.md", "caption_references.md", "group_references.md"):
        h.update((PROMPTS / name).read_bytes())
    h.update(llm.model_id.encode())
    return f"{REFERENCES_VERSION}-{h.hexdigest()[:8]}"


# --- preview cache ----------------------------------------------------------------

class RunLocalImageCache:
    """Bounded, run-local preview cache that guarantees the exact bytes evaluated by
    the model are the exact bytes stored.

    Bound by total byte size. Uses temporary files to avoid holding all images in
    memory while still preventing re-downloads.
    """

    def __init__(self, images: Images, max_bytes: int = 200 * 1024 * 1024):
        self._images = images
        self._max_bytes = max_bytes
        self._cache: dict[str, Path] = {}   # url -> temp file path
        self._total_bytes = 0
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self.hits = 0
        self.downloads = 0

    def _ensure_tmpdir(self) -> Path:
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="harness_ref_cache_")
        return Path(self._tmpdir.name)

    def fetch(self, url: str) -> bytes:
        """Fetch preview bytes, using cache on repeat access."""
        cached = self._cache.get(url)
        if cached is not None and cached.exists():
            self.hits += 1
            return cached.read_bytes()

        data = self._images.fetch(url)
        self.downloads += 1

        if self._total_bytes + len(data) <= self._max_bytes:
            tmpdir = self._ensure_tmpdir()
            fname = hashlib.sha256(url.encode()).hexdigest()[:16]
            path = tmpdir / fname
            path.write_bytes(data)
            self._cache[url] = path
            self._total_bytes += len(data)

        return data

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._cache.clear()
            self._total_bytes = 0

    def verify_term(self, term: str) -> TermHit | None:
        return self._images.verify_term(term)

    def search_images(self, term: str, limit: int, region: str | None = None,
                      region_title: str | None = None) -> list[ImageHit]:
        return self._images.search_images(term, limit, region=region, region_title=region_title)


# --- main entry -------------------------------------------------------------------

def suggest_references(ctx: RefCtx, project_id: str, scope_ref: str, *, per_term: int = 6,
                       max_images: int = 32, dry_run: bool = False,
                       terms_only: bool = False,
                       bypass_verification: bool = False) -> ReferenceReport:
    t_total = time.monotonic()
    scope_id = resolve_scope(ctx.store, project_id, scope_ref)
    entity = ctx.store.get_entity(project_id, scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"references are per location; {scope_ref!r} resolved to a {type(entity).__name__.lower()}")

    pack = get_references_context(ctx.store, project_id, scope_id)
    context_md = render_references_context_md(pack)
    report = ReferenceReport(scope_id=scope_id, location=entity.name)
    version = references_version(ctx.llm)

    cache = RunLocalImageCache(ctx.images)
    cached_ctx = RefCtx(store=ctx.store, blobs=ctx.blobs, llm=ctx.llm,
                        images=cache, settings=ctx.settings)
    try:
        t0 = time.monotonic()
        vocab = _vocabulary(cached_ctx, context_md, report)
        report.stage_timings.append(StageTiming("vocabulary", time.monotonic() - t0))

        t0 = time.monotonic()
        verified = _verify(cached_ctx, vocab, report, bypass=bypass_verification)
        report.stage_timings.append(StageTiming("verification", time.monotonic() - t0))
        if not verified:
            report.warn("no search terms survived verification; nothing to retrieve")
            return report

        if terms_only:
            return report

        t0 = time.monotonic()
        hits = _retrieve(cached_ctx, vocab, verified, per_term, max_images, report)
        report.stage_timings.append(StageTiming("retrieval", time.monotonic() - t0, includes_downloads=True))
        if not hits:
            report.warn("no licensed images found for the verified terms")
        curated = _curate(cached_ctx, context_md, hits, report) if hits else ([], {}, {})

        if dry_run:
            return report
        _write(cached_ctx, project_id, scope_id, version, vocab, verified, hits, curated, report)
    finally:
        report.cache_hits = cache.hits
        report.cache_downloads = cache.downloads
        report.stage_timings.append(StageTiming("total", time.monotonic() - t_total, includes_downloads=True))
        _collect_token_usage(ctx.llm, report)
        cache.cleanup()
    return report


def _collect_token_usage(llm: LLM, report: ReferenceReport) -> None:
    usage = getattr(llm, "accumulated_usage", None)
    if isinstance(usage, dict):
        report.token_usage = dict(usage)
    report.http_retries = getattr(llm, "http_retries", 0)


# --- passes -------------------------------------------------------------------------

def _vocabulary(ctx: RefCtx, context_md: str, report: ReferenceReport) -> VocabularyOut:
    report.model_calls += 1
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


def _verify(ctx: RefCtx, vocab: VocabularyOut, report: ReferenceReport, *,
            bypass: bool = False) -> list[tuple[str, TermHit]]:
    """A term the encyclopedia does not recognise is usually invented, and searching it
    returns plausible-looking rubbish, so it is dropped rather than downweighted.

    When bypass=True, terms without Wikipedia hits are accepted using synthetic fallbacks
    rather than dropped, but successful Wikipedia hits (such as canonical region titles like
    "Kinnaur district" for "Kinnaur") are preserved for Commons category resolution.
    If a region term lacks canonical resolution, a warning is recorded.
    """
    verified: list[tuple[str, TermHit]] = []
    for t in vocab.terms:
        log.info("verifying term %r", t.term)
        try:
            hit = ctx.images.verify_term(t.term)
        except VerificationServiceError as exc:
            report.verification_failures.append(t.term)
            if bypass:
                # accept the term without verified metadata
                hit = TermHit(title=t.term, url="", snippet="")
                verified.append((t.term, hit))
                report.terms_bypassed.append(t.term)
                log.info("verification bypassed for %r (service error: %s)", t.term, exc)
                if t.kind == "region":
                    report.warn(f"region {t.term!r} lacks canonical resolution (service error: {exc}); using fallback title which may degrade category resolution")
            else:
                report.warn(f"verification service unavailable for {t.term!r}: {exc}")
            continue
        if hit is None:
            if bypass:
                hit = TermHit(title=t.term, url="", snippet="")
                verified.append((t.term, hit))
                report.terms_bypassed.append(t.term)
                log.info("verification bypassed for %r (no match)", t.term)
                if t.kind == "region":
                    report.warn(f"region {t.term!r} lacks canonical resolution; using fallback title which may degrade category resolution")
            else:
                report.terms_dropped.append(t.term)
            continue
        verified.append((t.term, hit))
        report.terms_verified.append(t.term)
    # if service failures prevented all verification and bypass is off, leave prior proposals intact
    if not verified and report.verification_failures and not bypass:
        report.warn("all terms failed service verification; previous proposals left intact")
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
        report.candidates_retrieved += len(found)
        per_term_hits.append(found)

        # build structured retrieval origins
        region_val = term if spec.kind == "region" else None
        origin = RetrievalOrigin(term=term, query=term, region=region_val, via=via)
        for h in found:
            key = h.image_url or h.page_url
            report.origins.setdefault(key, f"{term} ({via})")
            report.retrieval_origins_map.setdefault(key, []).append(origin)

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
            report.failed_downloads += 1
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

    report.model_calls += 1
    try:
        out = ctx.llm.generate(system=load_prompt("caption_references"), parts=parts,
                               schema=CaptionsOut)
    except OutputTruncated:
        report.truncation_retries += 1
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
    report.candidates_evaluated = len(hits)
    t0 = time.monotonic()
    captions, facets = caption_all(ctx, context_md, hits, report)
    report.stage_timings.append(StageTiming("captioning", time.monotonic() - t0, includes_downloads=True))
    order = list(range(len(hits)))

    uncaptioned = [i for i in order if i not in captions]
    if uncaptioned:
        report.warn(f"{len(uncaptioned)} image(s) judged irrelevant and left uncaptioned: "
                    + ", ".join(hits[i].title[:40] for i in uncaptioned))
    if not captions:
        report.images_kept = 0
        report.candidates_omitted = len(hits)
        report.images_uncaptioned = len(uncaptioned)
        return [], {}, {}

    t0 = time.monotonic()
    listing = "\n".join(f"[{i}] ({facets.get(i, 'place')}) {captions[i]}" for i in sorted(captions))
    report.model_calls += 1
    out = ctx.llm.generate(system=load_prompt("group_references"),
                           parts=[Text(context_md), Text("Captioned images:\n" + listing)],
                           schema=CurateOut)
    report.stage_timings.append(StageTiming("grouping", time.monotonic() - t0))

    directions: list[tuple[str, str, list[int]]] = []
    placed: set[int] = set()
    for d in out.directions:
        members = [i for i in d.images if i in captions and i not in placed]
        if not members:
            report.warn(f"direction {d.name.strip() or 'unnamed'!r} named no usable images; dropped")
            continue
        placed |= set(members)
        directions.append((d.name.strip() or "Untitled direction", d.why.strip(), members))

    # a captioned image the model forgot to file is still a usable reference
    unplaced = [i for i in captions if i not in placed]
    if unplaced:
        report.warn(f"{len(unplaced)} captioned image(s) were not placed in a direction; "
                    f"kept under {UNSORTED!r}")
        directions.append((UNSORTED, "", unplaced))
        placed |= set(unplaced)

    report.images_kept = len(placed)
    report.candidates_omitted = len(hits) - len(placed)
    report.kept = [(hits[i].title, report.origins.get(hits[i].image_url or hits[i].page_url, ""))
                   for _, _, members in directions for i in members]
    report.images_uncaptioned = len(uncaptioned)
    report.directions = [(name, len(members)) for name, _, members in directions]
    report.facets = Counter(facets[i] for i in placed if i in facets)
    return directions, captions, facets


# --- write-back -----------------------------------------------------------------------

def _store_image(ctx: RefCtx, project_id: str, hit: ImageHit,
                 report: ReferenceReport | None = None) -> Source | None:
    try:
        data = ctx.images.fetch(hit.preview_url)
    except Exception as exc:
        if report is not None:
            report.failed_downloads += 1
        log.warning("could not fetch %s: %s", hit.preview_url, exc)
        return None
    sid = hashlib.sha256(data).hexdigest()
    name = Path(hit.title).name or "reference"
    uri = source_uri(ctx.settings, project_id, sid, "original" + extension_for(name, hit.mime_type))
    ctx.blobs.put(uri, data, hit.mime_type)
    src = Source(id=sid, filename=name, mime_type=hit.mime_type, kind="image", doc_type="reference",
                 size_bytes=len(data), storage_path=uri, status="digested",
                 source_purpose="reference",
                 origin_url=hit.page_url or hit.image_url, license=hit.license, attribution=hit.attribution)
    # put_source_if_absent: if the same bytes were already stored (e.g. as an ingest-purpose
    # source or a user-uploaded reference), keep that record's purpose/status/provenance.
    # The blob write above is idempotent (content-addressed key). Return whichever record
    # exists so note provenance always points at the canonical stored Source.
    stored, _ = ctx.store.put_source_if_absent(project_id, src)
    return stored


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
        provenance=[Provenance(url=hit.url, title=hit.title)
                    for _, hit in verified if hit.url and hit.url.strip()][:8],
        origin=NoteOrigin(producer=PRODUCER, scope=scope_id, digest_version=version),
    )


def _write(ctx: RefCtx, project_id: str, scope_id: str, version: str, vocab: VocabularyOut,
           verified: list[tuple[str, TermHit]], hits: list[ImageHit],
           curated: tuple[list, dict, dict], report: ReferenceReport) -> None:
    t0 = time.monotonic()
    directions, captions, facets = curated

    # Phase 1: prepare all images first, before touching any notes
    prepared_notes: list[Note] = [_vocabulary_note(scope_id, version, vocab, verified)]
    for name, why, members in directions:
        for i in members:
            group = FACET_LABELS.get(facets.get(i, "place"), "Place")
            body = f"{name} \u2014 {captions[i]}" if name != UNSORTED else captions[i]
            src = _store_image(ctx, project_id, hits[i], report)
            if src is None:
                report.warn(f"image {i} could not be stored; skipped")
                continue
            key = hits[i].image_url or hits[i].page_url
            origins = report.retrieval_origins_map.get(key, [])
            prepared_notes.append(Note(
                kind="reference_image", body=body[:2000], owner_id=scope_id,
                author="agent", group=group,
                direction=name if name != UNSORTED else None,
                direction_rationale=why if why else None,
                provenance=[Provenance(source_id=src.id, url=hits[i].page_url or None,
                                       title=hits[i].title or None,
                                       retrieval_origins=origins)],
                origin=NoteOrigin(producer=PRODUCER, scope=scope_id, digest_version=version),
            ))

    # Phase 2: re-read current state to protect reviews made during generation
    existing = [n for n in ctx.store.notes_for_owners(project_id, [scope_id])
                if n.origin and n.origin.producer == PRODUCER and n.origin.scope == scope_id]
    stale_ids = [n.id for n in existing if n.status == "proposed"]
    expected_status = {n.id: n.status for n in existing}
    reviewed = {_key(n) for n in existing if n.status != "proposed"}
    fresh = [n for n in prepared_notes if _key(n) not in reviewed]

    # Phase 3: atomic replacement
    ctx.store.replace_notes(project_id, stale_ids, fresh, expected_status=expected_status)

    report.notes_replaced = len(stale_ids)
    report.notes_written = len(fresh)
    report.stage_timings.append(StageTiming("persistence", time.monotonic() - t0, includes_downloads=True))


def _key(n: Note) -> tuple:
    """A reviewed image keeps its verdict across re-runs; its identity is the file, not the caption."""
    src = next((p.source_id for p in n.provenance if p.source_id), None)
    return (n.kind, src) if n.kind == "reference_image" else (n.kind, " ".join(n.body.lower().split()))


# --- HTML export ------------------------------------------------------------------

def _safe_url(url: str | None) -> str | None:
    """Return url only if it uses a safe scheme; prevents javascript: etc in HTML output."""
    if url is None:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        return url if parsed.scheme in _SAFE_URL_SCHEMES else None
    except Exception:
        return None


def _mime_ext(mime_type: str) -> str:
    """Return a file extension matching the MIME type, not blindly .jpg."""
    ext = mimetypes.guess_extension(mime_type or "image/jpeg")
    return ext or ".jpg"


def strip_heading_prefix(caption: str, heading: str | None = None) -> str:
    """Remove repeated group-name prefix from displayed captions when the heading already supplies the name.

    Preserves legacy stored content in the store while presenting clean card text.
    """
    if not caption:
        return ""
    if heading and heading != UNSORTED:
        pattern = rf"^{re.escape(heading.strip())}\s*[\u2014\u2013\-\:]+\s*"
        if re.match(pattern, caption, flags=re.IGNORECASE):
            return re.sub(pattern, "", caption, count=1, flags=re.IGNORECASE).strip()
    return caption


def export_references_html(store: Store, blobs: Blobs, project_id: str, scope_ref: str,
                           out_dir: str | Path, *, confirmed_only: bool = False,
                           include_rejected: bool = False) -> tuple[Path, int, str | None]:
    """Export stored references as a self-contained HTML file with local images.

    Reads stored notes; does not repeat search or model calls.
    Returns ``(html_path, count, hint)`` where:
    - ``count`` is the number of references written to the file.
    - ``hint`` is ``None`` when count > 0, or a short sentence explaining why
      count is zero (no stored references vs. filtered to zero).
    """
    from .retrieval import get_context, _reference, _all_notes

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scope_id = resolve_scope(store, project_id, scope_ref)
    entity = store.get_entity(project_id, scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"export is per location; {scope_ref!r} resolved to a {type(entity).__name__.lower()}")

    pack = get_context(store, project_id, scope_id)
    all_notes = _all_notes(pack)
    sources_map = store.get_sources(project_id, [p.source_id for n in all_notes for p in n.provenance if p.source_id])

    # collect reference notes by status
    all_ref_notes = [n for n in all_notes
                     if n.kind == "reference_image"
                     and n.origin is not None and n.origin.producer == PRODUCER]
    ref_notes = list(all_ref_notes)
    if confirmed_only:
        ref_notes = [n for n in ref_notes if n.status == "confirmed"]
    elif not include_rejected:
        ref_notes = [n for n in ref_notes if n.status != "rejected"]

    # group by direction (or fallback for older notes without structured direction)
    groups: dict[str, list[tuple[Note, Source | None]]] = {}
    for n in ref_notes:
        direction = n.direction or n.group or "Unsorted"
        src = sources_map.get(n.provenance[0].source_id) if n.provenance else None
        groups.setdefault(direction, []).append((n, src))

    # download images locally
    local_images: dict[str, str] = {}  # source_id -> local relative path
    for n, src in [(n, s) for ns in groups.values() for n, s in ns]:
        if src is None:
            continue
        sid = src.id
        if sid in local_images:
            continue
        ext = _mime_ext(src.mime_type)
        local_path = images_dir / f"{sid}{ext}"
        if not local_path.exists():
            try:
                data = blobs.get(src.storage_path)
                local_path.write_bytes(data)
            except Exception as exc:
                log.warning("could not download image %s: %s", sid, exc)
                continue
        local_images[sid] = f"images/{sid}{ext}"

    # build HTML
    esc = html.escape
    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(entity.name)} \u2014 Visual References</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em; background: #fafafa; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ color: #555; margin-top: 1.5em; }}
.card {{ display: inline-block; vertical-align: top; width: 320px; margin: 12px;
         background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12);
         overflow: hidden; }}
.card .img-link {{ display: block; background: #f0f0f0; text-decoration: none; border-bottom: 1px solid #eee; }}
.card img {{ width: 100%; height: auto; max-height: 280px; object-fit: contain; display: block; }}
.card .placeholder {{ width: 100%; height: 200px; background: #ddd; display: flex;
                      align-items: center; justify-content: center; color: #999; font-size: 14px; }}
.card .body {{ padding: 10px; font-size: 13px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
          margin-right: 4px; }}
.badge-proposed {{ background: #fff3cd; color: #856404; }}
.badge-confirmed {{ background: #d4edda; color: #155724; }}
.badge-rejected {{ background: #f8d7da; color: #721c24; }}
.meta {{ color: #888; font-size: 11px; margin-top: 6px; }}
.guidance {{ color: #0056b3; font-style: italic; margin-top: 4px; }}
a {{ color: #0066cc; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{esc(entity.name)} \u2014 Visual References</h1>
"""]

    facet_order = ["Place", "Terrain", "Architecture", "Material"]

    def sort_key(name):
        try:
            return (0, facet_order.index(name))
        except ValueError:
            return (1, name)

    for direction in sorted(groups, key=sort_key):
        items = groups[direction]
        parts.append(f"<h2>{esc(direction)}</h2>\n")
        rationale = next((n.direction_rationale for n, _ in items if n.direction_rationale), None)
        if rationale:
            parts.append(f"<p><em>{esc(rationale)}</em></p>\n")
        for note, src in items:
            parts.append('<div class="card">\n')
            display_cap = strip_heading_prefix(note.body, direction)
            if src and src.id in local_images:
                img_path = local_images[src.id]
                parts.append(f'  <a class="img-link" href="{esc(img_path)}" target="_blank" rel="noopener" title="View full preview">\n'
                             f'    <img src="{esc(img_path)}" alt="{esc(display_cap[:80])}">\n'
                             f'  </a>\n')
            else:
                parts.append('  <div class="placeholder">Image unavailable</div>\n')

            parts.append('  <div class="body">\n')
            badge_cls = f"badge-{note.status}"
            parts.append(f'    <span class="badge {badge_cls}">{esc(note.status)}</span>')
            facet = note.group
            if facet:
                parts.append(f' <span class="badge">{esc(facet)}</span>')
            parts.append(f'\n    <p>{esc(display_cap[:500])}</p>\n')

            if note.guidance:
                parts.append(f'    <p class="guidance">Guidance: {esc(note.guidance)}</p>\n')

            meta_parts = []
            if src:
                safe = _safe_url(src.origin_url)
                if safe:
                    meta_parts.append(f'<a href="{esc(safe)}" target="_blank" rel="noopener">{esc(src.filename)}</a>')
                else:
                    meta_parts.append(esc(src.filename))
                if src.attribution:
                    meta_parts.append(esc(src.attribution))
                if src.license:
                    meta_parts.append(esc(src.license))
            meta_parts.append(f"<code>{esc(note.id)}</code>")
            parts.append(f'    <p class="meta">{" \u00b7 ".join(meta_parts)}</p>\n')
            parts.append('  </div>\n</div>\n')

    # empty-state message when no references matched
    if not ref_notes:
        if not all_ref_notes:
            hint = "No references have been generated for this location yet. Run 'harness-memory references' first."
        else:
            active = [n for n in all_ref_notes if n.status != "rejected"]
            if confirmed_only and not any(n.status == "confirmed" for n in all_ref_notes):
                hint = f"No confirmed references yet ({len(all_ref_notes)} proposed). Confirm some with 'harness-memory confirm'."
            elif not include_rejected and len(active) == 0:
                hint = f"All {len(all_ref_notes)} reference(s) have been rejected."
            else:
                hint = f"All references were excluded by the active filters ({len(all_ref_notes)} total stored)."
        parts.append(
            f'<p style="color:#888;font-style:italic;margin-top:2em">{esc(hint)}</p>\n'
        )
    else:
        hint = None

    parts.append("</body>\n</html>\n")

    html_path = out_dir / f"{re.sub(r'[^a-z0-9]+', '-', entity.name.lower()).strip('-')}-references.html"
    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path, len(ref_notes), hint

# --- user-uploaded location reference images ----------------------------------

def _attachment_note_id(location_id: str, source_id: str) -> str:
    """Deterministic note ID for a (location, source) attachment pair.

    Stable across processes and restarts. The same image bytes attached to the same
    location always produces the same ID, which allows put_note_if_absent to be
    idempotent and makes concurrent duplicate uploads safe.

    Different locations produce different IDs, so two locations that reference the
    same underlying image retain fully independent review state.
    """
    digest = hashlib.sha256(f"user-ref\x00{location_id}\x00{source_id}".encode()).hexdigest()
    return f"note_{digest[:12]}"


def upload_reference_image(
    store: Store,
    blobs: Blobs,
    settings: Settings,
    project_id: str,
    location_id: str,
    data: bytes,
    filename: str,
) -> tuple[Note, bool]:
    """Attach a user-uploaded image as a location reference note.

    Returns (note, created) where created=True means a new attachment note was written
    for this location. created=False means the same image was already attached to this
    location; the existing note (with its current status/revision/guidance) is returned
    unchanged.

    Does NOT call any LLM. No caption, direction, attribution, or licensing is
    fabricated. Body is set to the filename as the visible provenance label.

    Cross-purpose behavior:
    - If the same bytes are already stored as an ingest-purpose source, the existing
      source is used as-is. It remains ingest-eligible and appears in GET /sources.
    - If bytes are new, a reference-purpose source is created.
    - Source purpose is never changed by this function.

    Image validation (ValueError -> HTTP 400):
    - MIME must be in ACCEPTED_REFERENCE_MIMES (jpeg, png, webp).
    - Pillow must fully decode the image within MAX_REFERENCE_PIXELS.
    - Animated images are rejected.
    - Byte-count limit is enforced upstream by the route's _read_body_bounded.
    """
    from .ingest import source_uri   # local import to avoid circular at module load

    mime = sniff_mime(data, filename)
    validate_reference_image(data, mime)   # raises ValueError on bad content

    sid = hashlib.sha256(data).hexdigest()
    uri = source_uri(settings, project_id, sid, "original" + extension_for(filename, mime))
    blobs.put(uri, data, mime)

    ref_src = Source(
        id=sid, filename=filename, mime_type=mime,
        kind=kind_for(mime), doc_type="reference",
        size_bytes=len(data), storage_path=uri,
        status="digested",          # reference images don't go through ingest_source
        source_purpose="reference",
        origin_url=None,            # user-uploaded: no external origin
        license=None, attribution=None,
    )
    # Use the existing source record if the same bytes are already stored; never
    # overwrite an ingest-purpose source's purpose or provenance.
    stored_src, _ = store.put_source_if_absent(project_id, ref_src)

    note_id = _attachment_note_id(location_id, sid)
    new_note = Note(
        id=note_id,
        kind="reference_image",
        owner_id=location_id,
        author="user",
        body=filename,          # visible provenance label; not an AI-generated caption
        provenance=[Provenance(source_id=stored_src.id)],
        origin=NoteOrigin(
            producer="user",
            scope=location_id,
            digest_version="user-upload-v1",
        ),
        # group, direction left None: the pipeline may later assign a direction when the
        # image is included in a generated run; this upload alone does not imply approval
        # of any visual concept.
    )
    # Atomic create-if-absent: if the note already exists (reviewed, guided, or from a
    # concurrent upload), return it unchanged without any write. This is the only safe
    # path - put_notes would silently overwrite a reviewed note.
    note, created = store.put_note_if_absent(project_id, new_note)
    return note, created
