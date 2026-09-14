"""Caption-pass evaluation over a fixed candidate set.

Live reference runs cannot tell a prompt change from sampling noise: the vocabulary, the
candidates and the verdicts all vary between runs of identical code. This freezes the
candidates and the context pack in a fixture, then runs only the caption pass over them
N times at production settings.

Two kinds of call, kept apart on purpose:
  - production runs use the shipped prompt, schema and batching unchanged. Kept, dropped
    and flipped counts come only from these.
  - one diagnostic run appends DIAGNOSTIC to ask for a verdict and a reason per image.
    Asking for reasons changes the verdicts, so its reasons are shown beside the
    production results, and the report counts how often it disagrees with them.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .ingest import load_prompt
from .models import Location
from .ports import LLM, ImageHit, Images, OutputTruncated
from .references import (CAPTION_BATCH, RefCtx, ReferenceReport, _retrieve, _verify, _vocabulary,
                         caption_all, caption_parts, references_version)
from .retrieval import get_context, render_context_md, resolve_scope
from .schemas import Out

FIXTURE_VERSION = 1

DIAGNOSTIC = """

## Diagnostic mode
For this run only, return a verdict for EVERY image, including the ones you would drop. keep is whether the image would appear in your normal captions output. caption is the caption you would write if kept, otherwise empty. reason explains the decision in one or two sentences. rule_quoted quotes, word for word, the instruction above that decided it."""


class Verdict(Out):
    index: int
    keep: bool
    facet: Literal["place", "architecture", "material", "terrain"] = "place"
    caption: str = ""
    reason: str = ""
    rule_quoted: str = ""


class VerdictsOut(Out):
    verdicts: list[Verdict] = []


# --- fixture --------------------------------------------------------------------------

@dataclass
class Fixture:
    location: str
    context_md: str
    hits: list[ImageHit]
    origins: list[str] = field(default_factory=list)     # which search found each hit
    included: list[str] = field(default_factory=list)    # titles appended by hand, not retrieved
    captured_at: str = ""
    references_version: str = ""


def save_fixture(path: str | Path, fx: Fixture) -> None:
    data = {"fixture_version": FIXTURE_VERSION, **asdict(fx)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_fixture(path: str | Path) -> Fixture:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.pop("fixture_version", None) != FIXTURE_VERSION:
        raise ValueError(f"{path}: unsupported fixture version")
    data["hits"] = [ImageHit(**h) for h in data["hits"]]
    return Fixture(**data)


def capture(ctx: RefCtx, project_id: str, scope_ref: str, *, include: list[str] = (),
            per_term: int = 6, max_images: int = 32) -> Fixture:
    """Vocabulary, verification and retrieval exactly as a references run does them; no captioning.

    include names Commons files to append, so images a question is about are in the set
    whatever the vocabulary happened to produce this time.
    """
    scope_id = resolve_scope(ctx.store, project_id, scope_ref)
    entity = ctx.store.get_entity(project_id, scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"references are per location; {scope_ref!r} resolved to a {type(entity).__name__.lower()}")
    context_md = render_context_md(get_context(ctx.store, project_id, scope_id))
    report = ReferenceReport(scope_id=scope_id, location=entity.name)
    vocab = _vocabulary(ctx, context_md, report)
    hits = _retrieve(ctx, vocab, _verify(ctx, vocab, report), per_term, max_images, report)
    origins = [report.origins.get(h.image_url or h.page_url, "") for h in hits]

    included = []
    titles = {h.title for h in hits}
    for title in include:
        name = title.removeprefix("File:")
        if name in titles:
            continue
        hit = ctx.images.get_file(name)
        if hit is None:
            raise LookupError(f"Commons file not found or not licensed for reuse: {name!r}")
        hits.append(hit)
        origins.append("included by hand")
        included.append(name)
    return Fixture(location=entity.name, context_md=context_md, hits=hits, origins=origins,
                   included=included, captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   references_version=references_version(ctx.llm))


class CachedImages:
    """Fetches through a local disk cache, so repeat runs see identical bytes without refetching."""

    def __init__(self, inner: Images, cache_dir: str | Path):
        self.inner, self.dir = inner, Path(cache_dir)

    def fetch(self, url: str) -> bytes:
        path = self.dir / hashlib.sha256(url.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        data = self.inner.fetch(url)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def verify_term(self, term):
        return self.inner.verify_term(term)

    def search_images(self, term, limit, region=None, region_title=None):
        return self.inner.search_images(term, limit, region=region, region_title=region_title)


# --- runs -----------------------------------------------------------------------------

Kept = dict[int, tuple[str, str]]       # index -> (facet, caption)


def production_run(llm: LLM, images: Images, fx: Fixture) -> tuple[Kept, list[str]]:
    """The shipped caption pass, unchanged: same prompt, schema, batches and truncation handling."""
    ctx = RefCtx(store=None, blobs=None, llm=llm, images=images, settings=None)
    report = ReferenceReport(scope_id="", location=fx.location)
    captions, facets = caption_all(ctx, fx.context_md, fx.hits, report)
    return {i: (facets.get(i, "place"), c) for i, c in captions.items()}, report.warnings


def diagnostic_run(llm: LLM, images: Images, fx: Fixture) -> tuple[dict[int, Verdict], list[str]]:
    """The production prompt plus DIAGNOSTIC, batched the same way, asking for a verdict on every image."""
    system = load_prompt("caption_references") + DIAGNOSTIC
    report = ReferenceReport(scope_id="", location=fx.location)
    verdicts: dict[int, Verdict] = {}

    def call(indices: list[int]) -> None:
        parts, present = caption_parts(images, fx.context_md, fx.hits, indices, report)
        if not present:
            return
        try:
            out = llm.generate(system=system, parts=parts, schema=VerdictsOut)
        except OutputTruncated:
            if len(present) == 1:
                report.warn(f"image {present[0]} could not be judged within the output budget")
                return
            mid = len(present) // 2
            call(present[:mid])
            call(present[mid:])
            return
        keep = set(present)
        verdicts.update({v.index: v for v in out.verdicts if v.index in keep})

    for start in range(0, len(fx.hits), CAPTION_BATCH):
        call(list(range(start, min(start + CAPTION_BATCH, len(fx.hits)))))
    return verdicts, report.warnings


# --- aggregation ----------------------------------------------------------------------

@dataclass
class ImageResult:
    index: int
    title: str
    origin: str
    kept_runs: int
    runs: int
    facets: Counter
    captions: list[str]
    verdict: Verdict | None = None

    @property
    def status(self) -> str:
        if self.kept_runs == self.runs:
            return "kept"
        return "dropped" if self.kept_runs == 0 else "flipped"

    @property
    def majority_keep(self) -> bool | None:
        if self.kept_runs * 2 == self.runs:
            return None
        return self.kept_runs * 2 > self.runs


@dataclass
class EvalReport:
    location: str
    runs: int
    images: list[ImageResult]
    diagnosed: bool

    def by_status(self, status: str) -> list[ImageResult]:
        return [r for r in self.images if r.status == status]

    @property
    def facet_flips(self) -> list[ImageResult]:
        """Kept every run but filed under different facets: noise even when the verdict is stable."""
        return [r for r in self.by_status("kept") if len(r.facets) > 1]

    @property
    def disagreements(self) -> list[ImageResult]:
        return [r for r in self.images if r.verdict is not None and r.majority_keep is not None
                and r.verdict.keep != r.majority_keep]

    def headline(self) -> str:
        n = len(self.images)
        line = (f"{len(self.by_status('flipped'))} of {n} images flipped across {self.runs} runs "
                f"({len(self.by_status('kept'))} always kept, {len(self.by_status('dropped'))} always dropped; "
                f"facet changed on {len(self.facet_flips)} always-kept)")
        if self.diagnosed:
            judged = sum(r.verdict is not None and r.majority_keep is not None for r in self.images)
            line += f"; diagnostic disagreed with the production majority on {len(self.disagreements)} of {judged}"
        return line


def aggregate(fx: Fixture, runs: list[Kept], verdicts: dict[int, Verdict] | None = None) -> EvalReport:
    images = []
    for i, hit in enumerate(fx.hits):
        kept = [run[i] for run in runs if i in run]
        images.append(ImageResult(
            index=i, title=hit.title, origin=fx.origins[i] if i < len(fx.origins) else "",
            kept_runs=len(kept), runs=len(runs), facets=Counter(f for f, _ in kept),
            captions=list(dict.fromkeys(c for _, c in kept)),
            verdict=(verdicts or {}).get(i)))
    return EvalReport(location=fx.location, runs=len(runs), images=images, diagnosed=verdicts is not None)


def render(report: EvalReport) -> str:
    lines = [f"# Caption eval: {report.location}", "", f"HEADLINE: {report.headline()}", ""]
    order = {"flipped": 0, "kept": 1, "dropped": 2}
    for r in sorted(report.images, key=lambda r: (order[r.status], r.index)):
        facets = ", ".join(f"{f} {n}" for f, n in r.facets.most_common()) or "-"
        flag = "  [facet changed]" if r in report.facet_flips else ""
        lines.append(f"[{r.index}] {r.status.upper()} {r.kept_runs}/{r.runs}  facets: {facets}{flag}")
        lines.append(f"     {r.title}  <- {r.origin}")
        for c in r.captions:
            lines.append(f"     caption: {c}")
        if r.verdict is not None:
            v = r.verdict
            mark = "  [DISAGREES with production majority]" if r in report.disagreements else ""
            lines.append(f"     diagnostic: {'keep' if v.keep else 'drop'} ({v.facet}){mark}: {v.reason}")
            if v.rule_quoted:
                lines.append(f"     rule: {v.rule_quoted}")
        lines.append("")
    return "\n".join(lines)


def run_eval(llm: LLM, images: Images, fx: Fixture, *, n: int = 3, reasons: bool = True
             ) -> tuple[EvalReport, list[str]]:
    warnings: list[str] = []
    runs = []
    for _ in range(n):
        kept, w = production_run(llm, images, fx)
        runs.append(kept)
        warnings += w
    verdicts = None
    if reasons:
        verdicts, w = diagnostic_run(llm, images, fx)
        warnings += w
    return aggregate(fx, runs, verdicts), warnings


def prompt_fingerprint() -> str:
    return hashlib.sha256(load_prompt("caption_references").encode()).hexdigest()[:8]
