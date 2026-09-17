"""Pure functions mapping harness.memory domain objects onto the DTOs in dto.py.

No writes happen here. Every read goes through harness.memory.retrieval/ports functions;
this module only reshapes what they return - it does not re-implement entity/containment
resolution (reuses retrieval._Graph, exactly like harness.memory.curate already does).
"""
from __future__ import annotations

from harness.memory.models import Location, Note, Provenance, Scene, Source
from harness.memory.ports import Store
from harness.memory.references import strip_heading_prefix
from harness.memory.resolver import natural_key
from harness.memory.retrieval import _Graph, _live, _reference, get_context, resolve_scope

from .dto import (
    CitationOut, IngestSourceResultOut, LocationDetail, LocationSummary, NoteOut,
    ReferenceListOut, ReferenceOut, SceneRefOut, SourceOut, SourceSummaryOut,
)


# --- locations --------------------------------------------------------------------------

def location_summaries(store: Store, project_id: str) -> list[LocationSummary]:
    """Sidebar nav data for every live location, in two store reads total.

    Deliberately does not build a ContextPack per location (get_context resolves
    inheritance, conditional notes, sources, and reference images - far more than a
    nav list needs); with N locations that would be N times the reads this does once.
    """
    entities = store.list_entities(project_id)
    graph = _Graph(entities)
    notes = store.list_notes(project_id)
    counts: dict[str, int] = {}
    for n in notes:
        if n.status != "rejected":
            counts[n.owner_id] = counts.get(n.owner_id, 0) + 1

    scenes = [e for e in entities if isinstance(e, Scene) and _live(e)]
    locations = [e for e in entities if isinstance(e, Location) and _live(e)]

    out: list[LocationSummary] = []
    for loc in sorted(locations, key=lambda l: l.name.lower()):
        scene_numbers = sorted(
            {s.number for s in scenes if s.number
             and any((t := graph.target(lid)) is not None and t.id == loc.id for lid in s.location_ids)},
            key=natural_key,
        )
        ancestors = graph.ancestors(loc.id)
        note_count = sum(counts.get(i, 0) for i in graph.own_ids(loc.id))
        out.append(LocationSummary(
            id=loc.id, name=loc.name, aliases=loc.aliases, status=loc.status,
            parent_name=ancestors[0].name if ancestors else None,
            scene_numbers=scene_numbers, note_count=note_count,
        ))
    return out


def location_detail(store: Store, project_id: str, location_id: str) -> LocationDetail:
    """Read-only brief-tab data: extracted notes grouped by kind, nothing approved/editable
    yet (Phase 1 has no working-brief model - see API_CONTRACT.md)."""
    pack = get_context(store, project_id, location_id, include_proposed=True)
    if not isinstance(pack.entity, Location):
        raise ValueError(f"{location_id!r} is a scene; location details are per location")

    def notes_of(kind: str) -> list[NoteOut]:
        return [_note_out(n, pack.sources) for n in pack.notes if n.kind == kind]

    return LocationDetail(
        id=pack.entity.id, name=pack.entity.name, aliases=pack.entity.aliases, status=pack.entity.status,
        ancestors=[a.name for a in pack.ancestors],
        scenes=[SceneRefOut(id=s.id, number=s.number, name=s.name) for s in pack.scenes],
        description_notes=notes_of("description"),
        constraint_notes=notes_of("constraint"),
        tone_notes=notes_of("tone"),
        superseded_sources=pack.superseded_sources,
    )


def _note_out(n: Note, sources: dict[str, str]) -> NoteOut:
    return NoteOut(id=n.id, kind=n.kind, body=n.body, status=n.status, revision=n.revision,
                   citations=[_citation(p, sources) for p in n.provenance])


def _citation(p: Provenance, sources: dict[str, str]) -> CitationOut:
    return CitationOut(source_id=p.source_id, filename=sources.get(p.source_id or ""),
                       page=p.page, quote=p.quote, url=p.url, title=p.title)


# --- references ---------------------------------------------------------------------------

def reference_list(store: Store, project_id: str, location_id: str, *,
                   include_rejected: bool) -> ReferenceListOut:
    """Owned + inherited reference_image notes, distinguished, with review state intact.

    Deliberately does not go through retrieval.get_references_context (built for feeding
    the pipeline its own prior output, not for a review UI: it hides proposed images) nor
    retrieval.get_context's pack.reference_images (owned only, and never includes rejected
    notes regardless of include_proposed) - this needs both owned and inherited, and an
    explicit choice about rejected notes, so it reads notes_for_owners directly.
    """
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; references are per location")

    statuses = {"proposed", "confirmed"} | ({"rejected"} if include_rejected else set())

    owned_notes = [n for n in store.notes_for_owners(project_id, graph.own_ids(scope_id))
                   if n.kind == "reference_image" and n.status in statuses]

    inherited_pairs: list[tuple[Location, Note]] = []
    for parent in graph.ancestors(scope_id):
        for n in store.notes_for_owners(project_id, graph.own_ids(parent.id)):
            if (n.kind == "reference_image" and n.status in statuses
                    and n.applicability.include_descendants and n.applicability.scene_id is None):
                inherited_pairs.append((parent, n))

    all_notes = owned_notes + [n for _, n in inherited_pairs]
    sources = store.get_sources(project_id, [p.source_id for n in all_notes for p in n.provenance if p.source_id])

    out: list[ReferenceOut] = []
    for n in owned_notes:
        if (ref := _reference(n, sources)) is not None:
            out.append(_reference_out(ref, n, project_id, owned=True, inherited_from=None))
    for parent, n in inherited_pairs:
        if (ref := _reference(n, sources)) is not None:
            out.append(_reference_out(ref, n, project_id, owned=False, inherited_from=parent.name))

    return ReferenceListOut(location_id=scope_id, references=out)


def _reference_out(ref, note: Note, project_id: str, *, owned: bool, inherited_from: str | None) -> ReferenceOut:
    caption = strip_heading_prefix(ref.caption, ref.direction)
    credit_bits = [b for b in (ref.attribution, ref.license) if b]
    return ReferenceOut(
        id=ref.note_id,
        title=ref.direction or (caption[:60].rstrip() or "Reference"),
        image=f"/projects/{project_id}/references/{ref.note_id}/image",
        category=ref.group or "Uploaded",
        facet=ref.group,
        reason=caption,
        direction=ref.direction,
        direction_rationale=ref.direction_rationale,
        source=ref.origin_url,
        credit=" · ".join(credit_bits) or None,
        license=ref.license,
        attribution=ref.attribution,
        selected=(ref.status == "confirmed"),
        status=ref.status,
        guidance=ref.guidance or "",
        owned=owned,
        inherited_from=inherited_from,
        revision=note.revision,
    )


# --- misc -----------------------------------------------------------------------------

def find_note(store: Store, project_id: str, note_id: str) -> Note | None:
    """O(n) over the project's notes - fine at the scale a single film project reaches
    (a few hundred notes; the reference export tooling already assumes this same scale)."""
    for n in store.list_notes(project_id):
        if n.id == note_id:
            return n
    return None


# --- sources / ingest -------------------------------------------------------------------

def source_out(src: Source, *, created: bool) -> SourceOut:
    return SourceOut(id=src.id, filename=src.filename, mime_type=src.mime_type,
                     size_bytes=src.size_bytes, status=src.status, created=created, error=src.error)


def uploaded_sources(store: Store, project_id: str) -> list[SourceSummaryOut]:
    """Sources eligible for ingestion - excludes reference-only assets.

    Uses Source.is_ingest_eligible (which reads effective_purpose) as the exclusion
    signal. This is backward-compatible:
    - Legacy Wikimedia-fetched sources (source_purpose=None, origin_url set) resolve
      to effective_purpose="reference" and are excluded, same as before.
    - Legacy screenplay/notes sources (source_purpose=None, origin_url=None) resolve
      to effective_purpose="ingest" and are included, same as before.
    - New sources have an explicit source_purpose="ingest"|"reference"|"both"; only
      "reference" is excluded.
    - "both" sources were reference-only images explicitly promoted via POST /sources;
      they are included here and eligible for ingest.

    doc_type alone would not be reliable: a genuinely uploaded lookbook/reference
    document can be classified doc_type="reference" by ingestion, which would wrongly
    hide it. is_ingest_eligible never makes that mistake.

    Ordered by (created_at, filename, id) for a deterministic response - neither
    MemoryStore's dict iteration nor Firestore's unordered stream() guarantee any
    order on their own.
    """
    sources = [s for s in store.list_sources(project_id) if s.is_ingest_eligible]
    sources.sort(key=lambda s: (s.created_at, s.filename, s.id))
    return [
        SourceSummaryOut(id=s.id, filename=s.filename, mime_type=s.mime_type,
                         size_bytes=s.size_bytes, status=s.status, error=s.error)
        for s in sources
    ]


def ingest_source_result_out(report: dict) -> IngestSourceResultOut:
    """report is dataclasses.asdict(IngestReport) - see JobRunner._report_to_dict."""
    return IngestSourceResultOut(
        source_id=report["source_id"], filename=report["filename"], status=report["status"],
        doc_type=report.get("doc_type"), entities_created=report.get("entities_created", 0),
        notes_written=report.get("notes_written", 0), notes_replaced=report.get("notes_replaced", 0),
        notes_reused=report.get("notes_reused", 0), warnings=report.get("warnings", []),
        error=report.get("error"),
    )
