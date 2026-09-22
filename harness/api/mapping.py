"""Pure functions mapping harness.memory domain objects onto the DTOs in dto.py.

No writes happen here. Every read goes through harness.memory.retrieval/ports functions;
this module only reshapes what they return - it does not re-implement entity/containment
resolution (reuses retrieval._Graph, exactly like harness.memory.curate already does).
"""
from __future__ import annotations

from harness.memory.concepts import CORE_NOTES_CAVEAT, CURRENT_SNAPSHOT_SCHEMA_VERSION, approval_staleness_reasons
from harness.memory.curate import CorrectionResult
from harness.memory.models import ConceptGenerationJob, ApprovedReference, ConceptApproval, ConceptVersion, Location, Note, Provenance, Scene, Source
from harness.memory.ports import Store
from harness.memory.references import strip_heading_prefix
from harness.memory.resolver import natural_key
from harness.memory.retrieval import BRIEF_NOTE_KINDS, _Graph, _live, _reference, get_context, resolve_scope

from .dto import (
    GenerationCandidateOut, GenerationJobOut, GenerationPreviewOut, GenerationReferenceOut,
    ApprovalInheritedOut, ApprovalPackageOut, ApprovalPreviewOut,
    ApprovalStateOut, ApprovedReferenceOut, CitationOut, ConceptUploadOut, ConceptVersionListOut,
    ConceptVersionOut, IngestSourceResultOut, LocationDetail, LocationSceneRequirementOut, LocationSummary, NoteCorrectionResultOut, NoteOut,
    ReferenceListOut, ReferenceOut, ReviewResultOut, SceneRefOut, SceneRequirementOut, ScopedNoteOut, SourceOut, SourceSummaryOut,
)

_CORE_NOTE_KINDS = ("description", "tone")
_PHYSICAL_NOTE_KINDS = ("constraint",)


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

    notes_by_scene = {c.scene_id: c.notes for c in pack.conditional}
    roster_ids = {s.id for s in pack.scenes}

    return LocationDetail(
        id=pack.entity.id, name=pack.entity.name, aliases=pack.entity.aliases, status=pack.entity.status,
        ancestors=[a.name for a in pack.ancestors],
        scenes=[SceneRefOut(id=s.id, number=s.number, name=s.name) for s in pack.scenes],
        description_notes=notes_of("description"),
        constraint_notes=notes_of("constraint"),
        tone_notes=notes_of("tone"),
        scene_requirements=[_location_scene_out(s.id, s.number, s.name, True, notes_by_scene.get(s.id, []),
                                                pack.sources) for s in pack.scenes],
        out_of_roster_scene_requirements=[
            _location_scene_out(c.scene_id, None, c.label, False, c.notes, pack.sources)
            for c in pack.conditional if c.scene_id not in roster_ids
            if any(n.kind in BRIEF_NOTE_KINDS for n in c.notes)
        ],
        superseded_sources=pack.superseded_sources,
    )


def _location_scene_out(scene_id: str, number: str | None, heading: str | None, linked: bool,
                        notes: list[Note], sources: dict[str, str]) -> LocationSceneRequirementOut:
    """Only get_context's own owned conditional notes are surfaced (pack.conditional) -
    it never yields an ancestor's scene-scoped note (inherited notes are unconditional
    only, by retrieval's own rule), so owned is always True today; the field exists so
    a future retrieval change can't silently make an inherited note look editable."""
    return LocationSceneRequirementOut(
        scene_id=scene_id, number=number, heading=heading, linked=linked,
        notes=[ScopedNoteOut(**_note_out(n, sources).model_dump(), owner_id=n.owner_id, owned=True,
                             inherited_from=None, editable=n.kind in BRIEF_NOTE_KINDS)
               for n in notes if n.kind in BRIEF_NOTE_KINDS],
    )


def _note_out(n: Note, sources: dict[str, str]) -> NoteOut:
    return NoteOut(id=n.id, kind=n.kind, body=n.body, status=n.status, revision=n.revision,
                   citations=[_citation(p, sources) for p in n.provenance],
                   scene_id=n.applicability.scene_id,
                   include_descendants=n.applicability.include_descendants)


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


# --- concept versions -------------------------------------------------------------------

def _concept_version_out(project_id: str, v: ConceptVersion, approved_id: str | None) -> ConceptVersionOut:
    return ConceptVersionOut(
        id=v.id, image=f"/projects/{project_id}/concepts/{v.id}/image",
        filename=v.filename, created_at=v.created_at, author=v.author,
        approved=(v.id == approved_id),
        promoted_from_note_id=v.promoted_from_note_id,
        promoted_from_note_revision=v.promoted_from_note_revision,
        generation_job_id=v.generation_job_id,
    )


def concept_version_list(store: Store, project_id: str, location_id: str) -> ConceptVersionListOut:
    """All uploaded concept versions for a location, plus which one (if any) is
    currently approved - selecting a candidate in the UI is purely client-side state,
    so there is nothing else here to persist or read back."""
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; concept versions are per location")

    current = store.get_current_approval(project_id, scope_id)
    approved_id = current.concept_version_id if current else None
    versions = store.list_concept_versions(project_id, scope_id)
    return ConceptVersionListOut(
        location_id=scope_id, approved_version_id=approved_id,
        versions=[_concept_version_out(project_id, v, approved_id) for v in versions],
    )


def concept_upload_out(store: Store, project_id: str, version: ConceptVersion, *, created: bool) -> ConceptUploadOut:
    current = store.get_current_approval(project_id, version.location_id)
    approved_id = current.concept_version_id if current else None
    return ConceptUploadOut(
        id=version.id, image=f"/projects/{project_id}/concepts/{version.id}/image",
        filename=version.filename, created=created, approved=(version.id == approved_id),
        promoted_from_note_id=version.promoted_from_note_id,
        promoted_from_note_revision=version.promoted_from_note_revision,
        generation_job_id=version.generation_job_id,
    )


# --- concept approval ---------------------------------------------------------------------

def _all_approval_notes(pkg) -> list[Note]:
    """pkg is a ConceptApproval or concepts.ApprovalSnapshot - the field is named
    brief_conditional on ConceptApproval and scene_requirements on ApprovalSnapshot
    (see concepts.ApprovalSnapshot), both carrying the same per-scene shape."""
    scenes = getattr(pkg, "scene_requirements", None)
    if scenes is None:
        scenes = pkg.brief_conditional
    return (pkg.brief_notes + [n for c in scenes for n in c.notes]
           + [n for i in pkg.brief_inherited for n in i.notes])


def _approval_sources(store: Store, project_id: str, pkg) -> dict[str, str]:
    notes = _all_approval_notes(pkg)
    source_ids = [p.source_id for n in notes for p in n.provenance if p.source_id]
    sources = store.get_sources(project_id, source_ids)
    return {sid: s.filename for sid, s in sources.items()}


def _core_and_physical(notes: list[Note]) -> tuple[list[Note], list[Note]]:
    """Splits a flat unconditional brief-note list into "core description" (kind in
    description/tone) and "physical/set-dressing" (kind == constraint) - pure
    presentation reuse of the existing kind field, no reclassification. See
    concepts.CORE_NOTES_CAVEAT for the one remaining, honestly-disclosed ambiguity this
    split does not resolve."""
    core = [n for n in notes if n.kind in _CORE_NOTE_KINDS]
    physical = [n for n in notes if n.kind in _PHYSICAL_NOTE_KINDS]
    return core, physical


def _scene_requirement_out(scenes, sources: dict[str, str]) -> list[SceneRequirementOut]:
    return [SceneRequirementOut(scene_id=s.scene_id, number=s.number, heading=s.heading,
                                notes=[_note_out(n, sources) for n in s.notes]) for s in scenes]


def _approval_inherited_out(inhs, sources: dict[str, str]) -> list[ApprovalInheritedOut]:
    return [ApprovalInheritedOut(entity_id=i.entity_id, name=i.name,
                                 notes=[_note_out(n, sources) for n in i.notes]) for i in inhs]


def _approved_reference_out(project_id: str, r: ApprovedReference) -> ApprovedReferenceOut:
    return ApprovedReferenceOut(
        note_id=r.note_id, revision=r.revision, status=r.status, guidance=r.guidance,
        direction=r.direction, caption=r.caption,
        image=f"/projects/{project_id}/references/{r.note_id}/image",
    )


def approval_package_out(store: Store, project_id: str, approval: ConceptApproval) -> ApprovalPackageOut:
    sources = _approval_sources(store, project_id, approval)
    core, physical = _core_and_physical(approval.brief_notes)
    return ApprovalPackageOut(
        id=approval.id, location_id=approval.location_id, revision=approval.revision,
        concept_version_id=approval.concept_version_id,
        concept_image=f"/projects/{project_id}/concepts/{approval.concept_version_id}/image",
        concept_filename=approval.concept_filename, depiction_label=approval.depiction_label,
        core_notes=[_note_out(n, sources) for n in core], core_notes_caveat=CORE_NOTES_CAVEAT,
        physical_notes=[_note_out(n, sources) for n in physical],
        scene_requirements=_scene_requirement_out(approval.brief_conditional, sources),
        scene_coverage_complete=(approval.snapshot_schema_version >= CURRENT_SNAPSHOT_SCHEMA_VERSION),
        brief_inherited=_approval_inherited_out(approval.brief_inherited, sources),
        brief_ancestors=approval.brief_ancestors, superseded_sources=approval.superseded_sources,
        references=[_approved_reference_out(project_id, r) for r in approval.references],
        context_token=approval.context_token, locked_by=approval.locked_by, locked_at=approval.locked_at,
    )


def approval_preview_out(store: Store, project_id: str, snapshot) -> ApprovalPreviewOut:
    """snapshot is a concepts.ApprovalSnapshot - see concepts.build_approval_snapshot.
    Always complete scene coverage (a preview is always computed fresh against current
    retrieval - there is no legacy/partial state for something that isn't persisted)."""
    sources = _approval_sources(store, project_id, snapshot)
    core, physical = _core_and_physical(snapshot.brief_notes)
    return ApprovalPreviewOut(
        location_id=snapshot.location_id, concept_version_id=snapshot.concept_version.id,
        concept_image=f"/projects/{project_id}/concepts/{snapshot.concept_version.id}/image",
        concept_filename=snapshot.concept_version.filename, depiction_label=snapshot.depiction_label,
        core_notes=[_note_out(n, sources) for n in core], core_notes_caveat=CORE_NOTES_CAVEAT,
        physical_notes=[_note_out(n, sources) for n in physical],
        scene_requirements=_scene_requirement_out(snapshot.scene_requirements, sources),
        brief_inherited=_approval_inherited_out(snapshot.brief_inherited, sources),
        brief_ancestors=snapshot.brief_ancestors, superseded_sources=snapshot.superseded_sources,
        references=[_approved_reference_out(project_id, r) for r in snapshot.references],
        context_token=snapshot.context_token,
    )


def approval_state(store: Store, project_id: str, location_id: str) -> ApprovalStateOut:
    """GET .../approval: the current approved package (or none) plus its revision (for
    the next lock's expectedRevision) and whether live inputs have since diverged from
    it. This is also the retrieval boundary a later "Send to 3D blockout" action would
    read from - nothing here calls out to anything, or marks a package as sent.

    Always returns the stored package even when it is stale, and even when every
    reference it named has since been rejected/removed or the location itself was
    later rejected/merged - approval_staleness_reasons never raises, so a diverged
    package is reported (isStale + staleReasons), never a 500 or a missing approval.
    """
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; concept approval is per location")

    current = store.get_current_approval(project_id, scope_id)
    if current is None:
        return ApprovalStateOut(location_id=scope_id, revision=0, approval=None, is_stale=False)

    reasons = approval_staleness_reasons(store, project_id, current)
    return ApprovalStateOut(location_id=scope_id, revision=current.revision,
                            approval=approval_package_out(store, project_id, current),
                            is_stale=bool(reasons), stale_reasons=reasons)


# --- note correction / splitting -------------------------------------------------------

def note_correction_result_out(store: Store, project_id: str, result: CorrectionResult) -> NoteCorrectionResultOut:
    all_notes = [result.original, *result.new_notes]
    source_ids = [p.source_id for n in all_notes for p in n.provenance if p.source_id]
    sources = {sid: s.filename for sid, s in store.get_sources(project_id, source_ids).items()}
    return NoteCorrectionResultOut(
        original=ReviewResultOut(
            id=result.original.id, status=result.original.status, revision=result.original.revision,
            reviewed_by=result.original.reviewed_by, reviewed_at=result.original.reviewed_at,
            guidance=result.original.guidance, review_reason=result.original.review_reason,
        ),
        new_notes=[_note_out(n, sources) for n in result.new_notes],
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
    """Sources eligible for ingestion - excludes reference-only and concept-only assets.

    Uses Source.is_ingest_eligible (which reads effective_purpose) as the exclusion
    signal. This is backward-compatible:
    - Legacy Wikimedia-fetched sources (source_purpose=None, origin_url set) resolve
      to effective_purpose="reference" and are excluded, same as before.
    - Legacy screenplay/notes sources (source_purpose=None, origin_url=None) resolve
      to effective_purpose="ingest" and are included, same as before.
    - New sources have an explicit source_purpose="ingest"|"reference"|"both"|"concept";
      only "reference" and "concept" are excluded.
    - "both" sources started as "reference" or "concept" and were explicitly promoted
      via POST /sources; they are included here and eligible for ingest either way.

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


# --- base-location concept generation -------------------------------------------------------

def generation_preview_out(project_id: str, location_id: str, prepared, provider_name: str) -> GenerationPreviewOut:
    snap = prepared.snapshot
    return GenerationPreviewOut(
        location_id=location_id, workflow_version=snap["workflowVersion"], prompt_version=snap["promptVersion"],
        provider=provider_name, model=snap["settings"]["model"], prompt=prepared.prompt,
        references=[GenerationReferenceOut(
            position=r["position"], note_id=r["noteId"], revision=r["revision"], source_id=r["sourceId"],
            guidance=r["guidance"], direction=r["direction"],
            image=f"/projects/{project_id}/references/{r['noteId']}/image") for r in snap["references"]],
        settings=snap["settings"], snapshot=snap, context_token=prepared.context_token,
    )


def generation_job_out(job: ConceptGenerationJob, *, resumable: bool) -> GenerationJobOut:
    reused = set(job.reused_candidate_ids)
    return GenerationJobOut(
        id=job.id, location_id=job.location_id, state=job.state,
        terminal=job.state in ("succeeded", "failed"), resumable=resumable,
        needs_attention=job.state == "submission_unknown",
        provider=job.provider, model=job.model, workflow_version=job.workflow_version,
        prompt_version=job.prompt_version, provider_generation_id=job.provider_generation_id,
        submit_attempts=job.submit_attempts, prompt=job.prompt, context_token=job.context_token,
        input_snapshot=job.input_snapshot,
        candidates=[GenerationCandidateOut(id=c, image=f"/projects/{job.project_id}/concepts/{c}/image",
                                           reused=c in reused) for c in job.candidate_ids],
        failure_stage=job.failure_stage, failure_code=job.failure_code, error=job.error,
        attempt_history=job.attempt_history, resolutions=job.resolutions,
        submit_started_at=job.submit_started_at,
        created_at=job.created_at, updated_at=job.updated_at, submitted_at=job.submitted_at,
        provider_completed_at=job.provider_completed_at, finished_at=job.finished_at,
    )
