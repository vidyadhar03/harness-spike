"""Human review actions over memory: merging duplicate entities, note review, and note
correction/splitting.

Ingest deliberately creates a new entity when it cannot confidently match an existing one,
so visible duplicates are expected and a person resolves them here. Notes are never
rewritten: retrieval follows merge chains, so a merge stays reversible by editing one field.
The same "never edit in place" rule applies to a note's own reviewed assertion
(body/owner/applicability) - see correct_note below, and models.Note's module docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    PROJECT_SCOPE, Applicability, Containment, Location, Note, NoteOrigin, RejectReason,
    ReviewStatus, Scene, utcnow,
)
from .ports import EntityDoc, Store
from .resolver import norm
from .retrieval import BRIEF_NOTE_KINDS, _Graph, _live, resolve_scope


@dataclass
class MergeResult:
    source: EntityDoc
    target: EntityDoc
    aliases_added: list[str] = field(default_factory=list)
    notes_moved: int = 0
    applied: bool = False

    def summary(self) -> str:
        verb = "merged" if self.applied else "would merge"
        line = (f"{verb} {self.source.name} `{self.source.id}` into "
                f"{self.target.name} `{self.target.id}`; "
                f"{self.notes_moved} note(s) now resolve to the target")
        if self.aliases_added:
            line += f"\n  aliases added: {', '.join(self.aliases_added)}"
        return line


def merge_entities(store: Store, project_id: str, source_ref: str, target_ref: str,
                   *, dry_run: bool = False) -> MergeResult:
    graph = _Graph(store.list_entities(project_id))
    source_id = resolve_scope(store, project_id, source_ref, graph)
    target_id = resolve_scope(store, project_id, target_ref, graph)

    if source_id == target_id:
        raise ValueError(f"{source_ref!r} and {target_ref!r} are already the same entity ({source_id})")
    source, target = graph.by_id[source_id], graph.by_id[target_id]
    if type(source) is not type(target):
        raise ValueError(f"cannot merge a {_kind(source)} into a {_kind(target)}")
    if target_id in graph.own_ids(source_id):
        raise ValueError(f"{target.name} already merges into {source.name}; merging would make a cycle")

    own = graph.own_ids(source_id)
    notes = store.notes_for_owners(project_id, own) if own else []

    existing = {norm(target.name)} | {norm(a) for a in target.aliases}
    added: list[str] = []
    for name in (source.name, *source.aliases):
        key = norm(name)
        if key and key not in existing:
            existing.add(key)
            added.append(name.strip())

    result = MergeResult(source=source, target=target, aliases_added=added, notes_moved=len(notes))
    if dry_run:
        return result

    updates: list[EntityDoc] = [source.touch(status="merged", merged_into=target_id)]
    if added:
        updates.append(target.touch(aliases=[*target.aliases, *added]))
    store.put_entities(project_id, updates)
    result.applied = True
    return result


def _kind(entity: EntityDoc) -> str:
    return "location" if isinstance(entity, Location) else "scene"


# --- note review ---------------------------------------------------------------------

@dataclass
class ReviewResult:
    note: Note
    action: str

    def summary(self) -> str:
        head = " ".join(self.note.body.split())[:70]
        extra = f" (duplicate of {self.note.duplicate_of})" if self.note.duplicate_of else ""
        return f"{self.action} `{self.note.id}`{extra}: {head}"


def _get_note(store: Store, project_id: str, note_id: str) -> Note:
    for n in store.list_notes(project_id):
        if n.id == note_id:
            return n
    raise LookupError(f"no note with id {note_id!r}")


def review_note(store: Store, project_id: str, note_id: str, decision: ReviewStatus, *,
                reviewer: str, reason: RejectReason | None = None,
                duplicate_of: str | None = None,
                guidance: str | None = None,
                expected_revision: int | None = None,
                expected_status: ReviewStatus | None = None) -> ReviewResult:
    """Record a human decision about one note.

    The decision applies to a specific revision of (body, owner, applicability). If that
    assertion later changes, `review_is_current` goes false rather than the approval
    silently carrying over to text nobody agreed to.

    guidance is restricted to confirmed reference_image notes. Omitting it
    preserves any existing guidance. Setting it bumps the revision so the
    confirmation applies to the resulting revision.

    expected_revision and expected_status, when both given, make the write conditional on
    the note still being at that (revision, status) - raises NoteReviewConflict otherwise.
    Covers two browser tabs reviewing the same note, and a pipeline re-run replacing it
    underneath a pending review. Both are required together (a status-only or
    revision-only check would not actually protect the case it looks like it protects -
    see NoteReviewConflict). The CLI leaves both unset and keeps its unconditional behavior.
    """
    note = _get_note(store, project_id, note_id)
    # A note replaced by correct_note is history, not a live claim: reviewing it would
    # reactivate (or re-reason) something a human deliberately superseded, and would
    # overwrite review_reason="corrected", breaking the lineage. Only refuse a caller
    # who sees it as it really is (no expectation, or expecting "rejected"); a caller
    # still expecting its old live status falls through to the CAS below and gets the
    # usual 409, since their view is stale.
    if note.superseded_by_note_ids and (expected_status is None or expected_status == note.status):
        raise ValueError(
            f"{note.id} was replaced by a correction ({', '.join(note.superseded_by_note_ids)}); "
            "review the replacement note instead"
        )
    if decision not in ("confirmed", "rejected"):
        raise ValueError("decision must be 'confirmed' or 'rejected'")
    if guidance is not None and (decision != "confirmed" or note.kind != "reference_image"):
        raise ValueError("guidance is only supported when confirming a reference_image note")
    if duplicate_of is not None:
        if decision != "rejected":
            raise ValueError("only a rejected note can be a duplicate of another")
        survivor = _get_note(store, project_id, duplicate_of)
        if survivor.id == note.id:
            raise ValueError("a note cannot be a duplicate of itself")
        reason = "duplicate"
    if decision == "rejected" and reason is None:
        raise ValueError("rejecting a note needs a reason: false, wrong_scope, duplicate, "
                         "not_useful, other")

    changes: dict = dict(status=decision, reviewed_by=reviewer,
                         reviewed_at=utcnow(),
                         review_reason=reason if decision == "rejected" else None,
                         duplicate_of=duplicate_of)
    # guidance edit: bump revision so the confirmation applies to the new state
    if guidance is not None:
        changes["guidance"] = guidance
        changes["revision"] = note.revision + 1
    changes["reviewed_revision"] = changes.get("revision", note.revision)

    if (expected_revision is None) != (expected_status is None):
        raise ValueError("expected_revision and expected_status must be given together")

    updated = note.touch(**changes)
    if expected_revision is not None:
        store.put_note_if_current(project_id, updated, expected_revision, expected_status)
    else:
        store.put_notes(project_id, [updated])
    return ReviewResult(note=updated, action=decision)


def review_containment(store: Store, project_id: str, child_ref: str, decision: ReviewStatus, *,
                       reviewer: str, parent_ref: str | None = None) -> Location:
    """Confirm, reject, or set the parent of a location.

    Only confirmed containment inherits notes downward, so this is the gate between a
    machine's guess about the world's shape and facts flowing into a child's context.
    """
    graph = _Graph(store.list_entities(project_id))
    child_id = resolve_scope(store, project_id, child_ref, graph)
    child = graph.by_id[child_id]
    if not isinstance(child, Location):
        raise ValueError(f"{child_ref!r} is a scene; containment is between locations")

    if parent_ref is not None:
        parent_id = resolve_scope(store, project_id, parent_ref, graph)
        parent = graph.by_id[parent_id]
        if not isinstance(parent, Location):
            raise ValueError(f"{parent_ref!r} is a scene; a location's parent must be a location")
        if parent_id == child_id:
            raise ValueError("a location cannot contain itself")
        if _would_cycle(graph, child_id, parent_id):
            raise ValueError(f"{parent.name} is already inside {child.name}; that would make a cycle")
        containment = Containment(parent_id=parent_id)
    elif child.containment is None:
        raise ValueError(f"{child.name} has no proposed parent; pass one to set it")
    else:
        containment = child.containment

    updated = child.touch(containment=containment.model_copy(update={
        "status": decision, "reviewed_by": reviewer, "reviewed_at": utcnow()}))
    store.put_entities(project_id, [updated])
    return updated


def _would_cycle(graph: _Graph, child_id: str, parent_id: str) -> bool:
    seen, node = {child_id}, graph.by_id.get(parent_id)
    while isinstance(node, Location) and node.containment is not None:
        nxt = node.containment.parent_id
        if nxt in seen:
            return True
        seen.add(node.id)
        node = graph.by_id.get(nxt)
    return False


# --- note correction / splitting ------------------------------------------------------

@dataclass
class NoteSuccessorSpec:
    """One replacement for correct_note - either the sole successor of a plain
    correction, or one half of a standing/scene-specific split."""
    kind: str
    body: str
    scene_id: str | None = None
    include_descendants: bool = False


@dataclass
class CorrectionResult:
    original: Note              # now status="rejected", review_reason="corrected"
    new_notes: list[Note]

    def summary(self) -> str:
        heads = "; ".join(" ".join(n.body.split())[:60] for n in self.new_notes)
        return f"corrected `{self.original.id}` -> {len(self.new_notes)} note(s): {heads}"


def _linked_scene_ids(graph: _Graph, owner_id: str) -> set[str]:
    """The exact rule retrieval.get_context uses to compute a location's own linked
    scenes (pack.scenes) - reused here rather than redefined, so a scene accepted by
    correct_note is guaranteed to be one get_context/build_approval_snapshot would also
    recognize as belonging to this location. PROJECT_SCOPE has no linked-scene concept
    - always an empty set, which correctly makes any scene_id invalid for a project
    note without a separate check."""
    if owner_id == PROJECT_SCOPE:
        return set()
    return {
        s.id for s in graph.by_id.values()
        if isinstance(s, Scene) and _live(s)
        and any((t := graph.target(lid)) is not None and t.id == owner_id for lid in s.location_ids)
    }


def correct_note(store: Store, project_id: str, note_id: str, successors: list[NoteSuccessorSpec], *,
                 reviewer: str, expected_revision: int, expected_status: ReviewStatus) -> CorrectionResult:
    """Replace one brief note (description/constraint/tone) with one corrected note, or
    split it into two (typically one standing, one scene-specific) - the human
    correction path for a note whose wording, kind, or scope needs fixing.

    Never edits the original note's body/kind/applicability in place - mirrors
    models.Note's own documented rule ("changing body/owner/applicability needs a new
    note that supersedes the old one"). Instead: the original is marked
    status="rejected", review_reason="corrected", superseded_by_note_ids=[the new
    note(s)]; each new note is a fresh Note with author="user",
    origin.producer="user", supersedes_note_id=original.id, status="proposed" (a
    changed assertion has no reviewer sign-off yet - it is never carried forward from
    the original, confirmed or not: see Note.assertion/review_is_current). Ownership
    (owner_id) is unconditionally inherited from the original and cannot be changed
    here - that is explicitly out of scope for this action.

    Concurrency: expected_revision/expected_status are required and checked atomically
    against the CURRENT original note by Store.correct_note (the exact same CAS
    contract as put_note_if_current) - a stale pair raises NoteReviewConflict (-> 409
    at the route), writing nothing at all, so the caller's draft (the successors list
    it already built) is never lost; retry with a fresh read. The reject-original and
    create-successor(s) writes happen in that one atomic call, so a split can never be
    observed half-done.

    Provenance is copied verbatim from the original onto every successor - this
    function has no parameter for editing a citation/quote, so an edited body can never
    masquerade as a different verbatim screenplay excerpt; Provenance.extracted_body
    keeps showing exactly what was originally extracted, next to the new, human-
    corrected Note.body, as the audit trail.

    Retrieval: once the original is "rejected", retrieval._visible excludes it
    unconditionally (regardless of include_proposed) - the existing visibility rule,
    not new filtering code - so get_context/build_approval_snapshot immediately stop
    showing it and start showing the new note(s) instead, with no further wiring.

    Re-ingestion: a later ingest run of the same source cannot resurrect or overwrite
    this correction. _commit's reconciliation only ever considers notes whose
    origin.producer is "ingest" (or the pre-origin legacy None) - our new notes have
    producer="user" and are invisible to it entirely - and for the original, whose
    origin.producer IS still "ingest", _commit's own rule already treats any non-
    "proposed" status (rejected included, for any reason) as a durable human decision
    and writes nothing for a re-matched candidate. Both protections are existing
    ingest.py behavior; nothing there needed to change for this to hold.

    Approval staleness: an already-locked ConceptApproval snapshot is never rewritten
    by this (see concepts.py) - GET .../approval will recompute the original as
    changed/removed and the new note(s) as added, via the existing generic diff, and
    report isStale accordingly. Re-approval is always a separate, explicit lock.

    Raises LookupError if note_id is unknown, ValueError for anything else invalid
    (unsupported kind, an already-rejected original, 1-2 successors required, a bad
    kind/scene_id on a successor, or an owner that is no longer a live entity).
    """
    original = _get_note(store, project_id, note_id)
    if original.kind not in BRIEF_NOTE_KINDS:
        raise ValueError(
            f"only {'/'.join(BRIEF_NOTE_KINDS)} notes can be corrected here; "
            f"{original.id} is {original.kind!r}"
        )
    # Checked against the CALLER's claimed expected_status, not the freshly-read
    # original.status: a caller who correctly expected "proposed"/"confirmed" but
    # finds the note is now actually rejected (raced by someone else's correction or
    # review) must get a concurrency conflict from the CAS below (409), not this
    # business-rule error - only a caller who explicitly claims expected_status ==
    # "rejected" is making the nonsensical request this guards against.
    if expected_status == "rejected":
        raise ValueError(f"{original.id} is already rejected; nothing to correct")
    if not (1 <= len(successors) <= 2):
        raise ValueError(f"correct_note takes 1 successor (a correction) or 2 (a split), "
                         f"not {len(successors)}")

    graph = _Graph(store.list_entities(project_id))
    if original.owner_id != PROJECT_SCOPE:
        owner = graph.by_id.get(original.owner_id)
        if owner is None or not _live(owner):
            raise ValueError(
                f"{original.owner_id!r} is no longer a live entity; cannot correct notes owned by it"
            )
    scene_ids = _linked_scene_ids(graph, original.owner_id)

    new_notes: list[Note] = []
    for spec in successors:
        if spec.kind not in BRIEF_NOTE_KINDS:
            raise ValueError(
                f"{spec.kind!r} is not a supported kind for a correction; use one of "
                f"{', '.join(BRIEF_NOTE_KINDS)}"
            )
        if spec.scene_id is not None and spec.scene_id not in scene_ids:
            raise ValueError(f"{spec.scene_id!r} is not a scene linked to {original.owner_id!r}")
        new_notes.append(Note(
            kind=spec.kind, body=spec.body, owner_id=original.owner_id,
            applicability=Applicability(scene_id=spec.scene_id, include_descendants=spec.include_descendants),
            author="user", mentions=list(original.mentions),
            provenance=[p.model_copy() for p in original.provenance],
            origin=NoteOrigin(producer="user", scope=original.owner_id, digest_version="user-correction-v1"),
            supersedes_note_id=original.id,
        ))

    updated_original = original.touch(
        status="rejected", review_reason="corrected",
        superseded_by_note_ids=[n.id for n in new_notes],
        reviewed_by=reviewer, reviewed_at=utcnow(), reviewed_revision=original.revision,
    )
    stored_original, stored_new = store.correct_note(
        project_id, updated_original, new_notes, expected_revision, expected_status,
    )
    return CorrectionResult(original=stored_original, new_notes=stored_new)
