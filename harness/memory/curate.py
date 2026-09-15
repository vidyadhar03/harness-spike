"""Human review actions over memory: merging duplicate entities.

Ingest deliberately creates a new entity when it cannot confidently match an existing one,
so visible duplicates are expected and a person resolves them here. Notes are never
rewritten: retrieval follows merge chains, so a merge stays reversible by editing one field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Containment, Location, Note, RejectReason, ReviewStatus, Scene, utcnow
from .ports import EntityDoc, Store
from .resolver import norm
from .retrieval import _Graph, resolve_scope


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
                guidance: str | None = None) -> ReviewResult:
    """Record a human decision about one note.

    The decision applies to a specific revision of (body, owner, applicability). If that
    assertion later changes, `review_is_current` goes false rather than the approval
    silently carrying over to text nobody agreed to.

    guidance is restricted to confirmed reference_image notes. Omitting it
    preserves any existing guidance. Setting it bumps the revision so the
    confirmation applies to the resulting revision.
    """
    note = _get_note(store, project_id, note_id)
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

    updated = note.touch(**changes)
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
