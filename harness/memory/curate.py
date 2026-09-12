"""Human review actions over memory: merging duplicate entities.

Ingest deliberately creates a new entity when it cannot confidently match an existing one,
so visible duplicates are expected and a person resolves them here. Notes are never
rewritten: retrieval follows merge chains, so a merge stays reversible by editing one field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Location, Scene, utcnow
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
    notes = store.notes_for_scopes(project_id, own) if own else []

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

    updates: list[EntityDoc] = [
        source.model_copy(update={"status": "merged", "merged_into": target_id, "updated_at": utcnow()})
    ]
    if added:
        updates.append(target.model_copy(update={"aliases": [*target.aliases, *added],
                                                 "updated_at": utcnow()}))
    store.put_entities(project_id, updates)
    result.applied = True
    return result


def _kind(entity: EntityDoc) -> str:
    return "location" if isinstance(entity, Location) else "scene"
