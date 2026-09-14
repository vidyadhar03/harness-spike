"""Context retrieval: store-only reads that assemble a ContextPack and render it as markdown.

No model calls, so the same memory state always yields the same pack.

The pack keeps three kinds of note visibly apart, because collapsing them is how a
generation step ends up treating a dream flood as the market square's permanent geometry:
  - owned, unconditional: what this place is
  - conditional: true only during one scene
  - inherited: owned by a confirmed ancestor, and marked as applying within it
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from pathlib import Path

from .models import (
    PROJECT_SCOPE, ConditionalNotes, ContextPack, InheritedNotes, Location, Note,
    ReferenceImage, Scene, Source,
)
from .ports import EntityDoc, Store
from .resolver import natural_key, norm, scene_key

_STATUS_ORDER = {"confirmed": 0, "proposed": 1}
_KIND_ORDER = {"constraint": 0, "description": 1, "tone": 2, "vocabulary": 3, "reference_image": 4}
_SECTIONS = [("constraint", "Constraints"), ("description", "Description"), ("tone", "Tone"),
             ("vocabulary", "Search vocabulary")]


# --- resolution -----------------------------------------------------------------

def _live(e: EntityDoc) -> bool:
    return e.status not in ("rejected", "merged")


class _Graph:
    """The project's entities, read once. Merges and containment resolve in memory."""

    def __init__(self, entities: list[EntityDoc]):
        self.by_id = {e.id: e for e in entities}
        self.children: dict[str, list[str]] = {}
        for e in entities:
            if e.status == "merged" and e.merged_into:
                self.children.setdefault(e.merged_into, []).append(e.id)

    def target(self, entity_id: str | None) -> EntityDoc | None:
        e, seen = self.by_id.get(entity_id or ""), set()
        while e is not None and e.status == "merged" and e.id not in seen:
            seen.add(e.id)
            e = self.by_id.get(e.merged_into or "")
        return e if e is not None and e.status != "merged" else None

    def own_ids(self, entity_id: str) -> list[str]:
        """The entity plus everything merged into it, including chains."""
        out, frontier = [entity_id], [entity_id]
        while frontier:
            frontier = [c for f in frontier for c in self.children.get(f, []) if c not in out]
            out += frontier
        return out

    def ancestors(self, entity_id: str) -> list[Location]:
        """Confirmed containment only, nearest parent first. Proposed containment is
        invisible until a human confirms it, so a guessed parent cannot leak facts."""
        out: list[Location] = []
        seen = {entity_id}
        node = self.by_id.get(entity_id)
        while isinstance(node, Location) and node.containment is not None:
            if node.containment.status != "confirmed":
                break
            parent = self.target(node.containment.parent_id)
            if not isinstance(parent, Location) or parent.id in seen or not _live(parent):
                break
            out.append(parent)
            seen.add(parent.id)
            node = parent
        return out


def resolve_scope(store: Store, project_id: str, ref: str, graph: _Graph | None = None) -> str:
    """Accepts an entity id, a location name or alias, a scene number ("12", "Scene 12A"),
    a scene heading, or "project". Merged entities resolve to their target."""
    ref = ref.strip()
    if ref == PROJECT_SCOPE:
        return ref
    graph = graph or _Graph(store.list_entities(project_id))
    if ref in graph.by_id:
        return (graph.target(ref) or graph.by_id[ref]).id
    key, skey = norm(ref), scene_key(ref)
    hits: list[str] = []
    for e in graph.by_id.values():
        if isinstance(e, Location):
            match = key in {norm(n) for n in (e.name, *e.aliases)}
        else:
            match = (bool(e.number) and scene_key(e.number) == skey) or norm(e.name) == key
        if match:
            target = graph.target(e.id) or e
            if target.id not in hits:
                hits.append(target.id)
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise LookupError(f"{ref!r} matches several entities: {', '.join(hits)}; pass an id")
    live = [e for e in graph.by_id.values() if _live(e)]
    names = [e.name for e in live if isinstance(e, Location)]
    names += [f"Scene {e.number}" for e in live if isinstance(e, Scene) and e.number]
    close = difflib.get_close_matches(ref, names, n=5, cutoff=0.5)
    raise LookupError(f"no entity matches {ref!r}" + (f"; did you mean: {', '.join(close)}" if close else ""))


# --- packs --------------------------------------------------------------------------

def _sort_key(n: Note):
    page = next((p.page for p in n.provenance if p.page is not None), 10**6)
    return _STATUS_ORDER.get(n.status, 9), _KIND_ORDER.get(n.kind, 9), page, n.created_at


def _visible(notes: list[Note], include_proposed: bool) -> list[Note]:
    keep = [n for n in notes if n.status == "confirmed" or (include_proposed and n.status == "proposed")]
    return sorted(keep, key=_sort_key)


def get_context(store: Store, project_id: str, scope_ref: str, *, include_proposed: bool = True) -> ContextPack:
    """Everything the harness knows about one location, scene, or the project."""
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, scope_ref, graph)
    project_notes = _visible(store.notes_for_owners(project_id, [PROJECT_SCOPE]), include_proposed)

    if scope_id == PROJECT_SCOPE:
        return _finish(store, project_id, ContextPack(
            scope_id=scope_id, entity=None, include_proposed=include_proposed, notes=project_notes))

    entity = graph.by_id[scope_id]
    own = graph.own_ids(scope_id)
    owned = _visible(store.notes_for_owners(project_id, own), include_proposed)

    # a fact true only during one scene is never the place's general state
    unconditional = [n for n in owned if n.applicability.scene_id is None]
    conditional: list[ConditionalNotes] = []
    by_scene: dict[str, list[Note]] = {}
    for n in owned:
        if n.applicability.scene_id is not None:
            by_scene.setdefault(n.applicability.scene_id, []).append(n)

    scenes: list[Scene] = []
    locations: list[Location] = []
    ancestors: list[Location] = []
    if isinstance(entity, Location):
        scenes = sorted(
            (s for s in graph.by_id.values() if isinstance(s, Scene) and _live(s)
             and any((t := graph.target(lid)) is not None and t.id == scope_id for lid in s.location_ids)),
            key=lambda s: natural_key(s.number or ""),
        )
        ancestors = graph.ancestors(scope_id)
    elif isinstance(entity, Scene):
        for lid in entity.location_ids:
            loc = graph.target(lid)
            if isinstance(loc, Location) and _live(loc) and all(l.id != loc.id for l in locations):
                locations.append(loc)

    labels = {s.id: f"{s.number} · {s.name}" if s.number else s.name for s in scenes}
    for scene_id, notes in by_scene.items():
        scene = graph.target(scene_id)
        label = labels.get(scene_id) or (scene.name if scene else scene_id)
        conditional.append(ConditionalNotes(scene_id=scene_id, label=label, notes=notes))
    conditional.sort(key=lambda c: natural_key(c.label))

    inherited: list[InheritedNotes] = []
    for parent in ancestors:
        parent_ids = graph.own_ids(parent.id)
        notes = [n for n in _visible(store.notes_for_owners(project_id, parent_ids), include_proposed)
                 if n.applicability.include_descendants and n.applicability.scene_id is None]
        if notes:
            inherited.append(InheritedNotes(entity_id=parent.id, name=parent.name, notes=notes))

    seen = {n.id for n in owned} | {n.id for i in inherited for n in i.notes}
    # unattributed reference images would flood every pack; they live in the project pack
    project_notes = [n for n in project_notes if n.kind != "reference_image" and n.id not in seen]

    return _finish(store, project_id, ContextPack(
        scope_id=scope_id, entity=entity, include_proposed=include_proposed, merged_ids=own[1:],
        notes=unconditional, conditional=conditional, inherited=inherited,
        project_notes=project_notes, scenes=scenes, locations=locations, ancestors=ancestors,
    ))


def _all_notes(pack: ContextPack) -> list[Note]:
    return (pack.notes + [n for c in pack.conditional for n in c.notes]
            + [n for i in pack.inherited for n in i.notes] + pack.project_notes)


def _finish(store: Store, project_id: str, pack: ContextPack) -> ContextPack:
    everything = _all_notes(pack)
    sources = store.get_sources(project_id, [p.source_id for n in everything for p in n.provenance if p.source_id])
    pack.sources = {sid: s.filename for sid, s in sources.items()}
    pack.superseded_sources = sorted({s.filename for s in sources.values() if s.superseded})
    pack.reference_images = [
        ref for n in pack.notes + [x for c in pack.conditional for x in c.notes]
        if n.kind == "reference_image" and (ref := _reference(n, sources)) is not None
    ]
    return pack


def _reference(n: Note, sources: dict[str, Source]) -> ReferenceImage | None:
    prov = n.provenance[0]
    src = sources.get(prov.source_id or "")
    if src is None:
        return None
    if prov.page is None:
        uri = src.storage_path if src.kind == "image" else None
    else:
        uri = f"{src.derived.pages_prefix}{prov.page:04d}.png" if src.derived.pages_prefix else None
    if uri is None:
        return None
    return ReferenceImage(note_id=n.id, uri=uri, caption=n.body, status=n.status, source_id=src.id,
                          page=prov.page, group=n.group, origin_url=src.origin_url, license=src.license,
                          attribution=src.attribution)


# --- rendering -------------------------------------------------------------------

def _cite(n: Note, sources: dict[str, str]) -> str:
    parts = []
    for p in n.provenance:
        if p.source_id:
            s = sources.get(p.source_id, p.source_id[:12])
        else:
            s = p.title or p.url or ""
        if p.page is not None:
            s += f" p.{p.page}"
        if p.quote:
            s += f': "{p.quote}"'
        if p.url and p.source_id is None and p.title:
            s += f" <{p.url}>"
        parts.append(s)
    return f" _({'; '.join(parts)})_" if parts else ""


def _bullet(n: Note, sources: dict[str, str]) -> str:
    flag = "[proposed] " if n.status == "proposed" else ""
    head, *rest = [" ".join(line.split()) for line in n.body.strip().splitlines() if line.strip()]
    lines = [f"- {flag}{head}"] + [f"  {line}" for line in rest]
    lines[-1] += f"{_cite(n, sources)} <!-- {n.id} -->"
    return "\n".join(lines)


def _kind_sections(notes: list[Note], sources: dict[str, str], level: str) -> list[str]:
    out = []
    for kind, title in _SECTIONS:
        group = [n for n in notes if n.kind == kind]
        if group:
            out += [f"{level} {title}", *[_bullet(n, sources) for n in group], ""]
    return out


def render_context_md(pack: ContextPack) -> str:
    src = pack.sources
    lines: list[str] = []
    e = pack.entity
    if e is None:
        lines += ["# Project-wide context", ""]
    else:
        kind = "Location" if isinstance(e, Location) else f"Scene {e.number}"
        lines += [f"# {e.name}", f"{kind} · `{e.id}` · {e.status}"]
        if isinstance(e, Location) and e.aliases:
            lines.append(f"Also called: {', '.join(e.aliases)}")
        if pack.ancestors:
            lines.append("Inside: " + " → ".join(a.name for a in reversed(pack.ancestors)))
        if pack.merged_ids:
            lines.append(f"Merged in: {', '.join(f'`{i}`' for i in pack.merged_ids)}")
        lines.append("")
    lines.append("Notes marked [proposed] are unreviewed." if pack.include_proposed
                 else "Confirmed notes only.")
    if pack.superseded_sources:
        lines.append(f"Some notes come from superseded drafts: {', '.join(pack.superseded_sources)}. "
                     "They have not been reconciled against the current version.")
    lines.append("")

    body = [n for n in pack.notes if n.kind != "reference_image"]
    lines += _kind_sections(body, src, "##") or ["_No notes yet._", ""]

    refs = pack.reference_images
    if refs:
        lines.append("## Reference images")
        groups: dict[str | None, list] = {}
        for r in refs:
            groups.setdefault(r.group, []).append(r)
        for group, items in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
            if group:
                lines.append(f"### {group}")
            for r in items:
                where = r.attribution or pack.sources.get(r.source_id, r.source_id[:12])
                if r.page:
                    where += f" p.{r.page}"
                if r.license:
                    where += f", {r.license}"
                if r.origin_url:
                    where += f" <{r.origin_url}>"
                flag = "[proposed] " if r.status == "proposed" else ""
                lines.append(f"- {flag}{' '.join(r.caption.split())} — `{r.uri}` _({where})_ <!-- {r.note_id} -->")
            lines.append("")

    if pack.conditional:
        lines += ["## Only during these scenes",
                  "_True while the scene plays, not the place's usual state._", ""]
        for group in pack.conditional:
            text = [n for n in group.notes if n.kind != "reference_image"]
            if text:
                lines += [f"### {group.label}", *[_bullet(n, src) for n in text], ""]

    if pack.inherited:
        lines.append("## Inherited")
        for group in pack.inherited:
            lines += [f"### From {group.name}", *[_bullet(n, src) for n in group.notes], ""]

    if isinstance(e, Location) and pack.scenes:
        lines += ["## Scenes set here",
                  *[f"- {s.number} · {s.name} `{s.id}`" for s in pack.scenes], ""]
    if isinstance(e, Scene) and pack.locations:
        lines += ["## Set in", *[f"- {l.name} `{l.id}`" for l in pack.locations], ""]

    if pack.project_notes:
        lines += ["## Project-wide", *[_bullet(n, src) for n in pack.project_notes], ""]
    return "\n".join(lines).rstrip() + "\n"


# --- index and export -------------------------------------------------------------

def _n(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def render_index_md(store: Store, project_id: str) -> str:
    """The root listing an agent reads first: what exists and how much is known about each."""
    graph = _Graph(store.list_entities(project_id))
    notes = [n for n in store.list_notes(project_id) if n.status != "rejected"]
    sources = store.list_sources(project_id)
    owned = Counter(n.owner_id for n in notes)

    def count(entity_id: str) -> int:
        ids = set(graph.own_ids(entity_id))
        return sum(1 for n in notes if n.owner_id in ids)

    live = [e for e in graph.by_id.values() if _live(e)]
    locs = sorted((e for e in live if isinstance(e, Location)), key=lambda e: e.name.lower())
    scenes = sorted((e for e in live if isinstance(e, Scene)), key=lambda e: natural_key(e.number or ""))
    scene_count = Counter(t.id for s in scenes for lid in set(s.location_ids) if (t := graph.target(lid)))

    def tally(c: Counter) -> str:
        return ", ".join(f"{v} {k}" for k, v in sorted(c.items())) or "none"

    lines = ["# Project memory", "",
             f"Sources: {len(sources)} ({tally(Counter(s.status for s in sources))})",
             f"Notes: {len(notes)} ({tally(Counter(n.status for n in notes))})",
             "", f"## Locations ({len(locs)})"]
    for l in locs:
        aka = f" · aka {', '.join(l.aliases)}" if l.aliases else ""
        inside = ""
        if l.containment is not None:
            parent = graph.target(l.containment.parent_id)
            if parent is not None:
                mark = "" if l.containment.status == "confirmed" else " (proposed)"
                inside = f" · inside {parent.name}{mark}"
        lines.append(f"- {l.name} `{l.id}` · {l.status} · {_n(count(l.id), 'note')} · "
                     f"{_n(scene_count[l.id], 'scene')}{inside}{aka}")
    lines += ["", f"## Scenes ({len(scenes)})"]
    lines += [f"- {s.number} · {s.name} `{s.id}` · {_n(count(s.id), 'note')}" for s in scenes]
    unattributed = sum(1 for n in notes if n.owner_id == PROJECT_SCOPE and n.kind == "reference_image")
    lines += ["", "## Project-wide", f"- {_n(owned[PROJECT_SCOPE] - unattributed, 'note')}, "
                                     f"{_n(unattributed, 'unattributed reference image')}"]
    return "\n".join(lines) + "\n"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "untitled"


def export_context(store: Store, project_id: str, out_dir: str | Path, *, include_proposed: bool = True) -> list[Path]:
    """Writes INDEX.md, project.md, locations/*.md, scenes/*.md."""
    root = Path(out_dir)
    (root / "locations").mkdir(parents=True, exist_ok=True)
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    written = []

    def write(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        written.append(path)

    write(root / "INDEX.md", render_index_md(store, project_id))
    write(root / "project.md", render_context_md(get_context(store, project_id, PROJECT_SCOPE,
                                                             include_proposed=include_proposed)))
    for e in store.list_entities(project_id):
        if not _live(e):
            continue
        pack = get_context(store, project_id, e.id, include_proposed=include_proposed)
        if isinstance(e, Location):
            path = root / "locations" / f"{_slug(e.name)}--{e.id}.md"
        else:
            path = root / "scenes" / f"{_slug(e.number or '')}-{_slug(e.name)}--{e.id}.md"
        write(path, render_context_md(pack))
    return written
