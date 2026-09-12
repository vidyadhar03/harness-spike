"""Context retrieval: store-only reads that assemble a ContextPack and render it as markdown.

No model calls, so the same memory state always yields the same pack. That is what
makes a pack safe to snapshot when a location is locked.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from pathlib import Path

from .models import PROJECT_SCOPE, ContextPack, Location, Note, ReferenceImage, Scene, Source
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
    """The project's entities, read once. Merge chains and scene-location links resolve in memory."""

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

def _owned_elsewhere(note: Note, graph: "_Graph", own: set[str], related_type: type) -> bool:
    """True when a related note belongs to a different entity of the pack's own type.

    A scene can be set at several locations, so its notes would otherwise appear in every
    one of those locations' packs. A note that names its own location belongs to that
    location's pack only. A note scoped to the scene alone has no better owner and stays.
    """
    for ref in note.scope_refs:
        target = graph.target(ref)
        if target is None or ref == PROJECT_SCOPE:
            continue
        if isinstance(target, related_type):
            continue                      # the link that pulled this note in
        if target.id not in own:
            return True
    return False


def _sort_key(n: Note):
    page = next((p.page for p in n.provenance if p.page is not None), 10**6)
    return _STATUS_ORDER.get(n.status, 9), _KIND_ORDER.get(n.kind, 9), page, n.created_at


def _visible(notes: list[Note], include_proposed: bool) -> list[Note]:
    keep = [n for n in notes if n.status == "confirmed" or (include_proposed and n.status == "proposed")]
    return sorted(keep, key=_sort_key)


def get_context(store: Store, project_id: str, scope_ref: str, *, include_proposed: bool = True) -> ContextPack:
    """Everything the harness knows about one location, scene, or the project.

    Location pack: its notes (plus merged-in entities), the scenes set there and their notes,
    project-wide notes. Scene pack: its notes, the locations it is set in and their notes,
    project-wide notes. Rejected notes never appear; proposed ones only when include_proposed.
    """
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, scope_ref, graph)
    project_notes = _visible(store.notes_for_scopes(project_id, [PROJECT_SCOPE]), include_proposed)

    if scope_id == PROJECT_SCOPE:
        return _finish(store, project_id, ContextPack(
            scope_id=scope_id, entity=None, include_proposed=include_proposed, notes=project_notes))

    entity = graph.by_id[scope_id]
    own = graph.own_ids(scope_id)
    notes = _visible(store.notes_for_scopes(project_id, own), include_proposed)

    scenes: list[Scene] = []
    locations: list[Location] = []
    if isinstance(entity, Location):
        scenes = sorted(
            (s for s in graph.by_id.values() if isinstance(s, Scene) and _live(s)
             and any((t := graph.target(lid)) is not None and t.id == scope_id for lid in s.location_ids)),
            key=lambda s: natural_key(s.number or ""),
        )
        groups = [(s.id, graph.own_ids(s.id)) for s in scenes]
    else:
        for lid in entity.location_ids:
            loc = graph.target(lid)
            if isinstance(loc, Location) and _live(loc) and all(l.id != loc.id for l in locations):
                locations.append(loc)
        groups = [(l.id, graph.own_ids(l.id)) for l in locations]

    seen = {n.id for n in notes}
    related_scope = [i for _, ids in groups for i in ids]
    related = []
    if related_scope:
        owner_type = Scene if isinstance(entity, Location) else Location
        related = [n for n in _visible(store.notes_for_scopes(project_id, related_scope), include_proposed)
                   if n.id not in seen and not _owned_elsewhere(n, graph, set(own), owner_type)]
    related_by: dict[str, list[str]] = {}
    placed: set[str] = set()
    for gid, ids in groups:
        members = [n.id for n in related if n.id not in placed and set(ids) & set(n.scope_refs)]
        if members:
            related_by[gid] = members
            placed |= set(members)

    seen |= {n.id for n in related}
    # unattributed reference images would flood every pack; they live in the project pack
    project_notes = [n for n in project_notes if n.kind != "reference_image" and n.id not in seen]

    return _finish(store, project_id, ContextPack(
        scope_id=scope_id, entity=entity, include_proposed=include_proposed, merged_ids=own[1:],
        notes=notes, related_notes=related, related_by=related_by, project_notes=project_notes,
        scenes=scenes, locations=locations,
    ))


def _finish(store: Store, project_id: str, pack: ContextPack) -> ContextPack:
    everything = pack.notes + pack.related_notes + pack.project_notes
    sources = store.get_sources(project_id, [p.source_id for n in everything for p in n.provenance if p.source_id])
    pack.sources = {sid: s.filename for sid, s in sources.items()}
    pack.reference_images = [
        ref for n in pack.notes + pack.related_notes
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
        if pack.merged_ids:
            lines.append(f"Merged in: {', '.join(f'`{i}`' for i in pack.merged_ids)}")
        lines.append("")
    lines.append("Notes marked [proposed] are unreviewed." if pack.include_proposed
                 else "Confirmed notes only.")
    lines.append("")

    body = [n for n in pack.notes if n.kind != "reference_image"]
    lines += _kind_sections(body, src, "##") or ["_No notes yet._", ""]

    if pack.reference_images:
        lines.append("## Reference images")
        groups: dict[str | None, list] = {}
        for r in pack.reference_images:
            groups.setdefault(r.group, []).append(r)
        for group, refs in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
            if group:
                lines.append(f"### {group}")
            for r in refs:
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

    if isinstance(e, Location) and pack.scenes:
        lines += ["## Scenes set here", *[f"- {s.number} · {s.name} `{s.id}`" for s in pack.scenes], ""]
        lines += _grouped(pack, [(s.id, f"{s.number} · {s.name}") for s in pack.scenes], "## Scene notes")
    if isinstance(e, Scene) and pack.locations:
        lines += ["## Set in", *[f"- {l.name} `{l.id}`" for l in pack.locations], ""]
        lines += _grouped(pack, [(l.id, l.name) for l in pack.locations], "## Location context")

    if pack.project_notes:
        lines += ["## Project-wide", *[_bullet(n, src) for n in pack.project_notes], ""]
    return "\n".join(lines).rstrip() + "\n"


def _grouped(pack: ContextPack, groups: list[tuple[str, str]], heading: str) -> list[str]:
    by_id = {n.id: n for n in pack.related_notes if n.kind != "reference_image"}
    out: list[str] = []
    for gid, title in groups:
        group = [by_id[i] for i in pack.related_by.get(gid, []) if i in by_id]
        if group:
            out += [f"### {title}", *[_bullet(n, pack.sources) for n in group], ""]
    return [heading, "", *out] if out else []


# --- index and export -------------------------------------------------------------

def render_index_md(store: Store, project_id: str) -> str:
    """The root listing an agent reads first: what exists and how much is known about each."""
    graph = _Graph(store.list_entities(project_id))
    notes = [n for n in store.list_notes(project_id) if n.status != "rejected"]
    sources = store.list_sources(project_id)
    by_scope = Counter(ref for n in notes for ref in n.scope_refs)

    def count(entity_id: str) -> int:
        ids = set(graph.own_ids(entity_id))
        return sum(1 for n in notes if ids & set(n.scope_refs))

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
        lines.append(f"- {l.name} `{l.id}` · {l.status} · {_n(count(l.id), 'note')} · {_n(scene_count[l.id], 'scene')}{aka}")
    lines += ["", f"## Scenes ({len(scenes)})"]
    lines += [f"- {s.number} · {s.name} `{s.id}` · {_n(count(s.id), 'note')}" for s in scenes]
    unattributed = sum(1 for n in notes if PROJECT_SCOPE in n.scope_refs and n.kind == "reference_image")
    lines += ["", "## Project-wide", f"- {_n(by_scope[PROJECT_SCOPE] - unattributed, 'note')}, "
                                     f"{_n(unattributed, 'unattributed reference image')}"]
    return "\n".join(lines) + "\n"


def _n(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "untitled"


def export_context(store: Store, project_id: str, out_dir: str | Path, *, include_proposed: bool = True) -> list[Path]:
    """Writes INDEX.md, project.md, locations/*.md, scenes/*.md. Diff two exports to compare digest versions."""
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
