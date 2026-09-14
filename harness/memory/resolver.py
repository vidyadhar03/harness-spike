"""Maps names the model uses onto entity ids.

Exact normalised name/alias matches only. Anything unmatched becomes a new
proposed entity: a visible duplicate a human can merge beats a silent wrong merge.
"""
from __future__ import annotations

import re

from .models import Containment, Location, Scene
from .ports import EntityDoc

_ARTICLE = re.compile(r"^(the|a|an)\s+")


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return _ARTICLE.sub("", " ".join(s.split()))


def scene_key(ref: str) -> str:
    key = norm(ref)
    return key[6:] if key.startswith("scene ") else key


class EntityResolver:
    def __init__(self, existing: list[EntityDoc]):
        self._by_id: dict[str, EntityDoc] = {e.id: e for e in existing}
        self._loc: dict[str, str] = {}
        self._scene: dict[str, str] = {}
        self._rejected: set[str] = set()
        self.new: list[EntityDoc] = []
        self.pending_parents: list[tuple[str, str]] = []
        for e in existing:
            self._index(e)

    def _target(self, entity_id: str | None) -> str | None:
        e, seen = self._by_id.get(entity_id or ""), set()
        while e is not None and e.status == "merged" and e.id not in seen:
            seen.add(e.id)
            e = self._by_id.get(e.merged_into or "")
        if e is None or e.status in ("rejected", "merged"):
            return None
        return e.id

    def _index(self, e: EntityDoc) -> None:
        if e.status == "rejected":
            if isinstance(e, Location):
                self._rejected.update(norm(n) for n in (e.name, *e.aliases))
            return
        target = self._target(e.id)
        if target is None:
            return
        if isinstance(e, Location):
            self._alias(target, e.name, *e.aliases)
        elif e.number:
            self._scene.setdefault(scene_key(e.number), target)

    def _alias(self, target: str, *names: str) -> None:
        for name in names:
            if key := norm(name or ""):
                self._loc.setdefault(key, target)

    def _add(self, e: EntityDoc) -> None:
        self._by_id[e.id] = e
        self.new.append(e)
        self._index(e)

    def location(self, ref: str, aliases: list[str] | tuple = (), existing_id: str | None = None,
                 inside: str | None = None) -> str | None:
        for candidate in (existing_id, ref):
            if candidate and isinstance(self._by_id.get(candidate), Location):
                target = self._target(candidate)
                if target:
                    self._alias(target, ref, *aliases)  # later refs by this name resolve too
                return target
        key = norm(ref or "")
        if not key:
            return None
        if key in self._loc:
            return self._loc[key]
        if key in self._rejected:
            return None
        extra = list(dict.fromkeys(a.strip() for a in aliases if a.strip() and norm(a) != key))
        loc = Location(name=ref.strip(), aliases=extra, author="agent")
        self._add(loc)
        if inside:
            self.pending_parents.append((loc.id, inside))
        return loc.id

    def apply_parents(self) -> list[str]:
        """Resolve proposed containment once every location exists. Parents are proposed,
        never confirmed: retrieval ignores them until a human confirms."""
        warnings: list[str] = []
        for child_id, parent_ref in self.pending_parents:
            child = self._by_id.get(child_id)
            if not isinstance(child, Location) or child.containment is not None:
                continue
            parent_id = self.location(parent_ref)
            if parent_id is None or parent_id == child_id or self._would_cycle(child_id, parent_id):
                warnings.append(f"ignored containment {child.name!r} inside {parent_ref!r}")
                continue
            updated = child.touch(containment=Containment(parent_id=parent_id))
            self._by_id[child_id] = updated
            for i, e in enumerate(self.new):
                if e.id == child_id:
                    self.new[i] = updated
                    break
        self.pending_parents.clear()
        return warnings

    def _would_cycle(self, child_id: str, parent_id: str) -> bool:
        seen, node = {child_id}, self._by_id.get(parent_id)
        while isinstance(node, Location) and node.containment is not None:
            if node.containment.parent_id in seen:
                return True
            seen.add(node.id)
            node = self._by_id.get(node.containment.parent_id)
        return False

    def scene(self, ref: str, *, heading: str | None = None, location_ids: list[str] | tuple = (),
              create: bool = False) -> str | None:
        if isinstance(self._by_id.get(ref), Scene):
            return self._target(ref)
        key = scene_key(ref or "")
        if key in self._scene:
            return self._scene[key]
        if not create or not key:
            return None
        sc = Scene(name=(heading or f"Scene {ref}").strip(), number=ref.strip(),
                   location_ids=list(dict.fromkeys(location_ids)), author="agent")
        self._add(sc)
        return sc.id

    def roster_text(self) -> str:
        live = [e for e in self._by_id.values() if self._target(e.id) == e.id]
        locs = sorted((e for e in live if isinstance(e, Location)), key=lambda e: e.name.lower())
        scenes = sorted((e for e in live if isinstance(e, Scene)), key=lambda e: natural_key(e.number or ""))
        lines = ["LOCATIONS"]
        lines += [f"- {l.id} | {l.name}" + (f" | aliases: {', '.join(l.aliases)}" if l.aliases else "")
                  for l in locs] or ["- (none yet)"]
        lines += ["SCENES"]
        lines += [f"- {s.id} | {s.number} | {s.name}" for s in scenes] or ["- (none yet)"]
        return "\n".join(lines)


def natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]
