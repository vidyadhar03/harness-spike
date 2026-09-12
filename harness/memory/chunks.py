"""Units of work for a notes pass, and how to split them when output is truncated.

Script units always hold whole scenes: a scene is never cut from its slugline unless it
alone overflows the model's output, and then the second half is told which scene it continues.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from .files import PDF, pdf_subset
from .ports import Blob, Part, Text

MARKER = re.compile(r"<<<PAGE (\d+)>>>")
MARKER_RULE = "use the number in the nearest preceding <<<PAGE n>>> marker."
NO_PAGES = "not applicable; leave page null."


@dataclass(frozen=True)
class SceneRef:
    number: str
    heading: str


def excerpt_rule(a: int, b: int) -> str:
    return (f"this PDF excerpt holds pages {a + 1}-{b} of the original. Report page as the "
            f"1-based position within the excerpt (1 = original page {a + 1}).")


def _scene_label(scenes: list[SceneRef]) -> str:
    if not scenes:
        return ""
    return f"scene {scenes[0].number}" if len(scenes) == 1 else f"scenes {scenes[0].number}-{scenes[-1].number}"


@dataclass
class TextUnit:
    text: str
    rule: str
    scenes: list[SceneRef] = field(default_factory=list)
    scene_offsets: list[int] = field(default_factory=list)  # where each scene starts in text
    continuation: SceneRef | None = None
    offset: int = 0  # text units carry absolute page markers

    @property
    def label(self) -> str:
        pages = [int(m.group(1)) for m in MARKER.finditer(self.text)]
        where = f"pages {pages[0]}-{pages[-1]}" if pages else f"{len(self.text)} chars"
        return " ".join(x for x in (_scene_label(self.scenes), f"({where})") if x)

    def parts(self, data: bytes) -> list[Part]:
        return [Text(self.text)]

    def split(self) -> list["TextUnit"] | None:
        if len(self.scenes) > 1:
            k = len(self.scenes) // 2
            cut = self.scene_offsets[k]
            if 0 < cut < len(self.text):
                right, shift = _reprefix(self.text, cut)
                return [
                    TextUnit(self.text[:cut], self.rule, self.scenes[:k], self.scene_offsets[:k], self.continuation),
                    TextUnit(right, self.rule, self.scenes[k:], [o - cut + shift for o in self.scene_offsets[k:]]),
                ]
            # scenes share one position in the text: narrow the scope, keep the text
            return [
                TextUnit(self.text, self.rule, self.scenes[:k], [0] * k, self.continuation),
                TextUnit(self.text, self.rule, self.scenes[k:], [0] * (len(self.scenes) - k)),
            ]
        cut = _cut_point(self.text)
        if cut is None:
            return None
        right, _ = _reprefix(self.text, cut)
        cont = self.scenes[0] if self.scenes else None
        return [
            TextUnit(self.text[:cut], self.rule, self.scenes, [0] * len(self.scenes), self.continuation),
            TextUnit(right, self.rule, self.scenes, [0] * len(self.scenes), cont),
        ]


@dataclass
class PdfUnit:
    a: int  # 0-based, [a, b)
    b: int
    scenes: list[SceneRef] = field(default_factory=list)
    scene_starts: list[int] = field(default_factory=list)  # 0-based page index per scene
    continuation: SceneRef | None = None

    @property
    def rule(self) -> str:
        return excerpt_rule(self.a, self.b)

    @property
    def offset(self) -> int:
        return self.a

    @property
    def label(self) -> str:
        return " ".join(x for x in (_scene_label(self.scenes), f"(pages {self.a + 1}-{self.b})") if x)

    def parts(self, data: bytes) -> list[Part]:
        return [Blob(pdf_subset(data, self.a, self.b), PDF)]

    def split(self) -> list["PdfUnit"] | None:
        if len(self.scenes) > 1:
            k = len(self.scenes) // 2
            mid = self.scene_starts[k]
            left_b = max(self.a + 1, min(self.b, mid + 1))   # boundary page belongs to both halves
            right_a = max(self.a, min(mid, self.b - 1))
            return [
                PdfUnit(self.a, left_b, self.scenes[:k], self.scene_starts[:k], self.continuation),
                PdfUnit(right_a, self.b, self.scenes[k:], self.scene_starts[k:]),
            ]
        if self.b - self.a > 1:
            m = (self.a + self.b) // 2
            cont = self.scenes[0] if self.scenes else None
            return [
                PdfUnit(self.a, m, self.scenes, self.scene_starts, self.continuation),
                PdfUnit(m, self.b, self.scenes, [m] * len(self.scenes), cont),
            ]
        return None


Unit = Union[TextUnit, PdfUnit]


def _reprefix(text: str, cut: int) -> tuple[str, int]:
    """Right half of a split, re-headed with its page marker so page numbers stay absolute."""
    right = text[cut:]
    if MARKER.match(right):
        return right, 0
    before = list(MARKER.finditer(text, 0, cut))
    if not before:
        return right, 0
    prefix = f"<<<PAGE {before[-1].group(1)}>>>\n"
    return prefix + right, len(prefix)


def _cut_point(text: str, min_len: int = 2000) -> int | None:
    mid = len(text) // 2
    for pattern, use_end in ((MARKER, False), (re.compile(r"\n\s*\n"), True)):
        cands = [m.end() if use_end else m.start() for m in pattern.finditer(text)]
        cands = [c for c in cands if 0 < c < len(text)]
        if cands:
            return min(cands, key=lambda c: abs(c - mid))
    if len(text) >= min_len:
        space = text.rfind(" ", 0, mid)
        return space if space > 0 else mid
    return None


def find_heading(text: str, heading: str, start: int) -> int | None:
    """Offset of the line holding this slugline, at or after start. Tolerant of case,
    punctuation, and dash differences between the model's heading and the PDF text."""
    tokens = re.findall(r"\w+", heading)
    if not tokens:
        return None
    m = re.compile(r"\W+".join(map(re.escape, tokens)), re.IGNORECASE).search(text, start)
    if not m:
        return None
    line_start = text.rfind("\n", 0, m.start()) + 1
    return max(line_start, start)


def pack(sizes: list[int], budget: int) -> list[list[int]]:
    """Greedy grouping of consecutive items (by index) up to budget; an oversized item stands alone."""
    groups, current, total = [], [], 0
    for i, size in enumerate(sizes):
        if current and total + size > budget:
            groups.append(current)
            current, total = [], 0
        current.append(i)
        total += size
    if current:
        groups.append(current)
    return groups
