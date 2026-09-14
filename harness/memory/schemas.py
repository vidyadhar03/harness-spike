"""What the model returns. Kept separate from storage models on purpose:
loose where the model needs room, converted and validated by the worker."""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import DocType


class Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OutLocation(Out):
    name: str
    aliases: list[str] = []
    existing_id: str | None = None
    inside: str | None = None           # the location physically containing this one


class OutScene(Out):
    number: str
    heading: str
    start_page: int | None = None
    locations: list[str] = []


class OutNote(Out):
    kind: Literal["description", "constraint", "tone"]
    body: str
    owner: str = ""                     # the one location this is about; "" with project_wide
    project_wide: bool = False
    applies_to_places_within: bool = False
    only_during_scene: str | None = None
    mentions: list[str] = []            # other places named; never inherits the note
    page: int | None = None
    quote: str | None = None


class OutReference(Out):
    caption: str
    location: str = ""
    page: int | None = None


class ClassifyOut(Out):
    doc_type: DocType


class RosterOut(Out):
    locations: list[OutLocation] = []
    scenes: list[OutScene] = []


class NotesOut(Out):
    locations: list[OutLocation] = []
    notes: list[OutNote] = []
    references: list[OutReference] = []


class ImageOut(NotesOut):
    doc_type: DocType


def gemini_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined and title/default keys dropped."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(raw)


def parse_json(text: str, schema: type[BaseModel]):
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        import json_repair  # same fallback the legacy workers use

        return schema.model_validate(json_repair.loads(text))


# --- reference suggestions ---

class OutTerm(Out):
    term: str
    kind: Literal["technique", "material", "landform", "vegetation", "settlement",
                  "craft", "period", "region", "other"] = "other"
    why: str = ""


class VocabularyOut(Out):
    script_phrases: list[str] = []
    description: str = ""
    terms: list[OutTerm] = []


class OutDirection(Out):
    name: str
    why: str = ""
    images: list[int] = []


class OutCaption(Out):
    index: int
    caption: str
    facet: Literal["place", "architecture", "material", "terrain"] = "place"


class CurateOut(Out):
    directions: list[OutDirection] = []
    captions: list[OutCaption] = []
