"""Regression set built from real failures on the Dehleez data.

Each test is a mistake the pipeline actually made, encoded so it cannot come back.
"""
import json

import pytest

from harness.memory.curate import review_containment, review_note
from harness.memory.ingest import Ctx, ingest_source, register_file
from harness.memory.config import Settings
from harness.memory.models import PROJECT_SCOPE, Location, Project, Scene
from harness.memory.ports import MemoryBlobs, MemoryStore, Text
from harness.memory.retrieval import get_context, render_context_md
from harness.memory.schemas import ClassifyOut, NotesOut, OutLocation, OutNote, OutScene, RosterOut

PID = "prj_reg"
FILLER = "\n".join(["The wind moves through the dry fields and the village sleeps."] * 6)
PAGES = [
    f"1 EXT. DEVGRAM - EVENING\nA bell rings across the village.\n{FILLER}",
    f"7 INT./EXT. PANDIT'S DREAM - NIGHT\nWater moves uphill through the market.\n{FILLER}",
]


def text_pdf(pages):
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for body in pages:
        y = 800
        for line in body.split("\n"):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


class FakeLLM:
    model_id = "fake"

    def __init__(self, handler):
        self.handler = handler

    def generate(self, *, system, parts, schema, fast=False):
        return self.handler(schema, system, parts)


def handler(schema, system, parts):
    if schema is ClassifyOut:
        return ClassifyOut(doc_type="script")
    if schema is RosterOut:
        return RosterOut(
            locations=[
                OutLocation(name="Devgram"),
                OutLocation(name="Market Square", aliases=["Market"], inside="Devgram"),
                OutLocation(name="Ghat Steps", inside="Devgram"),
                OutLocation(name="Tree Temple", aliases=["the temple"]),   # across the river
            ],
            scenes=[
                OutScene(number="1", heading="EXT. DEVGRAM - EVENING", start_page=1,
                         locations=["Devgram"]),
                OutScene(number="7", heading="INT./EXT. PANDIT'S DREAM - NIGHT", start_page=2,
                         locations=["Market Square", "Ghat Steps", "Tree Temple"]),
            ],
        )
    text = " ".join(p.text for p in parts if isinstance(p, Text))
    if "- 1 | " in text:
        return NotesOut(notes=[
            OutNote(kind="constraint", body="Streets empty after the evening bell.",
                    owner="Devgram", applies_to_places_within=True, page=1),
            OutNote(kind="description", body="The village has three hundred houses.",
                    owner="Devgram", page=1),
            OutNote(kind="tone", body="Longer lenses for the first day.", project_wide=True, page=1),
        ])
    return NotesOut(notes=[
        OutNote(kind="description", body="Floodwater climbs the stone steps from the river.",
                owner="Ghat Steps", only_during_scene="7", page=2),
        OutNote(kind="description", body="Fruit baskets float uphill past the stalls.",
                owner="Market Square", only_during_scene="7", page=2,
                mentions=["Ghat Steps"]),
        OutNote(kind="description", body="Grain is weighed on scales each morning.",
                owner="Market Square", page=2),
    ])


@pytest.fixture
def ingested():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    # one unit per scene, so each scene's notes come from its own call
    ctx = Ctx(store, MemoryBlobs(), FakeLLM(handler),
              Settings(gcp_project="t", bucket="b", script_chunk_chars=200))
    src, _ = register_file(ctx, PID, text_pdf(PAGES), "scripts/Dehleez_Ep1.pdf")
    report = ingest_source(ctx, PID, src.id)
    assert report.status == "digested"
    by_name = {e.name: e for e in store.list_entities(PID)}
    # containment is proposed by extraction; a human confirms the real ones
    review_containment(store, PID, "Market Square", "confirmed", reviewer="vd")
    review_containment(store, PID, "Ghat Steps", "confirmed", reviewer="vd")
    return ctx, store, by_name, report


def bodies(notes):
    return {n.body for n in notes}


def test_dream_details_never_become_another_location_geometry(ingested):
    _, store, e, _ = ingested
    market = get_context(store, PID, "Market Square")
    assert "Floodwater climbs the stone steps from the river." not in bodies(market.notes)
    assert all("Floodwater" not in n.body for c in market.conditional for n in c.notes)
    ghat = get_context(store, PID, "Ghat Steps")
    assert any("Floodwater" in n.body for c in ghat.conditional for n in c.notes)


def test_a_scene_fact_is_never_the_places_general_state(ingested):
    _, store, _, _ = ingested
    market = get_context(store, PID, "Market Square")
    assert bodies(market.notes) == {"Grain is weighed on scales each morning."}
    assert any("Fruit baskets" in n.body for c in market.conditional for n in c.notes)
    md = render_context_md(market)
    assert md.index("Grain is weighed") < md.index("## Only during these scenes")


def test_village_rule_reaches_the_market_but_not_the_temple(ingested):
    _, store, _, _ = ingested
    market = get_context(store, PID, "Market Square")
    assert bodies(n for i in market.inherited for n in i.notes) == {
        "Streets empty after the evening bell."}
    assert "The village has three hundred houses." not in bodies(
        n for i in market.inherited for n in i.notes)
    temple = get_context(store, PID, "Tree Temple")
    assert temple.inherited == []
    assert "Streets empty" not in render_context_md(temple)


def test_project_scope_holds_only_production_rules(ingested):
    _, store, _, _ = ingested
    project = get_context(store, PID, PROJECT_SCOPE)
    assert bodies(project.notes) == {"Longer lenses for the first day."}


def test_extraction_is_archived_before_anything_is_replaced(ingested):
    ctx, store, _, report = ingested
    key = next(k for k in ctx.blobs.objects if "extractions/" in k)
    payload = json.loads(ctx.blobs.get(key))
    assert payload["extraction_id"] == report.extraction_id
    assert len(payload["candidates"]) == len(store.list_notes(PID))
    assert {c["body"] for c in payload["candidates"]} >= {"Grain is weighed on scales each morning."}
    assert all(c["provenance"][0]["extracted_owner_id"] == c["owner_id"] for c in payload["candidates"])


def test_rerun_keeps_reviewed_decisions_and_does_not_repropose(ingested):
    ctx, store, _, _ = ingested
    grain = next(n for n in store.list_notes(PID) if n.body.startswith("Grain"))
    houses = next(n for n in store.list_notes(PID) if n.body.startswith("The village has"))
    review_note(store, PID, grain.id, "confirmed", reviewer="vd")
    review_note(store, PID, houses.id, "rejected", reviewer="vd", reason="false")

    src = store.list_sources(PID)[0]
    report = ingest_source(ctx, PID, src.id, force=True)
    after = {n.body: n for n in store.list_notes(PID)}
    assert after["Grain is weighed on scales each morning."].status == "confirmed"
    assert after["The village has three hundred houses."].status == "rejected"
    assert report.notes_reused == 2
    assert sum(n.status == "proposed" for n in after.values()) == len(after) - 2


def test_failed_rerun_leaves_the_previous_result_intact(ingested):
    ctx, store, _, _ = ingested
    before = {n.id: n.body for n in store.list_notes(PID)}

    def failing(schema, system, parts):
        if schema is NotesOut:
            raise RuntimeError("quota")
        return handler(schema, system, parts)

    ctx.llm.handler = failing
    src = store.list_sources(PID)[0]
    report = ingest_source(ctx, PID, src.id, force=True)
    assert report.status == "failed"
    assert {n.id: n.body for n in store.list_notes(PID)} == before


def test_two_ingests_cannot_run_at_once(ingested):
    from harness.memory.ingest import ingest_lock

    ctx, store, _, _ = ingested
    with ingest_lock(ctx, PID):
        with pytest.raises(RuntimeError, match="another ingest is running"):
            with ingest_lock(ctx, PID):
                pass
    with ingest_lock(ctx, PID):          # released, so it can be taken again
        pass


def test_a_new_draft_marks_the_old_one_without_transferring_confirmations(ingested):
    ctx, store, _, _ = ingested
    old = store.list_sources(PID)[0]
    grain = next(n for n in store.list_notes(PID) if n.body.startswith("Grain"))
    review_note(store, PID, grain.id, "confirmed", reviewer="vd")

    new, created = register_file(ctx, PID, text_pdf(PAGES + ["8 EXT. BRIDGE - DAY\nNew scene."]),
                                 "scripts/Dehleez_Ep1_draft2.pdf", revision_label="Draft 2",
                                 supersedes=old.id)
    assert created and new.supersedes_source_id == old.id
    assert store.get_source(PID, old.id).superseded_by_source_id == new.id

    # the old confirmation stands but is visibly from a superseded draft, not silently current
    pack = get_context(store, PID, "Market Square")
    assert "scripts/Dehleez_Ep1.pdf" in pack.superseded_sources
    assert "superseded drafts" in render_context_md(pack)
    assert next(n for n in store.list_notes(PID) if n.id == grain.id).status == "confirmed"
