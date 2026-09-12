import pytest

from harness.memory.models import (
    PROJECT_SCOPE, Derived, Location, Note, NoteOrigin, Project, Provenance, Scene, Source,
)
from harness.memory.ports import MemoryStore
from harness.memory.retrieval import (
    export_context, get_context, render_context_md, render_index_md, resolve_scope,
)

PID = "prj_r"
SCRIPT = "a" * 64
IMG = "b" * 64
LOOKBOOK = "c" * 64


def note(body, scope, *, kind="description", status="proposed", source=SCRIPT, page=None, quote=None):
    return Note(kind=kind, body=body, scope_refs=scope, status=status, author="agent",
                provenance=[Provenance(source_id=source, page=page, quote=quote)],
                origin=NoteOrigin(source_id=source, digest_version="v"))


@pytest.fixture
def world():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    store.put_source(PID, Source(id=SCRIPT, filename="Dehleez_Ep1.pdf", mime_type="application/pdf",
                                 kind="document", size_bytes=1, storage_path=f"gs://b/{SCRIPT}/original.pdf",
                                 status="digested"))
    store.put_source(PID, Source(id=IMG, filename="devgram_well/IMG_1.jpg", mime_type="image/jpeg",
                                 kind="image", size_bytes=1, storage_path=f"gs://b/{IMG}/original.jpg",
                                 status="digested"))
    store.put_source(PID, Source(id=LOOKBOOK, filename="lookbook.pdf", mime_type="application/pdf",
                                 kind="document", size_bytes=1, storage_path=f"gs://b/{LOOKBOOK}/original.pdf",
                                 status="digested", derived=Derived(page_count=9, pages_prefix=f"gs://b/{LOOKBOOK}/pages/")))

    well = Location(name="Devgram well", aliases=["the well"], status="confirmed", author="agent")
    old = Location(name="Old well", status="merged", merged_into=well.id, author="agent")
    older = Location(name="Kuan", status="merged", merged_into=old.id, author="agent")
    court = Location(name="Temple courtyard", author="agent")
    s1 = Scene(name="EXT. DEVGRAM WELL - NIGHT", number="1", location_ids=[old.id], author="agent")
    s2 = Scene(name="EXT. TEMPLE COURTYARD - DAY", number="2", location_ids=[court.id], author="agent")
    s10 = Scene(name="EXT. DEVGRAM WELL - DAWN", number="10", location_ids=[well.id], author="agent")
    store.put_entities(PID, [well, old, older, court, s1, s2, s10])

    n = {
        "constraint": note("No electricity poles near the well.", [well.id], kind="constraint", status="confirmed", page=2),
        "merged": note("Rope worn smooth.", [older.id], page=5),
        "scene_only": note("A fissure opens in the tree beside the well.", [s1.id], page=1, quote="the ground splits"),
        "both": note("Stone well, waist high.", [well.id, s1.id], page=1),
        "rejected": note("Well is made of steel.", [well.id], status="rejected"),
        "tone": note("Muted ochre palette.", [PROJECT_SCOPE], kind="tone", status="confirmed"),
        "loose_ref": note("Unattributed dusk photo.", [PROJECT_SCOPE], kind="reference_image", source=IMG),
        "photo": note("Well at dusk, rope coiled on the rim.", [well.id], kind="reference_image", source=IMG),
        "page": note("Mud wall behind the well.", [well.id], kind="reference_image", source=LOOKBOOK, page=3),
        "court": note("Slate floor, uneven.", [court.id, s2.id], page=3),
        "dawn": note("Mist over the well at dawn.", [s10.id], page=9),
    }
    store.put_notes(PID, list(n.values()))
    return store, dict(well=well, old=old, older=older, court=court, s1=s1, s2=s2, s10=s10), n


def ids(notes):
    return [x.id for x in notes]


def test_resolve_scope_by_id_name_alias_merged_and_scene_number(world):
    store, e, _ = world
    well = e["well"].id
    assert resolve_scope(store, PID, "Devgram well") == well
    assert resolve_scope(store, PID, "the WELL") == well
    assert resolve_scope(store, PID, "Kuan") == well              # merged twice
    assert resolve_scope(store, PID, e["old"].id) == well
    assert resolve_scope(store, PID, "Scene 1") == e["s1"].id
    assert resolve_scope(store, PID, "project") == PROJECT_SCOPE
    with pytest.raises(LookupError, match="did you mean: Devgram well"):
        resolve_scope(store, PID, "Devgram wel")


def test_location_pack(world):
    store, e, n = world
    pack = get_context(store, PID, "Devgram well")
    assert pack.scope_id == e["well"].id
    assert set(pack.merged_ids) == {e["old"].id, e["older"].id}
    assert ids(pack.notes) == [n["constraint"].id, n["both"].id, n["merged"].id, n["page"].id, n["photo"].id]
    assert n["rejected"].id not in ids(pack.notes + pack.related_notes + pack.project_notes)
    assert [s.number for s in pack.scenes] == ["1", "10"]         # scene 1 links via a merged id; natural order
    assert ids(pack.related_notes) == [n["scene_only"].id, n["dawn"].id]
    assert pack.related_by == {e["s1"].id: [n["scene_only"].id], e["s10"].id: [n["dawn"].id]}
    assert ids(pack.project_notes) == [n["tone"].id]             # unattributed images stay out
    assert [r.uri for r in pack.reference_images] == [f"gs://b/{LOOKBOOK}/pages/0003.png", f"gs://b/{IMG}/original.jpg"]
    assert pack.sources[SCRIPT] == "Dehleez_Ep1.pdf"


def test_confirmed_only_pack(world):
    store, _, n = world
    pack = get_context(store, PID, "Devgram well", include_proposed=False)
    assert ids(pack.notes) == [n["constraint"].id]
    assert pack.related_notes == [] and pack.reference_images == []
    assert ids(pack.project_notes) == [n["tone"].id]


def test_scene_pack_pulls_location_context_through_merges(world):
    store, e, n = world
    pack = get_context(store, PID, "1")
    assert ids(pack.notes) == [n["scene_only"].id, n["both"].id]
    assert [l.id for l in pack.locations] == [e["well"].id]
    assert set(ids(pack.related_notes)) == {n["constraint"].id, n["merged"].id, n["page"].id, n["photo"].id}
    assert set(pack.related_by[e["well"].id]) == set(ids(pack.related_notes))
    assert len(pack.reference_images) == 2


def test_project_pack_includes_unattributed_images(world):
    store, _, n = world
    pack = get_context(store, PID, PROJECT_SCOPE)
    assert set(ids(pack.notes)) == {n["tone"].id, n["loose_ref"].id}
    assert [r.note_id for r in pack.reference_images] == [n["loose_ref"].id]  # triage view for unattributed images
    md = render_context_md(pack)
    assert md.startswith("# Project-wide context") and "## Reference images" in md and "## Tone" in md


def test_render_context_md(world):
    store, e, n = world
    md = render_context_md(get_context(store, PID, "Devgram well"))
    assert md.startswith("# Devgram well\nLocation · `")
    assert "Also called: the well" in md and "Merged in:" in md
    assert md.index("## Constraints") < md.index("## Description") < md.index("## Reference images")
    assert "- No electricity poles near the well. _(Dehleez_Ep1.pdf p.2)_" in md
    assert "- [proposed] Stone well, waist high." in md
    assert '_(Dehleez_Ep1.pdf p.1: "the ground splits")_' in md
    assert "### 1 · EXT. DEVGRAM WELL - NIGHT" in md and "### 10 · EXT. DEVGRAM WELL - DAWN" in md
    assert f"<!-- {n['both'].id} -->" in md
    assert "steel" not in md and "Unattributed dusk photo" not in md
    assert md.rstrip().endswith(f"<!-- {n['tone'].id} -->")

    scene_md = render_context_md(get_context(store, PID, "Scene 1"))
    assert "## Set in" in scene_md and "## Location context" in scene_md and "### Devgram well" in scene_md


def test_index_and_export(world, tmp_path):
    store, e, _ = world
    index = render_index_md(store, PID)
    assert "## Locations (2)" in index and "## Scenes (3)" in index
    assert f"- Devgram well `{e['well'].id}` · confirmed · 5 notes · 2 scenes · aka the well" in index
    assert "Old well" not in index
    assert "- 1 note, 1 unattributed reference image" in index

    paths = export_context(store, PID, tmp_path)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in paths)
    assert names[:2] == ["INDEX.md", "locations/devgram-well--" + e["well"].id + ".md"]
    assert any(p.startswith("scenes/10-ext-devgram-well-dawn--") for p in names)
    assert len(names) == 2 + 2 + 3


# --- scene notes stay with the location that owns them --------------------------

@pytest.fixture
def dream(world):
    """Scene 7 is set at two locations, as a dream sequence would be."""
    store, e, n = world
    s7 = Scene(name="INT./EXT. PANDIT'S DREAM - NIGHT", number="7",
               location_ids=[e["well"].id, e["court"].id], author="agent")
    store.put_entities(PID, [s7])
    notes = {
        "well_owned": note("Water climbs the well rim.", [e["well"].id, s7.id], page=13),
        "court_owned": note("Slate floor vanishes under water.", [e["court"].id, s7.id], page=13),
        "merged_owned": note("The old rope floats free.", [e["older"].id, s7.id], page=13),
        "unowned": note("The bell rings with no sound.", [s7.id], page=13),
    }
    store.put_notes(PID, list(notes.values()))
    return store, e, {**n, **notes}, s7


def test_scene_note_owned_by_another_location_is_excluded(dream):
    store, e, n, s7 = dream
    well = get_context(store, PID, e["well"].id)
    court = get_context(store, PID, e["court"].id)

    assert n["court_owned"].id not in ids(well.related_notes)   # belongs to the courtyard
    assert n["well_owned"].id in ids(well.notes)                # its own note, direct scope
    assert n["merged_owned"].id in ids(well.notes)              # merged ids count as its own
    assert n["well_owned"].id not in ids(court.related_notes)
    assert n["court_owned"].id in ids(court.notes)


def test_scene_note_with_no_location_appears_in_every_location_pack(dream):
    store, e, n, s7 = dream
    for loc in (e["well"].id, e["court"].id):
        pack = get_context(store, PID, loc)
        assert n["unowned"].id in ids(pack.related_notes)
        assert pack.related_by[s7.id] == [n["unowned"].id]


def test_scene_pack_excludes_notes_owned_by_another_scene(dream):
    store, e, n, s7 = dream
    pack = get_context(store, PID, s7.id)
    assert n["scene_only"].id not in ids(pack.related_notes)    # scoped to scene 1
    assert n["constraint"].id in ids(pack.related_notes)        # location-only note
    assert n["both"].id not in ids(pack.related_notes)          # scoped to scene 1 as well


def test_related_by_never_points_at_a_filtered_note(dream):
    store, e, n, s7 = dream
    for scope in (e["well"].id, e["court"].id, s7.id, "1"):
        pack = get_context(store, PID, scope)
        present = set(ids(pack.related_notes))
        assert all(i in present for members in pack.related_by.values() for i in members)
