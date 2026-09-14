import pytest

from harness.memory.models import (
    PROJECT_SCOPE, Applicability, Containment, Derived, Location, Note, NoteOrigin, Project,
    Provenance, Scene, Source,
)
from harness.memory.ports import MemoryStore
from harness.memory.retrieval import (
    export_context, get_context, render_context_md, render_index_md, resolve_scope,
)

PID = "prj_r"
SCRIPT = "a" * 64
IMG = "b" * 64
LOOKBOOK = "c" * 64


def note(body, owner, *, kind="description", status="proposed", source=SCRIPT, page=None,
         quote=None, within=False, scene=None, mentions=(), reviewed=False):
    extra = {}
    if status != "proposed" or reviewed:
        extra = dict(reviewed_revision=1, reviewed_by="vd")
    return Note(kind=kind, body=body, owner_id=owner, status=status, author="agent",
                applicability=Applicability(include_descendants=within, scene_id=scene),
                mentions=list(mentions),
                provenance=[Provenance(source_id=source, page=page, quote=quote)],
                origin=NoteOrigin(digest_version="v", source_id=source),
                review_reason="false" if status == "rejected" else None, **extra)


@pytest.fixture
def world():
    """Devgram contains the market square and the lanes. The temple is across the river,
    outside the village. Scene 7 is a dream that visits several of them."""
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    for sid, name, kind, derived in [
        (SCRIPT, "Dehleez_Ep1.pdf", "document", Derived()),
        (IMG, "devgram_well/IMG_1.jpg", "image", Derived()),
        (LOOKBOOK, "lookbook.pdf", "document", Derived(page_count=9, pages_prefix=f"gs://b/{LOOKBOOK}/pages/")),
    ]:
        store.put_source(PID, Source(id=sid, filename=name, mime_type="application/pdf" if kind == "document" else "image/jpeg",
                                     kind=kind, size_bytes=1, storage_path=f"gs://b/{sid}/original.x",
                                     status="digested", derived=derived))

    devgram = Location(name="Devgram", aliases=["the village"], status="confirmed", author="user")
    market = Location(name="Market Square", aliases=["Market"], status="confirmed", author="agent",
                      containment=Containment(parent_id=devgram.id, status="confirmed"))
    lanes = Location(name="Devgram Lanes", author="agent",
                     containment=Containment(parent_id=devgram.id, status="proposed"))
    temple = Location(name="Tree Temple", aliases=["the temple"], author="agent")
    sanctum = Location(name="Inner Sanctum", author="agent",
                       containment=Containment(parent_id=temple.id, status="confirmed"))
    old_market = Location(name="Bazaar", status="merged", merged_into=market.id, author="agent")
    s7 = Scene(name="INT./EXT. PANDIT'S DREAM - NIGHT", number="7",
               location_ids=[market.id, temple.id], author="agent")
    s10 = Scene(name="EXT. MARKET SQUARE - MORNING", number="10", location_ids=[market.id], author="agent")
    store.put_entities(PID, [devgram, market, lanes, temple, sanctum, old_market, s7, s10])

    notes = {
        "curfew": note("Streets empty after the evening bell.", devgram.id, kind="constraint",
                       status="confirmed", within=True, page=2),
        "houses": note("The village has three hundred houses.", devgram.id, page=2),
        "grain": note("Grain is weighed on scales each morning.", market.id, page=15),
        "merged": note("Stalls are roofed with corrugated sheet.", old_market.id, page=15),
        "flood": note("Floodwater surges uphill through the square.", market.id, scene=s7.id, page=13),
        "temple_roots": note("Stone walls absorbed by ancient roots.", temple.id, within=False, page=18),
        "temple_rule": note("No idols or treasure anywhere in the temple.", temple.id,
                            kind="constraint", within=True, page=13),
        "bridge_ref": note("The temple is visible across the river.", market.id,
                           mentions=[temple.id], page=4),
        "rejected": note("The market is roofed in glass.", market.id, status="rejected"),
        "tone": note("Muted ochre palette throughout.", PROJECT_SCOPE, kind="tone",
                     status="confirmed"),
        "loose_ref": note("Unattributed dusk photo.", PROJECT_SCOPE, kind="reference_image", source=IMG),
        "photo": note("Well at dusk, rope coiled on the rim.", market.id, kind="reference_image", source=IMG),
        "page_ref": note("Mud wall behind the stalls.", market.id, kind="reference_image",
                         source=LOOKBOOK, page=3),
    }
    store.put_notes(PID, list(notes.values()))
    e = dict(devgram=devgram, market=market, lanes=lanes, temple=temple, sanctum=sanctum,
             old_market=old_market, s7=s7, s10=s10)
    return store, e, notes


def ids(notes):
    return [n.id for n in notes]


def test_resolve_scope(world):
    store, e, _ = world
    assert resolve_scope(store, PID, "Market") == e["market"].id
    assert resolve_scope(store, PID, "Bazaar") == e["market"].id          # merged
    assert resolve_scope(store, PID, "Scene 7") == e["s7"].id
    assert resolve_scope(store, PID, PROJECT_SCOPE) == PROJECT_SCOPE
    with pytest.raises(LookupError, match="did you mean: Devgram"):
        resolve_scope(store, PID, "Devgran")


def test_owned_notes_only_and_merged_ids(world):
    store, e, n = world
    pack = get_context(store, PID, "Market")
    assert ids(pack.notes) == [n["bridge_ref"].id, n["grain"].id, n["merged"].id,   # by page
                               n["page_ref"].id, n["photo"].id]
    assert pack.merged_ids == [e["old_market"].id]
    assert n["rejected"].id not in ids(pack.notes)
    # a note that merely mentions the temple is the market's note, not the temple's
    assert n["bridge_ref"].id not in ids(get_context(store, PID, "Tree Temple").notes)


def test_scene_conditional_notes_are_kept_separate(world):
    store, e, n = world
    pack = get_context(store, PID, "Market")
    assert n["flood"].id not in ids(pack.notes)          # never the place's general state
    assert [c.scene_id for c in pack.conditional] == [e["s7"].id]
    assert ids(pack.conditional[0].notes) == [n["flood"].id]
    assert pack.conditional[0].label.startswith("7 · ")
    md = render_context_md(pack)
    assert md.index("## Description") < md.index("## Only during these scenes")
    assert "not the place's usual state" in md


def test_confirmed_containment_inherits_only_marked_notes(world):
    store, e, n = world
    pack = get_context(store, PID, "Market")
    assert [i.entity_id for i in pack.inherited] == [e["devgram"].id]
    assert ids(pack.inherited[0].notes) == [n["curfew"].id]   # within=True
    assert n["houses"].id not in ids(pack.inherited[0].notes)  # the parent's own extent
    assert [a.name for a in pack.ancestors] == ["Devgram"]
    assert "Inside: Devgram" in render_context_md(pack)


def test_proposed_containment_does_not_inherit(world):
    store, e, n = world
    pack = get_context(store, PID, "Devgram Lanes")
    assert pack.inherited == [] and pack.ancestors == []
    assert n["curfew"].id not in ids(pack.notes)


def test_notes_do_not_cross_to_an_unrelated_location(world):
    """The temple is across the river, so the village curfew must not reach it, and the
    temple's own rule must not reach the village."""
    store, e, n = world
    temple = get_context(store, PID, "Tree Temple")
    assert n["curfew"].id not in ids(temple.notes + [x for i in temple.inherited for x in i.notes])
    devgram = get_context(store, PID, "Devgram")
    assert n["temple_rule"].id not in ids(devgram.notes)
    assert devgram.inherited == []


def test_inheritance_chains_through_two_levels(world):
    store, e, n = world
    pack = get_context(store, PID, "Inner Sanctum")
    assert [i.name for i in pack.inherited] == ["Tree Temple"]
    assert ids(pack.inherited[0].notes) == [n["temple_rule"].id]
    assert n["temple_roots"].id not in ids(pack.inherited[0].notes)


def test_scene_pack_lists_locations_without_borrowing_their_notes(world):
    store, e, n = world
    pack = get_context(store, PID, "Scene 7")
    assert ids(pack.notes) == []                       # scene owns nothing itself
    assert {l.name for l in pack.locations} == {"Market Square", "Tree Temple"}
    assert n["grain"].id not in ids(pack.notes)


def test_confirmed_only_pack(world):
    store, _, n = world
    pack = get_context(store, PID, "Devgram", include_proposed=False)
    assert ids(pack.notes) == [n["curfew"].id]
    assert ids(pack.project_notes) == [n["tone"].id]
    assert "Confirmed notes only." in render_context_md(pack)


def test_project_pack_holds_unattributed_images(world):
    store, _, n = world
    pack = get_context(store, PID, PROJECT_SCOPE)
    assert set(ids(pack.notes)) == {n["tone"].id, n["loose_ref"].id}
    assert [r.note_id for r in pack.reference_images] == [n["loose_ref"].id]


def test_reference_images_resolve_to_uris(world):
    store, _, n = world
    pack = get_context(store, PID, "Market")
    assert [r.uri for r in pack.reference_images] == [
        f"gs://b/{LOOKBOOK}/pages/0003.png", f"gs://b/{IMG}/original.x"]
    assert "## Reference images" in render_context_md(pack)


def test_superseded_sources_are_flagged(world):
    store, e, n = world
    old = store.get_source(PID, SCRIPT)
    store.put_source(PID, old.touch(superseded_by_source_id="d" * 64))
    md = render_context_md(get_context(store, PID, "Market"))
    assert "superseded drafts: Dehleez_Ep1.pdf" in md
    assert "not been reconciled" in md


def test_index_and_export(world, tmp_path):
    store, e, _ = world
    index = render_index_md(store, PID)
    assert "## Locations (5)" in index and "## Scenes (2)" in index
    assert "inside Devgram ·" in index and "inside Devgram (proposed)" in index
    assert "Bazaar" not in index

    paths = export_context(store, PID, tmp_path)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in paths)
    assert names[0] == "INDEX.md"
    assert sum(p.startswith("locations/") for p in names) == 5
    assert sum(p.startswith("scenes/") for p in names) == 2
