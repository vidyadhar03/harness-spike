import pytest

from harness.memory.curate import merge_entities
from harness.memory.models import Location, Note, NoteOrigin, Project, Provenance, Scene, Source
from harness.memory.ports import MemoryStore
from harness.memory.retrieval import get_context, resolve_scope

PID = "prj_m"
SRC = "d" * 64


def note(body, owner):
    return Note(kind="description", body=body, owner_id=owner, author="agent",
                provenance=[Provenance(source_id=SRC)], origin=NoteOrigin(digest_version="v", source_id=SRC))


@pytest.fixture
def world():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Dehleez"))
    store.put_source(PID, Source(id=SRC, filename="script.pdf", mime_type="application/pdf", kind="document",
                                 size_bytes=1, storage_path=f"gs://b/{SRC}/original.pdf", status="digested"))
    approach = Location(name="Approach Road", aliases=["Devgram Approach"], author="agent")
    village = Location(name="Village Road", aliases=["Main road"], author="agent")
    third = Location(name="Lower Road", author="agent")
    scene = Scene(name="EXT. VILLAGE ROAD - DAY", number="3", location_ids=[approach.id], author="agent")
    store.put_entities(PID, [approach, village, third, scene])
    notes = {
        "approach": note("Dust and loose gravel on the verge.", approach.id),
        "village": note("Shops with pull-down shutters.", village.id),
        "third": note("A culvert crosses under the road.", third.id),
    }
    store.put_notes(PID, list(notes.values()))
    return store, approach, village, third, scene, notes


def test_merge_by_alias_sets_status_and_folds_aliases(world):
    store, approach, village, _, _, _ = world
    result = merge_entities(store, PID, "Devgram Approach", "Village Road")

    merged = store.get_entity(PID, approach.id)
    target = store.get_entity(PID, village.id)
    assert result.applied and merged.status == "merged" and merged.merged_into == village.id
    assert target.aliases == ["Main road", "Approach Road", "Devgram Approach"]
    assert result.notes_moved == 1
    assert "merged Approach Road" in result.summary()


def test_merged_notes_and_scenes_resolve_to_the_target(world):
    store, approach, village, _, scene, notes = world
    merge_entities(store, PID, approach.id, village.id)

    assert resolve_scope(store, PID, "Approach Road") == village.id
    pack = get_context(store, PID, "Village Road")
    assert {n.id for n in pack.notes} == {notes["approach"].id, notes["village"].id}
    assert [s.id for s in pack.scenes] == [scene.id]   # scene linked via the merged id
    assert pack.merged_ids == [approach.id]


def test_chain_merge_resolves_through_both_hops(world):
    store, approach, village, third, _, notes = world
    merge_entities(store, PID, approach.id, village.id)
    merge_entities(store, PID, village.id, third.id)

    assert resolve_scope(store, PID, "Devgram Approach") == third.id
    pack = get_context(store, PID, third.id)
    assert {n.id for n in pack.notes} == {n.id for n in notes.values()}
    assert set(pack.merged_ids) == {approach.id, village.id}


def test_dry_run_changes_nothing(world):
    store, approach, village, _, _, _ = world
    result = merge_entities(store, PID, "Approach Road", "Village Road", dry_run=True)

    assert not result.applied and result.aliases_added == ["Approach Road", "Devgram Approach"]
    assert result.notes_moved == 1 and "would merge" in result.summary()
    assert store.get_entity(PID, approach.id).status == "proposed"
    assert store.get_entity(PID, village.id).aliases == ["Main road"]


def test_rejections(world):
    store, approach, village, _, scene, _ = world

    with pytest.raises(ValueError, match="already the same entity"):
        merge_entities(store, PID, "Approach Road", "Devgram Approach")
    with pytest.raises(ValueError, match="cannot merge a scene into a location"):
        merge_entities(store, PID, "Scene 3", "Village Road")
    with pytest.raises(LookupError, match="did you mean"):
        merge_entities(store, PID, "Approach Rd", "Village Road")

    merge_entities(store, PID, approach.id, village.id)
    with pytest.raises(ValueError, match="already the same entity"):
        merge_entities(store, PID, village.id, approach.id)   # reverse merge is a no-op, not a cycle


def test_cycle_in_existing_data_is_refused(world):
    """Resolution normally returns live roots, so this only happens if the store is
    already inconsistent. The merge must refuse rather than deepen the loop."""
    store, approach, village, _, _, _ = world
    store.put_entities(PID, [
        approach.model_copy(update={"status": "merged", "merged_into": village.id}),
        village.model_copy(update={"status": "merged", "merged_into": approach.id}),
    ])
    with pytest.raises(ValueError, match="cycle"):
        merge_entities(store, PID, approach.id, village.id)


# --- note review ------------------------------------------------------------------

from harness.memory.curate import review_containment, review_note


def test_confirm_records_who_and_which_revision(world):
    store, approach, _, _, _, notes = world
    result = review_note(store, PID, notes["approach"].id, "confirmed", reviewer="vd")
    stored = next(n for n in store.list_notes(PID) if n.id == notes["approach"].id)
    assert stored.status == "confirmed" and stored.reviewed_by == "vd"
    assert stored.reviewed_revision == stored.revision and stored.review_is_current
    assert "confirmed" in result.summary()


def test_a_confirmed_note_stops_being_current_when_its_assertion_changes(world):
    store, _, _, _, _, notes = world
    review_note(store, PID, notes["approach"].id, "confirmed", reviewer="vd")
    stored = next(n for n in store.list_notes(PID) if n.id == notes["approach"].id)
    edited = stored.touch(body="Dust, gravel and a broken culvert.", revision=stored.revision + 1)
    assert not edited.review_is_current      # approval does not carry to text nobody agreed to


def test_rejection_needs_a_reason_and_duplicates_name_the_survivor(world):
    store, _, village, _, _, notes = world
    with pytest.raises(ValueError, match="needs a reason"):
        review_note(store, PID, notes["approach"].id, "rejected", reviewer="vd")

    review_note(store, PID, notes["approach"].id, "rejected", reviewer="vd",
                duplicate_of=notes["village"].id)
    stored = next(n for n in store.list_notes(PID) if n.id == notes["approach"].id)
    assert stored.status == "rejected" and stored.review_reason == "duplicate"
    assert stored.duplicate_of == notes["village"].id     # not "false"; the fact still stands

    with pytest.raises(ValueError, match="duplicate of itself"):
        review_note(store, PID, notes["village"].id, "rejected", reviewer="vd",
                    duplicate_of=notes["village"].id)
    with pytest.raises(ValueError, match="only a rejected note"):
        review_note(store, PID, notes["third"].id, "confirmed", reviewer="vd",
                    duplicate_of=notes["village"].id)


def test_set_and_confirm_parent(world):
    store, approach, village, third, _, _ = world
    loc = review_containment(store, PID, "Approach Road", "confirmed", reviewer="vd",
                             parent_ref="Village Road")
    assert loc.containment.parent_id == village.id and loc.containment.status == "confirmed"
    assert loc.containment.reviewed_by == "vd"

    with pytest.raises(ValueError, match="cannot contain itself"):
        review_containment(store, PID, approach.id, "confirmed", reviewer="vd", parent_ref=approach.id)
    with pytest.raises(ValueError, match="cycle"):
        review_containment(store, PID, village.id, "confirmed", reviewer="vd", parent_ref=approach.id)
    with pytest.raises(ValueError, match="containment is between locations"):
        review_containment(store, PID, "Scene 3", "confirmed", reviewer="vd", parent_ref=village.id)
    with pytest.raises(ValueError, match="no proposed parent"):
        review_containment(store, PID, third.id, "confirmed", reviewer="vd")
