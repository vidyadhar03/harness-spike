"""Concept-art versions and their explicit visual-direction approval.

Uploading or listing a version never approves anything - only lock_approval does, and
only after re-deriving the exact snapshot fresh and checking it both against what the
caller was shown (context_token) and against whatever is currently locked
(expected_revision). See API_CONTRACT.md for the endpoint contract this backs.

Concept versions are deliberately not Notes: keeping them out of the Note collection is
what keeps them invisible to note review, retrieval context packs, and the references
pipeline's stale-note cleanup, with no filtering code needed anywhere for that.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .config import Settings
from .files import extension_for, kind_for, sniff_mime, validate_reference_image
from .models import (
    ApprovedReference, ConceptApproval, ConceptVersion, ConditionalNoteSnapshot,
    InheritedNoteSnapshot, Location, Note, Source,
)
from .ports import Blobs, Store
from .retrieval import BRIEF_NOTE_KINDS as _BRIEF_KINDS
from .retrieval import _Graph, get_context, resolve_scope

# Bumped when the *shape/completeness semantics* of a persisted ConceptApproval's
# snapshot changes in a way that makes old data ambiguous under new logic - see
# models.ConceptApproval.snapshot_schema_version. lock_approval always stamps this
# current value; nothing else ever writes to that field.
CURRENT_SNAPSHOT_SCHEMA_VERSION = 2

# Small explicit label, not a paragraph - see build_approval_snapshot's
# depiction_label parameter and API_CONTRACT.md.
MAX_DEPICTION_LABEL_LENGTH = 200

# Attached to the "core description" section wherever it's rendered (mapping.py).
# Static and always present when that section has notes - not computed per-note,
# because nothing in the stored data reliably distinguishes which notes (if any) need
# it: see build_approval_snapshot/_brief_from_pack's module docstring for the full
# reasoning. Kept here, next to the code whose limitation it describes.
CORE_NOTES_CAVEAT = (
    "These are the location's own standing notes. Extraction doesn't always separate "
    "a permanent fact about the place from something that's really only true in one "
    "scene (for example, an object simply being present, versus how a photo's "
    "reflection happens to look in one particular scene). If a note here should "
    "actually be scoped to a scene, or split into a standing part and a scene-specific "
    "part, use \"Correct this note\" to fix it - nothing here does that automatically."
)


class StaleApprovalContext(RuntimeError):
    """The recomputed context_token for (concept_version_id, reference_ids) does not match
    what the caller was shown - the brief or a selected reference changed since they last
    fetched a preview. Distinct from ports.ApprovalConflict, which guards a race between
    concurrent lock attempts rather than staleness of the content being locked.
    """
    def __init__(self, location_id: str, expected_token: str, actual_token: str):
        super().__init__(
            f"the approval context for {location_id} has changed since it was fetched "
            f"(expected token {expected_token[:12]}..., now {actual_token[:12]}...); "
            "re-fetch the preview and try again"
        )
        self.location_id = location_id
        self.expected_token, self.actual_token = expected_token, actual_token


# --- concept versions -----------------------------------------------------------------

def _concept_version_id(location_id: str, source_id: str) -> str:
    """Deterministic per (location, bytes) - a repeat upload of the same image to the
    same location always resolves to this same version id, mirroring
    references._attachment_note_id's exact rationale."""
    digest = hashlib.sha256(f"concept-version\x00{location_id}\x00{source_id}".encode()).hexdigest()
    return f"cvn_{digest[:12]}"


def register_concept_bytes(store: Store, blobs: Blobs, settings: Settings, project_id: str,
                           data: bytes, filename: str, *, max_pixels: int | None = None) -> Source:
    """Validate image bytes and register them as a concept-purpose Source (content-addressed,
    create-if-absent - identical bytes already stored under ANY purpose are reused untouched, so
    source purpose/ingest eligibility is never changed here). Shared by the upload path and the
    generated-image import so both get identical validation and dedup behavior.
    max_pixels overrides the default pixel bound (files.MAX_REFERENCE_PIXELS) - still a bound.
    Raises files.ImageValidationError (a ValueError) on invalid image content."""
    from .ingest import source_uri   # local import: avoids a circular import at module load

    mime = sniff_mime(data, filename)
    validate_reference_image(data, mime, max_pixels=max_pixels)

    sid = hashlib.sha256(data).hexdigest()
    uri = source_uri(settings, project_id, sid, "original" + extension_for(filename, mime))
    blobs.put(uri, data, mime)

    concept_src = Source(
        id=sid, filename=filename, mime_type=mime, kind=kind_for(mime), doc_type="concept",
        size_bytes=len(data), storage_path=uri, status="digested",
        source_purpose="concept",
    )
    stored_src, _ = store.put_source_if_absent(project_id, concept_src)
    return stored_src


def upload_concept_version(store: Store, blobs: Blobs, settings: Settings, project_id: str,
                           location_id: str, data: bytes, filename: str) -> tuple[ConceptVersion, bool]:
    """Store a user-uploaded concept-art image as a new version for a location.

    `location_id` must already be a resolved, validated Location id - this function
    does not resolve scope or check entity type, matching upload_reference_image's
    convention of leaving that to the caller (see routes/concepts.py).

    Returns (version, created). created=False means these exact bytes were already
    uploaded as a version for this location - the existing record is returned
    unchanged. Never touches any existing approval either way: an approval only ever
    points at a version id, and uploading a new candidate cannot rewrite that pointer.

    Reuses the same validation as reference-image uploads (ACCEPTED_REFERENCE_MIMES,
    MAX_REFERENCE_PIXELS, animated-image rejection) - see files.validate_reference_image.
    Raises ValueError on invalid image content (-> HTTP 400 at the route).
    """
    stored_src = register_concept_bytes(store, blobs, settings, project_id, data, filename)
    sid = stored_src.id

    version = ConceptVersion(
        id=_concept_version_id(location_id, sid), location_id=location_id,
        source_id=stored_src.id, filename=filename, author="user",
    )
    return store.put_concept_version_if_absent(project_id, version)


def promote_reference_to_concept(store: Store, project_id: str, location_id: str,
                                 reference_note_id: str) -> tuple[ConceptVersion, bool]:
    """Create a concept-candidate version from an existing, applicable reference image.

    No bytes are downloaded or re-uploaded through this call at all: the new version
    points directly at the reference note's own already-stored Source
    (note.provenance[0].source_id), the exact same object the reference's own preview
    image route serves. Never mutates the reference note - no confirm/reject, no
    revision bump; this is a read of the reference, not a review of it.

    `location_id` must already be a resolved, validated Location id - matches
    upload_concept_version's convention (see routes/concepts.py).

    Applicability: the reference must be owned by this location, or inherited from a
    confirmed ancestor with include_descendants - the same rule
    _reference_notes_for_location/build_approval_snapshot use for approval selection.
    Unlike approval selection, a PROPOSED reference is permitted here (creating a
    candidate is exploratory, not an approval decision) - only a REJECTED reference is
    refused, with a clear ValueError naming why. This is deliberately a *different*,
    looser gate than build_approval_snapshot's confirmed-only one: using a reference as
    a concept candidate and selecting a confirmed reference as supporting evidence for
    an approval are two independent actions on two independent fields
    (ConceptApproval.concept_version_id vs. .references) - nothing stops the same
    reference note from playing both roles in the same approval, and nothing here
    requires it to.

    Idempotent and concurrency-safe by construction, not by any new locking: the
    resulting version id is deterministic from (location_id, source_id) - see
    _concept_version_id - the exact id upload_concept_version would also produce for
    the same bytes. store.put_concept_version_if_absent (already thread-safe/
    transactional) is what actually makes a retry, a race between two promotions of
    the same reference, or a race between a promotion and a raw upload of the same
    bytes all resolve to one record. Because that call never overwrites an existing
    record, provenance (promoted_from_note_id/_revision below) is set only by whichever
    creation attempt wins first and is never rewritten by a later, different call that
    happens to resolve to the same (location, bytes) - including a later promotion
    naming a *different* reference note that happens to share the same underlying
    image; the first-recorded provenance is preserved, not silently replaced.

    Returns (version, created) - see upload_concept_version's docstring for the same
    contract. Never touches any existing approval either way.
    """
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; concept art is per location")

    any_status = _reference_notes_for_location(store, project_id, scope_id, graph, require_confirmed=False)
    note = any_status.get(reference_note_id)
    if note is None:
        raise ValueError(f"{reference_note_id!r} is not a reference applicable to location {scope_id!r}")
    if note.status == "rejected":
        raise ValueError(
            f"{reference_note_id!r} is rejected and cannot be promoted to a concept candidate"
        )

    source_id = note.provenance[0].source_id if note.provenance else None
    if source_id is None:
        raise ValueError(f"{reference_note_id!r} has no stored image to promote")
    source = store.get_source(project_id, source_id)
    if source is None:
        raise LookupError(f"the image behind reference {reference_note_id!r} is unavailable")

    version = ConceptVersion(
        id=_concept_version_id(scope_id, source_id), location_id=scope_id,
        source_id=source_id, filename=source.filename, author="user",
        promoted_from_note_id=note.id, promoted_from_note_revision=note.revision,
    )
    return store.put_concept_version_if_absent(project_id, version)


# --- approval context: snapshot, token, staleness --------------------------------------

@dataclass
class ApprovalSnapshot:
    """What a lock right now would capture, for one (concept_version, reference_ids)
    pair - shared by the preview endpoint (read-only) and lock_approval (which checks
    it, then persists it) so the two can never drift apart."""
    location_id: str
    concept_version: ConceptVersion
    brief_notes: list[Note]
    scene_requirements: list[ConditionalNoteSnapshot]
    brief_inherited: list[InheritedNoteSnapshot]
    brief_ancestors: list[str]
    superseded_sources: list[str]
    references: list[ApprovedReference]
    depiction_label: str | None
    context_token: str


def _validate_depiction_label(label: str | None) -> str | None:
    """Strips whitespace, treats an empty result as None. Raises ValueError if over
    length (-> 400 at the route). Purely a string - never parsed, never checked
    against any location/room hierarchy; see build_approval_snapshot's docstring."""
    if label is None:
        return None
    stripped = label.strip()
    if not stripped:
        return None
    if len(stripped) > MAX_DEPICTION_LABEL_LENGTH:
        raise ValueError(
            f"depictionLabel is {len(stripped)} characters; the limit is "
            f"{MAX_DEPICTION_LABEL_LENGTH}"
        )
    return stripped


def _brief_from_pack(pack) -> tuple[list[Note], list[ConditionalNoteSnapshot], list[InheritedNoteSnapshot]]:
    """The brief exactly as the Brief tab (LocationDetail) shows it, plus complete
    scene coverage:
    - unconditional description/constraint/tone notes (the "brief" proper);
    - one entry per scene in pack.scenes - the location's authoritative linked-scene
      roster, already computed by retrieval.get_context/resolve_scope and reused as-is
      here, never recomputed - regardless of whether that scene has any
      scene-conditional notes. A scene with none gets notes=[], which is a real,
      positive fact ("nothing scene-specific was extracted for this scene"), not an
      omission - this is what actually fixes a location whose linked scenes include
      some with no scene-conditional notes: previously only pack.conditional (which
      skips note-less scenes entirely) fed this, silently dropping them;
    - the same notes when inherited from a confirmed ancestor - retaining citations and
      applicability throughout.
    """
    notes = [n for n in pack.notes if n.kind in _BRIEF_KINDS]

    notes_by_scene: dict[str, list[Note]] = {c.scene_id: c.notes for c in pack.conditional}
    roster_ids = {s.id for s in pack.scenes}
    scene_requirements: list[ConditionalNoteSnapshot] = []
    for s in pack.scenes:
        kept = [n for n in notes_by_scene.get(s.id, []) if n.kind in _BRIEF_KINDS]
        label = f"{s.number} · {s.name}" if s.number else s.name
        scene_requirements.append(ConditionalNoteSnapshot(
            scene_id=s.id, label=label, notes=kept, number=s.number, heading=s.name,
        ))
    # Defensive, not expected in practice: a scene-conditional note whose scene_id does
    # not resolve to any scene in the location's own linked-scene roster (pack.scenes)
    # would previously still have shown up via pack.conditional. Keep surfacing it
    # rather than silently dropping it just because this function now iterates the
    # roster first - this is exactly the "don't silently drop ambiguous requirements"
    # case the roster-based rewrite must not introduce as a regression.
    for c in pack.conditional:
        if c.scene_id not in roster_ids:
            kept = [n for n in c.notes if n.kind in _BRIEF_KINDS]
            if kept:
                scene_requirements.append(ConditionalNoteSnapshot(
                    scene_id=c.scene_id, label=c.label, notes=kept, number=None, heading=None,
                ))

    inherited: list[InheritedNoteSnapshot] = []
    for i in pack.inherited:
        kept = [n for n in i.notes if n.kind in _BRIEF_KINDS]
        if kept:
            inherited.append(InheritedNoteSnapshot(entity_id=i.entity_id, name=i.name, notes=kept))

    return notes, scene_requirements, inherited


def _reference_notes_for_location(store: Store, project_id: str, scope_id: str, graph: _Graph, *,
                                  require_confirmed: bool = True) -> dict[str, Note]:
    """reference_image notes applicable to this location - owned, or inherited from a
    confirmed ancestor with include_descendants - matching the exact ownership/
    inheritance rule api.mapping.reference_list uses (kept domain-only here on purpose:
    this module must not depend on the API layer).

    require_confirmed=True (the default, used to validate what may be *selected* into
    an approval) restricts to status=="confirmed" only - a proposed or rejected note is
    not "applicable" for approval purposes, regardless of ownership. Pass False (used
    only by approval_staleness_reasons, to explain *why* a previously-selected
    reference is no longer eligible) to see a note regardless of its current status.
    """
    found: dict[str, Note] = {}
    for n in store.notes_for_owners(project_id, graph.own_ids(scope_id)):
        if n.kind == "reference_image" and (require_confirmed is False or n.status == "confirmed"):
            found[n.id] = n
    for parent in graph.ancestors(scope_id):
        for n in store.notes_for_owners(project_id, graph.own_ids(parent.id)):
            if (n.kind == "reference_image" and n.applicability.include_descendants
                    and n.applicability.scene_id is None
                    and (require_confirmed is False or n.status == "confirmed")):
                found.setdefault(n.id, n)
    return found


def _context_token(concept_version_id: str, concept_source_id: str, brief_notes: list[Note],
                   scene_requirements: list[ConditionalNoteSnapshot], inherited: list[InheritedNoteSnapshot],
                   references: list[ApprovedReference], depiction_label: str | None) -> str:
    """sha256 over exactly the fields that decide whether an approval is stale: concept
    identity, the user-supplied depiction label, and (id, revision, status) for every
    brief/scene note and selected reference - never the note bodies themselves, so an
    edit that bumps revision is always caught without this function needing to know
    what changed.

    Also hashes each scene's own (scene_id, number, heading) - not just its notes -
    so relinking a scene to/from this location, or a heading/number change, moves the
    token even when that scene's own note set happens to be unaffected (e.g. it has no
    notes at all). This is what "context tokens cover the scene roster" means in
    practice: the roster's *membership and identity*, not only its note contents.

    Deterministic/canonical by construction, independent of caller-supplied or
    retrieval-order input:
    - every list is wrapped in sorted(...) here, so the *order* build_approval_snapshot
      happened to gather brief/scene/inherited notes or the caller's reference_ids in
      never affects the resulting hash;
    - only id/revision/status (plus, for scenes, number/heading) are hashed - never
      created_at/updated_at/locked_at or any other timestamp, so re-fetching unchanged
      data twice (or a plain re-save that only bumps updated_at without changing
      revision) always yields the same token.
    json.dumps(..., sort_keys=True) additionally canonicalizes key order, though the
    sorted() calls above are what actually make the *content* order-independent.
    """
    payload = {
        "conceptVersionId": concept_version_id,
        "conceptSourceId": concept_source_id,
        "depictionLabel": depiction_label,
        "brief": sorted((n.id, n.revision, n.status) for n in brief_notes),
        "sceneRequirements": sorted(
            (s.scene_id, s.number, s.heading, sorted((n.id, n.revision, n.status) for n in s.notes))
            for s in scene_requirements
        ),
        "inherited": sorted(
            (i.entity_id, sorted((n.id, n.revision, n.status) for n in i.notes)) for i in inherited
        ),
        "references": sorted((r.note_id, r.revision, r.status) for r in references),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_approval_snapshot(store: Store, project_id: str, location_id: str,
                            concept_version_id: str, reference_ids: list[str],
                            depiction_label: str | None = None) -> ApprovalSnapshot:
    """Read-only: resolves scope itself (like retrieval.get_context), so both the
    preview endpoint and lock_approval can call this directly. Raises LookupError for
    an unknown location/concept version, ValueError for a scene id, an unrelated
    reference id, a reference id that is not confirmed, or an over-length
    depiction_label (-> 404 / 400 at the route).

    A reference may only be selected if it is BOTH confirmed AND applicable to this
    location through the existing ownership/inheritance rule (owned, or inherited from
    a confirmed ancestor with include_descendants) - ties approval selection to the
    same review gate the rest of the system already uses for "this reference is
    trustworthy," rather than letting a still-proposed or already-rejected note into
    an approved package. This function never itself confirms anything; it only reads.
    (Note: promote_reference_to_concept, used to *create a candidate* rather than
    select supporting evidence, deliberately applies a looser gate - see its own
    docstring for why those are two independent actions.)

    depiction_label is a small, free-text, user-supplied description of what the
    concept image depicts (e.g. "whole-house exterior", "bedroom interior - top
    view") - validated only for length (_validate_depiction_label), never parsed or
    checked against any location/room relationship, and never used to change
    concept_version.location_id or imply spatial accuracy. It is folded into the
    context_token like everything else here, so changing it (or clearing it) between
    a preview and a lock is caught as staleness like any other change.

    Duplicate reference_ids are silently deduplicated, keeping first-seen order.
    """
    graph = _Graph(store.list_entities(project_id))
    scope_id = resolve_scope(store, project_id, location_id, graph)
    entity = graph.by_id.get(scope_id)
    if not isinstance(entity, Location):
        raise ValueError(f"{location_id!r} is a scene; concept approval is per location")

    version = store.get_concept_version(project_id, concept_version_id)
    if version is None or version.location_id != scope_id:
        raise LookupError(f"no concept version {concept_version_id!r} for location {scope_id!r}")

    label = _validate_depiction_label(depiction_label)

    pack = get_context(store, project_id, scope_id, include_proposed=True)
    brief_notes, scene_requirements, inherited = _brief_from_pack(pack)

    # Gathered without a status filter first, purely so a "you picked something
    # unrelated" 400 can be told apart from a "you picked something not confirmed yet"
    # 400 - both are clear, distinct messages rather than one generic rejection.
    any_status = _reference_notes_for_location(store, project_id, scope_id, graph, require_confirmed=False)
    seen: set[str] = set()
    references: list[ApprovedReference] = []
    for rid in reference_ids:
        if rid in seen:
            continue
        seen.add(rid)
        note = any_status.get(rid)
        if note is None:
            raise ValueError(f"{rid!r} is not a reference applicable to location {scope_id!r}")
        if note.status != "confirmed":
            raise ValueError(
                f"{rid!r} is {note.status}, not confirmed; only confirmed references "
                "may be selected for approval"
            )
        references.append(ApprovedReference(
            note_id=note.id, revision=note.revision, status=note.status,
            guidance=note.guidance, direction=note.direction, caption=note.body,
        ))

    token = _context_token(version.id, version.source_id, brief_notes, scene_requirements,
                           inherited, references, label)
    return ApprovalSnapshot(
        location_id=scope_id, concept_version=version, brief_notes=brief_notes,
        scene_requirements=scene_requirements, brief_inherited=inherited,
        brief_ancestors=[a.name for a in pack.ancestors], superseded_sources=pack.superseded_sources,
        references=references, depiction_label=label, context_token=token,
    )


# --- locking ----------------------------------------------------------------------------

def lock_approval(store: Store, project_id: str, location_id: str, *, concept_version_id: str,
                  reference_ids: list[str], context_token: str, expected_revision: int,
                  locked_by: str | None, depiction_label: str | None = None) -> tuple[ConceptApproval, bool]:
    """Explicitly lock a concept version + brief/context + selected references as the
    approved visual direction.

    THE EXACT VALIDATION/COMMIT BOUNDARY (read this before changing either half):

    1. Validate (this function, once): recompute the snapshot fresh from live store
       state and require it to hash to the same context_token the caller was shown -
       raises StaleApprovalContext otherwise (-> 409). This is the only point where
       underlying Note state is read and checked against anything.
    2. Build: the ConceptApproval to persist is constructed directly from the fields of
       THAT SAME validated `snapshot` object - never re-read from the store, never
       rebuilt by a second call to build_approval_snapshot. This is deliberate: it is
       the only way to guarantee what gets committed is byte-for-byte the thing whose
       token was just checked, not a second, independently-fetched read that could
       already differ from it.
    3. Commit (Store.put_approval_if_current): atomic, but it does NOT re-validate
       brief/reference freshness - it only guards two things, both about the "current
       approval" pointer, never about whether the notes underneath changed again since
       step 1: (a) idempotency - if the pointer's current approval already has this
       exact context_token, return it unchanged; (b) a revision CAS against
       expected_revision, guarding a race between two *concurrent lock attempts*.

    What this means in practice: there is a real, if narrow, window between step 1 and
    step 3 in which a concurrent request (on another thread, process, or replica) could
    edit a note that this snapshot already captured. Nothing re-checks that window, and
    nothing here rebuilds the snapshot to "catch up" if it did happen - this system
    approves the exact, previously-previewed snapshot the caller confirmed, not
    "whatever the absolute latest state is at the instant of commit." Any drift that
    landed in that window is not silently hidden: it shows up afterward as
    approval_staleness_reasons()/GET .../approval's isStale, exactly as if the edit had
    happened a minute later. For this local, single-process slice that is an accepted,
    documented trade-off, not a bug - implementing true commit-time context
    revalidation would mean re-running build_approval_snapshot's store reads *inside*
    Store.put_approval_if_current's own lock/transaction, which the Store protocol does
    not support today (see ports.Store.put_approval_if_current's docstring).

    Never writes to a Note: guidance/status shown here are recorded as seen, not
    changed. See models.ConceptApproval for the persisted shape and history contract.
    """
    snapshot = build_approval_snapshot(store, project_id, location_id, concept_version_id,
                                       reference_ids, depiction_label)
    if snapshot.context_token != context_token:
        raise StaleApprovalContext(snapshot.location_id, context_token, snapshot.context_token)

    approval = ConceptApproval(
        location_id=snapshot.location_id,
        concept_version_id=snapshot.concept_version.id,
        concept_source_id=snapshot.concept_version.source_id,
        concept_filename=snapshot.concept_version.filename,
        brief_notes=snapshot.brief_notes, brief_conditional=snapshot.scene_requirements,
        brief_inherited=snapshot.brief_inherited, brief_ancestors=snapshot.brief_ancestors,
        superseded_sources=snapshot.superseded_sources, references=snapshot.references,
        depiction_label=snapshot.depiction_label, context_token=snapshot.context_token,
        locked_by=locked_by, snapshot_schema_version=CURRENT_SNAPSHOT_SCHEMA_VERSION,
    )
    return store.put_approval_if_current(project_id, snapshot.location_id, approval, expected_revision)


def _flatten_notes(*groups: list[Note]) -> dict[str, Note]:
    flat: dict[str, Note] = {}
    for group in groups:
        for n in group:
            flat[n.id] = n
    return flat


def _note_set_diff_reasons(stored: dict[str, Note], live: dict[str, Note], where: str,
                           all_notes_by_id: dict[str, Note] | None = None) -> list[str]:
    """Generic (id, revision, status) diff between a stored note set and its live
    counterpart, worded with `where` (e.g. "the brief", "scene 7") so the same helper
    serves both the unconditional brief and each scene's own requirements.

    A note missing from `live` usually means genuinely gone from retrieval (deleted, or
    plainly rejected) - but a corrected note is *also* invisible to `live` (rejected
    notes are never visible, by the same rule as any other rejection - see
    retrieval._visible), even though it still exists in the store with a clear record
    of why and what replaced it. all_notes_by_id (every note in the project, unfiltered
    by status - see approval_staleness_reasons) lets this say that precisely instead of
    the generic, more alarming-sounding "removed".
    """
    reasons: list[str] = []
    for nid, stored_note in stored.items():
        live_note = live.get(nid)
        if live_note is None:
            corrected = all_notes_by_id.get(nid) if all_notes_by_id else None
            if corrected is not None and corrected.review_reason == "corrected":
                successors = ", ".join(corrected.superseded_by_note_ids) or "none recorded"
                reasons.append(
                    f"a note in {where} ({nid}) was corrected since this was approved "
                    f"- replaced by: {successors}"
                )
            else:
                reasons.append(f"a note in {where} ({nid}) was removed since this was approved")
        elif live_note.revision != stored_note.revision or live_note.status != stored_note.status:
            reasons.append(
                f"a note in {where} ({nid}) changed since this was approved "
                f"(was revision {stored_note.revision}/{stored_note.status}, "
                f"now {live_note.revision}/{live_note.status})"
            )
    added = sorted(set(live) - set(stored))
    if added:
        plural = "s" if len(added) != 1 else ""
        reasons.append(f"{len(added)} new note{plural} in {where} since this was approved")
    return reasons


def _scene_diff_reasons(stored_scenes: list[ConditionalNoteSnapshot],
                        live_scenes: list[ConditionalNoteSnapshot],
                        all_notes_by_id: dict[str, Note] | None = None) -> list[str]:
    """Only meaningful when both sides represent a COMPLETE scene roster (current
    schema on both) - see approval_staleness_reasons, which skips this entirely for a
    legacy (partial-roster) approval rather than diffing against an admittedly
    incomplete baseline."""
    stored_by_id = {s.scene_id: s for s in stored_scenes}
    live_by_id = {s.scene_id: s for s in live_scenes}
    reasons: list[str] = []
    for sid, stored in stored_by_id.items():
        live = live_by_id.get(sid)
        heading = stored.heading or stored.label
        if live is None:
            reasons.append(f"scene {sid} ({heading}) is no longer linked to this location")
            continue
        if live.number != stored.number or live.heading != stored.heading:
            reasons.append(
                f"scene {sid} heading/number changed since this was approved "
                f"(was {stored.number!r} {stored.heading!r}, now {live.number!r} {live.heading!r})"
            )
        reasons += _note_set_diff_reasons(
            {n.id: n for n in stored.notes}, {n.id: n for n in live.notes}, f"scene {stored.number or sid}",
            all_notes_by_id,
        )
    added = sorted(set(live_by_id) - set(stored_by_id))
    if added:
        plural = "s" if len(added) != 1 else ""
        reasons.append(f"{len(added)} newly linked scene{plural} since this was approved")
    return reasons


def _reference_diff_reasons(store: Store, project_id: str, scope_id: str, graph: _Graph,
                            approved_references: list[ApprovedReference]) -> list[str]:
    reasons: list[str] = []
    if not approved_references:
        return reasons
    any_status = _reference_notes_for_location(store, project_id, scope_id, graph, require_confirmed=False)
    for r in approved_references:
        note = any_status.get(r.note_id)
        if note is None:
            reasons.append(f"reference {r.note_id} no longer applies to this location")
        elif note.status != "confirmed":
            reasons.append(f"reference {r.note_id} is no longer confirmed (now {note.status})")
        elif note.revision != r.revision:
            reasons.append(f"reference {r.note_id} changed since this was approved (now revision {note.revision})")
    return reasons


def approval_staleness_reasons(store: Store, project_id: str, approval: ConceptApproval) -> list[str]:
    """Human-readable reasons the live inputs behind this already-locked package have
    diverged from what was approved. Empty list means not stale.

    Never raises and never touches (or requires re-deriving) the stored approval -
    GET .../approval must stay fully readable even when the location was later
    rejected/merged, or when every reference this package named has since been
    rejected, edited, or stopped applying to this location. Deliberately does not
    reuse build_approval_snapshot: that function is written to validate a *prospective*
    selection (and raises on anything not confirmed/applicable, by design - see its
    docstring), which is the wrong tool for explaining why an *already-approved*
    package no longer matches live state without blowing up the read.

    A legacy approval (snapshot_schema_version < CURRENT_SNAPSHOT_SCHEMA_VERSION) always
    gets exactly one explicit reason about that, in place of a detailed scene-by-scene
    diff: its own stored scene list is, by construction, potentially missing scenes
    that never had a scene-conditional note (see ConditionalNoteSnapshot's docstring) -
    diffing that admittedly-incomplete baseline against the current, complete roster
    would report every such scene as "newly linked", which is not true and would be
    actively misleading. The brief-notes and reference diffs below are unaffected by
    that gap and still run normally even for a legacy approval.
    """
    graph = _Graph(store.list_entities(project_id))
    raw = graph.by_id.get(approval.location_id)
    if raw is None:
        return ["the approved location no longer exists"]
    if raw.status == "rejected":
        return ["the approved location was rejected"]
    if raw.status == "merged":
        return [f"the approved location was merged into {raw.merged_into}"]
    if not isinstance(raw, Location):
        return ["the approved location is no longer a location"]

    reasons: list[str] = []
    if approval.snapshot_schema_version < CURRENT_SNAPSHOT_SCHEMA_VERSION:
        reasons.append(
            f"this approval predates full scene-roster coverage (schema v"
            f"{approval.snapshot_schema_version}); scenes shown with no requirements "
            "may simply be missing from this old snapshot, not confirmed empty - "
            "re-lock to capture complete, current scene coverage"
        )

    pack = get_context(store, project_id, raw.id, include_proposed=True)
    live_notes, live_scenes, live_inherited = _brief_from_pack(pack)
    # Unfiltered by status, project-wide - lets a "missing from live" note be told
    # apart from a genuinely corrected one (see _note_set_diff_reasons). One extra
    # store read per staleness check; this project's note counts stay small enough
    # (see mapping.find_note's own O(n) precedent) that this isn't worth caching.
    all_notes_by_id = {n.id: n for n in store.list_notes(project_id)}

    stored_brief = _flatten_notes(approval.brief_notes)
    live_brief = _flatten_notes(live_notes)
    reasons += _note_set_diff_reasons(stored_brief, live_brief, "the brief", all_notes_by_id)

    stored_inherited = _flatten_notes(*(i.notes for i in approval.brief_inherited))
    live_inherited_flat = _flatten_notes(*(i.notes for i in live_inherited))
    reasons += _note_set_diff_reasons(stored_inherited, live_inherited_flat, "inherited context",
                                      all_notes_by_id)

    if approval.snapshot_schema_version >= CURRENT_SNAPSHOT_SCHEMA_VERSION:
        reasons += _scene_diff_reasons(approval.brief_conditional, live_scenes, all_notes_by_id)

    reasons += _reference_diff_reasons(store, project_id, raw.id, graph, approval.references)
    return reasons
