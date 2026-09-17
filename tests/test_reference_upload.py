import io
import hashlib
import threading
from unittest.mock import Mock
import pytest
from PIL import Image

from harness.api.main import create_app
from harness.api.settings import ApiSettings
from harness.memory.cli import main as cli_main
from harness.memory.config import Settings
from harness.memory.files import (
    ACCEPTED_REFERENCE_MIMES,
    MAX_REFERENCE_PIXELS,
    sniff_mime,
    validate_reference_image,
)
from harness.memory.ingest import Ctx as IngestCtx, register_file
from harness.memory.models import Location, Note, Project, Provenance, Scene, Source
from harness.memory.ports import (
    ImageHit,
    MemoryBlobs,
    MemoryImages,
    MemoryStore,
)
from harness.memory.references import (
    RefCtx,
    _attachment_note_id,
    _store_image,
    upload_reference_image,
)
from starlette.testclient import TestClient

PID = "prj_upload_test"


def _make_image(fmt="JPEG", size=(80, 80), color="blue"):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_animated_webp():
    buf = io.BytesIO()
    f1 = Image.new("RGB", (50, 50), color="red")
    f2 = Image.new("RGB", (50, 50), color="blue")
    f1.save(buf, format="WEBP", save_all=True, append_images=[f2], duration=100)
    return buf.getvalue()


@pytest.fixture
def harness_env():
    store = MemoryStore()
    store.put_project(Project(id=PID, name="Upload Test Film"))
    loc = Location(name="Mountain Pass", status="confirmed", author="agent")
    scene = Scene(name="EXT. MOUNTAIN PASS - DAY", number="1", location_ids=[loc.id], author="agent")
    store.put_entities(PID, [loc, scene])
    blobs = MemoryBlobs()
    images = MemoryImages({}, {}, {})
    settings = Settings(gcp_project="test-gcp", bucket="test-bucket")
    return store, blobs, images, settings, loc, scene


def make_client(store, blobs, images, *, max_upload_bytes=10_000_000):
    app = create_app(
        settings=Settings(gcp_project="t", bucket="b"),
        api_settings=ApiSettings(
            allowed_hosts=("testserver", "127.0.0.1", "localhost"),
            max_upload_bytes=max_upload_bytes,
        ),
        store=store,
        blobs=blobs,
        images=images,
    )
    return TestClient(app, base_url="http://testserver")


# ==========================================================================================
# Domain Validation & Attachment Creation Tests
# ==========================================================================================

def test_validate_reference_image_formats():
    for fmt in ("JPEG", "PNG", "WEBP"):
        data = _make_image(fmt=fmt, size=(120, 90))
        mime = sniff_mime(data, f"img.{fmt.lower()}")
        w, h = validate_reference_image(data, mime)
        assert (w, h) == (120, 90)

    # Unsupported format (GIF)
    gif_buf = io.BytesIO()
    Image.new("RGB", (50, 50)).save(gif_buf, format="GIF")
    with pytest.raises(ValueError, match="unsupported image format"):
        validate_reference_image(gif_buf.getvalue(), "image/gif")

    # Animated WebP
    anim_data = _make_animated_webp()
    anim_mime = sniff_mime(anim_data, "anim.webp")
    with pytest.raises(ValueError, match="animated images are not supported"):
        validate_reference_image(anim_data, anim_mime)

    # Corrupt data
    corrupt_data = bytes.fromhex("ffd8ffe000104a46494600010101006000600000") + b"garbage" * 20
    with pytest.raises(ValueError, match="invalid image"):
        validate_reference_image(corrupt_data, "image/jpeg")

    # Mismatched format
    jpeg_data = _make_image(fmt="JPEG")
    with pytest.raises(ValueError, match="declared.*image/png.*decoded as.*JPEG"):
        validate_reference_image(jpeg_data, "image/png")

    # Limit on pixels
    import harness.memory.files as mf
    orig_limit = mf.MAX_REFERENCE_PIXELS
    try:
        mf.MAX_REFERENCE_PIXELS = 100
        with pytest.raises(ValueError, match="image too large"):
            validate_reference_image(jpeg_data, "image/jpeg")
    finally:
        mf.MAX_REFERENCE_PIXELS = orig_limit


def test_upload_reference_image_domain_logic(harness_env):
    store, blobs, _, settings, loc, _ = harness_env
    img_data = _make_image(fmt="JPEG", color="green")
    note, created = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data, "mountain.jpg"
    )
    assert created is True
    assert note.owner_id == loc.id
    assert note.kind == "reference_image"
    assert note.status == "proposed"
    assert note.body == "mountain.jpg"
    assert note.revision == 1

    src_id = note.provenance[0].source_id
    src = store.get_source(PID, src_id)
    assert src is not None
    assert src.source_purpose == "reference"
    assert src.status == "digested"
    assert src.is_ingest_eligible is False

    # Second upload of identical bytes to same location: existing note, created=False
    note2, created2 = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data, "duplicate.jpg"
    )
    assert created2 is False
    assert note2.id == note.id

    # Upload to different location: distinct note, reuses source
    loc2 = Location(name="Valley", status="confirmed", author="agent")
    store.put_entities(PID, [loc2])
    note3, created3 = upload_reference_image(
        store, blobs, settings, PID, loc2.id, img_data, "valley.jpg"
    )
    assert created3 is True
    assert note3.id != note.id
    assert note3.owner_id == loc2.id
    assert note3.provenance[0].source_id == src_id


def test_put_note_if_absent_concurrency_barrier(harness_env):
    store, _, _, _, loc, _ = harness_env
    note_id = "note_barrier_test"
    base_note = Note(
        id=note_id,
        kind="reference_image",
        owner_id=loc.id,
        status="proposed",
        revision=1,
        author="user",
        body="barrier.jpg",
        provenance=[Provenance(source_id="a" * 64)],
    )

    barrier_created = threading.Barrier(2)
    barrier_reviewed = threading.Barrier(2)
    thread_b_out = []

    def thread_a():
        n, created = store.put_note_if_absent(PID, base_note)
        assert created is True
        barrier_created.wait()

    def thread_b():
        barrier_reviewed.wait()
        res = store.put_note_if_absent(PID, base_note)
        thread_b_out.append(res)

    t_a = threading.Thread(target=thread_a)
    t_b = threading.Thread(target=thread_b)

    t_a.start()
    barrier_created.wait()

    # Main thread reviews the note
    reviewed = base_note.touch(
        status="confirmed",
        guidance="use the sky",
        reviewed_by="director",
        reviewed_at=base_note.created_at,
    )
    store.put_note_if_current(PID, reviewed, expected_revision=1, expected_status="proposed")

    t_b.start()
    barrier_reviewed.wait()
    t_a.join(timeout=2)
    t_b.join(timeout=2)

    assert len(thread_b_out) == 1
    returned_note, returned_created = thread_b_out[0]
    assert returned_created is False
    assert returned_note.status == "confirmed"
    assert returned_note.guidance == "use the sky"

    in_store = store.notes[(PID, note_id)]
    assert in_store.status == "confirmed"
    assert in_store.guidance == "use the sky"
    assert in_store.revision == 1


def test_promote_source_to_ingest_delayed_recheck(harness_env):
    store, _, _, _, _, _ = harness_env
    sid = "d" * 64
    src = Source(
        id=sid,
        filename="ref.jpg",
        mime_type="image/jpeg",
        kind="image",
        size_bytes=100,
        storage_path=f"gs://b/{sid}.jpg",
        status="digested",
        source_purpose="reference",
    )
    store.put_source(PID, src)

    promoted = store.promote_source_to_ingest(PID, sid)
    assert promoted.source_purpose == "both"
    assert promoted.status == "uploaded"
    assert promoted.digest_version is None

    # Ingestion processes it
    digested = promoted.touch(status="digested", digest_version="digest-v2")
    store.put_source(PID, digested)

    # Delayed second promotion: re-checks inside lock, preserves state
    delayed = store.promote_source_to_ingest(PID, sid)
    assert delayed.status == "digested"
    assert delayed.digest_version == "digest-v2"
    assert delayed.source_purpose == "both"


def test_competing_source_writers_preserve_metadata(harness_env):
    store, blobs, _, settings, _, _ = harness_env
    img_data = _make_image(fmt="JPEG", color="yellow")
    sid = hashlib.sha256(img_data).hexdigest()

    ingest_ctx = IngestCtx(store=store, blobs=blobs, llm=None, settings=settings)
    src, created = register_file(ingest_ctx, PID, img_data, "original_script.jpg")
    assert created is True
    assert src.source_purpose == "ingest"
    assert src.status == "uploaded"

    img_hit = ImageHit(
        title="Wiki_Image.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Wiki_Image.jpg",
        image_url="https://upload/Wiki_Image.jpg",
        preview_url="https://preview/Wiki_Image.jpg",
        description="A wiki photo",
        license="CC BY 3.0",
        attribution="Photographer",
        mime_type="image/jpeg",
    )
    ref_images = MemoryImages({}, {}, {img_hit.preview_url: img_data})
    ref_ctx = RefCtx(store=store, blobs=blobs, llm=None, images=ref_images, settings=settings)

    stored_ref_src = _store_image(ref_ctx, PID, img_hit)
    assert stored_ref_src.id == sid

    current = store.get_source(PID, sid)
    assert current.source_purpose == "ingest"
    assert current.status == "uploaded"
    assert current.filename == "original_script.jpg"


# ==========================================================================================
# HTTP API Tests: Upload, Serve Image, Review, Validation, Cross-purpose
# ==========================================================================================

def test_api_upload_reference_image_full_lifecycle(harness_env):
    store, blobs, images, _, loc, _ = harness_env
    client = make_client(store, blobs, images)
    img_data = _make_image(fmt="JPEG", color="red")

    # 1. Successful upload
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=cliff.jpg",
        content=img_data,
    )
    assert r.status_code == 201
    body = r.json()
    note_id = body["noteId"]
    assert body["imagePath"] == f"/projects/{PID}/references/{note_id}/image"
    assert body["created"] is True
    assert body["status"] == "proposed"
    assert body["revision"] == 1

    # 2. Preview endpoint returns image bytes
    r_img = client.get(f"/projects/{PID}/references/{note_id}/image")
    assert r_img.status_code == 200
    assert r_img.headers["content-type"] == "image/jpeg"
    assert r_img.content == img_data

    # 3. Location references listing includes new upload
    r_list = client.get(f"/projects/{PID}/locations/{loc.id}/references")
    assert r_list.status_code == 200
    refs = r_list.json()["references"]
    match = [ref for ref in refs if ref["id"] == note_id]
    assert len(match) == 1
    assert match[0]["title"] == "cliff.jpg"
    assert match[0]["category"] == "Uploaded"
    assert match[0]["status"] == "proposed"
    assert match[0]["owned"] is True

    # 4. Review workflow confirms the note
    r_rev = client.post(
        f"/projects/{PID}/notes/{note_id}/review",
        json={
            "decision": "confirmed",
            "expectedRevision": 1,
            "expectedStatus": "proposed",
            "guidance": "match the red lighting",
            "by": "cinematographer",
        },
    )
    assert r_rev.status_code == 200
    rev_data = r_rev.json()
    assert rev_data["status"] == "confirmed"
    assert rev_data["guidance"] == "match the red lighting"

    # 5. Idempotent re-upload preserves confirmed review state
    r_dup = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=cliff_again.jpg",
        content=img_data,
    )
    assert r_dup.status_code == 201
    dup_body = r_dup.json()
    assert dup_body["noteId"] == note_id
    assert dup_body["created"] is False
    assert dup_body["status"] == "confirmed"


def test_api_upload_reference_validation_errors(harness_env):
    store, blobs, images, _, loc, scene = harness_env
    client = make_client(store, blobs, images, max_upload_bytes=5000)

    # 1. Missing project
    r = client.post(
        f"/projects/unknown_prj/locations/{loc.id}/references/upload?filename=test.jpg",
        content=_make_image(),
    )
    assert r.status_code == 404

    # 2. Missing location
    r = client.post(
        f"/projects/{PID}/locations/unknown_loc/references/upload?filename=test.jpg",
        content=_make_image(),
    )
    assert r.status_code == 404

    # 3. Passing scene ID instead of location ID -> 400
    r = client.post(
        f"/projects/{PID}/locations/{scene.id}/references/upload?filename=test.jpg",
        content=_make_image(),
    )
    assert r.status_code == 400
    assert "scene" in r.json()["detail"].lower()

    # 4. Empty upload body -> 400
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=test.jpg",
        content=b"",
    )
    assert r.status_code == 400

    # 5. Unsupported mime (GIF) -> 400
    gif_buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(gif_buf, format="GIF")
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=test.gif",
        content=gif_buf.getvalue(),
    )
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"]

    # 6. Animated WebP -> 400
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=anim.webp",
        content=_make_animated_webp(),
    )
    assert r.status_code == 400
    assert "animated" in r.json()["detail"]

    # 7. Corrupted image -> 400
    corrupt = bytes.fromhex("ffd8ffe000104a46494600010101006000600000") + b"garbagedata" * 10
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=corrupt.jpg",
        content=corrupt,
    )
    assert r.status_code == 400

    # 8. Upload exceeding max_upload_bytes -> 413
    big_data = _make_image(size=(100, 100))
    assert len(big_data) < 10000
    r = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=large.jpg",
        content=b"X" * 6000,
    )
    assert r.status_code == 413


def test_cross_purpose_reference_and_ingest_flows(harness_env):
    store, blobs, images, _, loc, _ = harness_env
    client = make_client(store, blobs, images)
    img_data = _make_image(fmt="PNG", color="purple")
    sid = hashlib.sha256(img_data).hexdigest()

    # Step 1: Upload as reference first
    r_up = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=purple.png",
        content=img_data,
    )
    assert r_up.status_code == 201

    # Check GET /sources: must NOT appear
    r_srcs = client.get(f"/projects/{PID}/sources")
    assert sid not in [s["id"] for s in r_srcs.json()]

    # Check explicit ingest: must be rejected with 422
    r_ing = client.post(f"/projects/{PID}/ingest", json={"sourceId": sid})
    assert r_ing.status_code == 422
    assert "reference" in r_ing.json()["detail"]

    # Step 2: Now upload the exact same image bytes via POST /sources
    r_src_up = client.post(
        f"/projects/{PID}/sources?filename=purple_ingest.png",
        content=img_data,
    )
    assert r_src_up.status_code == 201
    assert r_src_up.json()["created"] is False  # Bytes were not new

    # Check GET /sources: NOW it appears!
    r_srcs2 = client.get(f"/projects/{PID}/sources")
    assert sid in [s["id"] for s in r_srcs2.json()]

    # Check explicit ingest: NOW accepted (201)
    r_ing2 = client.post(f"/projects/{PID}/ingest", json={"sourceId": sid})
    assert r_ing2.status_code == 201


def test_cli_ingest_guards(harness_env, monkeypatch):
    store, blobs, _, settings, loc, _ = harness_env
    img_data = _make_image(fmt="JPEG", color="cyan")
    note, _ = upload_reference_image(store, blobs, settings, PID, loc.id, img_data, "cyan.jpg")
    ref_sid = note.provenance[0].source_id

    ctx = IngestCtx(store=store, blobs=blobs, llm=None, settings=settings)
    monkeypatch.setattr("harness.memory.cli.build_ctx", lambda s: ctx)
    monkeypatch.setenv("GCP_PROJECT", "test-prj")
    monkeypatch.setenv("MEMORY_BUCKET", "test-bucket")

    # Explicit CLI ingest of reference-only source exits with 1
    exit_code = cli_main(["ingest", PID, "--source", ref_sid])
    assert exit_code == 1

    # Batch ingest of project runs without error and does NOT ingest the reference source
    exit_code_batch = cli_main(["ingest", PID])
    assert exit_code_batch == 0


# ==========================================================================================
# Focused Verification for Remaining Edge Cases
# ==========================================================================================

from harness.memory.references import suggest_references
from harness.api.mapping import uploaded_sources
from harness.memory.ingest import ingest_source
from harness.memory.schemas import VocabularyOut, OutTerm


def test_user_uploaded_notes_survive_suggest_references(harness_env):
    """User-uploaded reference notes (producer='user'), both proposed and confirmed,
    must never be treated as stale_ids or deleted by the generated-references pipeline."""
    store, blobs, _, settings, loc, _ = harness_env
    
    # 1. Create a user-uploaded note in "proposed" status
    img_data1 = _make_image(fmt="JPEG", color="black")
    user_note_prop, _ = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data1, "prop.jpg"
    )
    assert user_note_prop.status == "proposed"

    # 2. Create a user-uploaded note and review it to "confirmed" with guidance
    img_data2 = _make_image(fmt="JPEG", color="white")
    user_note_conf, _ = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data2, "conf.jpg"
    )
    user_note_conf = user_note_conf.touch(
        status="confirmed",
        guidance="keep monochrome",
        reviewed_by="director",
        reviewed_at=user_note_conf.created_at,
    )
    store.put_note_if_current(PID, user_note_conf, expected_revision=1, expected_status="proposed")

    # 3. Prepare RefCtx for suggest_references with a mock LLM and verified hit
    term = "mountain pass"
    image_hit = ImageHit(
        title="Pass.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Pass.jpg",
        image_url="https://upload/Pass.jpg",
        preview_url="https://preview/Pass.jpg",
        description="A pass",
        license="CC BY 4.0",
        attribution="Photographer",
        mime_type="image/jpeg",
    )
    test_blobs = {image_hit.preview_url: _make_image(fmt="JPEG", color="blue")}
    mem_images = MemoryImages({"mountain pass": None}, {"mountain pass": [image_hit]}, test_blobs)
    
    from harness.memory.schemas import CaptionsOut, CurateOut, OutCaption, OutDirection
    class DummyLLM:
        model_id = "test-model"
        def generate(self, *, system, parts, schema, fast=False):
            if schema is VocabularyOut:
                return VocabularyOut(terms=[OutTerm(term=term, kind="landform")])
            if schema is CurateOut:
                return CurateOut(directions=[OutDirection(name="Pass Direction", why="Dramatic", images=[0])])
            if schema is CaptionsOut:
                return CaptionsOut(captions=[OutCaption(index=0, caption="Pass caption")])
            return None

    ref_ctx = RefCtx(
        store=store,
        blobs=blobs,
        llm=DummyLLM(),
        images=mem_images,
        settings=settings,
    )

    # 4. Run suggest_references
    report = suggest_references(
        ref_ctx, PID, loc.id, per_term=1, max_images=1, dry_run=False, bypass_verification=True
    )
    assert report.notes_written > 0

    # 5. Verify that BOTH user-uploaded notes survived intact
    stored_notes = {n.id: n for n in store.list_notes(PID)}
    assert user_note_prop.id in stored_notes
    assert stored_notes[user_note_prop.id].status == "proposed"
    assert stored_notes[user_note_prop.id].author == "user"

    assert user_note_conf.id in stored_notes
    assert stored_notes[user_note_conf.id].status == "confirmed"
    assert stored_notes[user_note_conf.id].guidance == "keep monochrome"
    assert stored_notes[user_note_conf.id].author == "user"


def test_legacy_reference_sources_without_source_purpose_excluded(harness_env):
    """Legacy Wikimedia sources have source_purpose=None and origin_url set.
    They must resolve to effective_purpose='reference' and remain excluded from:
    1) GET /sources (uploaded_sources)
    2) POST /ingest explicit sourceId (returns 422)
    3) ingest_source domain function (returns status='skipped')
    4) CLI ingest (exits with 1)"""
    store, blobs, images, settings, loc, _ = harness_env
    legacy_sid = "f" * 64
    legacy_src = Source(
        id=legacy_sid,
        filename="Wikimedia_Legacy.jpg",
        mime_type="image/jpeg",
        kind="image",
        size_bytes=500,
        storage_path=f"gs://b/{legacy_sid}.jpg",
        status="digested",
        source_purpose=None,  # Legacy: no explicit purpose
        origin_url="https://commons.wikimedia.org/wiki/File:Wikimedia_Legacy.jpg",
        license="CC BY-SA 4.0",
        attribution="Legacy Photographer",
    )
    store.put_source(PID, legacy_src)

    # 1. Check model property
    assert legacy_src.effective_purpose == "reference"
    assert legacy_src.is_ingest_eligible is False

    # 2. Check uploaded_sources listing
    sources_list = uploaded_sources(store, PID)
    assert legacy_sid not in [s.id for s in sources_list]

    # 3. Check GET /sources API
    client = make_client(store, blobs, images)
    r_get = client.get(f"/projects/{PID}/sources")
    assert legacy_sid not in [s["id"] for s in r_get.json()]

    # 4. Check POST /ingest with explicit sourceId -> 422
    r_ing = client.post(f"/projects/{PID}/ingest", json={"sourceId": legacy_sid})
    assert r_ing.status_code == 422
    assert "reference" in r_ing.json()["detail"]

    # 5. Check ingest_source worker -> skipped
    class DummyIngestLLM:
        model_id = "test-ingest-model"
    ingest_ctx = IngestCtx(store=store, blobs=blobs, llm=DummyIngestLLM(), settings=settings)
    report = ingest_source(ingest_ctx, PID, legacy_sid)
    assert report.status == "skipped"

    # 6. Check CLI ingest -> exit code 1
    import sys
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("harness.memory.cli.build_ctx", lambda s: ingest_ctx)
    monkeypatch.setenv("GCP_PROJECT", "test-prj")
    monkeypatch.setenv("MEMORY_BUCKET", "test-bucket")
    exit_code = cli_main(["ingest", PID, "--source", legacy_sid])
    assert exit_code == 1
    monkeypatch.undo()


def test_duplicate_upload_preserves_rejected_status_reason_guidance_and_revision(harness_env):
    """A duplicate upload must preserve 'rejected' status, rejection reason, guidance,
    and revision unconditionally."""
    store, blobs, images, settings, loc, _ = harness_env
    img_data = _make_image(fmt="JPEG", color="gray")

    # 1. Initial upload
    note, created = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data, "photo.jpg"
    )
    assert created is True
    note_id = note.id

    # 2. Review to rejected with reason and guidance
    rejected_note = note.touch(
        status="rejected",
        review_reason="not_useful",
        guidance="too modern for 19th century setting",
        reviewed_by="historian",
        reviewed_at=note.created_at,
    )
    store.put_note_if_current(PID, rejected_note, expected_revision=1, expected_status="proposed")

    # 3. Duplicate upload via domain function
    dup_note, dup_created = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data, "photo_again.jpg"
    )
    assert dup_created is False
    assert dup_note.id == note_id
    assert dup_note.status == "rejected"
    assert dup_note.review_reason == "not_useful"
    assert dup_note.guidance == "too modern for 19th century setting"
    assert dup_note.revision == 1

    # 4. Duplicate upload via HTTP route
    client = make_client(store, blobs, images)
    r_dup = client.post(
        f"/projects/{PID}/locations/{loc.id}/references/upload?filename=photo_http.jpg",
        content=img_data,
    )
    assert r_dup.status_code == 201
    body = r_dup.json()
    assert body["noteId"] == note_id
    assert body["created"] is False
    assert body["status"] == "rejected"
    assert body["revision"] == 1

    # Confirm store still holds all fields
    in_store = store.notes[(PID, note_id)]
    assert in_store.status == "rejected"
    assert in_store.review_reason == "not_useful"
    assert in_store.guidance == "too modern for 19th century setting"
    assert in_store.revision == 1


def test_blob_deduplication_preserves_winning_source_metadata(harness_env):
    """When duplicate bytes are uploaded, the winning Source's storage_path,
    MIME type, and original metadata must be preserved."""
    store, blobs, _, settings, loc, _ = harness_env
    img_data = _make_image(fmt="JPEG", color="brown")
    sid = hashlib.sha256(img_data).hexdigest()

    # 1. First upload via register_file
    ingest_ctx = IngestCtx(store=store, blobs=blobs, llm=None, settings=settings)
    src_first, created_first = register_file(ingest_ctx, PID, img_data, "first_winner.jpg")
    assert created_first is True
    original_storage_path = src_first.storage_path
    original_mime = src_first.mime_type
    original_filename = src_first.filename

    # 2. Second upload via upload_reference_image with identical bytes but different filename
    note, created_ref = upload_reference_image(
        store, blobs, settings, PID, loc.id, img_data, "second_attempt.jpeg"
    )
    # The note was created for this location, but points to the winning Source
    assert created_ref is True
    assert note.provenance[0].source_id == sid

    # 3. Verify that the Source in store was NOT overwritten by the second attempt
    stored_src = store.get_source(PID, sid)
    assert stored_src.storage_path == original_storage_path
    assert stored_src.mime_type == original_mime
    assert stored_src.filename == original_filename
