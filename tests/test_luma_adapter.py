"""Luma adapter tests against httpx.MockTransport - no network, no credentials, no paid call.
They pin the request the adapter builds from the documented contract and, above all, that a POST
is never retried and every POST failure is classified as rejected vs. outcome-unknown."""
from __future__ import annotations

import base64
import json
import logging

import httpx
import pytest

from harness.memory.imagegen import (
    ImageInput, ImageRequest, OutputUnavailable, ProviderTransientError, SubmissionOutcomeUnknown,
    SubmissionRejected, UnsupportedRequest,
)
from harness.memory.luma import LumaImageProvider

KEY = "sk-test-secret-key"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def make(handler):
    calls = []

    def wrapped(request: httpx.Request):
        calls.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return LumaImageProvider(KEY, client=client), calls


def req(**kw):
    return ImageRequest(prompt="a stone house", model="uni-1", **kw)


def test_submit_builds_documented_payload_and_returns_id():
    p, calls = make(lambda r: httpx.Response(201, json={"id": "gen-1", "state": "queued"}))
    gid = p.submit(req(references=(ImageInput(PNG, "image/png"),), aspect_ratio="16:9", output_format="png"))
    assert gid == "gen-1" and len(calls) == 1
    c = calls[0]
    assert str(c.url) == "https://agents.lumalabs.ai/v1/generations" and c.method == "POST"
    assert c.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(c.content)
    assert body == {"type": "image", "model": "uni-1", "prompt": "a stone house", "web_search": False,
                    "image_ref": [{"data": base64.b64encode(PNG).decode(), "media_type": "image/png"}],
                    "aspect_ratio": "16:9", "output_format": "png"}


def test_no_unsupported_parameters_are_sent():
    p, calls = make(lambda r: httpx.Response(201, json={"id": "g"}))
    p.submit(req())
    body = json.loads(calls[0].content)
    assert set(body) == {"type", "model", "prompt", "web_search"}     # no seed/weights/refs/etc.


@pytest.mark.parametrize("model", ["ray-3.2", "ray-flash", "dream-machine", "uni-2"])
def test_video_and_unknown_models_rejected_before_any_request(model):
    p, calls = make(lambda r: httpx.Response(201, json={"id": "g"}))
    with pytest.raises(UnsupportedRequest):
        p.submit(ImageRequest(prompt="x", model=model))
    assert calls == []


def test_capability_limits_checked_locally():
    p, calls = make(lambda r: httpx.Response(201, json={"id": "g"}))
    ten = tuple(ImageInput(PNG, "image/png") for _ in range(10))
    for bad in (req(references=ten), req(aspect_ratio="4:3"), req(output_format="gif"),
                req(references=(ImageInput(PNG, "image/heic"),))):
        with pytest.raises(UnsupportedRequest):
            p.submit(bad)
    p.submit(req(references=ten[:9]))          # exactly nine is fine
    assert len(calls) == 1


def test_post_is_never_retried_and_5xx_is_outcome_unknown():
    p, calls = make(lambda r: httpx.Response(503, text="upstream"))
    with pytest.raises(SubmissionOutcomeUnknown):
        p.submit(req())
    assert len(calls) == 1


def test_read_timeout_after_send_is_outcome_unknown_not_retried():
    def boom(r):
        raise httpx.ReadTimeout("slow", request=r)
    p, calls = make(boom)
    with pytest.raises(SubmissionOutcomeUnknown):
        p.submit(req())
    assert len(calls) == 1


def test_connect_failure_is_definitely_not_sent():
    def boom(r):
        raise httpx.ConnectError("refused", request=r)
    p, _ = make(boom)
    with pytest.raises(SubmissionRejected) as e:
        p.submit(req())
    assert e.value.code == "not_sent"


def test_success_without_id_is_unknown():
    p, _ = make(lambda r: httpx.Response(201, json={"state": "queued"}))
    with pytest.raises(SubmissionOutcomeUnknown):
        p.submit(req())


def test_rate_limit_and_validation_errors_are_definite_rejections():
    p, _ = make(lambda r: httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "slow down"}))
    with pytest.raises(SubmissionRejected) as e:
        p.submit(req())
    assert (e.value.code, e.value.retry_after_s) == ("rate_limited", 7.0)
    p, _ = make(lambda r: httpx.Response(422, json={"detail": "bad aspect_ratio"}))
    with pytest.raises(SubmissionRejected) as e:
        p.submit(req())
    assert e.value.code == "http_422" and "bad aspect_ratio" in str(e.value)


def test_get_parses_states_outputs_and_failures():
    p, calls = make(lambda r: httpx.Response(200, json={
        "id": "g", "state": "completed", "output": [{"type": "image", "url": "https://s.example/o.png?X-Amz-Signature=SECRET"}],
        "failure_reason": None, "failure_code": None}))
    j = p.get("g")
    assert j.state == "completed" and j.output_urls == ("https://s.example/o.png?X-Amz-Signature=SECRET",)
    assert str(calls[0].url) == "https://agents.lumalabs.ai/v1/generations/g"
    p, _ = make(lambda r: httpx.Response(200, json={"id": "g", "state": "failed", "output": None,
                                                     "failure_code": "content_moderated", "failure_reason": "flagged"}))
    j = p.get("g")
    assert (j.state, j.failure_code, j.failure_reason, j.output_urls) == ("failed", "content_moderated", "flagged", ())


@pytest.mark.parametrize("resp", [httpx.Response(503), httpx.Response(429), httpx.Response(200, json={"state": "weird"}),
                                  httpx.Response(200, text="not json")])
def test_get_read_failures_are_transient(resp):
    p, _ = make(lambda r: resp)
    with pytest.raises(ProviderTransientError):
        p.get("g")


def test_download_sends_no_credentials_and_hides_query_string():
    seen = {}

    def handler(r: httpx.Request):
        seen["auth"] = r.headers.get("authorization")
        return httpx.Response(403)
    p, _ = make(handler)
    url = "https://s.example/o.png?X-Amz-Signature=SECRET&X-Amz-Expires=3600"
    with pytest.raises(OutputUnavailable) as e:
        p.download(url)
    assert seen["auth"] is None
    assert "SECRET" not in str(e.value) and "X-Amz" not in str(e.value) and "https://s.example/o.png" in str(e.value)


def test_download_returns_bytes_and_mime():
    p, _ = make(lambda r: httpx.Response(200, content=PNG, headers={"content-type": "image/png; charset=x"}))
    d = p.download("https://s.example/o.png?sig=1")
    assert (d.data, d.mime_type) == (PNG, "image/png")


def test_nothing_sensitive_is_logged_or_raised(caplog):
    caplog.set_level(logging.DEBUG)
    p, _ = make(lambda r: httpx.Response(503, text="x"))
    with pytest.raises(SubmissionOutcomeUnknown) as e:
        p.submit(req(references=(ImageInput(PNG, "image/png"),)))
    text = caplog.text + str(e.value)
    assert KEY not in text and base64.b64encode(PNG).decode() not in text


def test_adapter_and_interface_import_no_domain_logic():
    import ast
    import pathlib
    root = pathlib.Path(__file__).parent.parent / "harness" / "memory"
    banned = {"models", "concepts", "concept_generation", "retrieval", "ports", "curate", "references", "ingest"}
    for name in ("luma.py", "imagegen.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                assert node.module not in banned, f"{name} imports {node.module}"
    tree = ast.parse((root / "concept_generation.py").read_text())
    for node in ast.walk(tree):     # the workflow depends on the neutral interface only
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            assert node.module != "luma"
        if isinstance(node, ast.Import):
            assert not any("luma" in a.name for a in node.names)


# --- hardening: request-size accounting, prompt limit, provider-reported identity ----------------------

def _refs(n_bytes, count=1):
    return tuple(ImageInput(b"\x00" * n_bytes, "image/png") for _ in range(count))


def test_request_limit_measures_the_serialized_request_including_base64_expansion():
    from harness.memory.imagegen import estimate_request_bytes
    p, _ = make(lambda r: httpx.Response(201, json={"id": "g"}), )
    r = req(references=_refs(3000, 2))
    actual = len(json.dumps(p._payload(r)))
    est = estimate_request_bytes(r)
    assert est >= actual and est - actual < 2048          # a conservative estimate, not wildly off
    assert actual > 2 * 3000 * 1.33                        # base64 growth is really there (~4/3)


def test_operational_limit_is_configurable_and_labelled_as_ours():
    raw = 3_000_000                                        # ~4.0 MB once encoded
    ok = LumaImageProvider(KEY, client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(201, json={"id": "g"}))), max_request_bytes=4_500_000)
    assert ok.submit(req(references=_refs(raw))) == "g"
    tight = LumaImageProvider(KEY, client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(201, json={"id": "g"}))), max_request_bytes=4_000_000)
    with pytest.raises(UnsupportedRequest) as e:
        tight.submit(req(references=_refs(raw)))
    msg = str(e.value)
    assert "base64" in msg and "33%" in msg and "4000000" in msg and "not a documented provider limit" in msg
    assert tight.capabilities.max_request_bytes == 4_000_000


def test_raw_reference_size_and_encoded_request_size_are_different_checks():
    p, calls = make(lambda r: httpx.Response(201, json={"id": "g"}))      # default cap: 32 MiB serialized
    with pytest.raises(UnsupportedRequest, match="operational limit"):
        p.submit(req(references=_refs(25 * 1024 * 1024)))                 # 25 MiB raw -> ~33.4 MiB encoded
    assert calls == []
    assert p.submit(req(references=_refs(20 * 1024 * 1024))) == "g"       # 20 MiB raw -> ~26.7 MiB encoded
    assert "per image" in _try(lambda: p.submit(req(references=_refs(51 * 1024 * 1024))))  # provider's 50 MB rule


def _try(fn):
    try:
        fn()
    except UnsupportedRequest as exc:
        return str(exc)
    return ""


def test_prompt_limit_is_the_documented_6000_characters():
    p, calls = make(lambda r: httpx.Response(201, json={"id": "g"}))
    assert p.submit(ImageRequest(prompt="x" * 6000, model="uni-1")) == "g"
    with pytest.raises(UnsupportedRequest, match="6000"):
        p.submit(ImageRequest(prompt="x" * 6001, model="uni-1"))
    assert len(calls) == 1


def test_limit_from_environment(monkeypatch):
    from harness.memory.luma import from_env
    p = from_env({"LUMA_AGENTS_API_KEY": KEY, "HARNESS_GENERATION_MAX_REQUEST_BYTES": "5000000"})
    assert p.capabilities.max_request_bytes == 5_000_000
    assert from_env({}) is None


def test_get_reports_kind_model_and_creation_time_when_present():
    p, _ = make(lambda r: httpx.Response(200, json={"id": "g", "type": "image", "model": "uni-1", "state": "queued",
                                                     "created_at": "2026-04-08T12:00:00Z"}))
    j = p.get("g")
    assert (j.kind, j.model) == ("image", "uni-1") and j.created_at.year == 2026
    p, _ = make(lambda r: httpx.Response(200, json={"id": "g", "state": "queued"}))
    j = p.get("g")
    assert (j.kind, j.model, j.created_at) == (None, None, None)


def test_oversized_output_download_is_its_own_error(monkeypatch):
    from harness.memory import luma
    from harness.memory.imagegen import OutputTooLarge
    monkeypatch.setattr(luma, "_MAX_DOWNLOAD_BYTES", 10)
    p, _ = make(lambda r: httpx.Response(200, content=b"x" * 100, headers={"content-type": "image/png"}))
    with pytest.raises(OutputTooLarge):
        p.download("https://s.example/o.png?sig=1")
