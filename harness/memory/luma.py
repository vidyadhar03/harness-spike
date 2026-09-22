"""Luma Agents image adapter (https://agents.lumalabs.ai/v1) implementing imagegen.ImageProvider.

Imports only the provider-neutral imagegen module - no location/concept/note/approval logic.

Contract used (from docs.agents.lumalabs.ai and the official luma-agents-go SDK, whose
ImageRefParam{data, file_id, generation_id, media_type, url} is the request type for image
generation; the OpenAPI document itself could not be retrieved - no live call has been made):
- POST /generations -> 201 with a job id. Body: {"type": "image", "model", "prompt" (1-6000 chars),
  "image_ref": [{"data": <base64>, "media_type": <mime>}, ...], "aspect_ratio", "output_format",
  "web_search": false}. Up to 9 references; 50 MB and 8000 px per side each; JPEG/PNG/WebP/still GIF.
  Exactly one of url/data/generation_id/file_id per reference; we only ever send data+media_type.
- GET /generations/{id} -> {"id", "state": queued|processing|completed|failed,
  "output": [{"type": "image", "url"}], "failure_code", "failure_reason"}.
  Output URLs are presigned for 1 hour; calling GET again mints a fresh one.
- Auth: "Authorization: Bearer <LUMA_AGENTS_API_KEY>".
- The docs describe NO idempotency key or duplicate-submission protection, so this adapter never
  retries a POST: it uses plain httpx (not an SDK that might retry on its own) with
  HTTPTransport(retries=0), and classifies every POST failure as either "definitely not sent /
  definitely rejected" (SubmissionRejected) or "may have been accepted" (SubmissionOutcomeUnknown).

Nothing here logs the API key, inline image payloads, or presigned URL query strings.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import replace
from datetime import datetime
from urllib.parse import urlsplit

import httpx

from .imagegen import (
    DownloadedImage, ImageCapabilities, ImageGenerationError, ImageRequest, OutputTooLarge, OutputUnavailable,
    ProviderJob, ProviderTransientError, SubmissionOutcomeUnknown, SubmissionRejected,
    UnsupportedRequest,
)

log = logging.getLogger(__name__)

BASE_URL = "https://agents.lumalabs.ai/v1"
IMAGE_MODELS = frozenset({"uni-1", "uni-1-max"})
DEFAULT_MODEL = "uni-1"
# Documented image aspect ratios (video ratios deliberately absent).
ASPECT_RATIOS = frozenset({"3:1", "2:1", "16:9", "3:2", "1:1", "2:3", "9:16", "1:2", "1:3"})
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
# Provider-documented: <=9 references, <=50 MB and <=8000 px per side each, JPEG/PNG/WebP/still GIF,
# prompt 1-6000 characters. OURS, not Luma's: max_request_bytes - a cap on the WHOLE serialized JSON
# request (reference images grow ~33% under base64). Luma documents no total body limit, so this is a
# conservative operational choice; it is configurable (HARNESS_GENERATION_MAX_REQUEST_BYTES).
LUMA_CAPABILITIES = ImageCapabilities(
    models=IMAGE_MODELS, max_references=9, aspect_ratios=ASPECT_RATIOS,
    reference_mime_types=frozenset({"image/jpeg", "image/png", "image/webp"}),
    max_reference_bytes=50 * 1024 * 1024, max_request_bytes=DEFAULT_MAX_REQUEST_BYTES,
    max_prompt_chars=6000, output_formats=frozenset({"png", "jpeg"}),
)
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


def safe_url(url: str) -> str:
    """scheme://host/path only - presigned URLs carry credentials in the query string."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        d = body.get("detail") if isinstance(body, dict) else None
        if isinstance(d, (str, list, dict)):
            return str(d)[:300]
    except ValueError:
        pass
    return f"HTTP {resp.status_code}"


class LumaImageProvider:
    name = "luma"
    capabilities = LUMA_CAPABILITIES

    def __init__(self, api_key: str, *, base_url: str = BASE_URL, client: httpx.Client | None = None,
                 connect_timeout_s: float = 5.0, read_timeout_s: float = 30.0,
                 write_timeout_s: float = 60.0, max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
                 account_scope: str | None = None):
        if not api_key:
            raise ValueError("LUMA_AGENTS_API_KEY is required")
        if max_request_bytes < 1024:
            raise ValueError("max_request_bytes is implausibly small")
        self.capabilities = replace(LUMA_CAPABILITIES, max_request_bytes=max_request_bytes)
        # Identity of the Luma account for provider-id uniqueness. Luma exposes no account id, so by
        # default it is a fingerprint of the API key (a rotated key looks like a new account); set
        # HARNESS_GENERATION_ACCOUNT_SCOPE to keep it stable across rotations. Never the key itself.
        self.account_scope = account_scope or "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s, write=write_timeout_s, pool=5.0),
            transport=httpx.HTTPTransport(retries=0),   # explicit: no transport-level retries
            follow_redirects=False,
        )

    # -- submit -----------------------------------------------------------------------
    def _payload(self, req: ImageRequest) -> dict:
        body: dict = {"type": "image", "model": req.model, "prompt": req.prompt, "web_search": False}
        if req.references:
            body["image_ref"] = [
                {"data": base64.b64encode(r.data).decode("ascii"), "media_type": r.mime_type}
                for r in req.references
            ]
        if req.aspect_ratio:
            body["aspect_ratio"] = req.aspect_ratio
        if req.output_format:
            body["output_format"] = req.output_format
        return body

    def submit(self, req: ImageRequest) -> str:
        self.capabilities.validate(req)
        payload = self._payload(req)
        try:
            resp = self._client.post(f"{self._base}/generations", json=payload, headers=self._headers)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise SubmissionRejected(f"could not reach Luma ({type(exc).__name__}); nothing was sent",
                                     code="not_sent") from None
        except httpx.HTTPError as exc:
            raise SubmissionOutcomeUnknown(
                f"submission outcome unknown ({type(exc).__name__}): the request may have reached Luma"
            ) from None
        finally:
            del payload   # drop the inline base64 promptly

        if resp.status_code in (200, 201):
            try:
                gid = resp.json().get("id")
            except (ValueError, AttributeError):
                gid = None
            if isinstance(gid, str) and gid:
                return gid
            raise SubmissionOutcomeUnknown("Luma answered success but no generation id could be read")
        if resp.status_code == 429:
            try:
                ra = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                ra = None
            raise SubmissionRejected("Luma rate limit reached", code="rate_limited", retry_after_s=ra)
        if 400 <= resp.status_code < 500:
            raise SubmissionRejected(f"Luma rejected the request: {_detail(resp)}",
                                     code=f"http_{resp.status_code}")
        raise SubmissionOutcomeUnknown(
            f"Luma returned HTTP {resp.status_code} on submit; the job may or may not exist")

    # -- read side ---------------------------------------------------------------------
    def get(self, provider_id: str) -> ProviderJob:
        try:
            resp = self._client.get(f"{self._base}/generations/{provider_id}", headers=self._headers)
        except httpx.HTTPError as exc:
            raise ProviderTransientError(f"poll failed ({type(exc).__name__})") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ProviderTransientError(f"poll returned HTTP {resp.status_code}")
        if resp.status_code in (401, 403):
            raise ImageGenerationError(f"Luma rejected our credentials on poll (HTTP {resp.status_code})")
        if resp.status_code == 404:
            raise ImageGenerationError(f"Luma has no generation {provider_id!r}")
        if resp.status_code != 200:
            raise ProviderTransientError(f"poll returned HTTP {resp.status_code}")
        try:
            body = resp.json()
            state = body["state"]
        except (ValueError, KeyError, TypeError):
            raise ProviderTransientError("poll response was not understood") from None
        if state not in ("queued", "processing", "completed", "failed"):
            raise ProviderTransientError(f"unknown generation state {state!r}")
        urls = tuple(o["url"] for o in (body.get("output") or [])
                     if isinstance(o, dict) and o.get("type", "image") == "image" and o.get("url"))
        created = None
        if isinstance(body.get("created_at"), str):
            try:
                created = datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))
            except ValueError:
                created = None
        return ProviderJob(provider_id=provider_id, state=state, output_urls=urls,
                           kind=body.get("type") if isinstance(body.get("type"), str) else None,
                           model=body.get("model") if isinstance(body.get("model"), str) else None,
                           created_at=created, failure_code=body.get("failure_code"),
                           failure_reason=(str(body["failure_reason"])[:500] if body.get("failure_reason") else None))

    def download(self, url: str) -> DownloadedImage:
        # Presigned URL: no Authorization header (it must not reach the storage host), no redirects.
        try:
            with self._client.stream("GET", url) as resp:
                if resp.status_code in (401, 403, 404, 410):
                    raise OutputUnavailable(
                        f"output URL is no longer usable (HTTP {resp.status_code}) at {safe_url(url)}")
                if resp.status_code != 200:
                    raise ProviderTransientError(f"download returned HTTP {resp.status_code} from {safe_url(url)}")
                chunks, size = [], 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > _MAX_DOWNLOAD_BYTES:
                        raise OutputTooLarge(f"output exceeds the {_MAX_DOWNLOAD_BYTES}-byte download bound")
                    chunks.append(chunk)
                mime = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        except httpx.HTTPError as exc:
            raise ProviderTransientError(f"download failed ({type(exc).__name__}) from {safe_url(url)}") from None
        return DownloadedImage(data=b"".join(chunks), mime_type=mime)


def from_env(env: dict | None = None, *, max_request_bytes: int | None = None) -> LumaImageProvider | None:
    import os
    e = env if env is not None else os.environ
    key = e.get("LUMA_AGENTS_API_KEY")
    if not key:
        return None
    limit = max_request_bytes or int(e.get("HARNESS_GENERATION_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
    return LumaImageProvider(key, max_request_bytes=limit,
                             account_scope=e.get("HARNESS_GENERATION_ACCOUNT_SCOPE") or None)


__all__ = ["LumaImageProvider", "from_env", "DEFAULT_MODEL", "IMAGE_MODELS", "safe_url", "UnsupportedRequest"]
