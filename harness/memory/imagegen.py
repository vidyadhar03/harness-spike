"""Provider-neutral image generation: request/result types, capability validation, and the
ImageProvider protocol.

Deliberately knows nothing about locations, notes, concepts or approvals, and nothing about
any one vendor's wire format - workflows build an ImageRequest and hand it to whatever
ImageProvider was injected; adapters (see luma.py) translate. Image-only on purpose: no video
types exist here, and an adapter rejects anything it cannot honestly do via
ImageCapabilities.validate rather than silently dropping parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

ProviderState = Literal["queued", "processing", "completed", "failed"]


class ImageGenerationError(RuntimeError):
    """Base for everything a provider adapter raises."""


class UnsupportedRequest(ImageGenerationError, ValueError):
    """The request asks for something the provider/model cannot do (checked locally, before
    any network call). A ValueError so callers' existing 400 handling applies."""


class SubmissionRejected(ImageGenerationError):
    """The provider definitely did NOT accept the submission (auth/validation/rate limit, or
    the request provably never left this machine). Safe to conclude no paid job exists."""
    def __init__(self, message: str, *, code: str = "rejected", retry_after_s: float | None = None):
        super().__init__(message)
        self.code, self.retry_after_s = code, retry_after_s


class SubmissionOutcomeUnknown(ImageGenerationError):
    """The request may have reached the provider and may have created a paid job, but no
    provider id was received (timeout after sending, dropped connection, 5xx, malformed
    reply). The caller must NOT blindly submit again."""


class ProviderTransientError(ImageGenerationError):
    """A read-side failure (poll/download) that is worth retrying within a bounded budget."""


class OutputTooLarge(ImageGenerationError):
    """The provider's output exceeds the download size bound we are willing to read."""


class OutputUnavailable(ImageGenerationError):
    """The output could not be downloaded (e.g. the presigned URL expired). Retrieving a fresh
    URL from the existing provider job fixes this; regenerating would not be needed."""


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    model: str
    references: tuple[ImageInput, ...] = ()
    aspect_ratio: str | None = None
    output_format: str | None = None      # provider-supported subset, e.g. "png" | "jpeg"


def estimate_request_bytes(req: ImageRequest) -> int:
    """Conservative estimate of the SERIALIZED request body for an inline-reference submission:
    every reference grows by 4/3 under base64 (plus padding), plus the prompt and a fixed
    allowance for JSON keys/envelope. An estimate on purpose - adapters may add a few bytes."""
    encoded = sum(4 * ((len(r.data) + 2) // 3) for r in req.references)
    return encoded + len(req.prompt.encode()) + 64 * len(req.references) + 1024


@dataclass(frozen=True)
class ImageCapabilities:
    """What a provider/model accepts. Fields are of two kinds, and messages say which:
    limits the PROVIDER documents (models, max_references, aspect ratios, per-reference bytes,
    max_prompt_chars) and limits WE choose to enforce (max_request_bytes - an operational cap
    on the whole serialized request, because the provider documents no total body limit)."""
    models: frozenset[str]
    max_references: int
    aspect_ratios: frozenset[str]
    reference_mime_types: frozenset[str]
    max_reference_bytes: int                  # raw bytes of one reference image
    max_request_bytes: int | None = None      # OUR operational limit: whole serialized request, base64 included
    max_prompt_chars: int | None = None
    output_formats: frozenset[str] = frozenset()

    def validate(self, req: ImageRequest) -> None:
        if not req.prompt.strip():
            raise UnsupportedRequest("prompt is empty")
        if self.max_prompt_chars is not None and len(req.prompt) > self.max_prompt_chars:
            raise UnsupportedRequest(
                f"the prompt is {len(req.prompt)} characters; the provider accepts at most "
                f"{self.max_prompt_chars} - shorten the brief notes or the extra direction")
        if req.model not in self.models:
            raise UnsupportedRequest(
                f"model {req.model!r} is not supported for image generation; "
                f"supported: {', '.join(sorted(self.models))}")
        if len(req.references) > self.max_references:
            raise UnsupportedRequest(
                f"{len(req.references)} reference images requested; this provider accepts at most "
                f"{self.max_references}")
        if req.aspect_ratio is not None and req.aspect_ratio not in self.aspect_ratios:
            raise UnsupportedRequest(
                f"aspect ratio {req.aspect_ratio!r} is not supported; "
                f"supported: {', '.join(sorted(self.aspect_ratios))}")
        if req.output_format is not None and req.output_format not in self.output_formats:
            raise UnsupportedRequest(
                f"output format {req.output_format!r} is not supported; "
                f"supported: {', '.join(sorted(self.output_formats)) or 'none selectable'}")
        for i, ref in enumerate(req.references, start=1):
            if ref.mime_type not in self.reference_mime_types:
                raise UnsupportedRequest(
                    f"reference image {i} is {ref.mime_type!r}; supported: "
                    f"{', '.join(sorted(self.reference_mime_types))}")
            if len(ref.data) > self.max_reference_bytes:
                raise UnsupportedRequest(
                    f"reference image {i} is {len(ref.data)} bytes; the provider limit is "
                    f"{self.max_reference_bytes} bytes per image")
        if self.max_request_bytes is not None:
            est = estimate_request_bytes(req)
            if est > self.max_request_bytes:
                raw = sum(len(r.data) for r in req.references)
                raise UnsupportedRequest(
                    f"the request would be about {est} bytes once the {raw} bytes of reference images are "
                    f"base64-encoded (about 33% larger); this server's operational limit is "
                    f"{self.max_request_bytes} bytes for the whole request (a local safety limit, not a "
                    f"documented provider limit) - select fewer or smaller references")


@dataclass(frozen=True)
class ProviderJob:
    provider_id: str
    state: ProviderState
    output_urls: tuple[str, ...] = ()     # presigned/expiring - never persist or log
    failure_code: str | None = None
    failure_reason: str | None = None
    # What the provider says this job IS, when it reports it - used to sanity-check a job a human
    # asks us to adopt (see ConceptGenerationService.resolve). None = the provider did not say.
    kind: str | None = None               # e.g. "image"
    model: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class DownloadedImage:
    data: bytes
    mime_type: str


class ImageProvider(Protocol):
    name: str
    account_scope: str      # provider account identity; provider generation ids are unique within (name, scope)
    capabilities: ImageCapabilities

    def submit(self, req: ImageRequest) -> str:
        """Returns the provider's job id. Raises SubmissionRejected (definitely not accepted),
        SubmissionOutcomeUnknown (maybe accepted - never retried inside the adapter),
        or UnsupportedRequest."""
        ...

    def get(self, provider_id: str) -> ProviderJob:
        """Idempotent read; also the way to obtain fresh output URLs. Raises
        ProviderTransientError for retryable read failures."""
        ...

    def download(self, url: str) -> DownloadedImage:
        """Raises OutputUnavailable / ProviderTransientError."""
        ...
