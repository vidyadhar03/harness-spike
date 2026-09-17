"""File sniffing and PDF utilities."""
from __future__ import annotations

import io
import mimetypes
from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image as _PILImage, UnidentifiedImageError as _UnidentifiedImageError

from .models import SourceKind

PDF = "application/pdf"
IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
TEXT_MIME = {"text/markdown", "text/plain"}
SUPPORTED_MIME = {PDF} | IMAGE_MIME | TEXT_MIME

# Accepted formats for user-uploaded location reference images.
# Excludes HEIC/HEIF: inconsistent browser read support and non-universal
# Pillow decode (requires libheif). The pipeline fetches HEIC from external
# sources but the upload path enforces a stricter known-good set.
ACCEPTED_REFERENCE_MIMES: frozenset[str] = frozenset({"image/jpeg", "image/png", "image/webp"})
# 4000x4000 = 16 MP cap. Enforced from header dimensions BEFORE any pixel decode so
# an attacker cannot force a multi-gigabyte decompression with a small upload.
MAX_REFERENCE_PIXELS: int = 4_000 * 4_000
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"}


def sniff_mime(data: bytes, filename: str) -> str:
    head = data[:16]
    if head.startswith(b"%PDF"):
        return PDF
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS:
        return "image/heic"
    suffix = Path(filename).suffix.lower()
    if suffix in (".md", ".markdown"):
        return "text/markdown"
    if suffix == ".txt":
        return "text/plain"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def kind_for(mime: str) -> SourceKind:
    if mime == PDF or mime.endswith(("wordprocessingml.document", "presentationml.presentation", "msword")):
        return "document"
    if mime.startswith("image/"):
        return "image"
    if mime in ("text/csv",) or "spreadsheet" in mime or "excel" in mime:
        return "table"
    if mime.startswith("text/"):
        return "text"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    return "other"


def extension_for(filename: str, mime: str) -> str:
    return Path(filename).suffix.lower() or mimetypes.guess_extension(mime) or ".bin"


def pdf_page_texts(data: bytes) -> list[str]:
    pdf = pdfium.PdfDocument(data)
    try:
        texts = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            texts.append(tp.get_text_range().replace("\r\n", "\n").strip())
            tp.close()
            page.close()
        return texts
    finally:
        pdf.close()


def pdf_subset(data: bytes, start: int, end: int) -> bytes:
    """Pages [start, end) as a standalone PDF."""
    src = pdfium.PdfDocument(data)
    dst = pdfium.PdfDocument.new()
    try:
        dst.import_pages(src, list(range(start, end)))
        buf = io.BytesIO()
        dst.save(buf)
        return buf.getvalue()
    finally:
        dst.close()
        src.close()


def render_pages_png(data: bytes, scale: float) -> Iterator[tuple[int, bytes]]:
    pdf = pdfium.PdfDocument(data)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            img = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page.close()
            yield i + 1, buf.getvalue()
    finally:
        pdf.close()


def is_visual(page_texts: list[str], min_avg_chars: int = 200) -> bool:
    """Lookbooks, recce decks, and scans have little or no text layer."""
    if not page_texts:
        return True
    return sum(len(t) for t in page_texts) / len(page_texts) < min_avg_chars


def page_marked(page_texts: list[str], start: int, end: int) -> str:
    return "\n\n".join(f"<<<PAGE {i + 1}>>>\n{page_texts[i]}" for i in range(start, end))


def page_ranges(count: int, size: int) -> list[tuple[int, int]]:
    return [(i, min(i + size, count)) for i in range(0, count, size)]


def text_chunks(text: str, max_chars: int) -> list[str]:
    chunks, current, size = [], [], 0
    for para in text.split("\n\n"):
        if size + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]

# --- user-uploaded reference image validation ---------------------------------

def validate_reference_image(data: bytes, mime: str) -> tuple[int, int]:
    """Decode-validate image bytes for user-uploaded location references.

    Returns (width, height) in pixels. Raises ValueError for:
    - MIME not in ACCEPTED_REFERENCE_MIMES
    - Corrupt or truncated data (Pillow OSError / UnidentifiedImageError)
    - Pixel count > MAX_REFERENCE_PIXELS (checked from header BEFORE decode)
    - Animated images (APNG, animated WebP)
    - Decoder format inconsistent with the sniffed MIME

    The streamed byte-count limit is enforced upstream by _read_body_bounded;
    this function validates content, not body size.
    """
    if mime not in ACCEPTED_REFERENCE_MIMES:
        raise ValueError(
            f"unsupported image format {mime!r}; accepted: jpeg, png, webp"
        )
    try:
        with _PILImage.open(io.BytesIO(data)) as img:
            # 1. Dimensions from header metadata - no pixel decode yet.
            w, h = img.size
            # 2. Pixel-count guard BEFORE any full decode operation.
            if w * h > MAX_REFERENCE_PIXELS:
                raise ValueError(
                    f"image too large ({w}\u00d7{h} = {w*h:,} px); "
                    f"limit is {MAX_REFERENCE_PIXELS:,} px total"
                )
            # 3. Reject animated images (APNG, animated WebP).
            if getattr(img, "n_frames", 1) > 1:
                raise ValueError("animated images are not supported")
            # 4. Force full pixel decode within known-safe bounds.
            #    Catches truncated bodies, decompression bombs (PIL raises
            #    DecompressionBombError at its own 178 MP limit, well above ours),
            #    and corrupt trailing data.
            img.load()
            # 5. Cross-check what Pillow actually decoded against sniffed MIME.
            _FMT_TO_MIME = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }
            decoded_mime = _FMT_TO_MIME.get(img.format or "")
            if decoded_mime != mime:
                raise ValueError(
                    f"declared {mime!r} but image decoded as {img.format!r}"
                )
    except (_UnidentifiedImageError, _PILImage.DecompressionBombError, OSError) as exc:
        raise ValueError(f"invalid image data: {exc}") from exc
    return w, h
