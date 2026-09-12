"""File sniffing and PDF utilities."""
from __future__ import annotations

import io
import mimetypes
from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium

from .models import SourceKind

PDF = "application/pdf"
IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
TEXT_MIME = {"text/markdown", "text/plain"}
SUPPORTED_MIME = {PDF} | IMAGE_MIME | TEXT_MIME
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
