"""Wikimedia adapter: Wikipedia for term verification, Commons for images.

Commons is the default archive because its files carry an explicit licence and a
credit line, which is what makes a reference safe to put in a deck or feed to a
generation step. Wikimedia's API policy requires a descriptive User-Agent.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

from .ports import ImageHit, TermHit

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MotionX-harness/0.5 (https://motionx.in; contact: tech@motionx.in)"
_TAGS = re.compile(r"<[^>]+>")
_LICENSE_OK = re.compile(r"^(cc|public domain|pd|cc0)", re.I)


def _clean(html: str | None) -> str:
    return " ".join(_TAGS.sub(" ", html or "").replace("&nbsp;", " ").split())


class WikimediaImages:
    def __init__(self, timeout: float = 20.0, user_agent: str = USER_AGENT,
                 preview_px: int = 1280, thumb_px: int = 512):
        self.timeout, self.user_agent = timeout, user_agent
        self.preview_px, self.thumb_px = preview_px, thumb_px

    # --- http ---

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def _api(self, endpoint: str, **params) -> dict:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        return json.loads(self._get(url).decode("utf-8"))

    def fetch(self, url: str) -> bytes:
        return self._get(url)

    # --- terms ---

    def verify_term(self, term: str) -> TermHit | None:
        """An article whose title or lead mentions the term. Unverified terms are dropped upstream."""
        try:
            data = self._api(WIKIPEDIA_API, action="query", list="search", srsearch=term, srlimit=3,
                             srprop="snippet")
        except Exception as exc:
            log.warning("term lookup failed for %r: %s", term, exc)
            return None
        results = data.get("query", {}).get("search", [])
        words = {w for w in re.findall(r"\w+", term.lower()) if len(w) > 2}
        for r in results:
            haystack = f"{r['title']} {_clean(r.get('snippet'))}".lower()
            if not words or sum(w in haystack for w in words) >= max(1, len(words) - 1):
                title = r["title"]
                return TermHit(title=title, snippet=_clean(r.get("snippet")),
                               url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")))
        return None

    # --- images ---

    def search_images(self, term: str, limit: int) -> list[ImageHit]:
        try:
            data = self._api(
                COMMONS_API, action="query", generator="search", gsrsearch=f"{term} -icon -logo -map",
                gsrnamespace=6, gsrlimit=max(1, min(limit * 2, 50)), prop="imageinfo",
                iiprop="url|extmetadata|size|mime", iiurlwidth=self.preview_px,
                iiextmetadatafilter="ImageDescription|LicenseShortName|Artist|Credit|License",
            )
        except Exception as exc:
            log.warning("image search failed for %r: %s", term, exc)
            return []
        hits: list[ImageHit] = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            mime = info.get("mime", "")
            if not mime.startswith("image/") or mime == "image/svg+xml":
                continue
            licence = _clean(meta.get("LicenseShortName", {}).get("value")) or None
            if licence and not _LICENSE_OK.match(licence):
                continue  # keep the library safe to reuse
            hits.append(ImageHit(
                title=re.sub(r"^File:", "", page.get("title", "")),
                page_url=info.get("descriptionurl", ""),
                image_url=info.get("url", ""),
                preview_url=info.get("thumburl") or info.get("url", ""),
                description=_clean(meta.get("ImageDescription", {}).get("value"))[:600],
                license=licence,
                attribution=_clean(meta.get("Artist", {}).get("value")) or
                            _clean(meta.get("Credit", {}).get("value")) or None,
                width=info.get("width"), height=info.get("height"), mime_type=mime,
            ))
            if len(hits) >= limit:
                break
        return hits

    def thumbnail_url(self, hit: ImageHit) -> str:
        return re.sub(r"/(\d+)px-", f"/{self.thumb_px}px-", hit.preview_url)
