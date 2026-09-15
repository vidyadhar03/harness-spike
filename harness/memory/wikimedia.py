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

from .ports import ImageHit, TermHit, VerificationServiceError

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MotionX-harness/0.5 (https://motionx.in; contact: tech@motionx.in)"
_TAGS = re.compile(r"<[^>]+>")
_LICENSE_OK = re.compile(r"^(cc|public domain|pd|cc0)", re.I)


def _norm(name: str) -> str:
    return " ".join(name.replace("_", " ").lower().split())


def _batch(limit: int) -> int:
    return max(1, min(limit * 2, 50))   # headroom for files the licence and type filters drop


def _clean(html: str | None) -> str:
    return " ".join(_TAGS.sub(" ", html or "").replace("&nbsp;", " ").split())


class WikimediaImages:
    def __init__(self, timeout: float = 20.0, user_agent: str = USER_AGENT,
                 preview_px: int = 1280, thumb_px: int = 512):
        self.timeout, self.user_agent = timeout, user_agent
        self.preview_px, self.thumb_px = preview_px, thumb_px
        self._categories: dict[tuple[str, str], str | None] = {}

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
        """An article whose title or lead mentions the term.

        Returns None when the encyclopedia has no matching article (genuine no-match).
        Raises VerificationServiceError when the API itself is unreachable.
        """
        try:
            data = self._api(WIKIPEDIA_API, action="query", list="search", srsearch=term, srlimit=3,
                             srprop="snippet")
        except Exception as exc:
            raise VerificationServiceError(f"term lookup failed for {term!r}: {exc}") from exc
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

    def search_images(self, term: str, limit: int, region: str | None = None,
                      region_title: str | None = None) -> list[ImageHit]:
        """Free-text search, or scoped to a region.

        Commons categories are curated by place, so a region resolves to its category
        where one exists and the search runs across that category tree: the region's own
        images with no other words, a term within it otherwise. deepcat keeps the
        category's precision but ranks by relevance; listing members alphabetises. Free
        text naming the region is the fallback, because a category that doesn't exist or
        is too large for deepcat must not make the term unsearchable.
        """
        if region is None:
            return self._search(term, limit)
        own = _norm(term) == _norm(region)
        category = self._category(region, region_title)
        if category:
            scope = f'deepcat:"{category.removeprefix("Category:")}"'
            query = scope if own else f"{term} {scope}"
            hits = self._search(query, limit)
            log.info("%r: %r returned %d licensed image(s)", term, query, len(hits))
            if hits:
                return hits
        query = term if own else f"{term} {region}"
        hits = self._search(query, limit)
        log.info("%r: free text %r returned %d licensed image(s)", term, query, len(hits))
        return hits

    def get_file(self, title: str) -> ImageHit | None:
        """One Commons file by title, through the same licence and type filters as search."""
        hits = self._images(1, titles="File:" + re.sub(r"^File:", "", title))
        return hits[0] if hits else None

    def _category(self, region: str, title: str | None) -> str | None:
        """The Commons category for a region, matched against the verified page title first.

        Search ranking alone is not enough: "Kinnaur" ranks Category:Kinnaur Kailash, a
        mountain, above Category:Kinnaur district. Only an exact name match is accepted.
        """
        key = (_norm(region), _norm(title or ""))
        if key in self._categories:
            return self._categories[key]
        wanted = [n for n in dict.fromkeys([_norm(title or ""), _norm(region)]) if n]
        found = None
        for query in wanted:
            try:
                data = self._api(COMMONS_API, action="query", list="search", srsearch=query,
                                 srnamespace=14, srlimit=20)
            except Exception as exc:
                log.warning("category search failed for %r: %s", query, exc)
                continue
            titles = {_norm(r["title"].removeprefix("Category:")): r["title"]
                      for r in data.get("query", {}).get("search", [])}
            found = next((titles[n] for n in wanted if n in titles), None)
            if found:
                break
        log.info("region %r (%s) resolved to %s", region, title or "no page", found or "no category")
        self._categories[key] = found
        return found

    def _search(self, query: str, limit: int) -> list[ImageHit]:
        return self._images(limit, generator="search", gsrsearch=f"{query} -icon -logo -map",
                            gsrnamespace=6, gsrlimit=_batch(limit))

    def _images(self, limit: int, **generator) -> list[ImageHit]:
        try:
            data = self._api(
                COMMONS_API, action="query", prop="imageinfo",
                iiprop="url|extmetadata|size|mime", iiurlwidth=self.preview_px,
                iiextmetadatafilter="ImageDescription|LicenseShortName|Artist|Credit|License",
                **generator,
            )
        except Exception as exc:
            log.warning("image search failed for %r: %s", generator, exc)
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
