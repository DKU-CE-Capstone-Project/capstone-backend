"""Read NAVER's publisher-assigned section labels before paid extraction.

Diffbot's normalized article HTML need not retain the page's classification UI,
so inspect the original NAVER HTML, never inferred LLM categories or URL sid.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
EXCLUDED = {"정치", "사회"}
_cache: OrderedDict[str, tuple[float, tuple[str, ...] | None]] = OrderedDict()
_images: dict[str, str] = {}


def page_image(url: str) -> str:
    cached = _cache.get(url)
    return _images.get(url, "") if cached and cached[0] > time.monotonic() else ""


def parse_categories(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    labels = [
        node.get_text(strip=True)
        for node in soup.select("em.media_end_categorize_item")
    ]
    return list(dict.fromkeys(label for label in labels if label))


async def fetch_categories(url: str) -> list[str] | None:
    """None means unverified: defer that article instead of guessing its category."""
    cached = _cache.get(url)
    if cached and cached[0] > time.monotonic():
        _cache.move_to_end(url)
        return list(cached[1]) if cached[1] is not None else None
    labels = None
    image = ""
    try:
        host = (urlparse(url).hostname or "").lower()
        if host != "news.naver.com" and not host.endswith(".news.naver.com"):
            return None
        async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
            response = await asyncio.wait_for(client.get(url), timeout=5)
        response.raise_for_status()
        final_host = (response.url.host or "").lower()
        if final_host == "news.naver.com" or final_host.endswith(".news.naver.com"):
            labels = parse_categories(response.text) or None
            node = BeautifulSoup(response.text, "html.parser").find("meta", property="og:image")
            if node and isinstance(node.get("content"), str):
                image = node["content"]
    except (httpx.HTTPError, TimeoutError, ValueError):
        logger.info("NAVER article category unavailable")
    _cache[url] = (
        time.monotonic() + (300 if labels else 15),
        tuple(labels) if labels else None,
    )
    _cache.move_to_end(url)
    _images[url] = image
    while len(_cache) > 256:
        expired_url, _ = _cache.popitem(last=False)
        _images.pop(expired_url, None)
    return labels


def allowed(labels: list[str] | None) -> bool:
    return bool(labels) and not EXCLUDED.intersection(labels)
