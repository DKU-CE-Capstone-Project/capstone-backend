"""Shared utility helpers."""
from __future__ import annotations

import hashlib
from typing import Any

from app import store


def make_news_id(url: str) -> str:
    """Generate a deterministic, URL-safe 12-char ID from a news article URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def cache_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign news_id to each article, persist to store, return enriched list."""
    enriched: list[dict[str, Any]] = []
    for art in articles:
        url = art.get("url") or ""
        nid = make_news_id(url) if url else make_news_id(art.get("title", ""))
        enriched_art = {**art, "news_id": nid}
        store.news_cache[nid] = enriched_art
        enriched.append(enriched_art)
    return enriched


_TIER_ORDER = {"FREE": 0, "BASIC": 1, "PAID": 2}


def tier_ok(user_tier: str, required: str) -> bool:
    """Return True if user_tier satisfies the required tier level."""
    return _TIER_ORDER.get(user_tier.upper(), 0) >= _TIER_ORDER.get(required.upper(), 0)
