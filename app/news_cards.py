"""Shared article presentation for search, related news and graph nodes."""
from __future__ import annotations

from typing import Any

from app.agents.diffbot_client import is_article_image_url
from app.schemas import NewsCard
from app.utils import make_news_id

_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1516937941344-00b4e0337589?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=520&q=80",
]


def thumbnail(art: dict[str, Any], idx: int = 0) -> tuple[str, bool]:
    """Return (thumbnail_url, fallback_used), excluding known QR images."""
    url = art.get("thumbnail_url") or ""
    if is_article_image_url(url):
        return url, False
    return _FALLBACK_IMAGES[idx % len(_FALLBACK_IMAGES)], True


def to_news_card(art: dict[str, Any], idx: int = 0) -> NewsCard:
    thumb, _ = thumbnail(art, idx)
    return NewsCard(
        news_id=art.get("news_id") or make_news_id(art.get("url", "")),
        title=art.get("title", ""),
        summary=art.get("description") or art.get("summary") or "",
        description=art.get("description") or "",
        thumbnail_url=thumb,
        source_name=art.get("source") or "",
        published_at=art.get("published_at") or "",
        source_url=art.get("naver_url") or art.get("url") or "",
        related_stock_names=art.get("related_stock_names") or [],
        keywords=art.get("keywords") or [],
        categories=art.get("categories") or [],
    )
