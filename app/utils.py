"""Shared utility helpers."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app import store
from app.config import settings


def make_news_id(url: str) -> str:
    """Generate a deterministic, URL-safe 12-char ID from a news article URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


async def cache_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign news_id, persist to in-memory store, and (if enabled) write-through to
    MongoDB with a Gemini embedding for vector search. Returns enriched list.

    in-memory store는 L1 캐시(즉시 응답용). MongoDB 영속화/임베딩은 use_mongodb 시에만.
    반환하는 dict 는 기존 형태 그대로다 — 설계 스키마 변환은 Mongo 저장 직전에만 한다.
    """
    enriched: list[dict[str, Any]] = []
    for art in articles:
        url = art.get("url") or ""
        nid = make_news_id(url) if url else make_news_id(art.get("title", ""))
        enriched_art = {**art, "news_id": nid}
        store.news_cache[nid] = enriched_art
        enriched.append(enriched_art)

    if settings.use_mongodb and enriched:
        await _persist_to_mongo(enriched)

    return enriched


async def _persist_to_mongo(enriched: list[dict[str, Any]]) -> None:
    """각 기사에 임베딩 부여 후 MongoDB news에 write-through (벡터검색 근거).

    in-memory dict 를 그대로 넣지 않고 database.news_doc_from_article() 로
    「구조크」news 스키마에 맞춰 변환한다. 그러지 않으면 mongo-init 의 validator 가
    거부한다(source 가 문자열, 날짜가 ISO 문자열, 필수 필드 누락).
    in-memory/API 응답 형태는 건드리지 않으므로 프론트는 영향 없다.
    """
    from app import database
    from app.agents.llm import embed

    async def _one(art: dict[str, Any]) -> None:
        text = f"{art.get('title','')} {art.get('summary') or art.get('description','')}".strip()
        vec = await embed(text)
        doc = database.news_doc_from_article(art)
        if vec:
            doc["embedding"] = vec
        await database.save_news(doc)

    try:
        await asyncio.gather(*(_one(a) for a in enriched))
    except Exception as exc:  # noqa: BLE001
        print(f"[utils] mongo persist skip: {type(exc).__name__}: {exc}")


_TIER_ORDER = {"FREE": 0, "BASIC": 1, "PAID": 2}


def tier_ok(user_tier: str, required: str) -> bool:
    """Return True if user_tier satisfies the required tier level."""
    return _TIER_ORDER.get(user_tier.upper(), 0) >= _TIER_ORDER.get(required.upper(), 0)
