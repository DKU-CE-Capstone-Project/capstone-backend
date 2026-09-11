"""Shared utility helpers."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app import store
from app.config import settings


def make_news_id(url: str) -> str:
    """Generate a deterministic, URL-safe 12-char ID from a news article URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    """외부 API의 발행일시 문자열을 datetime으로 파싱한다. 실패하면 None.

    news_fetcher가 GDELT("20260525T010000Z")를 ISO로 정규화해 넘기지만,
    NewsAPI/Currents 등 소스마다 형식이 달라 방어적으로 처리한다.
    설계 스키마의 published_at은 date 타입이라, 문자열을 그대로 넣으면
    MongoDB validator가 거부한다.
    """
    if isinstance(value, datetime):
        return value
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _domain(url: str) -> str:
    try:
        return (urlparse(url or "").netloc or "").lower()
    except ValueError:
        return ""


# ── 앱 내부 형태 ↔ 설계 스키마 매핑 ───────────────────────────────────────────
#
# store.news_cache는 수집기가 만든 평평한 dict를 그대로 들고 있고, API 레이어와
# graph_builder가 그 형태에 의존한다. 반면 MongoDB에는 설계 문서(구조크)의
# `news` 스키마로 저장해야 한다. 두 형태를 여기서만 변환해, 앱 내부 구조를
# 건드리지 않으면서 DB는 설계를 지키도록 한다.


def to_news_document(art: dict[str, Any]) -> dict[str, Any]:
    """앱 내부 기사 dict → 설계 `news` 컬렉션 문서."""
    url = art.get("url") or ""
    now = _now()
    return {
        "news_id": art.get("news_id") or make_news_id(url),
        "title": art.get("title") or "",
        "summary": art.get("summary") or "",
        # Diffbot이 추출한 본문은 cleaned_content에, 없으면 description에 들어 있다.
        "content": art.get("cleaned_content") or art.get("description") or "",
        "url": url,
        "source": {
            "name": art.get("source") or "",
            "domain": _domain(url),
        },
        "thumbnail_url": art.get("thumbnail_url") or "",
        "published_at": _parse_dt(art.get("published_at")),
        "collected_at": now,
        # TODO(문주안): 뉴스 수집 단계에서 키워드/카테고리 추출 미구현
        "keywords": art.get("keywords") or [],
        "categories": art.get("categories") or [],
        # TODO(김성민): 종목 추출 에이전트 미구현 — related_stock_names가 항상 빈 배열인 원인
        "related_tickers": art.get("related_tickers") or [],
        "status": art.get("status") or "collected",
        "language": art.get("language") or "ko",
        "is_deleted": False,
        "created_at": now,
        "updated_at": now,
        "_search_keyword": art.get("_search_keyword") or "",
    }


def from_news_document(doc: dict[str, Any]) -> dict[str, Any]:
    """설계 `news` 문서 → 앱 내부 기사 dict (캐시 미스 시 복원용)."""
    source = doc.get("source") or {}
    published = doc.get("published_at")
    return {
        "news_id": doc.get("news_id", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "source": source.get("name") or source.get("domain") or "",
        "published_at": published.isoformat() if isinstance(published, datetime) else (published or ""),
        "description": doc.get("content") or doc.get("summary") or "",
        "summary": doc.get("summary") or "",
        "thumbnail_url": doc.get("thumbnail_url") or "",
        "_search_keyword": doc.get("_search_keyword", ""),
    }


async def cache_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign news_id, persist to in-memory store, and (if enabled) write-through to
    MongoDB with a Gemini embedding for vector search. Returns enriched list.

    in-memory store는 L1 캐시(즉시 응답용). MongoDB 영속화/임베딩은 use_mongodb 시에만.
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
    """각 기사에 임베딩 부여 후 MongoDB news에 write-through (벡터검색 근거)."""
    from app import database
    from app.agents.llm import embed

    async def _one(art: dict[str, Any]) -> None:
        text = f"{art.get('title','')} {art.get('summary') or art.get('description','')}".strip()
        vec = await embed(text)
        doc = to_news_document(art)
        if vec:
            # 설계상 embedding은 news_analysis 소속이지만 news에 둔다.
            # 임베딩 대상이 분석 결과가 아니라 뉴스 본문이고, 벡터 인덱스를
            # 한 컬렉션에 모아야 $vectorSearch 경로가 단순해지기 때문.
            doc["embedding"] = vec
        await database.save_news(doc)

    try:
        await asyncio.gather(*(_one(a) for a in enriched))
    except Exception as exc:  # noqa: BLE001
        print(f"[utils] mongo persist skip: {type(exc).__name__}: {exc}")


async def resolve_news(news_id: str) -> dict[str, Any] | None:
    """news_id로 기사를 찾는다. in-memory 미스 시 MongoDB로 폴백한다.

    api 팟을 재시작하거나 여러 개로 띄우면 in-memory 캐시가 비어 404가 나던 문제를
    막는다. 찾은 문서는 L1 캐시에 다시 채워 넣는다.
    """
    art = store.news_cache.get(news_id)
    if art:
        return art

    from app import database

    doc = await database.get_news(news_id)
    if not doc:
        return None

    restored = from_news_document(doc)
    store.news_cache[news_id] = restored
    return restored


_TIER_ORDER = {"FREE": 0, "BASIC": 1, "PAID": 2}


def tier_ok(user_tier: str, required: str) -> bool:
    """Return True if user_tier satisfies the required tier level."""
    return _TIER_ORDER.get(user_tier.upper(), 0) >= _TIER_ORDER.get(required.upper(), 0)
