"""All /api/v1/news/* endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.agents.filter_agent import _content_tokens, _overlap_coefficient
from app.agents.graph_builder import _relevance_score, build_graph
from app.agents.news_fetcher import fetch_news
from app.agents.orchestrator import run_analysis
from app.schemas import (
    GraphResponse,
    NewsCard,
    NewsSelectionRequest,
    NewsSelectionResponse,
    RelatedNewsItem,
    RelationScore,
    RelationsResponse,
    SearchResponse,
    SourceResponse,
    ThumbnailResponse,
)
from app import store
from app.utils import cache_articles, make_news_id, tier_ok

router = APIRouter()

# ── helpers ───────────────────────────────────────────────────────────────────

_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1516937941344-00b4e0337589?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=520&q=80",
]


def _thumb(art: dict[str, Any], idx: int = 0) -> tuple[str, bool]:
    """Return (thumbnail_url, fallback_used)."""
    url = art.get("thumbnail_url") or ""
    if url:
        return url, False
    return _FALLBACK_IMAGES[idx % len(_FALLBACK_IMAGES)], True


def _to_news_card(art: dict[str, Any], idx: int = 0) -> NewsCard:
    thumb, _ = _thumb(art, idx)
    return NewsCard(
        news_id=art.get("news_id") or make_news_id(art.get("url", "")),
        title=art.get("title", ""),
        summary=art.get("summary") or art.get("description", ""),
        thumbnail_url=thumb,
        source_name=art.get("source", ""),
        published_at=art.get("published_at", ""),
        related_stock_names=[],
    )


async def _fetch_and_cache(keyword: str) -> list[dict[str, Any]]:
    """Run full analysis pipeline and cache results. Returns enriched articles."""
    result = await run_analysis(keyword)
    raw_articles = [
        {
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "published_at": a.published_at,
            "summary": a.summary,
            "description": a.summary,
            "thumbnail_url": "",
            "_search_keyword": keyword,  # 재검색 시 활용
        }
        for a in result.articles
    ]
    return cache_articles(raw_articles)


async def _related_articles(news_id: str, extra: int = 8) -> list[dict[str, Any]]:
    """
    Return related articles for a cached article.

    Strategy:
    1. Return other cached articles from the same search (same _search_keyword).
    2. If fewer than 3, re-fetch using the original search keyword.
    """
    center = store.news_cache.get(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found. Search first.")

    original_keyword = center.get("_search_keyword", "")

    # 1) Already-cached articles from the same search
    same_search = [
        art for nid, art in store.news_cache.items()
        if nid != news_id and art.get("_search_keyword") == original_keyword
    ]
    if len(same_search) >= 3:
        return same_search[:extra]

    # 2) Re-fetch using the original keyword (or first meaningful title word as fallback)
    if not original_keyword:
        tokens = list(_content_tokens(center.get("title", "")))
        original_keyword = tokens[0] if tokens else center.get("title", "")[:15]

    raw = await fetch_news(original_keyword, page_size=extra + 2)
    enriched = cache_articles([{**a, "_search_keyword": original_keyword} for a in raw])
    return [a for a in enriched if a.get("news_id") != news_id]


# ── GET /search ───────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResponse)
async def search_news(
    q: str = Query(min_length=1, max_length=100),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
    sort: str = Query(default="relevance", pattern="^(relevance|latest)$"),
) -> SearchResponse:
    """검색어로 뉴스 카드 목록을 조회합니다."""
    articles = await _fetch_and_cache(q)

    if sort == "latest":
        articles = sorted(articles, key=lambda a: a.get("published_at", ""), reverse=True)

    total = len(articles)
    start = (page - 1) * size
    paged = articles[start: start + size]

    cards = [_to_news_card(art, i) for i, art in enumerate(paged)]
    return SearchResponse(news_cards=cards, total_count=total)


# ── GET /cards ────────────────────────────────────────────────────────────────

@router.get("/cards", response_model=SearchResponse)
async def news_cards(
    keyword: str = Query(min_length=1, max_length=100),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
) -> SearchResponse:
    """키워드 기반 뉴스 카드 목록을 조회합니다."""
    articles = await _fetch_and_cache(keyword)
    total = len(articles)
    start = (page - 1) * size
    paged = articles[start: start + size]
    cards = [_to_news_card(art, i) for i, art in enumerate(paged)]
    return SearchResponse(news_cards=cards, total_count=total)


# ── GET /{news_id}/thumbnail ──────────────────────────────────────────────────

@router.get("/{news_id}/thumbnail", response_model=ThumbnailResponse)
async def get_thumbnail(news_id: str) -> ThumbnailResponse:
    """뉴스 카드 썸네일 이미지를 반환합니다."""
    art = store.news_cache.get(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")
    thumb, fallback = _thumb(art)
    return ThumbnailResponse(news_id=news_id, thumbnail_url=thumb, fallback_used=fallback)


# ── GET /{news_id}/source ─────────────────────────────────────────────────────

@router.get("/{news_id}/source", response_model=SourceResponse)
async def get_source(news_id: str) -> SourceResponse:
    """뉴스 원문 출처 및 링크를 반환합니다."""
    art = store.news_cache.get(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")
    return SourceResponse(
        news_id=news_id,
        source_name=art.get("source", ""),
        source_url=art.get("url", ""),
        published_at=art.get("published_at", ""),
        original_title=art.get("title", ""),
    )


# ── GET /{news_id}/graph ──────────────────────────────────────────────────────

@router.get("/{news_id}/graph", response_model=GraphResponse)
async def get_graph(
    news_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    limit: int = Query(default=10, ge=1, le=30),
    include_distance: bool = Query(default=True),
) -> GraphResponse:
    """마인드맵 데이터를 반환합니다."""
    center = store.news_cache.get(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    related = await _related_articles(news_id, extra=limit + 2)
    graph = build_graph(center, related[:limit], include_distance=include_distance)

    return GraphResponse(**graph)


# ── GET /{news_id}/related ────────────────────────────────────────────────────

@router.get("/{news_id}/related")
async def get_related(
    news_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_relevance: float = Query(default=0.0, ge=0.0, le=1.0),
    include_score: bool = Query(default=False),
    tier: str = Query(default="FREE", pattern="^(FREE|BASIC|PAID)$"),
) -> dict[str, Any]:
    """
    연관 뉴스를 반환합니다.
    FREE: 최대 3개, relevance_score 미포함.
    PAID: 제한 없음 + relevance_score 포함.
    """
    center = store.news_cache.get(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    is_paid = tier_ok(tier, "PAID")
    effective_limit = limit if is_paid else min(limit, 3)

    related = await _related_articles(news_id, extra=effective_limit + 5)

    items: list[RelatedNewsItem] = []
    for i, art in enumerate(related[:effective_limit]):
        score = _relevance_score(center, art) if (is_paid or include_score) else None
        if min_relevance > 0 and score is not None and score < min_relevance:
            continue
        thumb, _ = _thumb(art, i)
        items.append(
            RelatedNewsItem(
                news_id=art.get("news_id") or make_news_id(art.get("url", "")),
                title=art.get("title", ""),
                summary=art.get("summary") or art.get("description", ""),
                thumbnail_url=thumb,
                relevance_score=score if is_paid else None,
                distance=1,
            )
        )

    return {"related_news": [item.model_dump() for item in items]}


# ── POST /selections (PAID) ───────────────────────────────────────────────────

@router.post("/selections", response_model=NewsSelectionResponse, status_code=201)
async def create_selection(
    body: NewsSelectionRequest,
    tier: str = Query(default="FREE", pattern="^(FREE|BASIC|PAID)$"),
) -> NewsSelectionResponse:
    """선택한 뉴스 카드 묶음을 저장합니다. (PAID 전용)"""
    if not tier_ok(tier, "PAID"):
        raise HTTPException(status_code=403, detail="PAID 플랜이 필요합니다.")

    selected = []
    for nid in body.news_ids:
        art = store.news_cache.get(nid)
        if art:
            selected.append({"news_id": nid, "title": art.get("title", "")})

    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.selection_cache[sid] = {
        "selection_id": sid,
        "news_ids": body.news_ids,
        "selected_news": selected,
        "created_at": now,
    }

    return NewsSelectionResponse(
        selection_id=sid,
        selected_news=selected,
        created_at=now,
    )


# ── GET /{news_id}/relations (PAID) ──────────────────────────────────────────

@router.get("/{news_id}/relations", response_model=RelationsResponse)
async def get_relations(
    news_id: str,
    target_news_ids: str = Query(default="", description="콤마 구분 뉴스 ID 목록"),
    tier: str = Query(default="FREE", pattern="^(FREE|BASIC|PAID)$"),
) -> RelationsResponse:
    """뉴스 간 연관도 점수를 반환합니다. (PAID 전용)"""
    if not tier_ok(tier, "PAID"):
        raise HTTPException(status_code=403, detail="PAID 플랜이 필요합니다.")

    source_art = store.news_cache.get(news_id)
    if not source_art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    targets = [t.strip() for t in target_news_ids.split(",") if t.strip()]
    relations: list[RelationScore] = []

    for tid in targets:
        target_art = store.news_cache.get(tid)
        if not target_art:
            continue
        score = _relevance_score(source_art, target_art)
        src_tokens = _content_tokens(source_art.get("title", ""))
        tgt_tokens = _content_tokens(target_art.get("title", ""))
        shared = list(src_tokens & tgt_tokens)[:5]
        relations.append(
            RelationScore(
                source_news_id=news_id,
                target_news_id=tid,
                relevance_score=score,
                relation_reason=f"공통 키워드 {len(shared)}개 공유",
                shared_keywords=shared,
            )
        )

    return RelationsResponse(relations=relations)
