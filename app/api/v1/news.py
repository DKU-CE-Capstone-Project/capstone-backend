"""All /api/v1/news/* endpoints."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import database, store
from app.agents import naver_categories
from app.agents.article_embeddings import EmbeddingUnavailable
from app.agents.filter_agent import _content_tokens
from app.agents.graph_builder import _relevance_score, build_graph
from app.agents.news_fetcher import fetch_naver_news_page, fetch_news
from app.agents.related_selector import article_id, select_related
from app.config import settings
from app.news_cards import thumbnail as _thumb
from app.news_cards import to_news_card as _to_news_card
from app.schemas import (
    GraphResponse,
    NewsSelectionRequest,
    NewsSelectionResponse,
    RelatedNewsItem,
    RelatedResponse,
    RelationScore,
    RelationsResponse,
    SearchResponse,
    SourceResponse,
    ThumbnailResponse,
)
from app.utils import cache_articles, tier_ok

router = APIRouter()

# ── helpers ───────────────────────────────────────────────────────────────────

async def _fetch_and_cache(keyword: str) -> list[dict[str, Any]]:
    """Fetch news list results and cache them without dropping source thumbnails."""
    raw_articles = [
        {**article, "_search_keyword": keyword}
        for article in await fetch_news(keyword, page_size=20)
    ]
    return await cache_articles(raw_articles)


async def _naver_search_response(keyword: str, page: int, size: int, sort: str) -> SearchResponse:
    result = await fetch_naver_news_page(keyword, page=page, size=size, sort=sort)
    articles = await cache_articles([{**a, "_search_keyword": keyword} for a in result.articles])
    return SearchResponse(news_cards=[_to_news_card(a, i) for i, a in enumerate(articles)],
                          total_count=result.total)


async def _related_articles(news_id: str) -> list[dict[str, Any]]:
    """
    Collect candidates, independently of the final display/tier limit.

    Strategy:
    1. Collect memory/MongoDB articles from the same search (_search_keyword).
    2. If fewer than 3, re-fetch using the original search keyword.
    """
    center = await store.get_news(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found. Search first.")

    original_keyword = center.get("_search_keyword", "")
    cap = settings.news_map_candidate_limit
    # Candidate budget is independent of FREE/PAID and the requested display limit.
    same_search = {
        article_id(art): art for nid, art in store.news_cache.items()
        if original_keyword and nid != news_id and art.get("_search_keyword") == original_keyword
    }
    for art in await database.news_candidates(original_keyword, cap + 1):
        if article_id(art) != news_id:
            same_search.setdefault(article_id(art), art)
    if len(same_search) >= 3:
        return [same_search[nid] for nid in sorted(same_search)[:cap]]

    # 2) Re-fetch using the original keyword (or first meaningful title word as fallback)
    if not original_keyword:
        tokens = sorted(_content_tokens(center.get("title", "")))
        original_keyword = tokens[0] if tokens else center.get("title", "")[:15]

    raw = await fetch_news(original_keyword, page_size=cap + 1)
    enriched = await cache_articles([{**a, "_search_keyword": original_keyword} for a in raw])
    for art in enriched:
        if article_id(art) != news_id:
            same_search[article_id(art)] = art
    return [same_search[nid] for nid in sorted(same_search)[:cap]]


async def _selected_related(
    center: dict[str, Any], limit: int, min_relevance: float = 0.0, tier: str = "FREE",
):
    limit = limit if tier_ok(tier, "PAID") else min(limit, 3)
    candidates = await _related_articles(article_id(center))
    try:
        return await select_related(center, candidates, limit=limit, min_relevance=min_relevance)
    except EmbeddingUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="연관 기사 임베딩을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
        ) from exc


# ── GET /search ───────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResponse)
async def search_news(
    q: str = Query(min_length=1, max_length=100),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=50),
    sort: str = Query(default="relevance", pattern="^(relevance|latest)$"),
) -> SearchResponse:
    """검색어로 뉴스 카드 목록을 조회합니다."""
    if settings.news_provider == "naver" and not settings.mock_news_active:
        return await _naver_search_response(q, page, size, sort)
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
    size: int = Query(default=20, ge=1, le=50),
) -> SearchResponse:
    """키워드 기반 뉴스 카드 목록을 조회합니다."""
    if settings.news_provider == "naver" and not settings.mock_news_active:
        return await _naver_search_response(keyword, page, size, "relevance")
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
    art = await store.get_news(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")
    thumb, fallback = _thumb(art)
    return ThumbnailResponse(news_id=news_id, thumbnail_url=thumb, fallback_used=fallback)


# ── GET /{news_id}/source ─────────────────────────────────────────────────────

@router.get("/{news_id}/source", response_model=SourceResponse)
async def get_source(news_id: str) -> SourceResponse:
    """뉴스 원문 출처 및 링크를 반환합니다."""
    art = await store.get_news(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    naver_url = art.get("naver_url")
    if naver_url:
        labels = await naver_categories.fetch_categories(naver_url)
        if labels is None:
            raise HTTPException(status_code=503, detail="기사 분류를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        if not naver_categories.allowed(labels):
            raise HTTPException(status_code=404, detail="정치·사회 기사는 제공하지 않습니다.")
    return SourceResponse(
        news_id=news_id,
        source_name=art.get("source", ""),
        source_url=naver_url or art.get("url", ""),
        published_at=art.get("published_at", ""),
        original_title=art.get("title", ""),
        original_body="",
        thumbnail_url=art.get("thumbnail_url", ""),
        description=art.get("description") or art.get("summary", ""),
        keywords=art.get("keywords", []),
        categories=art.get("categories", []),
    )


# ── GET /{news_id}/graph ──────────────────────────────────────────────────────

@router.get("/{news_id}/graph", response_model=GraphResponse)
async def get_graph(
    news_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    limit: int = Query(default=10, ge=1, le=50),
    include_distance: bool = Query(default=True),
    min_relevance: float = Query(default=0.0, ge=0.0, le=1.0),
    include_score: bool = Query(default=False),
    tier: str = Query(default="FREE", pattern="^(FREE|BASIC|PAID)$"),
) -> GraphResponse:
    """마인드맵 데이터를 반환합니다."""
    center = await store.get_news(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    related = await _selected_related(center, limit, min_relevance, tier)
    scores = {article_id(item.article): round(item.score, 6) for item in related} if tier_ok(tier, "PAID") else {}
    graph = build_graph(
        center, [item.article for item in related], include_distance=include_distance, scores=scores,
    )

    return GraphResponse(**graph)


# ── GET /{news_id}/related ────────────────────────────────────────────────────

@router.get("/{news_id}/related", response_model=RelatedResponse)
async def get_related(
    news_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_relevance: float = Query(default=0.0, ge=0.0, le=1.0),
    include_score: bool = Query(default=False),
    tier: str = Query(default="FREE", pattern="^(FREE|BASIC|PAID)$"),
) -> RelatedResponse:
    """
    연관 뉴스를 반환합니다.
    FREE: 최대 3개, relevance_score 미포함.
    PAID: 요청 limit 적용 + relevance_score 포함.
    모든 요금제는 같은 임베딩 기반 필터·정렬을 수행합니다.
    """
    center = await store.get_news(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    is_paid = tier_ok(tier, "PAID")
    related = await _selected_related(center, limit, min_relevance, tier)

    items: list[RelatedNewsItem] = []
    for i, ranked in enumerate(related):
        art, score = ranked.article, ranked.score
        items.append(
            RelatedNewsItem(
                **_to_news_card(art, i).model_dump(),
                relevance_score=round(score, 6) if is_paid else None,
                distance=1,
            )
        )

    return RelatedResponse(related_news=items)


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
        art = await store.get_news(nid)
        if art:
            selected.append({"news_id": nid, "title": art.get("title", "")})

    sid = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
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

    source_art = await store.get_news(news_id)
    if not source_art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    targets = [t.strip() for t in target_news_ids.split(",") if t.strip()]
    relations: list[RelationScore] = []

    for tid in targets:
        target_art = await store.get_news(tid)
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
