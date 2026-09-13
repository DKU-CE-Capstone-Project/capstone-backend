"""All /api/v1/news/* endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.agents.diffbot_client import extract_articles_with_diffbot
from app.agents.filter_agent import _content_tokens, _overlap_coefficient
from app.agents.graph_builder import _relevance_score, build_graph
from app.agents.news_fetcher import fetch_news
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
from app import session as session_store, store
from app.utils import cache_articles, make_news_id, resolve_news, tier_ok

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
    title = art.get("title", "")
    return NewsCard(
        news_id=art.get("news_id") or make_news_id(art.get("url", "")),
        title=title,
        summary=title or art.get("summary") or art.get("description", ""),
        thumbnail_url=thumb,
        source_name=art.get("source", ""),
        published_at=art.get("published_at", ""),
        related_stock_names=[],
    )


async def _fetch_and_cache(keyword: str) -> list[dict[str, Any]]:
    """Fetch news list results and cache them without dropping source thumbnails."""
    raw_articles = [
        {**article, "_search_keyword": keyword}
        for article in await fetch_news(keyword, page_size=10)
    ]
    return await cache_articles(raw_articles)


async def _related_articles(news_id: str, extra: int = 8) -> list[dict[str, Any]]:
    """
    Return related articles for a cached article.

    Strategy:
    1. Return other cached articles from the same search (same _search_keyword).
    2. If fewer than 3, re-fetch using the original search keyword.
    """
    center = await resolve_news(news_id)
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

    # 1-b) in-memory가 비어 있으면(재시작 직후 등) MongoDB에서 같은 검색어 기사를 복원
    if original_keyword:
        from app import database
        from app.utils import from_news_document

        docs = await database.find_news_by_keyword(original_keyword, limit=extra + 5)
        restored = []
        for doc in docs:
            nid = doc.get("news_id", "")
            if not nid or nid == news_id:
                continue
            art = store.news_cache.get(nid) or from_news_document(doc)
            store.news_cache[nid] = art
            restored.append(art)
        if len(restored) >= 3:
            return restored[:extra]

    # 2) Re-fetch using the original keyword (or first meaningful title word as fallback)
    if not original_keyword:
        tokens = list(_content_tokens(center.get("title", "")))
        original_keyword = tokens[0] if tokens else center.get("title", "")[:15]

    raw = await fetch_news(original_keyword, page_size=extra + 2)
    enriched = await cache_articles([{**a, "_search_keyword": original_keyword} for a in raw])
    return [a for a in enriched if a.get("news_id") != news_id]


# 연관도 점수를 news_relations에 캐시한다 (12주차 회의 "릴레이션까지는 저장하자" 결정).
# 같은 (source, target) 쌍은 uniq_news_relations_pair 인덱스로 upsert된다.
_RELATION_MIN_SCORE = 0.0


async def _persist_relation(
    source_id: str,
    target_id: str,
    score: float,
    reason: str,
    shared_keywords: list[str],
) -> None:
    """계산된 연관도를 MongoDB에 저장한다. use_mongodb 비활성 시 no-op."""
    if score <= _RELATION_MIN_SCORE:
        return  # 겹치는 토큰이 없으면 관계로 보지 않는다

    from app import database

    now = datetime.now(timezone.utc)
    await database.save_relation({
        "source_news_id": source_id,
        "target_news_id": target_id,
        "relation": {
            # 토큰 중복만으로 판별 가능한 유형은 same_topic 하나뿐이다.
            # TODO(김성민): cause_effect / same_company / same_industry /
            #   opposite_view / follow_up 판별은 미구현 (설계 6종 중 5종).
            "type": "same_topic",
            "score": score,
            "reason": reason,
        },
        "shared_keywords": shared_keywords,
        "created_at": now,
        "updated_at": now,
    })


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
    art = await resolve_news(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")
    thumb, fallback = _thumb(art)
    return ThumbnailResponse(news_id=news_id, thumbnail_url=thumb, fallback_used=fallback)


# ── GET /{news_id}/source ─────────────────────────────────────────────────────

@router.get("/{news_id}/source", response_model=SourceResponse)
async def get_source(news_id: str) -> SourceResponse:
    """뉴스 원문 출처 및 링크를 반환합니다."""
    art = await resolve_news(news_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    original_body = art.get("cleaned_content", "")
    if not original_body and art.get("url"):
        extracted = await extract_articles_with_diffbot([art], max_articles=1, concurrency=1)
        if extracted:
            next_art = {**art, **extracted[0]}
            original_body = next_art.get("cleaned_content", "")
            if original_body:
                next_art["description"] = original_body
                store.news_cache[news_id] = next_art
                await cache_articles([next_art])
                art = next_art

    return SourceResponse(
        news_id=news_id,
        source_name=art.get("source", ""),
        source_url=art.get("url", ""),
        published_at=art.get("published_at", ""),
        original_title=art.get("title", ""),
        original_body=original_body,
    )


# ── GET /{news_id}/graph ──────────────────────────────────────────────────────

@router.get("/{news_id}/graph", response_model=GraphResponse)
async def get_graph(
    request: Request,
    news_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    limit: int = Query(default=10, ge=1, le=30),
    include_distance: bool = Query(default=True),
) -> GraphResponse:
    """마인드맵 데이터를 반환합니다.

    세션(쿠키)에 기록된 확장 노드가 있으면 그 노드의 이웃까지 펼쳐서 돌려줍니다.
    로그인이 없으므로 사용자 구분은 세션 쿠키로만 이뤄지며, 같은 뉴스라도
    세션마다 다른 마인드맵이 나옵니다.
    """
    center = await resolve_news(news_id)
    if not center:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    related = await _related_articles(news_id, extra=limit + 2)
    graph = build_graph(center, related[:limit], include_distance=include_distance)

    # 세션에 중심 노드를 기록하고(중심이 바뀌면 확장 상태 초기화), 확장 노드를 반영한다.
    sid = getattr(request.state, "session_id", "")
    state = await session_store.record_center(
        sid, news_id, query=center.get("_search_keyword", "")
    )
    expanded = (state.get("mindmap") or {}).get("expanded_news_ids") or []
    if expanded:
        await _apply_expansions(graph, expanded, depth=depth, limit=limit)

    return GraphResponse(**graph)


async def _apply_expansions(
    graph: dict[str, Any],
    expanded_ids: list[str],
    depth: int,
    limit: int,
) -> None:
    """확장된 노드의 이웃을 그래프에 덧붙인다 (마인드맵 확장 기능).

    12주차 회의에서 "확장 기능은 아직 안 들어가 있다"고 확인된 부분이다.
    확장 결과는 저장하지 않고, 어떤 노드를 펼쳤는지만 세션에 남긴다.
    depth=1이면 확장하지 않는다(중심 + 1홉만 보기).
    """
    if depth < 2:
        return

    known = {n["news_id"] for n in graph["nodes"]}

    for eid in expanded_ids:
        if eid not in known:
            continue  # 현재 그래프에 없는 노드는 무시 (중심이 바뀐 경우 등)
        parent = await resolve_news(eid)
        if not parent:
            continue

        try:
            # _related_articles는 "같은 검색어로 수집된 기사"를 캐시 순서대로 돌려준다.
            # 그대로 쓰면 기본 그래프가 이미 가져간 상위 N개와 겹쳐 확장이 아무것도
            # 추가하지 못한다. 풀을 넓게 뽑은 뒤 '확장한 노드' 기준 연관도로 재정렬해서,
            # 아직 화면에 없는 기사 중 그 노드와 가장 가까운 것부터 붙인다.
            neighbors = await _related_articles(eid, extra=limit * 3 + 5)
        except HTTPException:
            continue

        neighbors = sorted(
            neighbors,
            key=lambda a: _relevance_score(parent, a),
            reverse=True,
        )

        added = 0
        for art in neighbors:
            nid = art.get("news_id") or make_news_id(art.get("url", ""))
            if not nid or nid in known:
                continue
            summary = art.get("title", "") or art.get("summary") or art.get("description", "")
            graph["nodes"].append({
                "news_id": nid,
                "title": art.get("title", ""),
                "summary": summary[:200],
                "distance": 2,
                "is_center": False,
            })
            graph["edges"].append({
                "source": eid,
                "target": nid,
                "relation_type": "expanded",
                "distance": 2,
            })
            known.add(nid)
            added += 1
            if added >= max(1, limit // 2):
                break  # 확장 노드당 상한 — 그래프가 폭발하지 않도록


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
    center = await resolve_news(news_id)
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
                summary=art.get("title", "") or art.get("summary") or art.get("description", ""),
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
        art = await resolve_news(nid)
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

    source_art = await resolve_news(news_id)
    if not source_art:
        raise HTTPException(status_code=404, detail=f"news_id '{news_id}' not found.")

    targets = [t.strip() for t in target_news_ids.split(",") if t.strip()]
    relations: list[RelationScore] = []

    for tid in targets:
        target_art = await resolve_news(tid)
        if not target_art:
            continue
        score = _relevance_score(source_art, target_art)
        src_tokens = _content_tokens(source_art.get("title", ""))
        tgt_tokens = _content_tokens(target_art.get("title", ""))
        shared = list(src_tokens & tgt_tokens)[:5]
        reason = f"공통 키워드 {len(shared)}개 공유"
        relations.append(
            RelationScore(
                source_news_id=news_id,
                target_news_id=tid,
                relevance_score=score,
                relation_reason=reason,
                shared_keywords=shared,
            )
        )
        await _persist_relation(news_id, tid, score, reason, shared)

    return RelationsResponse(relations=relations)
