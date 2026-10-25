"""GET /api/v1/keywords/recommended and GET /api/v1/keywords/map"""
from __future__ import annotations

import time
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Query, Request

from app.agents import news_fetcher
from app.agents.keyword_map_extractor import (
    KeywordMapEmbeddingUnavailable,
    extract_keyword_map,
)
from app.config import settings
from app.schemas import (
    KeywordMapResponse,
    RecommendedKeyword,
    RecommendedKeywordsResponse,
)
from app.usage_limits import check_quota, client_key

router = APIRouter()

_KEYWORDS: list[RecommendedKeyword] = [
    RecommendedKeyword(keyword="엔비디아",    category="기업",      rank=1),
    RecommendedKeyword(keyword="삼성전자",    category="기업",      rank=2),
    RecommendedKeyword(keyword="트럼프 관세", category="경제·정치", rank=3),
    RecommendedKeyword(keyword="반도체",      category="산업",      rank=4),
    RecommendedKeyword(keyword="AI 서버",     category="기술",      rank=5),
    RecommendedKeyword(keyword="원유",        category="원자재",    rank=6),
    RecommendedKeyword(keyword="금리",        category="금융",      rank=7),
    RecommendedKeyword(keyword="중동 전황",   category="지정학",    rank=8),
    RecommendedKeyword(keyword="전기차",      category="산업",      rank=9),
    RecommendedKeyword(keyword="HBM",         category="기술",      rank=10),
]

_MAX_CACHE_ENTRIES = 128
_map_cache: OrderedDict[tuple[object, ...], tuple[float, KeywordMapResponse]] = (
    OrderedDict()
)


def clear_keyword_map_cache() -> None:
    """Clear the bounded in-memory keyword map response cache."""
    _map_cache.clear()


async def _limit_keyword_map(request: Request) -> None:
    peer = client_key(request.client.host if request.client else None)
    await check_quota(
        peer,
        "news_search",
        per_session=settings.news_search_client_hourly_limit,
        global_limit=settings.news_search_global_hourly_limit,
    )


@router.get("/recommended", response_model=RecommendedKeywordsResponse)
async def recommended_keywords(
    limit: int = Query(default=8, ge=1, le=20),
) -> RecommendedKeywordsResponse:
    """홈 화면에 노출할 추천 키워드 목록을 반환합니다."""
    return RecommendedKeywordsResponse(keywords=_KEYWORDS[:limit])


@router.get("/map", response_model=KeywordMapResponse)
async def get_keyword_map(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100, description="검색어"),
) -> KeywordMapResponse:
    """검색어 관련 실시간 뉴스 제목·요약을 기반으로 Zero-LLM(`gemini-embedding-001` 배치 + Kiwi) 키워드 맵을 반환합니다."""
    cleaned = q.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="검색어를 입력해 주세요.")

    provider_key = "mock" if settings.mock_news_active else settings.news_provider
    cache_key = (
        provider_key,
        settings.keyword_map_embedding_model,
        settings.keyword_map_embedding_dimensions,
        settings.keyword_map_embedding_task_type,
        settings.keyword_map_sim_threshold,
        settings.keyword_map_sim_threshold_max,
        settings.keyword_map_anisotropy_gain,
        settings.keyword_map_candidate_size,
        cleaned.casefold(),
    )
    now = time.monotonic()
    ttl = settings.keyword_map_cache_ttl_seconds
    if ttl > 0:
        cached = _map_cache.get(cache_key)
        if cached is not None:
            expires_at, payload = cached
            if expires_at > now:
                _map_cache.move_to_end(cache_key)
                return payload
            _map_cache.pop(cache_key, None)

    await _limit_keyword_map(request)

    if not settings.mock_news_active and settings.news_provider == "naver":
        page = await news_fetcher.fetch_search_page(
            cleaned,
            start=1,
            size=settings.keyword_map_candidate_size,
        )
        raw_articles = page.articles
    else:
        raw_articles = await news_fetcher.fetch_news(cleaned)

    try:
        extracted = await extract_keyword_map(cleaned, raw_articles)
    except KeywordMapEmbeddingUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="임베딩 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
        ) from exc

    response = KeywordMapResponse(**extracted)
    if ttl > 0:
        _map_cache[cache_key] = (now + ttl, response)
        _map_cache.move_to_end(cache_key)
        while len(_map_cache) > _MAX_CACHE_ENTRIES:
            _map_cache.popitem(last=False)
    return response
