"""GET /api/v1/keywords/recommended · GET /api/v1/keywords/related"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query

from app.agents.article_metadata import ALIASES, STOPWORDS
from app.agents.naver_client import NaverNewsError
from app.api.v1.news import _limit_search, search_news
from app.schemas import (
    RecommendedKeyword,
    RecommendedKeywordsResponse,
    RelatedKeyword,
    RelatedKeywordsResponse,
)

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


@router.get("/recommended", response_model=RecommendedKeywordsResponse)
async def recommended_keywords(
    limit: int = Query(default=8, ge=1, le=20),
) -> RecommendedKeywordsResponse:
    """홈 화면에 노출할 추천 키워드 목록을 반환합니다."""
    return RecommendedKeywordsResponse(keywords=_KEYWORDS[:limit])


# Alias spelling → canonical keyword ("현대차" → "현대자동차"), compared case-insensitively.
_CANONICAL = {alias.casefold(): name for name, aliases in ALIASES.items() for alias in (name, *aliases)}
_MIN_ARTICLES = 2


def _canonical(keyword: str) -> str:
    text = " ".join(keyword.split())
    return _CANONICAL.get(text.casefold(), text)


@router.get("/related", response_model=RelatedKeywordsResponse, dependencies=[Depends(_limit_search)])
async def related_keywords(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=12),
) -> RelatedKeywordsResponse:
    """검색어의 연관 키워드. 검색 기사 20건의 keywords를 기사 수 기준으로 집계하고,
    부족하면 고정 추천 목록으로 채웁니다(source로 구분). 추가 LLM 호출은 없습니다."""
    query = " ".join(q.split())
    try:
        found = await search_news(q=query, page=1, size=20, sort="relevance")
    except NaverNewsError as exc:
        raise HTTPException(status_code=502, detail="뉴스 검색에 실패했습니다. 잠시 후 다시 시도해 주세요.") from exc

    skip = {_canonical(query).casefold(), *(word.casefold() for word in STOPWORDS)}
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for rank, card in enumerate(found.news_cards):
        for keyword in {_canonical(k) for k in card.keywords if k and k.strip()}:
            if keyword.casefold() in skip:
                continue
            counts[keyword] += 1
            first_seen.setdefault(keyword, rank)
    chosen = sorted((k for k, n in counts.items() if n >= _MIN_ARTICLES),
                    key=lambda k: (-counts[k], first_seen[k], k))[:limit]
    keywords = [RelatedKeyword(keyword=k, article_count=counts[k], source="articles") for k in chosen]

    taken = {k.casefold() for k in chosen} | skip
    for item in _KEYWORDS:
        if len(keywords) >= limit:
            break
        if item.keyword.casefold() not in taken:
            keywords.append(RelatedKeyword(keyword=item.keyword, article_count=0, source="recommended"))
            taken.add(item.keyword.casefold())
    return RelatedKeywordsResponse(query=query, keywords=keywords, article_total=len(found.news_cards))
