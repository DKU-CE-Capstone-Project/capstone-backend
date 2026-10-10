"""GET /api/v1/keywords/recommended"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas import RecommendedKeyword, RecommendedKeywordsResponse

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
