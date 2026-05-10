import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "news_mock.json"

# ── 한국어 → 영어 키워드 변환 테이블 ─────────────────────────────────────────
# NewsAPI.org는 영문 전용이므로, 한국어 주요 키워드를 영문으로 변환해 검색
KO_TO_EN: dict[str, str] = {
    # 기업
    "엔비디아": "nvidia",
    "삼성": "samsung",
    "삼성전자": "samsung electronics",
    "애플": "apple",
    "테슬라": "tesla",
    "구글": "google",
    "마이크로소프트": "microsoft",
    "메타": "meta",
    "아마존": "amazon",
    "sk하이닉스": "SK hynix",
    "하이닉스": "SK hynix",
    "현대": "hyundai",
    "기아": "kia",
    "lg": "LG",
    "카카오": "kakao",
    "네이버": "naver",
    # 산업·기술
    "반도체": "semiconductor",
    "ai서버": "AI server",
    "ai 서버": "AI server",
    "데이터센터": "data center",
    "hbm": "HBM memory",
    "전기차": "electric vehicle",
    "배터리": "battery",
    "태양광": "solar energy",
    "방산": "defense industry",
    "원전": "nuclear power",
    "바이오": "biotech",
    # 금융·경제
    "금리": "interest rate",
    "환율": "exchange rate",
    "달러": "US dollar",
    "원화": "Korean won",
    "코스피": "KOSPI",
    "코스닥": "KOSDAQ",
    "주식": "stock market Korea",
    "원유": "crude oil",
    "유가": "oil price",
    "인플레이션": "inflation",
    "관세": "tariff",
    # 지역·정치
    "중동": "middle east",
    "중동전황": "middle east conflict",
    "중동 전황": "middle east conflict",
    "러시아": "russia",
    "우크라이나": "ukraine",
    "이란": "iran",
    "트럼프": "trump",
    "바이든": "biden",
    "연준": "federal reserve",
    "한국": "south korea",
    "중국": "china",
    "일본": "japan",
}

_KO_RE = re.compile(r"[가-힣]")


def _translate_keyword(keyword: str) -> tuple[str, bool]:
    """한국어 키워드를 영문으로 변환. (영문 검색어, 한국어 여부) 반환.

    - 테이블에 전체 구절이 있으면 바로 변환
    - 없으면 단어 단위로 분리해 각 단어를 번역 후 재결합 (예: '트럼프 관세' → 'trump tariff')
    - 변환된 단어가 하나도 없으면 원문 그대로 반환 (고유명사 등)
    """
    stripped = keyword.strip().lower()
    # 1) 전체 구절 매칭
    if stripped in KO_TO_EN:
        return KO_TO_EN[stripped], True
    # 2) 단어별 번역
    words = stripped.split()
    translated = [KO_TO_EN.get(w, w) for w in words]
    any_translated = any(t != o for t, o in zip(translated, words))
    if any_translated:
        return " ".join(translated), True
    # 3) 한글 포함 여부만 확인 — 원문 그대로 시도
    is_korean = bool(_KO_RE.search(keyword))
    return keyword, is_korean


def _is_relevant(article: dict[str, Any], terms: list[str]) -> bool:
    """제목 또는 설명에 검색어(또는 주요 단어)가 하나라도 포함되는지 확인."""
    text = (article.get("title", "") + " " + article.get("description", "")).lower()
    return any(t.lower() in text for t in terms if len(t) > 2)


def _load_mock() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data.get("articles", [])


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": raw.get("title", "") or "",
        "url": raw.get("url", "") or "",
        "source": (raw.get("source") or {}).get("name", "") or "",
        "published_at": raw.get("publishedAt", "") or "",
        "description": raw.get("description", "") or "",
    }


async def fetch_news(keyword: str, page_size: int = 12) -> list[dict[str, Any]]:
    """Fetch raw articles for `keyword`.

    - 한국어 키워드는 영문으로 변환 후 NewsAPI 호출
    - NEWSAPI_KEY 없거나 USE_MOCK_NEWS=true 이면 fixture 반환
    """
    if settings.mock_news_active:
        articles = _load_mock()
        return [_normalize(a) for a in articles]

    search_term, _ = _translate_keyword(keyword)

    params = {
        "q": search_term,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en",
        "searchIn": "title,description",  # 제목/설명에만 매칭 → 관련 없는 기사 차단
        "apiKey": settings.newsapi_key,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(NEWSAPI_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    articles = payload.get("articles", [])

    # 결과가 없으면 language/searchIn 제한 없이 재시도
    if not articles:
        params.pop("language", None)
        params.pop("searchIn", None)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(NEWSAPI_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        articles = payload.get("articles", [])

    # 관련성 필터: 검색어 주요 단어가 제목/설명에 포함된 기사만 유지
    # (서로 무관한 기사가 혼입되는 경우 제거)
    terms = search_term.split()
    relevant = [a for a in articles if _is_relevant(a, terms)]
    # 너무 많이 걸러지면 원본 유지 (최소 2건 보장)
    if len(relevant) >= 2:
        articles = relevant

    return [_normalize(a) for a in articles]
