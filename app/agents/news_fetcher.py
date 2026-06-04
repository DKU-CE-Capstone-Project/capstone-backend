import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.agents.gdelt_client import fetch_gdelt_articles
from app.agents.content_extractor import extract_articles_batch

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
    text = ((article.get("title") or "") + " " + (article.get("description") or "")).lower()
    return any(t.lower() in text for t in terms if len(t) > 2)


def _load_mock() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data.get("articles", [])


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """NewsAPI 응답을 공통 형식으로 정규화합니다."""
    return {
        "title": raw.get("title", "") or "",
        "url": raw.get("url", "") or "",
        "source": (raw.get("source") or {}).get("name", "") or "",
        "published_at": raw.get("publishedAt", "") or "",
        "description": raw.get("description", "") or "",
        "thumbnail_url": raw.get("urlToImage") or "",
    }


def _normalize_gdelt(art: dict[str, Any], cleaned_body: str) -> dict[str, Any]:
    """GDELT + 본문 추출 결과를 공통 형식으로 정규화합니다.

    cleaned_body가 있으면 description으로 사용 (Gemini 요약 품질 향상).
    """
    description = cleaned_body.strip() if cleaned_body else art.get("title", "")
    # seendate 형식: "20260525T010000Z" → "2026-05-25T01:00:00Z"
    raw_date = art.get("published_at", "")
    if len(raw_date) == 16 and "T" in raw_date:
        published_at = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            f"T{raw_date[9:11]}:{raw_date[11:13]}:{raw_date[13:15]}Z"
        )
    else:
        published_at = raw_date

    return {
        "title": art.get("title", "") or "",
        "url": art.get("url", "") or "",
        "source": art.get("source_domain", "") or "",
        "published_at": published_at,
        "description": description,
        "thumbnail_url": art.get("image_url", "") or "",
    }


async def _fetch_newsapi(keyword: str, page_size: int) -> list[dict[str, Any]]:
    """NewsAPI 호출 (GDELT 실패 시 fallback)."""
    search_term, _ = _translate_keyword(keyword)
    params: dict[str, Any] = {
        "q": search_term,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en",
        "searchIn": "title,description",
        "apiKey": settings.newsapi_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(NEWSAPI_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        articles = payload.get("articles", [])

        if not articles:
            params.pop("language", None)
            params.pop("searchIn", None)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(NEWSAPI_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
            articles = payload.get("articles", [])

        terms = search_term.split()
        relevant = [a for a in articles if _is_relevant(a, terms)]
        if len(relevant) >= 2:
            articles = relevant
        return [_normalize(a) for a in articles]
    except Exception as exc:
        print(f"[newsapi] fallback also failed: {exc}")
        return []


async def fetch_news(keyword: str, page_size: int = 12) -> list[dict[str, Any]]:
    """기사 목록을 가져옵니다.

    전략:
    1. GDELT DOC API → URL 수집 + 중복 제거 → 본문 추출 (DOM/trafilatura)
    2. GDELT 실패 또는 결과 없음 → NewsAPI fallback
    3. USE_MOCK_NEWS=true → fixture 반환

    반환 형식: [{title, url, source, published_at, description, thumbnail_url}]
    description에 기사 full body가 담겨 Gemini 요약 품질이 크게 향상됩니다.
    """
    if settings.mock_news_active:
        return [_normalize(a) for a in _load_mock()]

    # ── 1) GDELT 시도 (글로벌 → 한국어 소스 순으로 시도) ────────────────────
    search_term, _ = _translate_keyword(keyword)

    # GDELT가 막힌 환경(429/타임아웃)에서는 use_gdelt=false로 두면 NewsAPI 직행
    gdelt_articles: list[dict[str, Any]] = []
    if settings.use_gdelt:
        # 1-a) 글로벌 검색 (언어 필터 없음 → 더 많은 결과)
        gdelt_articles = await fetch_gdelt_articles(
            keyword=search_term,
            source_lang="",        # 전 언어 대상
            maxrecords=page_size + 5,
            timespan="1d",
        )
        # 1-b) 결과 없으면 한국어 소스 한정으로 재시도
        if not gdelt_articles:
            gdelt_articles = await fetch_gdelt_articles(
                keyword=search_term,
                source_lang="korean",
                maxrecords=page_size + 5,
                timespan="3d",     # 기간을 3일로 넓혀 결과 확보
            )

    if gdelt_articles:
        # 본문 추출 (동시 최대 5개) — 느린 사이트가 있어도 전체를 블록하지 않음
        enriched = await extract_articles_batch(
            gdelt_articles[:page_size],
            concurrency=5,
        )
        # 본문 있으면 full body, 없으면 제목을 description으로 사용
        normalized = [_normalize_gdelt(art, art.get("cleaned_content", "")) for art in enriched]
        result = [a for a in normalized if a["title"]]
        if result:
            has_body = sum(1 for a in result if len(a["description"]) > 100)
            print(f"[news_fetcher] GDELT: {len(result)} articles, {has_body} with full body for '{keyword}'")
            return result

    # ── 2) NewsAPI fallback ──────────────────────────────────────────────────
    print(f"[news_fetcher] GDELT unavailable — NewsAPI fallback for '{keyword}'")
    return await _fetch_newsapi(keyword, page_size)
