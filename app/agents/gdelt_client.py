"""
GDELT DOC 2.0 API 클라이언트 + 중복 제거.

capstone-news-logic/test.py 및 tavily_api/tavily_extract.py의
GDELT 관련 로직을 백엔드 패키지용으로 이식.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_HEADERS = {
    "User-Agent": "capstone-gdelt-doc/1.0",
    "Accept": "application/json",
}
GDELT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".gdelt_cache"
GDELT_LAST_REQ_FILE = GDELT_CACHE_DIR / "last_request.txt"
GDELT_CACHE_TTL = 86400   # 24h
GDELT_MIN_INTERVAL = 12   # GDELT 속도제한 회피 (초)
GDELT_MAX_RETRIES = 5

_TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "source"}


# ── URL / 제목 정규화 (중복 제거용) ──────────────────────────────────────────

def _norm_url(url: str) -> str:
    """www./m. 차이 및 추적 파라미터를 제거한 URL 키를 반환합니다."""
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)
    ]
    path = parsed.path.rstrip("/") or parsed.path
    query = urlencode(sorted(pairs))
    return f"{host}{path}?{query}" if query else f"{host}{path}"


def _norm_title(title: str) -> str:
    """중복 판단용 제목 키 — 특수문자/공백 제거."""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", (title or "").lower())


def _dedupe(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """GDELT 결과에서 동일 기사로 보이는 항목을 제거합니다."""
    unique: list[dict] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for art in articles:
        uk = _norm_url(art.get("url", ""))
        tk = _norm_title(art.get("title", ""))

        if uk and uk in seen_urls:
            continue
        if len(tk) >= 12 and tk in seen_titles:
            continue

        if uk:
            seen_urls.add(uk)
        if len(tk) >= 12:
            seen_titles.add(tk)
        unique.append(art)

    return unique


# ── 로컬 캐시 ─────────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()
    return GDELT_CACHE_DIR / f"{digest}.json"


def _read_cache(url: str) -> dict | None:
    p = _cache_path(url)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > GDELT_CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(url: str, payload: dict) -> None:
    GDELT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ── 속도제한 보호 ─────────────────────────────────────────────────────────────

def _wait_rate_limit() -> None:
    if not GDELT_LAST_REQ_FILE.exists():
        return
    try:
        last = float(GDELT_LAST_REQ_FILE.read_text(encoding="utf-8"))
        wait = GDELT_MIN_INTERVAL - (time.time() - last)
        if wait > 0:
            print(f"[gdelt] rate limit wait {wait:.1f}s")
            time.sleep(wait)
    except ValueError:
        pass


def _mark_request() -> None:
    GDELT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GDELT_LAST_REQ_FILE.write_text(str(time.time()), encoding="utf-8")


# ── GDELT 동기 호출 (asyncio.to_thread 전용) ─────────────────────────────────

def _fetch_gdelt_sync(
    query: str,
    source_lang: str = "korean",
    sort: str = "hybridrel",
    maxrecords: int = 20,
    timespan: str = "1d",
) -> dict[str, Any]:
    import requests as req  # sync 전용 — asyncio.to_thread 안에서만 실행

    if source_lang and "sourcelang:" not in query.lower():
        full_query = f"{query} sourcelang:{source_lang}"
    else:
        full_query = query
    params: dict[str, Any] = {
        "query": full_query,
        "mode": "artlist",
        "format": "json",
        "sort": sort,
        "maxrecords": maxrecords,
        "timespan": timespan,
    }

    # 캐시 확인
    prepared_url = req.Request("GET", GDELT_URL, params=params).prepare().url or GDELT_URL
    cached = _read_cache(prepared_url)
    if cached is not None:
        print("[gdelt] cache hit")
        return cached

    last_response = None
    for attempt in range(GDELT_MAX_RETRIES):
        _wait_rate_limit()
        resp = req.get(GDELT_URL, params=params, headers=GDELT_HEADERS, timeout=30)
        _mark_request()
        last_response = resp

        if resp.status_code != 429:
            resp.raise_for_status()
            payload = resp.json()
            _write_cache(prepared_url, payload)
            return payload

        wait = int(resp.headers.get("Retry-After", GDELT_MIN_INTERVAL * (attempt + 1)))
        print(f"[gdelt] 429 rate limit — retry in {wait}s (attempt {attempt+1}/{GDELT_MAX_RETRIES})")
        if attempt < GDELT_MAX_RETRIES - 1:
            time.sleep(wait)

    detail = last_response.text.strip()[:200] if last_response else "no response"
    raise RuntimeError(f"GDELT rate limit exceeded after {GDELT_MAX_RETRIES} retries: {detail}")


# ── 공개 비동기 API ───────────────────────────────────────────────────────────

async def fetch_gdelt_articles(
    keyword: str,
    source_lang: str = "korean",
    maxrecords: int = 20,
    timespan: str = "1d",
) -> list[dict[str, Any]]:
    """GDELT DOC API에서 기사 목록을 비동기로 가져오고 중복 제거합니다.

    반환 형식:
        [{"title", "url", "source_domain", "published_at", "language", "image_url"}]
    """
    try:
        data = await asyncio.to_thread(
            _fetch_gdelt_sync,
            query=keyword,
            source_lang=source_lang,
            maxrecords=maxrecords,
            timespan=timespan,
        )
    except Exception as exc:
        print(f"[gdelt] fetch failed: {type(exc).__name__}: {exc}")
        return []

    raw = data.get("articles") or []
    articles = [
        {
            "title": art.get("title", "(제목 없음)"),
            "url": art.get("url", ""),
            "source_domain": art.get("domain", ""),
            "published_at": art.get("seendate", ""),
            "language": art.get("language", "Korean"),
            "image_url": art.get("socialimage", ""),
        }
        for art in raw
        if art.get("url")
    ]
    deduped = _dedupe(articles)
    print(f"[gdelt] {len(raw)} → {len(deduped)} articles after dedup")
    return deduped
