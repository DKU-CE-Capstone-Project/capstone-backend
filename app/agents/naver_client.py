"""NCP NAVER API HUB News Search with Korean queries and normalized news fields."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings

NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"


class NaverNewsError(Exception):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass
class NaverNewsPage:
    articles: list[dict[str, Any]]
    total: int


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parser = _PlainText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _http_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        url = value.strip()
        parsed = urlparse(url)
        return url if parsed.scheme in ("http", "https") and parsed.hostname else ""
    except ValueError:
        return ""


def normalize_naver_article(item: dict[str, Any]) -> dict[str, Any] | None:
    naver_url = _http_url(item.get("link"))
    if not naver_url:
        return None
    parsed = urlparse(naver_url)
    host = (parsed.hostname or "").lower()
    if (
        (host != "news.naver.com" and not host.endswith(".news.naver.com"))
        or parsed.username
        or parsed.password
    ):
        return None
    # Keep publisher attribution; images and report bodies use the NAVER URL.
    url = _http_url(item.get("originallink")) or naver_url
    title = plain_text(item.get("title"))
    if not url or not title:
        return None
    published_at = ""
    try:
        date = parsedate_to_datetime(item.get("pubDate", ""))
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        published_at = date.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError):
        pass
    description = plain_text(item.get("description"))
    return {
        "title": title,
        "url": url,
        "naver_url": naver_url,
        "source": (urlparse(url).hostname or "").removeprefix("www."),
        "published_at": published_at,
        "description": description,
        "summary": description,
        "thumbnail_url": "",  # NAVER page og:image is attached after category filtering.
        "_news_provider": "naver",
    }


async def fetch_naver_page(
    keyword: str, *, page: int = 1, size: int = 20, sort: str = "relevance"
) -> NaverNewsPage:
    query = keyword.strip()
    start = (page - 1) * size + 1
    if not query or page < 1 or not 1 <= size <= 100 or not 1 <= start <= 1000:
        raise NaverNewsError(
            "검색어·페이지 범위를 확인해 주세요. 네이버 검색 시작 위치는 최대 1000입니다.",
            422,
        )
    if sort not in ("relevance", "latest"):
        raise NaverNewsError("지원하지 않는 뉴스 정렬 방식입니다.", 422)
    if not settings.naver_client_id.strip() or not settings.naver_client_secret.strip():
        raise NaverNewsError(
            "NCP NAVER API HUB의 Client ID와 Client Secret을 백엔드 .env에 설정해 주세요.",
            503,
        )
    headers = {
        "X-NCP-APIGW-API-KEY-ID": settings.naver_client_id.strip(),
        "X-NCP-APIGW-API-KEY": settings.naver_client_secret.strip(),
    }
    params = {
        "query": query,
        "display": size,
        "start": start,
        "sort": "date" if sort == "latest" else "sim",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.naver_timeout_seconds) as client:
            response = await asyncio.wait_for(
                client.get(NAVER_NEWS_URL, params=params, headers=headers),
                timeout=settings.naver_timeout_seconds,
            )
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise NaverNewsError(
            "네이버 뉴스 검색 응답 시간이 초과되었습니다.", 504
        ) from exc
    except httpx.HTTPError as exc:
        raise NaverNewsError("네이버 뉴스 검색에 연결하지 못했습니다.") from exc
    # Never expose response/request bodies or credentials in client-facing errors.
    if response.status_code in (401, 403):
        raise NaverNewsError(
            "NCP NAVER API HUB의 인증 정보와 뉴스 API 사용 권한을 확인해 주세요."
        )
    if response.status_code == 429:
        raise NaverNewsError(
            "네이버 뉴스 검색 호출 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.",
            503,
        )
    if response.status_code >= 400:
        raise NaverNewsError("네이버 뉴스 검색 요청에 실패했습니다.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise NaverNewsError("네이버 뉴스 검색 응답 형식이 올바르지 않습니다.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise NaverNewsError("네이버 뉴스 검색 응답 형식이 올바르지 않습니다.")
    total = payload.get("total")
    if type(total) is not int or total < 0:
        raise NaverNewsError("네이버 뉴스 검색 결과 수가 올바르지 않습니다.")
    articles, seen = [], set()
    for item in payload["items"][:size]:
        article = normalize_naver_article(item) if isinstance(item, dict) else None
        if article and article["url"] not in seen:
            articles.append(article)
            seen.add(article["url"])
    return NaverNewsPage(articles=articles, total=total)
