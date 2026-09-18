"""
Optional Diffbot Article API client.

Ported from /Users/munjuan/Documents/New project 3/diffbot/diffbot_extract.py for
backend use. If no token is configured, callers get the original GDELT article
list unchanged.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

DIFFBOT_ARTICLE_URL = "https://api.diffbot.com/v3/article"
DEFAULT_REQUEST_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_MS = 30000
MAX_URLS = 10

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_FILES = (
    _WORKSPACE_ROOT / "token.env",
    _BACKEND_ROOT / "token.env",
    _BACKEND_ROOT / "diffbot" / "token.txt",
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def normalize_diffbot_text(text: str) -> str:
    paragraphs = []
    for line in (text or "").splitlines():
        line = normalize_space(line)
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None

    value = value.strip()
    if value.isdigit():
        return float(value)

    duration_match = re.fullmatch(
        r"(?:(\d+)\s+days?,\s*)?(\d{1,2}):(\d{2}):(\d{2})",
        value,
        re.IGNORECASE,
    )
    if duration_match:
        days = int(duration_match.group(1) or 0)
        hours = int(duration_match.group(2))
        minutes = int(duration_match.group(3))
        seconds = int(duration_match.group(4))
        return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _parse_token_file(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            return value.strip().strip("'\"")
        return line
    return ""


def load_diffbot_token(token_file: Path | None = None) -> str:
    for key in ("DIFFBOT_API_KEY", "DIFFBOT_TOKEN"):
        token = os.getenv(key, "").strip()
        if token:
            return token

    files = (token_file,) if token_file is not None else _TOKEN_FILES
    for path in files:
        try:
            token = _parse_token_file(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if token:
            return token
    return ""


def build_article_params(
    url: str,
    token: str,
    timeout_ms: int,
    render_delay_ms: int | None = None,
    scroll: str | None = None,
    discussion: bool = False,
    use_proxy: bool = False,
    natural_language: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "token": token,
        "url": url,
        "timeout": timeout_ms,
        "discussion": str(discussion).lower(),
    }
    if use_proxy:
        params["useProxy"] = "default"
    if natural_language:
        params["naturalLanguage"] = natural_language
    if render_delay_ms is not None:
        params["renderDelay"] = render_delay_ms
    if scroll:
        params["scroll"] = scroll
    return params


def call_diffbot_article(
    url: str,
    token: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    render_delay_ms: int | None = None,
    scroll: str | None = None,
    discussion: bool = False,
    use_proxy: bool = False,
    natural_language: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    import requests

    request_params = build_article_params(
        url=url,
        token=token,
        timeout_ms=timeout_ms,
        render_delay_ms=render_delay_ms,
        scroll=scroll,
        discussion=discussion,
        use_proxy=use_proxy,
        natural_language=natural_language,
    )
    timeout_seconds = max(30, int(timeout_ms / 1000) + 20)

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                DIFFBOT_ARTICLE_URL,
                params=request_params,
                timeout=timeout_seconds,
            )
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(DEFAULT_REQUEST_INTERVAL_SECONDS * (attempt + 1))
            continue

        if response.status_code == 429 and attempt < max_retries:
            wait_seconds = retry_after_seconds(response.headers.get("Retry-After"))
            if wait_seconds is None:
                wait_seconds = DEFAULT_REQUEST_INTERVAL_SECONDS * (attempt + 1)
            time.sleep(max(1.0, wait_seconds))
            continue

        if response.status_code >= 400:
            detail = response.text.strip().replace(token, "<TOKEN>")
            if len(detail) > 300:
                detail = detail[:300] + "..."
            raise RuntimeError(f"Diffbot HTTP {response.status_code}: {detail}")

        payload = response.json()
        if payload.get("error") or payload.get("errorCode"):
            detail = payload.get("error") or payload.get("message") or payload.get("errorCode")
            raise RuntimeError(f"Diffbot API error: {detail}")
        return payload

    raise RuntimeError("Diffbot API request failed after retries.")


def extract_primary_object(payload: dict[str, Any]) -> dict[str, Any]:
    objects = payload.get("objects") or []
    if not objects:
        return {}
    return objects[0]


def _extract_one_sync(
    article: dict[str, Any],
    token: str,
    timeout_ms: int,
    max_retries: int,
) -> dict[str, Any]:
    url = article.get("url", "")
    if not url:
        return article

    payload = call_diffbot_article(
        url=url,
        token=token,
        timeout_ms=timeout_ms,
        discussion=False,
        max_retries=max(0, max_retries),
    )
    obj = extract_primary_object(payload)
    diffbot_text = normalize_diffbot_text(obj.get("text", ""))
    if not diffbot_text:
        return article

    return {
        **article,
        "diffbot_text_length": len(diffbot_text),
        "cleaned_content": diffbot_text,
        "cleaned_content_length": len(diffbot_text),
    }


async def extract_articles_with_diffbot(
    articles: list[dict[str, Any]],
    token: str | None = None,
    max_articles: int = MAX_URLS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    concurrency: int = 2,
) -> list[dict[str, Any]]:
    token = token if token is not None else load_diffbot_token()
    if not token or not articles:
        return articles

    limit = max(0, min(max_articles, len(articles)))
    if limit == 0:
        return articles

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(article: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            try:
                return await asyncio.to_thread(
                    _extract_one_sync,
                    article,
                    token,
                    timeout_ms,
                    max_retries,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[diffbot] extract skip: {type(exc).__name__}: {str(exc).replace(token, '<TOKEN>')[:160]}")
                return article

    extracted = await asyncio.gather(*(_bounded(article) for article in articles[:limit]))
    return [*extracted, *articles[limit:]]
