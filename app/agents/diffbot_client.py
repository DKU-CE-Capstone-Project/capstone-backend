"""
Optional Diffbot Article API client.

Uses the shared capstone-news-logic package for extraction. If no token is configured, callers get the original GDELT article
list unchanged.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from diffbot.diffbot_extract import (
    call_diffbot_article, extract_primary_object, normalize_diffbot_text,
)

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
