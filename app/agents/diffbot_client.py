"""
Optional Diffbot Article API client.

Ported from /Users/munjuan/Documents/New project 3/diffbot/diffbot_extract.py for
backend use. If no token is configured, callers get the original GDELT article
list unchanged.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from app.agents import naver_categories
from app.config import settings

logger = logging.getLogger("econmind.diffbot")

DIFFBOT_ARTICLE_URL = "https://api.diffbot.com/v3/article"
DEFAULT_REQUEST_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_MS = 30000
MAX_URLS = 10
_extraction_cache: OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]] = OrderedDict()
_CACHE_FIELDS = ("cleaned_content", "cleaned_content_length", "diffbot_text_length", "thumbnail_url", "content_source_url")

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_FILES = (
    _WORKSPACE_ROOT / "token.env",
    _BACKEND_ROOT / "token.env",
    _BACKEND_ROOT / "diffbot" / "token.txt",
)


class DiffbotError(RuntimeError):
    """Provider failure with a log-safe classification.

    The message never carries the request URL, token or response body:
    ``requests`` exception text can embed the full query string.
    """

    def __init__(self, category: str, status: int | None = None, attempts: int = 0, elapsed_ms: int = 0, *, provider_error_code: int | None = None, key_slot: int | None = None):
        super().__init__(f"Diffbot {category}" + (f" (HTTP {status})" if status else ""))
        self.category = category
        self.status = status
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        self.provider_error_code = provider_error_code
        self.key_slot = key_slot


_last_call = threading.local()
_semaphore_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_size = 0
_token_rotation_lock = threading.Lock()
_token_rotation_pool: tuple[str, ...] = ()
_token_rotation_index = 0


def _provider_semaphore() -> threading.BoundedSemaphore:
    """Process-wide cap on in-flight Diffbot requests (calls run in worker threads)."""
    global _semaphore, _semaphore_size
    size = max(1, int(settings.diffbot_max_concurrency))
    with _semaphore_lock:
        if _semaphore is None or size != _semaphore_size:
            _semaphore, _semaphore_size = threading.BoundedSemaphore(size), size
        return _semaphore


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _interrupted(cancel_event: threading.Event | None, deadline: float | None) -> str | None:
    if cancel_event is not None and cancel_event.is_set():
        return "cancelled"
    if deadline is not None and time.monotonic() >= deadline:
        return "deadline"
    return None


@contextmanager
def _provider_slot(cancel_event: threading.Event | None, deadline: float | None):
    """Do not leave a queued thread able to send after its async caller expires."""
    semaphore = _provider_semaphore()
    while True:
        if reason := _interrupted(cancel_event, deadline):
            raise DiffbotError(reason)
        wait = 0.05 if deadline is None else min(0.05, max(0, deadline - time.monotonic()))
        if semaphore.acquire(timeout=wait):
            break
    try:
        if reason := _interrupted(cancel_event, deadline):
            raise DiffbotError(reason)
        yield
    finally:
        semaphore.release()


def classify_status(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status >= 500:
        return "server_error"
    return "http_error"


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float | None:
    """Wait before retry number ``attempt`` (0-based); None means do not retry.

    Exponential from DIFFBOT_BACKOFF_BASE_SECONDS, capped at
    DIFFBOT_BACKOFF_MAX_SECONDS. A server Retry-After above the cap is not waited
    out: retrying earlier would burn an attempt that is certain to fail.
    """
    cap = max(0.0, settings.diffbot_backoff_max_seconds)
    if retry_after is not None:
        return None if retry_after > cap else max(1.0, retry_after)
    return min(cap, max(0.0, settings.diffbot_backoff_base_seconds) * (2**attempt))


def _payload_error_category(payload: dict[str, Any]) -> tuple[str, int | None]:
    code = payload.get("errorCode")
    status = code if isinstance(code, int) else None
    return (classify_status(status) if status else "api_error"), status


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
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


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

    for token in (settings.diffbot_api_key, settings.diffbot_token):
        if token.strip():
            return token.strip()

    files = (token_file,) if token_file is not None else _TOKEN_FILES
    for path in files:
        try:
            token = _parse_token_file(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if token:
            return token
    return ""


def load_diffbot_tokens() -> tuple[str, ...]:
    """Keep legacy primary-key precedence and optionally add one distinct key."""
    primary = load_diffbot_token()
    secondary = os.getenv("DIFFBOT_TOKEN_SECONDARY", settings.diffbot_token_secondary).strip()
    return tuple(dict.fromkeys(token for token in (primary, secondary) if token))


def _next_diffbot_token(tokens: tuple[str, ...]) -> str:
    """Balance real extractions across keys, including separate v2 singletons.

    Rotation is process-local; it does not raise provider plan/account quotas.
    Explicit ``token=`` callers continue to use only that token.
    """
    global _token_rotation_pool, _token_rotation_index
    if len(tokens) == 1:
        return tokens[0]
    identity = tuple(hashlib.sha256(token.encode()).hexdigest() for token in tokens)
    with _token_rotation_lock:
        if identity != _token_rotation_pool:
            _token_rotation_pool, _token_rotation_index = identity, 0
        token = tokens[_token_rotation_index % len(tokens)]
        _token_rotation_index = (_token_rotation_index + 1) % len(tokens)
        return token


def _extraction_cache_key(url: str, token: str) -> tuple[str, str]:
    return url, hashlib.sha256((token + "|image-selection-v2").encode()).hexdigest()


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
    host = (urlparse(url).hostname or "").lower()
    if host == "news.naver.com" or host.endswith(".news.naver.com"):
        # NAVER photos are often lazy-loaded while its visible QR codes are not.
        params["fields"] = "meta"
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
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
    token_pool: tuple[str, ...] = (),
    news_id: str | None = None,
) -> dict[str, Any]:
    """Stop queued/new attempts on cancellation or deadline.

    A request already sent cannot be recalled; its transport timeout still applies.
    ``deadline`` uses the same monotonic clock as the async extraction caller.
    A 429 may switch to another supplied key even when same-key retries are off.
    HTTP attempts across all keys share DIFFBOT_MAX_TOTAL_ATTEMPTS. A key that
    returns 429 is not revisited within a multi-key call.
    """
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
    timeout_seconds = max(5, timeout_ms / 1000 + 5)
    tokens = tuple(dict.fromkeys((token, *(key for key in token_pool if key))))
    key_order = tuple(dict.fromkeys((*token_pool, token)))
    active_token = token
    max_retries = max(0, max_retries)
    # Same-key retries plus at most one visit to each alternate key, under one cap.
    max_attempts = max(1, min(max_retries + len(tokens), settings.diffbot_max_total_attempts))
    retries = 0
    rate_limited: set[str] = set()
    started = time.monotonic()
    _last_call.attempts, _last_call.elapsed_ms = 0, 0

    def fail(category: str, status: int | None, attempts: int, provider_error_code: int | None = None) -> DiffbotError:
        slot = key_order.index(active_token) + 1
        return DiffbotError(category, status, attempts, int((time.monotonic() - started) * 1000), provider_error_code=provider_error_code, key_slot=slot)

    attempts = 0
    for attempt in range(max_attempts):
        retry_after: float | None = None
        provider_error_code: int | None = None
        try:
            with _provider_slot(cancel_event, deadline):
                request_timeout = timeout_seconds if deadline is None else min(timeout_seconds, max(0.001, deadline - time.monotonic()))
                attempts += 1
                response = requests.get(
                    DIFFBOT_ARTICLE_URL,
                    params={**request_params, "token": active_token},
                    timeout=request_timeout,
                )
        except DiffbotError as exc:
            raise fail(exc.category, None, attempts) from None
        except requests.Timeout:
            category, status = "timeout", None
        except requests.RequestException:
            category, status = "network", None
        else:
            status = response.status_code
            if status < 400:
                try:
                    payload = response.json()
                except ValueError:
                    raise fail("invalid_response", status, attempts) from None
                if not isinstance(payload, dict):
                    raise fail("invalid_response", status, attempts)
                if payload.get("error") or payload.get("errorCode"):
                    category, provider_error_code = _payload_error_category(payload)
                else:
                    _last_call.attempts = attempts
                    _last_call.elapsed_ms = int((time.monotonic() - started) * 1000)
                    return payload
            else:
                category = classify_status(status)
            if category in ("rate_limit", "server_error"):
                retry_after = retry_after_seconds(response.headers.get("Retry-After"))

        if category == "rate_limit" and len(tokens) > 1:
            rate_limited.add(active_token)
            alternate = next((key for key in tokens if key not in rate_limited), None)
            if alternate is None or attempts >= max_attempts:
                raise fail(category, status, attempts, provider_error_code)
            if reason := _interrupted(cancel_event, deadline):
                raise fail(reason, status, attempts, provider_error_code)
            logger.warning(
                "[diffbot] key switch: news_id=%s category=rate_limit http_status=%s provider_error_code=%s from_key_slot=%d to_key_slot=%d attempts=%d",
                news_id or "-", status, provider_error_code if provider_error_code is not None else "-",
                key_order.index(active_token) + 1, key_order.index(alternate) + 1, attempts,
            )
            active_token = alternate
            continue

        if category not in ("rate_limit", "server_error", "timeout", "network") or attempts >= max_attempts or retries >= max_retries:
            raise fail(category, status, attempts, provider_error_code)
        wait = backoff_seconds(retries, retry_after)
        if wait is None:
            raise fail(category, status, attempts, provider_error_code)
        if reason := _interrupted(cancel_event, deadline):
            raise fail(reason, status, attempts, provider_error_code)
        retries += 1
        if deadline is not None and time.monotonic() + wait >= deadline:
            raise fail("deadline", status, attempts, provider_error_code)
        if cancel_event is None:
            _sleep(wait)
        else:
            cancel_event.wait(wait)
        if reason := _interrupted(cancel_event, deadline):
            raise fail(reason, status, attempts, provider_error_code)

    raise fail("exhausted", None, max_attempts)


def extract_primary_object(payload: dict[str, Any]) -> dict[str, Any]:
    objects = payload.get("objects") or []
    if not objects:
        return {}
    return objects[0] if isinstance(objects[0], dict) else {}


def is_article_image_url(url: Any, label: str = "") -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
    except ValueError:
        return False
    text = unquote(url) + " " + label
    return not re.search(r"qr[-_\s]?code|(?:^|[/_.?=&\s-])qr(?:$|[/_.?=&\s-])|큐알|QR\s*코드", text, re.IGNORECASE)


def extract_image_url(obj: dict[str, Any], article_url: str, meta: Any = None) -> str:
    meta = meta if isinstance(meta, dict) else obj.get("meta", {})
    meta_candidates = []
    if isinstance(meta, dict):
        for group, field in (("og", "og:image"), ("twitter", "twitter:image")):
            nested = meta.get(group)
            value = nested.get(field) if isinstance(nested, dict) else None
            meta_candidates.extend([value, meta.get(field)])

    def resolve(raw_url: Any, label: str = "") -> str:
        if not isinstance(raw_url, str) or not raw_url.strip():
            return ""
        try:
            url = urljoin(article_url, raw_url.strip())
            return url if is_article_image_url(url, label) else ""
        except ValueError:
            return ""

    host = (urlparse(article_url).hostname or "").lower()
    if host == "news.naver.com" or host.endswith(".news.naver.com"):
        for candidate in meta_candidates:
            if url := resolve(candidate):
                return url
    images = obj.get("images")
    candidates = [item for item in images if isinstance(item, dict)] if isinstance(images, list) else []
    candidates.sort(key=lambda item: item.get("primary") is not True)
    for item in candidates:
        if url := resolve(item.get("url"), str(item.get("title") or "")):
            return url
    for candidate in meta_candidates:
        if url := resolve(candidate):
            return url
    return ""


def _extract_one_sync(
    article: dict[str, Any],
    token: str,
    timeout_ms: int,
    max_retries: int,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
    token_pool: tuple[str, ...] = (),
) -> dict[str, Any]:
    url = article.get("naver_url") or article.get("url", "")
    if not url:
        return article

    payload = call_diffbot_article(
        url=url,
        token=token,
        timeout_ms=timeout_ms,
        discussion=False,
        max_retries=max(0, max_retries),
        cancel_event=cancel_event,
        deadline=deadline,
        token_pool=token_pool,
        news_id=article.get("news_id"),
    )
    obj = extract_primary_object(payload)
    text = obj.get("text")
    diffbot_text = normalize_diffbot_text(text) if isinstance(text, str) else ""
    if not diffbot_text:
        logger.warning(
            "[diffbot] extract empty: news_id=%s category=empty http_status=200 objects=%d attempts=%s elapsed_ms=%s",
            article.get("news_id") or "-",
            len(payload.get("objects") or []),
            getattr(_last_call, "attempts", "-"),
            getattr(_last_call, "elapsed_ms", "-"),
        )
    image_url = extract_image_url(obj, url, obj.get("meta") or payload.get("meta"))
    fields = {}
    if diffbot_text:
        fields.update(diffbot_text_length=len(diffbot_text), cleaned_content=diffbot_text,
                      cleaned_content_length=len(diffbot_text))
    # Explicitly clear a previous QR thumbnail when the extractor has no usable photo.
    if image_url or not is_article_image_url(article.get("thumbnail_url")):
        fields["thumbnail_url"] = image_url
    if fields:
        fields["content_source_url"] = url
    return {**article, **fields} if fields else article


def _log_extract_failure(article: dict[str, Any], exc: Exception, started: float, key_slot: int) -> None:
    """One structured line per failed extraction; only classification fields, never URLs or bodies."""
    if isinstance(exc, DiffbotError):
        category, status, attempts, elapsed_ms = exc.category, exc.status, exc.attempts, exc.elapsed_ms
        provider_error_code = exc.provider_error_code
        key_slot = exc.key_slot or key_slot
    else:
        category, status, attempts = "unexpected", None, "-"
        provider_error_code = None
        elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.warning(
        "[diffbot] extract skip: news_id=%s category=%s http_status=%s provider_error_code=%s attempts=%s elapsed_ms=%s error_type=%s key_slot=%s",
        article.get("news_id") or "-",
        category,
        status if status is not None else "-",
        provider_error_code if provider_error_code is not None else "-",
        attempts,
        elapsed_ms,
        type(exc).__name__,
        key_slot,
    )


async def extract_articles_with_diffbot(
    articles: list[dict[str, Any]],
    token: str | None = None,
    max_articles: int = MAX_URLS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    concurrency: int = 2,
    use_cache: bool = True,
    deadline: float | None = None,
    raise_errors: bool = False,
) -> list[dict[str, Any]]:
    tokens = ((token,) if token else ()) if token is not None else load_diffbot_tokens()
    if not articles:
        return articles

    limit = max(0, min(max_articles, len(articles)))
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(index: int, article: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            if article.get("thumbnail_url") and not is_article_image_url(article["thumbnail_url"]):
                article = {**article, "thumbnail_url": ""}
            naver_url = article.get("naver_url")
            if naver_url:
                labels = await naver_categories.fetch_categories(naver_url)
                if not naver_categories.allowed(labels):
                    return None
                article = {**article, "naver_categories": labels}
                if article.get("content_source_url") != naver_url:
                    article = {k: v for k, v in article.items() if k not in _CACHE_FIELDS}
            if not tokens or index >= limit:
                return article
            url = naver_url or article.get("url", "")
            if use_cache:
                # A cached body from either configured key avoids another billed call.
                for configured_token in tokens:
                    cache_key = _extraction_cache_key(url, configured_token)
                    cached = _extraction_cache.get(cache_key)
                    if cached and cached[0] > time.monotonic() and (cached[1].get("cleaned_content") or cached[1].get("thumbnail_url")):
                        _extraction_cache.move_to_end(cache_key)
                        return {**article, **deepcopy(cached[1])}
            selected_token = _next_diffbot_token(tokens)
            cache_key = _extraction_cache_key(url, selected_token)
            cached = _extraction_cache.get(cache_key)
            if use_cache and cached and cached[0] > time.monotonic():
                _extraction_cache.move_to_end(cache_key)
                return {**article, **deepcopy(cached[1])}
            fields = {}
            started = time.monotonic()
            cancel_event = threading.Event()
            try:
                extracted = await asyncio.to_thread(
                    _extract_one_sync,
                    article,
                    selected_token,
                    timeout_ms,
                    max_retries,
                    cancel_event=cancel_event,
                    deadline=deadline,
                    token_pool=tokens,
                )
                if extracted is not article:
                    fields = {key: extracted[key] for key in _CACHE_FIELDS if key in extracted}
            except asyncio.CancelledError:
                cancel_event.set()
                raise
            except Exception as exc:
                if token is not None and isinstance(exc, DiffbotError):
                    configured = load_diffbot_tokens()
                    if selected_token in configured:
                        exc.key_slot = configured.index(selected_token) + 1
                _log_extract_failure(article, exc, started, tokens.index(selected_token) + 1)
                if raise_errors:
                    raise
            finally:
                cancel_event.set()
            usable = fields.get("thumbnail_url") or fields.get("cleaned_content")
            _extraction_cache[cache_key] = (time.monotonic() + (3600 if usable else 30), deepcopy(fields))
            _extraction_cache.move_to_end(cache_key)
            while len(_extraction_cache) > 256:
                _extraction_cache.popitem(last=False)
            return {**article, **fields}

    extracted = await asyncio.gather(*(_bounded(i, article) for i, article in enumerate(articles)))
    return [article for article in extracted if article is not None]
