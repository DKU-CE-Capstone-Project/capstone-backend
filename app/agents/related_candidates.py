"""Grounded supplemental queries and bounded, process-local search page single-flight."""
from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any
from weakref import WeakKeyDictionary

from app.agents.article_embeddings import clean_text
from app.agents.article_metadata import STOPWORDS, _grounded
from app.agents.naver_client import NaverNewsError, NaverNewsPage
from app.config import settings

_cache: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
_inflight: WeakKeyDictionary = WeakKeyDictionary()
FAILURE_TTL_SECONDS = 5
CACHE_SIZE = 128


def query_key(value: str) -> str:
    return " ".join(value.casefold().split())


def grounded_queries(center: dict[str, Any]) -> list[str]:
    """Use supplied title/description/grounded metadata only, no invented angles."""
    title = clean_text(center.get("title"))[:500]
    description = clean_text(center.get("description"))[:6000]
    text = f"{title} {description}"
    # Keep literal model numbers, proper nouns and their original spelling/order.
    title_words = re.findall(r"[가-힣a-zA-Z0-9]+(?:[-+][a-zA-Z0-9]+)*", title)
    anchors = [w for w in title_words if len(w) >= 2 and w.casefold() not in STOPWORDS
               and not re.search(r"공개|발표|선보|출시|했다|한다|밝혔|신제품", w)]
    metadata = [clean_text(w) for w in center.get("keywords") or []
                if isinstance(w, str) and _grounded(w, text) and w.casefold() not in STOPWORDS]
    # Metadata can only add text actually present, even when a canonical alias is grounded.
    metadata = [w for w in metadata if w.casefold() in text.casefold()]
    proposals = [" ".join(anchors[:4]), " ".join(dict.fromkeys([*anchors[:2], *metadata[:3]])),
                 " ".join(title_words[:10])]
    if not title_words:
        proposals = [" ".join(metadata[:4])]
    seen = {query_key(clean_text(center.get("_search_keyword")))}
    queries = []
    for proposal in proposals:
        # Whole supplied terms only, within the search API's 100-character contract.
        words = []
        for word in proposal.split():
            if len(" ".join([*words, word])) > 100:
                break
            words.append(word)
        query = " ".join(words)
        key = query_key(query)
        if len(query) >= 2 and key not in seen:
            seen.add(key)
            queries.append(query)
    return queries


async def cached_search(
    query: str, *, start: int, size: int, fetch: Callable[..., Awaitable[NaverNewsPage]],
    stats: dict[str, int] | None = None,
) -> NaverNewsPage:
    """Process-local TTL cache and single-flight for one raw search page.

    The key includes the explicit start and size, so continuing a search with a
    different page size neither repeats nor skips raw positions.
    """
    key = (settings.news_provider, settings.mock_news_active, query_key(query), start, size)
    cached = _cache.get(key)
    if cached and cached[0] > time.monotonic():
        _cache.move_to_end(key)
        if stats is not None:
            stats["search_cache_hits"] = stats.get("search_cache_hits", 0) + 1
        if isinstance(cached[1], Exception):
            raise cached[1]  # A cached failure remains an error, not an empty success.
        return deepcopy(cached[1])
    tasks = _inflight.setdefault(asyncio.get_running_loop(), {})
    task = tasks.get(key)
    if stats is not None:
        name = "searches" if task is None else "search_cache_hits"
        stats[name] = stats.get(name, 0) + 1
    if task is None:
        async def resolve():
            try:
                async with asyncio.timeout(settings.news_map_supplement_timeout_seconds):
                    result = await fetch(query, start=start, size=size)
            except TimeoutError as exc:
                error = NaverNewsError("연관 기사 추가 검색 응답 시간이 초과되었습니다.", 504)
                _remember(key, error, FAILURE_TTL_SECONDS)
                raise error from exc
            except NaverNewsError as exc:
                _remember(key, exc, FAILURE_TTL_SECONDS)
                raise
            _remember(key, deepcopy(result), settings.news_map_supplement_cache_ttl_seconds)
            return result

        task = asyncio.create_task(resolve())
        tasks[key] = task

        def finished(done):
            if tasks.get(key) is done:
                tasks.pop(key, None)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
    return deepcopy(await asyncio.shield(task))


def _remember(key: tuple, result: Any, ttl: float) -> None:
    _cache[key] = (time.monotonic() + ttl, result)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
