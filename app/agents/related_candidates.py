"""Bounded, process-local original-search page cache and single-flight."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any
from weakref import WeakKeyDictionary

from app.agents.naver_client import NaverNewsError, NaverNewsPage
from app.config import settings

_cache: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
_inflight: WeakKeyDictionary = WeakKeyDictionary()
FAILURE_TTL_SECONDS = 5
CACHE_SIZE = 128


def query_key(value: str) -> str:
    return " ".join(value.casefold().split())


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
