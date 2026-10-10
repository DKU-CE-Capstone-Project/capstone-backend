"""Bounded news-map candidate collection, staged expansion and selection status.

Order: cached original-search pool (about 20 unique) -> identity de-duplication
-> Gemini relevance + minimum gate -> direct repetition exclusion + MMR -> display
limit. Only when meaningful neighbours are short does expansion add candidates,
from the cached remainder, the center's first grounded query, the original
query's next raw page, then further grounded queries, up to one per-request
total of unique evaluated candidates and
separate search-call and time budgets. Expansion never extracts bodies, writes
reports, generates metadata or creates RAG vectors.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app import database, store
from app.agents import news_fetcher
from app.agents.article_embeddings import EmbeddingUnavailable, map_text, track_usage
from app.agents.decision_client import DecisionClient, DecisionUnavailable
from app.agents.decision_selector import DecisionNewsMapSelector
from app.agents.naver_client import NaverNewsError
from app.agents.related_candidates import cached_search, grounded_queries
from app.agents.related_selector import (
    NewsMapSelector,
    RankedArticle,
    article_id,
    identity_keys,
)
from app.config import settings
from app.database import DatabasePersistenceError
from app.utils import cache_articles, tier_ok

logger = logging.getLogger("econmind.news_map")

COMPLETE, INSUFFICIENT, PARTIAL, EXPANDABLE = "complete", "insufficient", "partial", "expandable"
# NAVER search accepts start positions up to 1000.
_MAX_START = 1000


@dataclass
class NewsMapResult:
    items: list[RankedArticle]
    status: str
    reason: str | None
    target: int
    stats: dict[str, Any] = field(default_factory=dict)
    excluded: int = 0


def display_target(limit: int, tier: str) -> int:
    """FREE/BASIC show at most 3 neighbours; PAID uses the requested limit."""
    return limit if tier_ok(tier, "PAID") else min(limit, 3)


def _pool_order(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Latest search session first, then its raw search rank; never an ID order.
    ordered = sorted(articles, key=lambda a: a.get("news_id") or "")
    ordered.sort(key=lambda a: a.get("published_at") or "", reverse=True)
    ordered.sort(key=lambda a: a.get("_search_rank") or 10 ** 9)
    ordered.sort(key=lambda a: a.get("_searched_at") or "", reverse=True)
    return ordered


async def cached_pool(center: dict[str, Any]) -> list[dict[str, Any]]:
    """Same-search candidates from memory and MongoDB, without an external request."""
    keyword = center.get("_search_keyword") or ""
    if not keyword:
        return []
    found = {article_id(a): a for a in store.news_cache.values() if a.get("_search_keyword") == keyword}
    for article in await database.news_candidates(keyword, settings.news_map_max_candidates * 2):
        found.setdefault(article_id(article), article)
    return _pool_order(list(found.values()))


def _initial_end(center: dict[str, Any], pool: list[dict[str, Any]], count: int,
                 exclude: set[str] = frozenset()) -> int:
    """Pool index after the first ``count`` unique candidates in search order."""
    seen, taken = identity_keys(center) | exclude, 0
    for index, article in enumerate(pool):
        keys = identity_keys(article)
        if keys & seen or not map_text(article):
            continue
        if taken >= count:
            return index
        seen |= keys
        taken += 1
    return len(pool)


def _latest_session(pool: list[dict[str, Any]], keyword: str) -> tuple[int, str] | None:
    """Last raw position requested by the latest session of the original search."""
    session = [a for a in pool if a.get("_search_keyword") == keyword]
    if not session:
        return None
    latest = session[0].get("_searched_at") or ""
    ends = [a.get("_search_end") or a.get("_search_rank") for a in session if (a.get("_searched_at") or "") == latest]
    ends = [value for value in ends if isinstance(value, int)]
    return (max(ends), latest) if ends else None


class _Expansion:
    def __init__(self, center: dict[str, Any], selector: NewsMapSelector | DecisionNewsMapSelector, stats: dict[str, Any]):
        self.center = center
        self.selector = selector
        self.stats = stats

    async def search(self, query: str, *, start: int, session: str | None = None) -> list[dict[str, Any]]:
        """One bounded raw page; category checks only for new candidates still needed."""
        page = await cached_search(query, start=start, size=settings.news_map_supplement_page_size,
                                   fetch=news_fetcher.fetch_search_page, stats=self.stats)
        self.stats["raw"] += len(page.articles)
        fresh = [a for a in page.articles if self.selector.is_new(a)]
        self.stats["raw_duplicates"] += len(page.articles) - len(fresh)
        needed = settings.news_map_max_candidates - self.selector.evaluated
        accepted, index = [], 0
        while index < len(fresh) and len(accepted) < needed:
            batch = fresh[index:index + needed - len(accepted)]
            index += len(batch)
            allowed = await news_fetcher.inspect_search_articles(batch)
            self.stats["filtered"] += len(batch) - len(allowed)
            accepted.extend(allowed)
        if not accepted:
            return []
        now = datetime.now(UTC).isoformat()
        prepared = []
        for article in accepted:
            previous = store.news_cache.get(article_id(article)) or {}
            if session is not None:
                # The original session continues; keep its timestamp so page 1 stays first.
                membership = {"_search_keyword": query, "_search_end": page.end, "_searched_at": session or now}
            elif previous.get("_search_keyword"):
                membership = {k: previous[k] for k in ("_search_keyword", "_search_rank", "_search_end",
                                                       "_searched_at") if k in previous}
            else:
                membership = {"_search_keyword": query, "_search_end": page.end, "_searched_at": now}
            prepared.append({**article, **membership})
        # Map-only cache: grounded rule metadata and map vectors, no Flex or RAG calls.
        return await cache_articles(prepared, news_map_only=True)

    def steps(self, pool: list[dict[str, Any]], consumed: int):
        """(name, external, fetch) in order; external fetches happen only when reached."""
        rest = pool[consumed:]
        if rest:
            async def cached():
                return rest
            yield "cached_pool", False, cached
        if not news_fetcher.expansion_supported():
            return
        keyword = self.center.get("_search_keyword") or ""
        latest = _latest_session(pool, keyword) if keyword else None
        queries = grounded_queries(self.center)
        for index, query in enumerate(queries):
            async def grounded(query=query):
                return await self.search(query, start=1)
            yield "grounded_query", True, grounded
            # The center's most specific grounded query first targets its own event and
            # background; the original query's next raw page then keeps the user's intent.
            if index == 0 and latest and latest[0] < _MAX_START:
                yield "original_next", True, self._original(keyword, latest)
        if not queries and latest and latest[0] < _MAX_START:
            yield "original_next", True, self._original(keyword, latest)

    def _original(self, keyword: str, latest: tuple[int, str]):
        async def original():
            return await self.search(keyword, start=latest[0] + 1, session=latest[1])
        return original


async def exclusion_keys(news_ids: list[str]) -> set[str]:
    """Identity keys (ID and article URLs) of articles the caller already shows."""
    keys: set[str] = set()
    for news_id in news_ids:
        keys.add(f"id:{news_id}")
        article = await store.get_news(news_id)
        if article:
            keys |= identity_keys(article)
    return keys


async def build_news_map(
    center: dict[str, Any], *, target: int, min_relevance: float = 0.0, expand: bool = True,
    exclude_ids: list[str] | None = None,
) -> NewsMapResult:
    exclude = await exclusion_keys(exclude_ids or [])
    if settings.news_map_selector == "decision":
        async with DecisionClient() as client:
            selector = DecisionNewsMapSelector(center, client, min_relevance=min_relevance, exclude=exclude)
            return await _build_news_map(center, selector, target=target, expand=expand, exclude=exclude)
    return await _build_news_map(center, NewsMapSelector(center, min_relevance=min_relevance, exclude=exclude),
                                 target=target, expand=expand, exclude=exclude)


async def _select(selector: NewsMapSelector | DecisionNewsMapSelector, target: int) -> list[RankedArticle]:
    if isinstance(selector, DecisionNewsMapSelector):
        return await selector.select(target)
    return selector.select(target)


async def _build_news_map(
    center: dict[str, Any], selector: NewsMapSelector | DecisionNewsMapSelector, *, target: int, expand: bool, exclude: set[str],
) -> NewsMapResult:
    """Select up to ``target`` neighbours; never lowers the relevance gate to fill slots."""
    started = time.monotonic()
    stats: dict[str, Any] = {"pool": 0, "initial": 0, "search_steps": 0, "searches": 0, "search_cache_hits": 0,
                             "raw": 0, "raw_duplicates": 0, "filtered": 0, "rounds": 0}
    usage = track_usage()
    pool = await cached_pool(center)
    stats["pool"] = len(pool)
    # The first round takes the first unique candidates in search order.
    consumed = _initial_end(center, pool, settings.news_map_initial_candidates, exclude)
    # A failed first round has no valid result and propagates (HTTP 503).
    await selector.add(pool[:consumed], limit=settings.news_map_initial_candidates)
    stats["initial"] = selector.evaluated
    stats["rounds"] = 1
    items = await _select(selector, target)
    status, reason = (COMPLETE, None) if len(items) >= target else (INSUFFICIENT, None)
    expansion = _Expansion(center, selector, stats)
    steps = list(expansion.steps(pool, consumed))
    if status == INSUFFICIENT and not expand:
        if selector.evaluated < settings.news_map_max_candidates and any(
                not external or settings.news_map_supplement_max_searches for _, external, _ in steps):
            status, reason = EXPANDABLE, None
    elif status == INSUFFICIENT:
        reason = "exhausted"
        try:
            async with asyncio.timeout(settings.news_map_supplement_timeout_seconds):
                for name, external, fetch in steps:
                    if selector.evaluated >= settings.news_map_max_candidates:
                        reason = "candidate_limit"
                        break
                    if external:
                        # Budgeted per step, so a warm or cold search cache selects the same.
                        if stats["search_steps"] >= settings.news_map_supplement_max_searches:
                            reason = "search_limit"
                            break
                        stats["search_steps"] += 1
                    articles = await fetch()
                    added = await selector.add(articles, limit=settings.news_map_max_candidates - selector.evaluated)
                    stats["rounds"] += 1
                    stats[f"added_{name}"] = stats.get(f"added_{name}", 0) + added
                    items = await _select(selector, target)
                    if len(items) >= target:
                        status, reason = COMPLETE, None
                        break
                else:
                    if not steps:
                        reason = "no_source"
        except TimeoutError:
            status, reason = PARTIAL, "timeout"
        except NaverNewsError:
            status, reason = PARTIAL, "search_failed"
        except EmbeddingUnavailable:
            status, reason = PARTIAL, "embedding_failed"
        except DecisionUnavailable:
            status, reason = PARTIAL, "decision_failed"
        except DatabasePersistenceError:
            status, reason = PARTIAL, "storage_failed"
        items = list(selector.selected)
    stats.update(
        selector=settings.news_map_selector,
        status=status, reason=reason or "-", target=target, evaluated=selector.evaluated,
        below_threshold=selector.below_threshold, entity_only=selector.entity_only, unconnected=selector.unconnected,
        center_repeats=selector.center_repeats, excluded=len(selector.excluded),
        neighbour_repeats=selector.neighbour_repeats, withheld=selector.withheld,
        remaining=len(selector.remaining), selected=len(items), **usage,
        elapsed_ms=round((time.monotonic() - started) * 1000),
    )
    if isinstance(selector, DecisionNewsMapSelector):
        stats.update(selector.client.stats)
    # Counts and the hashed article ID only: no titles, queries, keys or vectors.
    logger.info("news_map center=%s %s", article_id(center),
                " ".join(f"{key}={value}" for key, value in stats.items()))
    return NewsMapResult(items, status, reason, target, stats, excluded=len(selector.excluded))
