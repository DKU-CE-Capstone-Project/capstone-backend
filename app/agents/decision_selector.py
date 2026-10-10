"""Repeat-first selection with useful roles and visible-representative comparisons."""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.article_embeddings import map_text
from app.agents.decision_client import DecisionClient
from app.agents.decision_policy import (
    CENTER_QUESTIONS,
    PAIR_QUESTIONS,
    POLICY,
    expected,
    redundant,
    route,
)
from app.agents.related_selector import (
    RankedArticle,
    article_id,
    article_key,
    identity_keys,
)


class DecisionNewsMapSelector:
    def __init__(self, center: dict[str, Any], client: DecisionClient, *, min_relevance: float = 0.0):
        self.center, self.client, self.threshold = center, client, min_relevance
        self.remaining: list[RankedArticle] = []
        self.selected: list[RankedArticle] = []
        self.center_repeats = self.neighbour_repeats = self.withheld = 0
        self.evaluated = self.below_threshold = self.entity_only = self.unconnected = 0
        self._seen = identity_keys(center)
        self._roles: dict[str, str] = {}

    def is_new(self, article: dict[str, Any]) -> bool:
        return not identity_keys(article) & self._seen

    async def add(self, candidates: list[dict[str, Any]], *, limit: int | None = None) -> int:
        fresh, seen = [], set(self._seen)
        for article in candidates:
            keys = identity_keys(article)
            if keys & seen or not map_text(article):
                continue
            if limit is not None and len(fresh) >= limit:
                break
            seen |= keys
            fresh.append(article)
        if not fresh:
            return 0
        tasks = [asyncio.create_task(self.client.ask(self.center, a, CENTER_QUESTIONS)) for a in fresh]
        try:
            answers = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        # Commit an evaluation round only after all evidence is valid.
        self._seen = seen
        self.evaluated += len(fresh)
        for article, answer in zip(fresh, answers):
            bucket, _, role = route(answer)
            if bucket == "center_group":
                self.center_repeats += 1
            elif bucket == "excluded":
                self.unconnected += 1
            elif bucket == "withheld":
                self.withheld += 1
            else:
                # The existing API score is 0..1. This mode reports expected
                # helpfulness / 3, not embedding cosine or a causal probability.
                score = min(1.0, expected(answer["helpfulness"]["probabilities"]) / 3)
                if score < self.threshold:
                    self.below_threshold += 1
                    continue
                self._roles[article_id(article)] = role
                self.remaining.append(RankedArticle(article, score, 0.0))
        return len(fresh)

    async def select(self, limit: int) -> list[RankedArticle]:
        # A failed/cancelled selection round preserves all previously visible
        # representatives. Discarded articles never exclude another candidate.
        selected, pending = list(self.selected), list(self.remaining)
        duplicates = 0

        def priority(item):
            role = self._roles[article_id(item.article)]
            same_role = sum(self._roles[article_id(s.article)] == role for s in selected)
            value = item.score * 3 - POLICY["role_penalty"] * same_role
            rank = item.article.get("_search_rank")
            rank = rank if isinstance(rank, int) and rank > 0 else 10 ** 9
            return -round(value, 12), rank, article_key(item.article)

        while pending and len(selected) < limit:
            best = min(pending, key=priority)
            pending.remove(best)
            for representative in selected:
                answer = await self.client.ask(representative.article, best.article, PAIR_QUESTIONS)
                if redundant(answer):
                    duplicates += 1
                    break
            else:
                selected.append(best)
        self.selected, self.remaining = selected, pending
        self.neighbour_repeats += duplicates
        return list(selected)
