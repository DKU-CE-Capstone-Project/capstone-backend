"""Relevance gate, direct repetition exclusion and deterministic MMR.

Only the center and selected neighbours can exclude a candidate. Discarded or
unselected articles never act as comparison anchors. Later rounds append to
the displayed selection without replacing earlier neighbours.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.agents.article_embeddings import article_vector, clean_text, map_text
from app.agents.article_metadata import (
    ALIASES,
    STOPWORDS,
    _canonical,
    _contains,
    _grounded,
)
from app.agents.repeated_coverage import (
    EventEvidence,
    Relation,
    compare,
    event_evidence,
    has_issue_connection,
)
from app.config import settings
from app.utils import make_news_id

ENTITIES = {"삼성전자", "SK하이닉스", "엔비디아", "테슬라", "현대자동차", "한국은행", "연준"}


def article_id(article: dict[str, Any]) -> str:
    return article.get("news_id") or make_news_id(article.get("url") or map_text(article))


def canonical_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        parts = urlsplit(value)
        # Keep article identity parameters (e.g. NAVER oid/aid); drop only tracking.
        query = sorted((k, v) for k, v in parse_qsl(parts.query)
                       if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"})
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), ""))
    except ValueError:
        return value


def identity_keys(article: dict[str, Any]) -> set[str]:
    """Same ID or same article URL only. Another outlet's copy is a separate candidate."""
    keys = {f"id:{article_id(article)}"}
    for field_name in ("url", "naver_url"):
        url = canonical_url(article.get(field_name))
        if url:
            keys.add(f"url:{url}")
    return keys


def article_key(article: dict[str, Any]) -> tuple[str, ...]:
    """A total order also for conflicting revisions with the same supplied ID."""
    return (article_id(article), canonical_url(article.get("url")), map_text(article),
            clean_text(article.get("published_at")), clean_text(article.get("source")))


def unique_candidates(center: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identity de-duplication that keeps the supplied (search-rank) order."""
    seen = identity_keys(center)
    unique = []
    for article in candidates:
        keys = identity_keys(article)
        if keys & seen or not map_text(article):
            continue
        seen.update(keys)
        unique.append(article)
    return unique


def keyword_tokens(article: dict[str, Any]) -> set[str]:
    text = clean_text(article.get("title")) + " " + clean_text(article.get("description"))
    normalized = text.casefold()
    tokens = set()
    # Replace confirmed aliases before tokenization; AI/HBM/HBM3E survive boundaries.
    alias_tokens = {}
    for index, (canonical, aliases) in enumerate(ALIASES.items()):
        placeholder = f"econalias{index}"
        alias_tokens[placeholder] = canonical.casefold()
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(alias.casefold()) + r"(?![a-z0-9])"
            normalized = re.sub(pattern, f" {placeholder} ", normalized)
        if any(_contains(text, alias) for alias in aliases):
            tokens.add(canonical.casefold())
    tokens.update(alias_tokens.get(token, token) for token in re.findall(r"[a-z][a-z0-9]*|[가-힣]{2,}", normalized))
    # Metadata is only a grounded union, never an independent extra score.
    for word in article.get("keywords") or []:
        if isinstance(word, str):
            word = _canonical(word)
            if _grounded(word, text):
                # Phrases contribute their existing terms, not a second phrase bonus.
                if word in ALIASES:
                    tokens.add(word.casefold())
                else:
                    tokens.update(re.findall(r"[a-z][a-z0-9]*|[가-힣]{2,}", word.casefold()))
    return tokens - {word.casefold() for word in STOPWORDS} - {"위해", "대한", "통해", "이번", "오늘"}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Incompatible embedding dimensions")
    score = sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    return max(-1.0, min(1.0, score))


@dataclass(frozen=True)
class RankedArticle:
    article: dict[str, Any]
    score: float   # Center relevance; MMR's selection score stays private.
    cosine: float


class NewsMapSelector:
    def __init__(self, center: dict[str, Any], *, min_relevance: float = 0.0,
                 exclude: set[str] | None = None):
        self.center = center
        self.threshold = max(settings.news_map_min_relevance, min_relevance)
        self.remaining: list[RankedArticle] = []
        self.selected: list[RankedArticle] = []
        self.center_repeats = 0
        self.neighbour_repeats = 0
        self.withheld = 0  # Unconfirmed repeat, no confirmed addition.
        self.evaluated = 0
        self.below_threshold = 0
        self.entity_only = 0
        self.unconnected = 0
        self._seen = identity_keys(center)
        # Articles already on the caller's map: never candidates, counted once each.
        self._exclude = set(exclude or ())
        self._seen |= self._exclude
        self.excluded: set[str] = set()
        self._center_vector: list[float] | None = None
        self._center_tokens = keyword_tokens(center)
        self._center_evidence = event_evidence(center)
        self._vectors: dict[str, list[float]] = {}
        self._evidence: dict[str, EventEvidence] = {}
        self._pairs: dict[tuple[str, str], float] = {}

    def is_new(self, article: dict[str, Any]) -> bool:
        keys = identity_keys(article)
        if keys & self._exclude:
            self.excluded.add(article_id(article))
        return not keys & self._seen

    async def add(self, candidates: list[dict[str, Any]], *, limit: int | None = None) -> int:
        """Evaluate up to ``limit`` new unique candidates, keeping the supplied order.

        All vectors are resolved before any state changes, so a failed or cancelled
        round leaves the earlier, complete evaluation intact.
        """
        fresh, seen = [], set(self._seen)
        for article in candidates:
            keys = identity_keys(article)
            if keys & self._exclude:
                self.excluded.add(article_id(article))
            if keys & seen or not map_text(article):
                continue
            if limit is not None and len(fresh) >= limit:
                break
            seen |= keys
            fresh.append(article)
        if not fresh:
            return 0
        if self._center_vector is None:
            self._center_vector = await article_vector(self.center)
        semaphore = asyncio.Semaphore(settings.news_map_embedding_concurrency)

        async def vector(article: dict[str, Any]) -> list[float]:
            async with semaphore:
                return await article_vector(article)

        # Failure of any vector fails the round, rather than silently losing evidence.
        vectors = await asyncio.gather(*(vector(article) for article in fresh))
        self._seen = seen
        self.evaluated += len(fresh)
        ranked = []
        for article, values in zip(fresh, vectors):
            self._vectors[article_id(article)] = values
            cosine = cosine_similarity(self._center_vector, values)
            score = self._relevance(article, cosine)
            if score < self.threshold:
                self.below_threshold += 1
                continue
            # The existing cosine/keyword score is unchanged. A registered
            # company alone is still not evidence of a connection to this issue.
            shared = self._center_tokens & keyword_tokens(article)
            if shared and shared <= {word.casefold() for word in ENTITIES}:
                self.entity_only += 1
                continue
            ranked.append(RankedArticle(article, score, cosine))
        ranked.sort(key=lambda item: (-item.score, -len(clean_text(item.article.get("description"))),
                                      article_key(item.article)))
        for item in ranked:
            self._place(item)
        return len(fresh)

    def _relevance(self, article: dict[str, Any], cosine: float) -> float:
        tokens = keyword_tokens(article)
        entities = {word.casefold() for word in ENTITIES}
        shared = self._center_tokens & tokens
        non_entity_a, non_entity_b = self._center_tokens - entities, tokens - entities
        union = non_entity_a | non_entity_b
        support = len(non_entity_a & non_entity_b) / len(union) if union else 0.0
        weight = settings.news_map_keyword_weight
        score = (1 - weight) * max(0.0, cosine) + weight * support
        if shared & entities and not (non_entity_a & non_entity_b):
            score -= settings.news_map_entity_only_penalty
        return max(0.0, min(1.0, score))

    def _evidence_of(self, item: RankedArticle) -> EventEvidence:
        key = article_id(item.article)
        if key not in self._evidence:
            self._evidence[key] = event_evidence(item.article)
        return self._evidence[key]

    def similarity(self, a: RankedArticle, b: RankedArticle) -> float:
        key = tuple(sorted((article_id(a.article), article_id(b.article))))
        if key not in self._pairs:
            self._pairs[key] = cosine_similarity(self._vectors[key[0]], self._vectors[key[1]])
        return self._pairs[key]

    def _place(self, item: RankedArticle) -> None:
        evidence = self._evidence_of(item)
        # Every candidate is judged directly against the center first.
        judgment = compare(self._center_evidence, evidence, item.cosine)
        if judgment.relation is Relation.REPEAT:
            self.center_repeats += 1
            return
        if judgment.withheld:
            self.withheld += 1
            return
        if not has_issue_connection(self._center_evidence, evidence, self.center.get("_search_keyword") or ""):
            self.unconnected += 1
            return
        self.remaining.append(item)

    def _excluded_by_selection(self, item: RankedArticle) -> bool:
        """Compare only with articles that will remain visible, never discarded copies."""
        evidence = self._evidence_of(item)
        for representative in self.selected:
            judgment = compare(self._evidence_of(representative), evidence, self.similarity(representative, item))
            if judgment.relation is Relation.REPEAT:
                self.neighbour_repeats += 1
                return True
            if judgment.withheld:
                self.withheld += 1
                return True
        return False

    def select(self, limit: int) -> list[RankedArticle]:
        """Continue MMR, verifying repetition before every additional visible node.

        MMR keeps its existing score and weight. Ties use center relevance,
        description completeness and the fixed article key. Each round retains
        previously displayed representatives, even if a later copy scores higher.
        """
        weight = settings.news_map_mmr_lambda

        def priority(item: RankedArticle):
            redundancy = [self.similarity(item, s) for s in self.selected]
            value = weight * item.score - (1 - weight) * max(redundancy) if redundancy else item.score
            return (-round(value, 12), -item.score, -len(clean_text(item.article.get("description"))),
                    article_key(item.article))

        while self.remaining:
            # Run even when the target has been reached: final selected nodes
            # exclude their copies consistently in diagnostics and later rounds.
            self.remaining = [item for item in self.remaining if not self._excluded_by_selection(item)]
            if not self.remaining or len(self.selected) >= limit:
                break
            best = min(self.remaining, key=priority)
            self.selected.append(best)
            self.remaining.remove(best)
        return list(self.selected)


async def select_related(
    center: dict[str, Any], candidates: list[dict[str, Any]], *, limit: int,
    min_relevance: float = 0.0,
) -> list[RankedArticle]:
    """One-round selection of useful neighbours, with repeated information excluded."""
    if limit <= 0:
        return []
    selector = NewsMapSelector(center, min_relevance=min_relevance)
    if not unique_candidates(center, candidates):
        return []
    await selector.add(candidates)
    return selector.select(limit)
