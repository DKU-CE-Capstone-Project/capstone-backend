"""Relevance gate, repeated-information cleanup, then deterministic MMR."""
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
from app.agents.repeated_coverage import event_evidence, repeated_information
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
    keys = {f"id:{article_id(article)}"}
    for field in ("url", "naver_url"):
        url = canonical_url(article.get(field))
        if url:
            keys.add(f"url:{url}")
    title, description = clean_text(article.get("title")).casefold(), clean_text(article.get("description")).casefold()
    if title:
        keys.add(f"text:{title}\n{description}")
    return keys


def article_key(article: dict[str, Any]) -> tuple[str, ...]:
    """A total order also for conflicting revisions with the same supplied ID."""
    return (article_id(article), canonical_url(article.get("url")), map_text(article),
            clean_text(article.get("published_at")), clean_text(article.get("source")))


def unique_candidates(center: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = identity_keys(center)
    unique = []
    for article in sorted(candidates, key=article_key):
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
    score: float
    cosine: float


async def select_related(
    center: dict[str, Any], candidates: list[dict[str, Any]], *, limit: int,
    min_relevance: float = 0.0,
) -> list[RankedArticle]:
    unique = unique_candidates(center, candidates)
    if not unique or limit <= 0:
        return []
    center_vector = await article_vector(center)
    vectors = {}
    semaphore = asyncio.Semaphore(settings.news_map_embedding_concurrency)
    center_tokens = keyword_tokens(center)
    entities = {word.casefold() for word in ENTITIES}

    async def evaluate(article: dict[str, Any]) -> RankedArticle:
        async with semaphore:
            vector = await article_vector(article)
        vectors[article_id(article)] = vector
        cosine = cosine_similarity(center_vector, vector)
        tokens = keyword_tokens(article)
        shared = center_tokens & tokens
        non_entity_a, non_entity_b = center_tokens - entities, tokens - entities
        union = non_entity_a | non_entity_b
        support = len(non_entity_a & non_entity_b) / len(union) if union else 0.0
        weight = settings.news_map_keyword_weight
        score = (1 - weight) * max(0.0, cosine) + weight * support
        if shared & entities and not (non_entity_a & non_entity_b):
            score -= settings.news_map_entity_only_penalty
        return RankedArticle(article, max(0.0, min(1.0, score)), cosine)

    # Failure of any vector fails the selection, rather than silently losing evidence.
    ranked = await asyncio.gather(*(evaluate(article) for article in unique))
    threshold = max(settings.news_map_min_relevance, min_relevance)
    ranked = [item for item in ranked if item.score >= threshold]
    ranked.sort(key=lambda item: (-item.score, -len(clean_text(item.article.get("description"))),
                                  article_key(item.article)))
    evidence = {article_id(item.article): event_evidence(item.article) for item in ranked}
    center_evidence = event_evidence(center)
    # Each candidate must be checked directly against the center, before MMR.
    ranked = [item for item in ranked if not repeated_information(
        center_evidence, evidence[article_id(item.article)], item.cosine)]
    pair_scores: dict[tuple[str, str], float] = {}

    def similarity(a: RankedArticle, b: RankedArticle) -> float:
        key = tuple(sorted((article_id(a.article), article_id(b.article))))
        if key not in pair_scores:
            pair_scores[key] = cosine_similarity(vectors[key[0]], vectors[key[1]])
        return pair_scores[key]

    groups: list[list[RankedArticle]] = []
    for item in ranked:
        for group in groups:
            # Complete-link admission, not connected components: A~B and B~C
            # alone cannot collapse A and C into one event/representative.
            if all(repeated_information(evidence[article_id(item.article)],
                                        evidence[article_id(member.article)], similarity(item, member))
                   for member in group):
                group.append(item)
                break
        else:
            groups.append([item])
    remaining = [group[0] for group in groups]
    selected: list[RankedArticle] = []
    weight = settings.news_map_mmr_lambda
    while remaining and len(selected) < limit:
        if not selected:
            best = remaining[0]  # Highest relevance, deterministic representative ties.
        else:
            best = min(remaining, key=lambda item: (
                -round(weight * item.score - (1 - weight) * max(similarity(item, prev) for prev in selected), 12),
                -item.score, article_key(item.article),
            ))
        selected.append(best)
        remaining.remove(best)
    # score is always center relevance; MMR's selection score is private.
    return selected
