"""Content/config-addressed embeddings; news-map and report RAG never share a slot."""
from __future__ import annotations

import asyncio
import hashlib
import html
import math
import re
import unicodedata
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from typing import Any
from weakref import WeakKeyDictionary

from app import database
from app.agents import llm
from app.config import settings

MAP_VERSION = "news-map-text-v1"
_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_inflight: WeakKeyDictionary = WeakKeyDictionary()
# Per-request counts only (never inputs or vector values), for news-map diagnostics.
_usage: ContextVar[dict[str, int] | None] = ContextVar("embedding_usage", default=None)


def track_usage() -> dict[str, int]:
    usage = {"embedding_calls": 0, "embedding_reused": 0}
    _usage.set(usage)
    return usage


def _count(usage: dict[str, int] | None, name: str) -> None:
    if usage is not None:
        usage[name] += 1


class EmbeddingUnavailable(RuntimeError):
    """Missing/invalid vectors are unavailable evidence, never a similarity of zero."""


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(value))
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def map_text(article: dict[str, Any]) -> str:
    # Only supplied search fields. No generated summary, body extraction or inference.
    title = clean_text(article.get("title"))[:500]
    description = clean_text(article.get("description"))[:6000]
    if not title and not description:
        return ""
    return f"Title: {title}\nDescription: {description}"


def _input(article: dict[str, Any], purpose: str) -> tuple[str, dict[str, Any]]:
    if purpose == "news_map":
        text = map_text(article)
        model, dimensions, task, version = (
            settings.news_map_embedding_model, settings.news_map_embedding_dimensions,
            settings.news_map_embedding_task_type, MAP_VERSION,
        )
    elif purpose == "rag":
        # Preserve the report's existing input format, model and dimension.
        text = f"{article.get('title', '')} {article.get('summary') or article.get('description', '')}".strip()[:8000]
        model, dimensions, task, version = settings.embedding_model, 768, None, "rag-text-v1"
    else:
        raise ValueError("Unknown embedding purpose")
    return text, {
        "input_hash": hashlib.sha256(text.encode()).hexdigest(),
        "model": model, "dimensions": dimensions, "task_type": task,
        "purpose": purpose, "preprocessing_version": version,
    }


def valid_vector(values: Any, dimensions: int) -> bool:
    return (
        isinstance(values, list) and len(values) == dimensions
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
        and sum(v * v for v in values) > 0
        and math.isfinite(sum(v * v for v in values))
    )


def _record(article: dict[str, Any], purpose: str) -> dict[str, Any] | None:
    if purpose == "news_map":
        return article.get("news_map_embedding")
    if article.get("embedding_metadata"):
        return {"metadata": article["embedding_metadata"], "values": article.get("embedding")}
    return None


def _matches(record: Any, metadata: dict[str, Any]) -> bool:
    return (isinstance(record, dict) and record.get("metadata") == metadata
            and valid_vector(record.get("values"), metadata["dimensions"]))


async def article_vector(article: dict[str, Any], purpose: str = "news_map", *, persist: bool = True) -> list[float]:
    text, metadata = _input(article, purpose)
    if not text:
        raise EmbeddingUnavailable("Article has no embedding input")
    # Metadata has fixed insertion order and contains no secrets.
    key = repr(metadata)
    usage = _usage.get()
    loop_tasks = _inflight.setdefault(asyncio.get_running_loop(), {})
    task = loop_tasks.get(key)
    if task is not None:
        _count(usage, "embedding_reused")
    else:
        async def resolve() -> dict[str, Any]:
            record = _record(article, purpose)
            if not _matches(record, metadata):
                record = _cache.get(key)
            if not _matches(record, metadata) and settings.use_mongodb:
                record = await database.get_news_embedding(article.get("news_id", ""), purpose)
            if _matches(record, metadata):
                _count(usage, "embedding_reused")
            else:
                _count(usage, "embedding_calls")
                try:
                    if purpose == "rag":
                        values = await llm.embed(text)
                    else:
                        values = await llm.embed(text, model=metadata["model"],
                                                 dimensions=metadata["dimensions"], task_type=metadata["task_type"])
                except Exception as exc:
                    raise EmbeddingUnavailable("Embedding provider failed") from exc
                if not valid_vector(values, metadata["dimensions"]):
                    raise EmbeddingUnavailable("Embedding provider returned no compatible vector")
                record = {"metadata": metadata, "values": values}
            _cache[key] = deepcopy(record)
            _cache.move_to_end(key)
            while len(_cache) > 512:
                _cache.popitem(last=False)
            return record

        task = asyncio.create_task(resolve())
        loop_tasks[key] = task
        def finished(done):
            if loop_tasks.get(key) is done:
                loop_tasks.pop(key, None)
            # If a client disconnects, shielded single-flight work can outlive the
            # waiter. Consume the exception without logging provider error details.
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
    record = deepcopy(await asyncio.shield(task))
    previous = _record(article, purpose)
    if purpose == "news_map":
        article["news_map_embedding"] = record
    else:
        article["embedding"], article["embedding_metadata"] = record["values"], record["metadata"]
    if persist and settings.use_mongodb and not _matches(previous, metadata):
        await database.save_news_embedding(article.get("news_id", ""), purpose, record)
    return list(record["values"])
