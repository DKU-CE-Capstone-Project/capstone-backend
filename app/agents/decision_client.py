"""Bounded Decisions REST calls with validated, content-keyed local caching."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any
from weakref import WeakKeyDictionary

import httpx

from app.agents.article_embeddings import clean_text
from app.config import settings

ENDPOINT = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_locks: WeakKeyDictionary = WeakKeyDictionary()


class DecisionUnavailable(RuntimeError):
    """No valid provider evidence; callers must not invent a selection."""


def payload_for(a: dict, b: dict, questions: dict) -> dict:
    def evidence(article):
        return {"title": clean_text(article.get("title"))[:500],
                "description": clean_text(article.get("description"))[:6000],
                "published_at": clean_text(article.get("published_at"))[:100]}

    converted = []
    for name, question in questions.items():
        item = {"name": name, "type": question["type"], "instructions": question["instructions"]}
        if question["type"] == "choice":
            item["choices"] = [{"value": k, "description": v} for k, v in question["criteria"].items()]
        elif question["type"] == "score":
            item["levels"] = [{"label": str(i), "description": v} for i, v in enumerate(question["criteria"])]
        else:
            raise ValueError("Unsupported Decision question type")
        converted.append(item)
    return {"model": MODEL, "input": json.dumps({"article_a": evidence(a), "article_b": evidence(b)},
                                                ensure_ascii=False), "questions": converted}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(body: Any, questions: dict) -> dict:
    """Reject refusals and malformed distributions before any selector state changes."""
    if not isinstance(body, dict) or body.get("model") != MODEL:
        raise ValueError("Unexpected Decision model")
    raw = body.get("answers")
    if not isinstance(raw, list) or len(raw) != len(questions) or any(not isinstance(a, dict) for a in raw):
        raise ValueError("Missing or invalid Decision answers")
    names = [a.get("name") for a in raw]
    if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names) or set(names) != set(questions):
        raise ValueError("Unexpected Decision question names")
    answers = {}
    for answer in raw:
        name = answer["name"]
        question = questions[name]
        if answer.get("type") != question["type"]:
            raise ValueError("Refused or invalid Decision answer type")
        expected = set(question["criteria"]) if question["type"] == "choice" else {
            str(i) for i in range(len(question["criteria"]))}
        rows = answer.get("probabilities")
        if not isinstance(rows, list) or any(not isinstance(p, dict) for p in rows):
            raise ValueError("Invalid Decision probabilities")
        probabilities = {}
        for row in rows:
            key, probability = str(row.get("value")), row.get("probability")
            if key in probabilities or not _number(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid Decision probability")
            probabilities[key] = probability
        if set(probabilities) != expected or abs(sum(probabilities.values()) - 1) > .04:
            raise ValueError("Incomplete Decision probability distribution")
        confidence = answer.get("confidence")
        if not _number(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid Decision confidence")
        if question["type"] == "score":
            score = answer.get("score")
            weighted = sum(int(k) * v for k, v in probabilities.items())
            if not _number(score) or not 0 <= score <= len(expected) - 1 or abs(score - weighted) > .04:
                raise ValueError("Decision score disagrees with distribution")
        elif not isinstance(answer.get("choice"), str) or answer["choice"] not in expected:
            raise ValueError("Unexpected Decision choice")
        answers[name] = {"probabilities": probabilities, "confidence": confidence}
    usage = body.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise ValueError("Invalid Decision usage")
    return answers


class DecisionClient:
    """One map owns its HTTP client, deadlines and attempt budget; no disk writes."""

    def __init__(self, *, transport=None):
        self.http = httpx.AsyncClient(timeout=settings.news_map_decision_timeout_seconds,
                                      trust_env=False, follow_redirects=False, transport=transport)
        self.semaphore = asyncio.Semaphore(settings.news_map_decision_concurrency)
        self.deadline = time.monotonic() + settings.news_map_decision_total_timeout_seconds
        self.stats = {"decision_calls": 0, "decision_attempts": 0, "decision_cache_hits": 0,
                      "decision_input_tokens": 0}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def ask(self, a: dict, b: dict, questions: dict) -> dict:
        key = settings.openai_api_key.get_secret_value().strip()
        if not key:
            raise DecisionUnavailable("Decision API key is not configured")
        payload = payload_for(a, b, questions)
        # Partition caches by credential, input, model and complete rubric. Hashes
        # and credentials stay internal; provider response bodies are never logged.
        digest = hashlib.sha256(json.dumps([key, ENDPOINT, payload], sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
        loop_locks = _locks.setdefault(asyncio.get_running_loop(), {})
        lock, users = loop_locks.get(digest, (asyncio.Lock(), 0))
        loop_locks[digest] = lock, users + 1
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise DecisionUnavailable("Decision deadline exceeded")
            async with asyncio.timeout(remaining):
                async with lock:
                    cached = _cache.get(digest)
                    if cached and time.monotonic() - cached[0] < settings.news_map_decision_cache_ttl_seconds:
                        _cache.move_to_end(digest)
                        self.stats["decision_cache_hits"] += 1
                        return deepcopy(cached[1])
                    async with self.semaphore:
                        for attempt in range(3):
                            if self.stats["decision_attempts"] >= settings.news_map_decision_max_calls:
                                raise DecisionUnavailable("Decision call budget exhausted")
                            self.stats["decision_attempts"] += 1
                            try:
                                response = await self.http.post(ENDPOINT, json=payload,
                                                                headers={"Authorization": f"Bearer {key}"})
                            except httpx.TransportError:
                                response = None
                            if response is not None and response.status_code == 200:
                                try:
                                    body = response.json()
                                    answers = normalize(body, questions)
                                except (ValueError, TypeError, KeyError):
                                    raise DecisionUnavailable("Invalid Decision response") from None
                                self.stats["decision_calls"] += 1
                                self.stats["decision_input_tokens"] += body["usage"]["input_tokens"]
                                _cache[digest] = time.monotonic(), deepcopy(answers)
                                _cache.move_to_end(digest)
                                while len(_cache) > 1024:
                                    _cache.popitem(last=False)
                                return answers
                            if response is not None and response.status_code not in (429, 500, 502, 503, 504):
                                raise DecisionUnavailable(f"Decision provider HTTP {response.status_code}")
                            if attempt < 2:
                                await asyncio.sleep(.5 * (attempt + 1))
                        raise DecisionUnavailable("Decision provider unavailable")
        except TimeoutError:
            raise DecisionUnavailable("Decision deadline exceeded") from None
        finally:
            current_lock, users = loop_locks[digest]
            if users == 1:
                loop_locks.pop(digest)
            else:
                loop_locks[digest] = current_lock, users - 1
