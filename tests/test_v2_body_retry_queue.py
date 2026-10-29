"""Sequential sticky-key extraction, pending-only retry and bounded deadlines."""
import asyncio
import time
from itertools import pairwise

import pytest
import requests
from pydantic import ValidationError

from app import store
from app.agents import diffbot_client as diffbot
from app.config import Settings, settings
from app.v2 import persistence, provider_budget, reports, state

PRIMARY, SECONDARY = "fixture-key-one", "fixture-key-two"


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(diffbot, "load_diffbot_token", lambda: PRIMARY)
    monkeypatch.setattr(settings, "diffbot_token_secondary", SECONDARY)
    monkeypatch.setattr(settings, "v2_body_batch_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "v2_state_memory", True)
    monkeypatch.setattr(state, "_memory", {})
    monkeypatch.setattr(persistence, "_memory", {"news_bodies": {}, "reports": {}, "strategies": {}})
    updates = []

    async def checkpoint(*args):
        return {}

    async def update(*args, **kwargs):
        updates.append(kwargs)

    async def article(news_id):
        return {"news_id": news_id, "url": f"https://example.com/{news_id}"}

    monkeypatch.setattr(reports, "_checkpoint", checkpoint)
    monkeypatch.setattr(reports, "_update", update)
    monkeypatch.setattr(store, "get_news", article)
    return updates


def rate_limited(news_id):
    return {"news_id": news_id, "code": "BODY_RATE_LIMITED", "message": "rate", "retryable": True}


async def test_sticky_key_gap_and_only_failed_article_is_requeued(monkeypatch, batch):
    calls = []
    monkeypatch.setattr(settings, "v2_body_request_interval_seconds", 0.01)

    async def extract(article, **kwargs):
        news_id = article["news_id"]
        kwargs["attempt_state"]["provider_called"] = True
        calls.append((news_id, kwargs["token"], kwargs["charge_budget"], time.monotonic()))
        if news_id == "b" and kwargs["token"] == PRIMARY:
            return None, rate_limited(news_id)
        return {"news_id": news_id}, None

    monkeypatch.setattr(reports, "extract_one", extract)
    successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a", "b", "c"])
    assert [(row[0], row[1], row[2]) for row in calls] == [("a", PRIMARY, True), ("b", PRIMARY, True), ("c", SECONDARY, True), ("b", SECONDARY, False)]
    assert all(b[3] - a[3] >= 0.009 for a, b in pairwise(calls))
    assert len(successes) == 3 and failures == []
    assert batch[-1]["progress"] == {"requested": 3, "processed": 3, "succeeded": 3, "failed": 0}


async def test_deadline_leaves_remaining_articles_as_failures(monkeypatch, batch):
    monkeypatch.setattr(settings, "v2_body_batch_timeout_seconds", 0.025)
    monkeypatch.setattr(settings, "v2_body_request_interval_seconds", 0.004)
    calls = []

    async def extract(article, **kwargs):
        kwargs["attempt_state"]["provider_called"] = True
        calls.append(kwargs["token"])
        return None, rate_limited(article["news_id"])

    monkeypatch.setattr(reports, "extract_one", extract)
    start = time.monotonic()
    successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a", "b"])
    assert time.monotonic() - start < 0.15
    assert successes == [] and {row["news_id"] for row in failures} == {"a", "b"}
    assert all(a != b for a, b in pairwise(calls))


async def test_busy_attempt_is_cancelled_at_overall_deadline(monkeypatch, batch):
    monkeypatch.setattr(settings, "v2_body_batch_timeout_seconds", 0.02)
    cancelled = asyncio.Event()

    async def extract(article, **kwargs):
        kwargs["attempt_state"]["provider_called"] = True
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    monkeypatch.setattr(reports, "extract_one", extract)
    successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a"])
    assert cancelled.is_set() and successes == []
    assert failures[0]["code"] == "EXTRACTION_TIMEOUT"


async def test_cancel_during_gap_starts_no_more_http(monkeypatch, batch):
    monkeypatch.setattr(settings, "v2_body_batch_timeout_seconds", 2)
    monkeypatch.setattr(settings, "v2_body_request_interval_seconds", 1)
    called = asyncio.Event()
    calls = []

    async def extract(article, **kwargs):
        kwargs["attempt_state"]["provider_called"] = True
        calls.append(article["news_id"])
        called.set()
        return None, rate_limited(article["news_id"])

    monkeypatch.setattr(reports, "extract_one", extract)
    task = asyncio.create_task(reports.extract_bodies("sid", "job", "lease", ["a", "b"]))
    await called.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["a"]


async def test_cached_bodies_need_no_interval_or_provider_work(monkeypatch, batch):
    monkeypatch.setattr(settings, "v2_body_request_interval_seconds", 1)

    async def cached(article, **kwargs):
        return {"news_id": article["news_id"]}, None

    monkeypatch.setattr(reports, "extract_one", cached)
    start = time.monotonic()
    successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a", "b"])
    assert time.monotonic() - start < 0.15
    assert len(successes) == 2 and failures == []


async def test_real_extract_path_bypasses_failure_cache_and_charges_body_once(monkeypatch, batch):
    seen = []
    failure_key = f"v2:body-failure:{reports.text_hash('https://example.com/a')}:{reports.EXTRACTOR_VERSION}"
    await state.put(failure_key, {"code": "BODY_EMPTY", "message": "old", "retryable": True}, 30)

    def get(*args, **kwargs):
        key = kwargs["params"]["token"]
        seen.append(key)
        result = requests.Response()
        result.status_code = 429 if key == PRIMARY else 200
        result._content = b'{"error":"rate"}' if key == PRIMARY else b'{"objects":[{"text":"Body"}]}'
        return result

    monkeypatch.setattr(requests, "get", get)
    with provider_budget.job_budget("test-retry-body"):
        successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a"])
    assert len(successes) == 1 and failures == [] and seen == [PRIMARY, SECONDARY]
    assert state._memory[provider_budget._job_key("test-retry-body")][1]["units"] == 1


async def test_permanent_failure_is_not_requeued(monkeypatch, batch):
    calls = []

    async def forbidden(article, **kwargs):
        calls.append(article["news_id"])
        return None, {"news_id": article["news_id"], "code": "ARTICLE_NOT_ALLOWED", "message": "no", "retryable": False}

    monkeypatch.setattr(reports, "extract_one", forbidden)
    successes, failures, _ = await reports.extract_bodies("sid", "job", "lease", ["a"])
    assert successes == [] and len(failures) == 1 and calls == ["a"]


def test_production_limits_reject_parallel_or_over_one_minute():
    assert Settings(_env_file=None).v2_body_batch_timeout_seconds == 60
    with pytest.raises(ValidationError):
        Settings(_env_file=None, v2_body_concurrency=2)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, v2_body_batch_timeout_seconds=61)
