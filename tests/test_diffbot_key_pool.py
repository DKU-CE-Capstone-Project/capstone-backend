"""Optional second credential: real call routing with all HTTP mocked."""
import asyncio
import threading
import time
from collections import Counter

import pytest
import requests

from app.agents import diffbot_client as diffbot
from app.config import settings
from app.v2 import persistence, reports, state

PRIMARY = "fixture-primary-secret"
SECONDARY = "fixture-secondary-secret"


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(diffbot, "load_diffbot_token", lambda: PRIMARY)
    monkeypatch.setattr(settings, "diffbot_token_secondary", SECONDARY)


def article(index):
    return {"news_id": f"article-{index}", "url": f"https://example.com/news/{index}"}


def record_extractions(monkeypatch):
    seen = []

    def extract(**kwargs):
        seen.append(kwargs["token"])
        return {"objects": [{"text": "Extracted article body."}]}

    monkeypatch.setattr(diffbot, "call_diffbot_article", extract)
    return seen


def test_secondary_uses_settings_and_environment_override(monkeypatch, pool):
    assert diffbot.load_diffbot_tokens() == (PRIMARY, SECONDARY)
    monkeypatch.setenv("DIFFBOT_TOKEN_SECONDARY", " override-secondary ")
    assert diffbot.load_diffbot_tokens() == (PRIMARY, "override-secondary")
    monkeypatch.setenv("DIFFBOT_TOKEN_SECONDARY", "")
    assert diffbot.load_diffbot_tokens() == (PRIMARY,)


def test_same_token_is_not_counted_as_two_keys(monkeypatch, pool):
    monkeypatch.setattr(settings, "diffbot_token_secondary", PRIMARY)
    assert diffbot.load_diffbot_tokens() == (PRIMARY,)


def test_secondary_can_be_used_without_primary(monkeypatch, pool):
    monkeypatch.setattr(diffbot, "load_diffbot_token", lambda: "")
    assert diffbot.load_diffbot_tokens() == (SECONDARY,)


async def test_rotation_spans_separate_single_article_calls(monkeypatch, pool):
    seen = record_extractions(monkeypatch)
    for index in range(4):
        rows = await diffbot.extract_articles_with_diffbot([article(index)], max_retries=0, use_cache=False)
        assert rows[0]["cleaned_content"] == "Extracted article body."
    assert seen == [PRIMARY, SECONDARY, PRIMARY, SECONDARY]


async def test_parallel_singletons_balance_both_keys(monkeypatch, pool):
    seen = record_extractions(monkeypatch)
    await asyncio.gather(*(diffbot.extract_articles_with_diffbot([article(index)], max_retries=0, use_cache=False) for index in range(10)))
    assert Counter(seen) == {PRIMARY: 5, SECONDARY: 5}


async def test_single_key_configuration_stays_compatible(monkeypatch, pool):
    monkeypatch.setattr(settings, "diffbot_token_secondary", "")
    seen = record_extractions(monkeypatch)
    await diffbot.extract_articles_with_diffbot([article(1), article(2)], max_retries=0, use_cache=False)
    assert seen == [PRIMARY, PRIMARY]


async def test_explicit_token_pins_call_to_one_key(monkeypatch, pool):
    seen = record_extractions(monkeypatch)
    await diffbot.extract_articles_with_diffbot([article(1), article(2)], token="explicit-key", max_retries=0, use_cache=False)
    assert seen == ["explicit-key", "explicit-key"]


async def test_explicit_empty_token_disables_both_configured_keys(monkeypatch, pool):
    seen = record_extractions(monkeypatch)
    assert await diffbot.extract_articles_with_diffbot([article(1)], token="") == [article(1)]
    assert seen == []


async def test_cache_hit_from_either_key_does_not_duplicate_extraction(monkeypatch, pool):
    seen = record_extractions(monkeypatch)
    await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0)
    await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0)
    await diffbot.extract_articles_with_diffbot([article(2)], max_retries=0)
    await diffbot.extract_articles_with_diffbot([article(2)], max_retries=0)
    assert seen == [PRIMARY, SECONDARY]


async def test_both_keys_429_stop_after_one_attempt_each_and_log_slots(monkeypatch, pool, caplog):
    seen = []

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        response = requests.Response()
        response.status_code = 429
        response._content = b'{"error":"Too many requests"}'
        return response

    monkeypatch.setattr(requests, "get", get)
    for index in range(2):
        assert await diffbot.extract_articles_with_diffbot([article(index)], max_retries=0, use_cache=False) == [article(index)]
    assert seen == [PRIMARY, SECONDARY, SECONDARY, PRIMARY]
    assert "key_slot=1" in caplog.text and "key_slot=2" in caplog.text
    assert "attempts=2" in caplog.text and "http_status=429" in caplog.text
    assert PRIMARY not in caplog.text and SECONDARY not in caplog.text


async def test_body_recovered_by_second_key_is_cached(monkeypatch, pool):
    seen = []

    def get(*args, **kwargs):
        key = kwargs["params"]["token"]
        seen.append(key)
        response = requests.Response()
        response.status_code = 429 if key == PRIMARY else 200
        response._content = b'{"error":"rate"}' if key == PRIMARY else b'{"objects":[{"text":"Usable body"}]}'
        return response

    monkeypatch.setattr(requests, "get", get)
    first = await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0)
    second = await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0)
    third = await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0)
    assert first[0]["cleaned_content"] == "Usable body"
    assert second[0]["cleaned_content"] == third[0]["cleaned_content"] == "Usable body"
    assert seen == [PRIMARY, SECONDARY]


def test_changed_pool_starts_with_current_primary():
    assert diffbot._next_diffbot_token((PRIMARY, SECONDARY)) == PRIMARY
    assert diffbot._next_diffbot_token((PRIMARY, SECONDARY)) == SECONDARY
    assert diffbot._next_diffbot_token((SECONDARY, "replacement-key")) == SECONDARY


def response(status=200, payload=b'{"objects":[{"text":"Recovered body"}]}'):
    result = requests.Response()
    result.status_code = status
    result._content = payload
    return result


@pytest.mark.parametrize("transport_status", [429, 200])
@pytest.mark.parametrize("start_secondary", [False, True])
async def test_rate_limit_switches_key_for_same_article_without_wait(monkeypatch, pool, caplog, transport_status, start_secondary):
    seen = []

    def get(*args, **kwargs):
        seen.append(dict(kwargs["params"]))
        if len(seen) == 1:
            result = response(transport_status, b'{"errorCode":429,"error":"SECRET provider body"}')
            result.headers["Retry-After"] = "3600"
            return result
        return response()

    def forbidden_sleep(*args):
        raise AssertionError("The alternate key must not wait for the failed key")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(diffbot, "_sleep", forbidden_sleep)
    if start_secondary:
        diffbot._next_diffbot_token((PRIMARY, SECONDARY))
    rows = await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0, use_cache=False)
    assert rows[0]["cleaned_content"] == "Recovered body"
    expected = [SECONDARY, PRIMARY] if start_secondary else [PRIMARY, SECONDARY]
    assert [item["token"] for item in seen] == expected
    assert all(item["url"] == article(1)["url"] for item in seen)
    assert "[diffbot] key switch: news_id=article-1" in caplog.text
    assert PRIMARY not in caplog.text and SECONDARY not in caplog.text
    assert "SECRET provider body" not in caplog.text and "example.com" not in caplog.text


async def test_global_one_attempt_cap_prevents_key_switch(monkeypatch, pool):
    seen = []

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        return response(429)

    monkeypatch.setattr(settings, "diffbot_max_total_attempts", 1)
    monkeypatch.setattr(requests, "get", get)
    assert await diffbot.extract_articles_with_diffbot([article(1)], max_retries=50, use_cache=False) == [article(1)]
    assert seen == [PRIMARY]


@pytest.mark.parametrize("status", [401, 403, 503])
async def test_non_rate_limit_does_not_switch_keys_when_retries_off(monkeypatch, pool, status):
    seen = []

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        return response(status)

    monkeypatch.setattr(requests, "get", get)
    await diffbot.extract_articles_with_diffbot([article(1)], max_retries=0, use_cache=False)
    assert seen == [PRIMARY]


async def test_explicit_key_429_does_not_use_configured_alternate(monkeypatch, pool):
    seen = []

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        return response(429)

    monkeypatch.setattr(requests, "get", get)
    await diffbot.extract_articles_with_diffbot([article(1)], token="explicit-key", max_retries=0, use_cache=False)
    assert seen == ["explicit-key"]


@pytest.mark.parametrize("stop", ["cancelled", "deadline"])
def test_stop_after_429_prevents_alternate_http(monkeypatch, pool, stop):
    seen = []
    cancelled = threading.Event()
    deadline = time.monotonic() + 0.02 if stop == "deadline" else None

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        if stop == "cancelled":
            cancelled.set()
        else:
            time.sleep(0.03)
        return response(429)

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(diffbot.DiffbotError) as error:
        diffbot.call_diffbot_article(article(1)["url"], PRIMARY, max_retries=0, token_pool=(PRIMARY, SECONDARY), cancel_event=cancelled, deadline=deadline)
    assert error.value.category == stop and error.value.attempts == 1
    assert seen == [PRIMARY]


def test_attempt_cap_is_shared_across_same_key_retries_and_switch(monkeypatch, pool):
    seen, waits = [], []
    outcomes = iter([response(503), response(429), response(503)])

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        return next(outcomes)

    monkeypatch.setattr(settings, "diffbot_max_total_attempts", 3)
    monkeypatch.setattr(diffbot, "_sleep", waits.append)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(diffbot.DiffbotError) as error:
        diffbot.call_diffbot_article(article(1)["url"], PRIMARY, max_retries=50, token_pool=(PRIMARY, SECONDARY))
    assert seen == [PRIMARY, PRIMARY, SECONDARY]
    assert error.value.attempts == 3 and error.value.key_slot == 2
    assert len(waits) == 1


def test_all_keys_rate_limited_do_not_ping_pong_with_large_retry_budget(monkeypatch, pool):
    seen = []

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["token"])
        return response(429)

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(diffbot.DiffbotError) as error:
        diffbot.call_diffbot_article(article(1)["url"], PRIMARY, max_retries=50, token_pool=(PRIMARY, SECONDARY))
    assert error.value.category == "rate_limit" and error.value.attempts == 2
    assert seen == [PRIMARY, SECONDARY]


async def test_v2_extract_one_saves_body_after_key_switch(monkeypatch, pool):
    seen = []

    def get(*args, **kwargs):
        key = kwargs["params"]["token"]
        seen.append(key)
        return response(429) if key == PRIMARY else response()

    monkeypatch.setattr(settings, "v2_state_memory", True)
    monkeypatch.setattr(state, "_memory", {})
    monkeypatch.setattr(persistence, "_memory", {"news_bodies": {}, "reports": {}, "strategies": {}})
    monkeypatch.setattr(requests, "get", get)
    body, failure = await reports.extract_one(article(1))
    assert failure is None and body["content"] == "Recovered body"
    assert len(persistence._memory["news_bodies"]) == 1
    assert seen == [PRIMARY, SECONDARY]


async def test_key_switches_share_process_http_concurrency_cap(monkeypatch, pool):
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0}

    def get(*args, **kwargs):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            time.sleep(0.002)
            return response(429) if kwargs["params"]["token"] == PRIMARY else response()
        finally:
            with lock:
                counts["active"] -= 1

    monkeypatch.setattr(settings, "diffbot_max_concurrency", 1)
    monkeypatch.setattr(requests, "get", get)
    batches = await asyncio.gather(*(diffbot.extract_articles_with_diffbot([article(index)], max_retries=0, use_cache=False) for index in range(10)))
    assert all(rows[0]["cleaned_content"] == "Recovered body" for rows in batches)
    assert counts["peak"] == 1
