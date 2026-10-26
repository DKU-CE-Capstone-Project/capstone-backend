"""Diffbot diagnostics and cancellation-aware provider protection (all mocked)."""
import asyncio
import logging
import threading
import time

import pytest
import requests

from app.agents import diffbot_client as diffbot
from app.config import settings
from app.v2 import persistence, reports, state

TOKEN = "SECRET-TOKEN-abc123"
URL = "https://example.com/news?key=URLSECRET&id=7"
ARTICLE = {"news_id": "news-42", "url": URL}


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text="BODYSECRET raw provider body"):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._payload is ValueError:
            raise ValueError("not json BODYSECRET")
        return self._payload


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr(diffbot, "_sleep", sleeps.append)
    monkeypatch.setattr(settings, "diffbot_max_concurrency", 3)
    monkeypatch.setattr(settings, "diffbot_max_total_attempts", 4)
    monkeypatch.setattr(settings, "diffbot_backoff_base_seconds", 10.0)
    monkeypatch.setattr(settings, "diffbot_backoff_max_seconds", 60.0)
    return sleeps


def call(max_retries=0):
    return diffbot.call_diffbot_article(URL, TOKEN, max_retries=max_retries)


def patch_get(monkeypatch, *outcomes):
    """Each request consumes the next outcome (response or exception); the last one repeats."""
    seen = []

    def get(*args, **kwargs):
        outcome = outcomes[min(len(seen), len(outcomes) - 1)]
        seen.append(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", get)
    return seen


@pytest.mark.parametrize(
    ("outcome", "category", "status"),
    [
        (FakeResponse(429), "rate_limit", 429),
        (FakeResponse(401), "auth", 401),
        (FakeResponse(403), "auth", 403),
        (FakeResponse(503), "server_error", 503),
        (FakeResponse(404), "http_error", 404),
        (requests.Timeout(f"timed out {URL}"), "timeout", None),
        (requests.ConnectionError(f"refused {URL}&token={TOKEN}"), "network", None),
        (FakeResponse(200, {"errorCode": 401, "error": "Not authorized API token."}), "auth", 200),
        (FakeResponse(200, {"errorCode": 429, "error": "rate"}), "rate_limit", 200),
        (FakeResponse(200, {"error": "boom"}), "api_error", 200),
        (FakeResponse(200, ValueError), "invalid_response", 200),
    ],
)
def test_failures_are_classified(monkeypatch, outcome, category, status):
    patch_get(monkeypatch, outcome)
    with pytest.raises(diffbot.DiffbotError) as info:
        call()
    assert (info.value.category, info.value.status, info.value.attempts) == (category, status, 1)
    # Message and chain carry nothing from the request or response.
    assert TOKEN not in str(info.value) and "URLSECRET" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None or info.value.__suppress_context__


def test_failure_log_has_diagnostics_and_no_secrets(monkeypatch, caplog):
    patch_get(monkeypatch, FakeResponse(429, text=f"quota exceeded {TOKEN} BODYSECRET"))
    monkeypatch.setattr(diffbot, "load_diffbot_token", lambda: TOKEN)
    caplog.set_level(logging.DEBUG)
    out = asyncio.run(diffbot.extract_articles_with_diffbot([dict(ARTICLE)], token=TOKEN, max_retries=0))
    assert out == [ARTICLE]
    text = caplog.text
    for part in ("news_id=news-42", "category=rate_limit", "http_status=429", "attempts=1", "elapsed_ms="):
        assert part in text
    for secret in (TOKEN, "URLSECRET", "BODYSECRET", "example.com"):
        assert secret not in text


def test_network_error_log_does_not_leak_url_or_token(monkeypatch, caplog):
    patch_get(monkeypatch, requests.ConnectionError(f"HTTPSConnectionPool url: /v3/article?token={TOKEN}&url={URL}"))
    caplog.set_level(logging.DEBUG)
    asyncio.run(diffbot.extract_articles_with_diffbot([dict(ARTICLE)], token=TOKEN, max_retries=0))
    assert "category=network" in caplog.text
    assert TOKEN not in caplog.text and "URLSECRET" not in caplog.text


def test_empty_body_is_logged_as_empty(monkeypatch, caplog):
    monkeypatch.setattr(diffbot, "call_diffbot_article", lambda **kw: {"objects": [{"text": "  "}]})
    caplog.set_level(logging.DEBUG)
    diffbot._extract_one_sync(dict(ARTICLE), TOKEN, 1000, 0)
    assert "category=empty" in caplog.text and "news_id=news-42" in caplog.text and "objects=1" in caplog.text
    assert "URLSECRET" not in caplog.text


def test_total_attempts_cap_overrides_max_retries(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(429))
    with pytest.raises(diffbot.DiffbotError) as info:
        call(max_retries=50)
    assert len(seen) == settings.diffbot_max_total_attempts == info.value.attempts
    assert len(fast) == len(seen) - 1


def test_default_v2_path_makes_a_single_attempt(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(503))
    with pytest.raises(diffbot.DiffbotError):
        call(max_retries=0)
    assert len(seen) == 1 and fast == []


def test_backoff_is_exponential_and_capped(monkeypatch, fast):
    monkeypatch.setattr(settings, "diffbot_max_total_attempts", 6)
    patch_get(monkeypatch, FakeResponse(503))
    with pytest.raises(diffbot.DiffbotError):
        call(max_retries=10)
    assert fast == [10.0, 20.0, 40.0, 60.0, 60.0]
    assert max(fast) <= settings.diffbot_backoff_max_seconds


def test_retry_after_beyond_cap_is_not_waited_out(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(429, headers={"Retry-After": "3600"}))
    with pytest.raises(diffbot.DiffbotError) as info:
        call(max_retries=3)
    assert len(seen) == 1 and fast == [] and info.value.category == "rate_limit"


def test_retry_after_within_cap_is_honoured_and_recovers(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"objects": []}))
    assert call(max_retries=3) == {"objects": []}
    assert len(seen) == 2 and fast == [7.0]


def test_auth_failure_is_not_retried(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(401))
    with pytest.raises(diffbot.DiffbotError):
        call(max_retries=3)
    assert len(seen) == 1 and fast == []


@pytest.mark.parametrize("code", [429, 503])
def test_retryable_payload_error_uses_same_bounded_retry_loop(monkeypatch, fast, code):
    seen = patch_get(monkeypatch, FakeResponse(200, {"errorCode": code, "error": "synthetic"}), FakeResponse(200, {"objects": []}))
    assert call(max_retries=3) == {"objects": []}
    assert len(seen) == 2 and fast == [10.0]


def test_payload_auth_is_not_retried_and_transport_status_is_preserved(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(200, {"errorCode": 401, "error": "synthetic"}))
    with pytest.raises(diffbot.DiffbotError) as info:
        call(max_retries=3)
    assert len(seen) == 1 and fast == []
    assert info.value.status == 200
    assert info.value.provider_error_code == 401


async def test_payload_failure_log_separates_transport_and_provider_status(monkeypatch, caplog):
    patch_get(monkeypatch, FakeResponse(200, {"errorCode": 429, "error": f"{TOKEN} BODYSECRET"}))
    caplog.set_level(logging.WARNING)
    await diffbot.extract_articles_with_diffbot([dict(ARTICLE)], token=TOKEN, max_retries=0, use_cache=False)
    assert "http_status=200 provider_error_code=429 attempts=1" in caplog.text
    assert TOKEN not in caplog.text and "BODYSECRET" not in caplog.text


def test_deadline_waiting_for_slot_never_starts_http(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    monkeypatch.setattr(diffbot, "_provider_semaphore", lambda: semaphore)
    seen = patch_get(monkeypatch, FakeResponse())
    try:
        with pytest.raises(diffbot.DiffbotError) as info:
            diffbot.call_diffbot_article(URL, TOKEN, max_retries=3, deadline=time.monotonic() + 0.03)
    finally:
        semaphore.release()
    assert info.value.category == "deadline" and info.value.attempts == 0
    assert seen == []


def test_deadline_does_not_start_a_retry_or_wait_out_its_backoff(monkeypatch, fast):
    seen = patch_get(monkeypatch, FakeResponse(503))
    with pytest.raises(diffbot.DiffbotError) as info:
        diffbot.call_diffbot_article(URL, TOKEN, max_retries=3, deadline=time.monotonic() + 0.1)
    assert info.value.category == "deadline" and info.value.attempts == 1
    assert len(seen) == 1 and fast == []


async def _wait_thread_event(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), 3)


async def test_cancelled_extraction_waiting_for_slot_never_starts_http(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    first_started, release_first = threading.Event(), threading.Event()
    second_waiting, second_finished = threading.Event(), threading.Event()
    seen = []
    sync = diffbot._extract_one_sync

    def provider_semaphore():
        if first_started.is_set():
            second_waiting.set()
        return semaphore

    def extract(article, *args, **kwargs):
        try:
            return sync(article, *args, **kwargs)
        finally:
            if article["news_id"] == "second":
                second_finished.set()

    def get(*args, **kwargs):
        seen.append(kwargs["params"]["url"])
        first_started.set()
        assert release_first.wait(2)
        return FakeResponse(200, {"objects": [{"text": "body"}]})

    monkeypatch.setattr(diffbot, "_provider_semaphore", provider_semaphore)
    monkeypatch.setattr(diffbot, "_extract_one_sync", extract)
    monkeypatch.setattr(requests, "get", get)
    first = asyncio.create_task(diffbot.extract_articles_with_diffbot([{"news_id": "first", "url": URL}], token=TOKEN, max_retries=0, use_cache=False))
    await _wait_thread_event(first_started)
    second = asyncio.create_task(diffbot.extract_articles_with_diffbot([{"news_id": "second", "url": URL + "second"}], token=TOKEN, max_retries=0, use_cache=False))
    try:
        await _wait_thread_event(second_waiting)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
    finally:
        release_first.set()
    await first
    await _wait_thread_event(second_finished)
    assert seen == [URL]
    # Cancelled waiters must not consume/leak a slot.
    assert semaphore.acquire(blocking=False)
    semaphore.release()


async def test_cancelled_extraction_during_backoff_never_starts_retry(monkeypatch):
    first_called, finished = threading.Event(), threading.Event()
    seen = []
    sync = diffbot._extract_one_sync

    def extract(*args, **kwargs):
        try:
            return sync(*args, **kwargs)
        finally:
            finished.set()

    def get(*args, **kwargs):
        seen.append(1)
        first_called.set()
        return FakeResponse(429)

    monkeypatch.setattr(diffbot, "_extract_one_sync", extract)
    monkeypatch.setattr(requests, "get", get)
    task = asyncio.create_task(diffbot.extract_articles_with_diffbot([dict(ARTICLE)], token=TOKEN, max_retries=3, use_cache=False))
    await _wait_thread_event(first_called)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_thread_event(finished)
    assert seen == [1]


def test_concurrency_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "diffbot_max_concurrency", 2)
    lock, state_ = threading.Lock(), {"now": 0, "peak": 0}

    def get(*args, **kwargs):
        with lock:
            state_["now"] += 1
            state_["peak"] = max(state_["peak"], state_["now"])
        time.sleep(0.05)
        with lock:
            state_["now"] -= 1
        return FakeResponse(200, {"objects": []})

    monkeypatch.setattr(requests, "get", get)
    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state_["peak"] == 2


async def test_v2_failure_code_stays_body_empty_and_is_logged(monkeypatch, caplog):
    puts = []

    async def get(key):
        return None

    async def put(key, value, ttl):
        puts.append(value)

    async def body(*args):
        return None

    async def extract(articles, **kwargs):
        return [{}]  # the client swallowed a provider error

    monkeypatch.setattr(state, "get", get)
    monkeypatch.setattr(state, "put", put)
    monkeypatch.setattr(persistence, "body", body)
    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", extract)
    caplog.set_level(logging.DEBUG)
    body_doc, failure = await reports.extract_one(dict(ARTICLE))
    assert body_doc is None
    assert failure["code"] == "BODY_EMPTY" and failure["news_id"] == "news-42"
    assert set(failure) == {"news_id", "code", "message", "retryable"}
    assert "news_id=news-42 code=BODY_EMPTY" in caplog.text
    assert "URLSECRET" not in caplog.text
