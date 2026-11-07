"""Provider usage events (econmind.usage): format, labels, safety. All providers mocked."""
import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
import requests
from pydantic import SecretStr

from app import usage_events
from app.agents import decision_client, diffbot_client, llm, naver_client
from app.agents.decision_client import DecisionClient, DecisionUnavailable
from app.agents.decision_policy import PAIR_QUESTIONS
from app.config import settings
from app.v2 import provider_budget
from tests.test_decision_news_map import article, response

SECRET = "unit-test-private-key"


@pytest.fixture(autouse=True)
def usage_on(monkeypatch):
    monkeypatch.setattr(settings, "usage_events_enabled", True)


def events(caplog) -> list[dict]:
    return [json.loads(r.getMessage().removeprefix("[usage] "))
            for r in caplog.records if r.name == "econmind.usage" and r.getMessage().startswith("[usage] {")]


def fake_gemini(monkeypatch, *, usage=None, text=" ok ", error=None):
    calls = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(text=text, usage_metadata=usage)

    monkeypatch.setattr(settings, "google_api_key", "unit-test-key")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))))
    return calls


# ── format and switches ──────────────────────────────────────────────────

def test_event_has_contract_fields_and_no_identity(caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    usage_events.emit("naver", "search", started=0.0, status="ok")
    [event] = events(caplog)
    assert set(event) == {"ts", "trace_id", "service", "operation", "model", "tokens_in", "tokens_out",
                          "calls", "latency_ms", "status", "est_cost_usd"}
    assert event["trace_id"] is None and event["calls"] == 1 and event["est_cost_usd"] is None


def test_disabled_writes_nothing(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    monkeypatch.setattr(settings, "usage_events_enabled", False)
    usage_events.emit("naver", "search", started=0.0, status="ok")
    assert events(caplog) == []


def test_trace_id_is_the_bound_job_only(caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    with provider_budget.job_budget("job-123"):
        usage_events.emit("diffbot", "body", started=0.0, status="ok")
    usage_events.emit("diffbot", "body", started=0.0, status="ok")
    assert [e["trace_id"] for e in events(caplog)] == ["job-123", None]


def test_worker_process_without_app_main_still_prints_events():
    # The v2 worker imports reports, not app.main, so nothing configures "econmind".
    code = ("import app.v2.report_worker, sys; from app.config import settings; from app import usage_events; "
            "assert 'app.main' not in sys.modules; settings.usage_events_enabled = True; "
            "usage_events.emit('gemini', 'analysis', started=0.0, status='ok')")
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True,
                            env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count('[usage] {"ts"') == 1


def test_cost_uses_price_table_and_zero_means_unknown(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    monkeypatch.setattr(settings, "usage_price_gemini_input_per_mtok", 0.10)
    monkeypatch.setattr(settings, "usage_price_gemini_output_per_mtok", 0.40)
    monkeypatch.setattr(settings, "usage_price_diffbot_per_call", 0.002)
    usage_events.emit("gemini", "analysis", started=0.0, status="ok", tokens_in=1_000_000, tokens_out=500_000)
    usage_events.emit("gemini", "analysis", started=0.0, status="error")
    usage_events.emit("diffbot", "body", started=0.0, status="ok")
    usage_events.emit("decision", "pair", started=0.0, status="ok")
    assert [e["est_cost_usd"] for e in events(caplog)] == [0.3, None, 0.002, None]


def test_recording_failure_is_swallowed(monkeypatch, caplog):
    def broken():
        raise RuntimeError("ledger down")

    monkeypatch.setattr(usage_events, "_current_trace_id", broken)
    usage_events.emit("naver", "search", started=0.0, status="ok")
    assert "event dropped: RuntimeError" in caplog.text


# ── Gemini via llm.generate ──────────────────────────────────────────────

async def test_generate_records_tokens_and_purpose(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    fake_gemini(monkeypatch, usage=SimpleNamespace(prompt_token_count=1200, candidates_token_count=340))
    assert await llm.generate("p", purpose="critic") == "ok"
    [event] = events(caplog)
    assert event["service"] == "gemini" and event["operation"] == "critic"
    assert (event["tokens_in"], event["tokens_out"], event["status"]) == (1200, 340, "ok")
    assert event["model"] == settings.gemini_model


async def test_existing_call_shape_is_unchanged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    calls = fake_gemini(monkeypatch, usage=None)
    assert await llm.generate("p") == "ok"
    assert calls[0]["config"].max_output_tokens == settings.llm_max_output_tokens
    [event] = events(caplog)
    assert event["operation"] == "unspecified" and event["tokens_in"] is None


@pytest.mark.parametrize(("requested", "sent"), [(4096, 4096), (100, 512), (99999, 8192)])
async def test_per_call_output_cap_is_bounded(monkeypatch, requested, sent):
    calls = fake_gemini(monkeypatch)
    await llm.generate("p", max_output_tokens=requested)
    assert calls[0]["config"].max_output_tokens == sent


async def test_provider_error_is_recorded_and_still_falls_back(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    fake_gemini(monkeypatch, error=RuntimeError("503"))
    assert await llm.generate("p", purpose="analysis") == ""
    assert [(e["status"], e["operation"]) for e in events(caplog)] == [("error", "analysis")]


async def test_broken_recorder_does_not_break_generation(monkeypatch):
    fake_gemini(monkeypatch, usage=SimpleNamespace(prompt_token_count=1, candidates_token_count=1))
    monkeypatch.setattr(usage_events, "_current_trace_id", lambda: 1 / 0)
    assert await llm.generate("p") == "ok"


async def test_generate_with_usage_returns_tokens(monkeypatch):
    fake_gemini(monkeypatch, usage=SimpleNamespace(prompt_token_count=10, candidates_token_count=4))
    text, usage = await llm.generate_with_usage("p", purpose="bull")
    assert text == "ok"
    assert usage == {"service": "gemini", "model": settings.gemini_model, "status": "ok",
                     "tokens_in": 10, "tokens_out": 4}
    _, after = await llm.generate_with_usage("p")
    assert after["tokens_in"] == 10  # each call gets its own sink


async def test_purpose_does_not_leak_between_calls(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    fake_gemini(monkeypatch)
    await llm.generate("p", purpose="strategy")
    await llm.generate("p")
    assert [e["operation"] for e in events(caplog)] == ["strategy", "unspecified"]


# ── Decision, NAVER, Diffbot ─────────────────────────────────────────────

async def test_decision_records_call_tokens_and_cache_hit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(SECRET))
    monkeypatch.setattr(settings, "news_map_selector", "decision")
    async with DecisionClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=response(PAIR_QUESTIONS)))) as client:
        await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
        await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
    first, hit = events(caplog)
    assert (first["service"], first["status"], first["calls"]) == ("decision", "ok", 1)
    assert first["tokens_in"] == client.stats["decision_input_tokens"]
    assert (hit["status"], hit["calls"]) == ("cache_hit", 0)
    assert SECRET not in caplog.text
    decision_client._cache.clear()


async def test_decision_records_each_failed_attempt(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(SECRET))

    async def no_wait(_):
        return None

    monkeypatch.setattr(decision_client.asyncio, "sleep", no_wait)
    async with DecisionClient(transport=httpx.MockTransport(lambda r: httpx.Response(429))) as client:
        with pytest.raises(DecisionUnavailable):
            await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
    assert [e["status"] for e in events(caplog)] == ["http_429"] * 3


@pytest.mark.parametrize(("handler", "status"), [
    (lambda r: httpx.Response(200, json={"total": 0, "items": []}), "ok"),
    (lambda r: httpx.Response(429), "http_429"),
])
async def test_naver_search_is_recorded(monkeypatch, caplog, handler, status):
    caplog.set_level(logging.INFO, "econmind.usage")
    monkeypatch.setattr(settings, "naver_client_id", "unit-test-id")
    monkeypatch.setattr(settings, "naver_client_secret", "unit-test-secret")
    real_client = httpx.AsyncClient
    monkeypatch.setattr(naver_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    try:
        await naver_client.fetch_naver_page("반도체")
    except naver_client.NaverNewsError:
        pass
    [event] = events(caplog)
    assert (event["service"], event["operation"], event["status"]) == ("naver", "search", status)
    assert "unit-test-secret" not in caplog.text


def test_diffbot_records_every_http_attempt_with_key_slot(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "econmind.usage")
    outcomes = [SimpleNamespace(status_code=429, headers={}, json=dict),
                SimpleNamespace(status_code=200, headers={}, json=lambda: {"objects": []})]
    monkeypatch.setattr(requests, "get", lambda *a, **k: outcomes.pop(0))
    diffbot_client.call_diffbot_article("https://example.com/a", "KEY-ONE-SECRET",
                                        token_pool=("KEY-ONE-SECRET", "KEY-TWO-SECRET"))
    got = [(e["status"], e["key_slot"]) for e in events(caplog)]
    assert got == [("http_429", 1), ("ok", 2)]
    assert "KEY-ONE-SECRET" not in caplog.text and "example.com" not in caplog.text
