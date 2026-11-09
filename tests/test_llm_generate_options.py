"""llm.generate options: telemetry stage, per-call output cap, returned usage."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from app import telemetry
from app.agents import llm
from app.config import settings


def events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"event_type": "econmind_usage"')]


def fake_gemini(monkeypatch, *, usage=None, text=" ok ", error=None, wait=None):
    calls = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        if wait is not None:
            await wait()
        return SimpleNamespace(text=text, usage_metadata=usage)

    monkeypatch.setattr(settings, "google_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))))
    return calls


def fake_claude(monkeypatch, *, provider="anthropic", text=" ok ", wait=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if wait is not None:
            await wait()
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: client))
    monkeypatch.setattr(settings, "anthropic_api_key", "unit-test-claude-key")
    monkeypatch.setattr(settings, "llm_provider", provider)
    return calls


async def test_existing_call_shape_is_unchanged(monkeypatch, capsys):
    calls = fake_gemini(monkeypatch)
    assert await llm.generate("p") == "ok"
    assert calls[0]["config"].max_output_tokens == settings.llm_max_output_tokens
    [event] = events(capsys)
    assert "stage" not in event


@pytest.mark.parametrize(("requested", "sent"), [(4096, 4096), (100, 512), (99999, 8192)])
async def test_per_call_output_cap_is_bounded(monkeypatch, requested, sent):
    calls = fake_gemini(monkeypatch)
    await llm.generate("p", max_output_tokens=requested)
    assert calls[0]["config"].max_output_tokens == sent


@pytest.mark.parametrize(("requested", "sent"), [(None, 1024), (100, 512), (4096, 4096), (99999, 8192)])
async def test_claude_per_call_output_cap_is_bounded(monkeypatch, requested, sent):
    calls = fake_claude(monkeypatch)
    assert await llm.generate("p", max_output_tokens=requested) == "ok"
    assert calls[0]["max_tokens"] == sent


async def test_purpose_is_the_telemetry_stage_and_does_not_leak(monkeypatch, capsys):
    fake_gemini(monkeypatch)
    await llm.generate("p", purpose="bull")
    await llm.generate("p")
    assert [e.get("stage") for e in events(capsys)] == ["bull", None]


async def test_caller_stage_stays_without_purpose(monkeypatch, capsys):
    fake_gemini(monkeypatch)

    @telemetry.stage("analysis")
    async def analyze():
        return await llm.generate("p")

    await analyze()
    assert events(capsys)[0]["stage"] == "analysis"


async def test_job_id_stays_on_purpose_events(monkeypatch, capsys):
    fake_gemini(monkeypatch)
    job_id = "a" * 32
    with telemetry.usage_context(job_id=job_id):
        await llm.generate("p", purpose="judge")
    [event] = events(capsys)
    assert (event["job_id"], event["stage"]) == (job_id, "judge")


async def test_generate_with_usage_returns_tokens(monkeypatch):
    fake_gemini(monkeypatch, usage=SimpleNamespace(prompt_token_count=10, candidates_token_count=4))
    text, usage = await llm.generate_with_usage("p", purpose="bull")
    assert text == "ok"
    assert usage == {"service": "gemini", "model": settings.gemini_model, "status": "ok",
                     "tokens_in": 10, "tokens_out": 4}


async def test_generate_with_usage_fills_a_new_dict_each_call(monkeypatch):
    fake_gemini(monkeypatch, usage=SimpleNamespace(prompt_token_count=10, candidates_token_count=4))
    _, first = await llm.generate_with_usage("p")
    _, second = await llm.generate_with_usage("p")
    assert first == second and first is not second


async def test_generate_with_usage_reports_missing_token_counts_as_none(monkeypatch):
    fake_gemini(monkeypatch, usage=None)
    _, usage = await llm.generate_with_usage("p")
    assert (usage["status"], usage["tokens_in"], usage["tokens_out"]) == ("ok", None, None)


async def test_generate_with_usage_reports_provider_error(monkeypatch):
    fake_gemini(monkeypatch, error=RuntimeError("503"))
    text, usage = await llm.generate_with_usage("p", purpose="judge")
    assert text == ""
    assert (usage["service"], usage["status"], usage["tokens_in"]) == ("gemini", "error", None)


@pytest.mark.parametrize("key", ["", "unit-test-key"])
async def test_generate_with_usage_keeps_fields_when_gemini_cannot_start(monkeypatch, key):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", key)
    monkeypatch.setattr(llm, "_client", lambda: None)
    text, usage = await llm.generate_with_usage("p")
    assert text == ""
    assert usage == {"service": None, "model": None, "status": "error",
                     "tokens_in": None, "tokens_out": None}


@pytest.mark.parametrize(("provider", "fake_provider"), [("gemini", fake_gemini), ("anthropic", fake_claude)])
async def test_generate_with_usage_reports_timeout(monkeypatch, capsys, provider, fake_provider):
    fake_provider(monkeypatch, wait=asyncio.Event().wait)
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 0.1)
    text, usage = await llm.generate_with_usage("p", purpose="judge")
    [event] = events(capsys)
    assert text == ""
    assert (event["provider"], event["stage"]) == (provider, "judge")
    assert usage == {"service": provider, "model": event["model"], "status": "timeout",
                     "tokens_in": None, "tokens_out": None}


async def test_auto_timeout_uses_fallback_provider_and_preserves_cap(monkeypatch, capsys):
    gemini_calls = fake_gemini(monkeypatch, wait=asyncio.Event().wait)
    claude_calls = fake_claude(monkeypatch, provider="auto", text="")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 0.1)
    text, usage = await llm.generate_with_usage("p", purpose="judge", max_output_tokens=8192)
    claude_event, gemini_event = events(capsys)
    assert text == ""
    assert claude_calls[0]["max_tokens"] == 8192
    assert gemini_calls[0]["config"].max_output_tokens == 8192
    assert (claude_event["provider"], gemini_event["provider"]) == ("anthropic", "gemini")
    assert usage == {"service": "gemini", "model": gemini_event["model"], "status": "timeout",
                     "tokens_in": None, "tokens_out": None}


async def test_generate_with_usage_propagates_caller_cancellation(monkeypatch, capsys):
    started = asyncio.Event()

    async def wait():
        started.set()
        await asyncio.Event().wait()

    fake_claude(monkeypatch, wait=wait)
    task = asyncio.create_task(llm.generate_with_usage("p"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [event] = events(capsys)
    assert (event["provider"], event["outcome"]) == ("anthropic", "cancelled")
