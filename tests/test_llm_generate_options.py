"""llm.generate options: telemetry stage, per-call output cap, returned usage."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app import telemetry
from app.agents import llm
from app.config import settings


def events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"event_type": "econmind_usage"')]


def fake_gemini(monkeypatch, *, usage=None, text=" ok ", error=None):
    calls = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(text=text, usage_metadata=usage)

    monkeypatch.setattr(settings, "google_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))))
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


async def test_generate_with_usage_reports_timeout(monkeypatch):
    async def hang(prompt, *, max_output_tokens=None):
        await asyncio.sleep(1)

    monkeypatch.setattr(settings, "gemini_timeout_seconds", 0.01)
    monkeypatch.setattr(llm, "_generate_with_provider", hang)
    text, usage = await llm.generate_with_usage("p")
    assert text == ""
    assert (usage["service"], usage["status"]) == ("gemini", "timeout")
