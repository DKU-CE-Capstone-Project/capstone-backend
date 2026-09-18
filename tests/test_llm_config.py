from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents import llm
from app.config import Settings, settings


def test_gemini_flex_defaults(monkeypatch):
    for key in (
        "GEMINI_MODEL",
        "GEMINI_SERVICE_TIER",
        "GEMINI_TIMEOUT_SECONDS",
        "METADATA_TIMEOUT_SECONDS",
        "LLM_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    config = Settings(_env_file=None)
    assert config.gemini_model == "gemini-3.5-flash-lite"
    assert config.gemini_service_tier == "flex"
    assert config.llm_provider == "gemini"
    assert config.gemini_timeout_seconds == 600
    assert config.metadata_timeout_seconds > config.gemini_timeout_seconds


def test_invalid_tier_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gemini_service_tier="flxe")


async def test_generation_sends_flex_and_timeout_even_with_claude_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "unit-test-claude-key")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "gemini_service_tier", "flex")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 600)
    calls = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=" extracted result ")

    async def forbidden_claude(prompt):
        pytest.fail("Gemini selection must not call Claude")

    monkeypatch.setattr(
        llm,
        "_client",
        lambda: SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        ),
    )
    monkeypatch.setattr(llm, "_generate_claude", forbidden_claude)
    assert await llm.generate("test article") == "extracted result"
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-3.5-flash-lite"
    config = calls[0]["config"]
    assert config.service_tier.value == "flex"
    assert config.http_options.timeout == 600000
    assert config.http_options.retry_options.http_status_codes == [429, 503]


async def test_flex_failure_does_not_switch_to_standard_or_claude(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "unit-test-claude-key")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_service_tier", "flex")
    calls = []

    async def unavailable(**kwargs):
        calls.append(kwargs["config"].service_tier.value)
        raise RuntimeError("503 capacity unavailable")

    async def forbidden_claude(prompt):
        pytest.fail("Flex failure must not switch providers")

    monkeypatch.setattr(
        llm,
        "_client",
        lambda: SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=unavailable))
        ),
    )
    monkeypatch.setattr(llm, "_generate_claude", forbidden_claude)
    assert await llm.generate("test article") == ""
    assert calls == ["flex"]
