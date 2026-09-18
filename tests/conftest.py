"""Offline defaults: tests never use developer credentials or production services."""

import socket

import pytest

from app import store
from app.agents import article_metadata, diffbot_client, naver_categories
from app.config import settings


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    for key, value in {
        "google_api_key": "",
        "anthropic_api_key": "",
        "newsapi_key": "",
        "naver_client_id": "",
        "naver_client_secret": "",
        "news_provider": "gdelt",
        "diffbot_token": "",
        "diffbot_api_key": "",
        "use_mongodb": False,
        "mongodb_required": False,
        "mongodb_uri": "",
        "use_mock_news": True,
        "demo_mode": False,
        "use_llm_summaries": False,
        "use_rag": False,
        "use_critic": False,
    }.items():
        monkeypatch.setattr(settings, key, value)

    def reject_network(*args, **kwargs):
        raise AssertionError("External network access is forbidden in unit tests")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    article_metadata._cache.clear()
    diffbot_client._extraction_cache.clear()
    naver_categories._cache.clear()
    naver_categories._images.clear()
    monkeypatch.delenv("DIFFBOT_TOKEN", raising=False)
    monkeypatch.delenv("DIFFBOT_API_KEY", raising=False)
    for cache in (
        store.news_cache,
        store.summary_cache,
        store.report_cache,
        store.strategy_cache,
        store.selection_cache,
        store.report_index,
    ):
        cache.clear()
    yield
    article_metadata._cache.clear()
    store.news_cache.clear()


@pytest.fixture
def metadata_ai(monkeypatch):
    # Tests enabling AI must stub article_metadata.generate; sockets remain blocked.
    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "use_llm_metadata", True)
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", "unit-test-not-a-real-key")
