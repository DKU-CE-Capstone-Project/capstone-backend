from fastapi.testclient import TestClient

from app.config import settings
from app.agents import news_fetcher
from app.agents.gdelt_client import build_gdelt_params
from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_gdelt_params_match_external_project_defaults() -> None:
    params = build_gdelt_params("semiconductor")

    assert params == {
        "query": "semiconductor sourcelang:korean",
        "mode": "artlist",
        "format": "json",
        "sort": "hybridrel",
        "maxrecords": 20,
        "timespan": "1d",
    }


async def test_fetch_news_uses_gdelt_article_list_only(monkeypatch) -> None:
    calls = {}

    async def fake_fetch_gdelt_articles(**kwargs):
        calls.update(kwargs)
        return [
            {
                "title": "Semiconductor supply chain update",
                "url": "https://example.com/news/1",
                "source_domain": "example.com",
                "published_at": "20260604T010203Z",
                "language": "Korean",
                "image_url": "https://example.com/image.jpg",
            }
        ]

    async def fail_newsapi(*args, **kwargs):
        raise AssertionError("NewsAPI should not be used when GDELT is enabled")

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "use_gdelt", True)
    monkeypatch.setattr(news_fetcher, "fetch_gdelt_articles", fake_fetch_gdelt_articles)
    monkeypatch.setattr(news_fetcher, "_fetch_newsapi", fail_newsapi)

    articles = await news_fetcher.fetch_news("반도체", page_size=5)

    assert calls == {
        "keyword": "semiconductor",
        "source_lang": "korean",
        "maxrecords": 20,
        "timespan": "1d",
    }
    assert articles == [
        {
            "title": "Semiconductor supply chain update",
            "url": "https://example.com/news/1",
            "source": "example.com",
            "published_at": "2026-06-04T01:02:03Z",
            "description": "Semiconductor supply chain update",
            "thumbnail_url": "https://example.com/image.jpg",
        }
    ]


def test_analyze_with_mocked_news(monkeypatch) -> None:
    """End-to-end: hits /analyze with mocked NewsAPI fixture and no Gemini key.
    Without an LLM key the summarizer/expander fall back to deterministic outputs,
    so we can still assert structure."""
    monkeypatch.setattr(settings, "use_mock_news", True)

    resp = client.post("/analyze", json={"keyword": "trump"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["keyword"] == "trump"

    articles = body["articles"]
    assert len(articles) >= 3, "expected several articles after dedup"

    # URL dedup actually fired (fixture has a duplicate URL)
    urls = [a["url"] for a in articles]
    assert len(urls) == len(set(urls)), "duplicate URL leaked through filter"

    # Title near-duplicates filtered (Reuters + Bloomberg headlines share most tokens)
    titles_lower = [a["title"].lower() for a in articles]
    tariff_pkg_titles = [t for t in titles_lower if "tariff package" in t]
    assert len(tariff_pkg_titles) <= 1, "near-duplicate tariff headlines not collapsed"

    for a in articles:
        for key in ("title", "url", "source", "summary", "published_at"):
            assert key in a

    related = body["related_keywords"]
    assert isinstance(related, list)
    assert len(related) >= 1
