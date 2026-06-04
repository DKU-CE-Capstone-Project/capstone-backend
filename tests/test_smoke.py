from fastapi.testclient import TestClient

from app.config import settings
from app.agents import diffbot_client
from app.agents import news_fetcher
from app.agents.graph_builder import build_graph
from app.api.v1.news import _to_news_card
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


def test_news_api_summary_prefers_title() -> None:
    article = {
        "news_id": "n1",
        "title": "Displayed title",
        "summary": "Generated summary",
        "description": "Diffbot body",
        "url": "https://example.com/news/1",
    }

    card = _to_news_card(article)
    graph = build_graph(article, [article])

    assert card.summary == "Displayed title"
    assert graph["center_node"]["summary"] == "Displayed title"
    assert graph["nodes"][1]["summary"] == "Displayed title"


async def test_fetch_news_uses_gdelt_and_diffbot_content_when_available(monkeypatch) -> None:
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

    async def fake_extract_articles_with_diffbot(articles, **kwargs):
        return [{**articles[0], "cleaned_content": "Full article body from Diffbot."}]

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "use_gdelt", True)
    monkeypatch.setattr(news_fetcher, "fetch_gdelt_articles", fake_fetch_gdelt_articles)
    monkeypatch.setattr(news_fetcher, "extract_articles_with_diffbot", fake_extract_articles_with_diffbot)
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
            "description": "Full article body from Diffbot.",
            "thumbnail_url": "https://example.com/image.jpg",
        }
    ]


async def test_diffbot_extraction_returns_original_articles_without_token(monkeypatch) -> None:
    articles = [{"title": "A", "url": "https://example.com/a"}]

    def fail_extract(*args, **kwargs):
        raise AssertionError("Diffbot should not be called without a token")

    monkeypatch.setattr(diffbot_client, "load_diffbot_token", lambda: "")
    monkeypatch.setattr(diffbot_client, "_extract_one_sync", fail_extract)

    assert await diffbot_client.extract_articles_with_diffbot(articles) == articles


async def test_diffbot_extraction_adds_cleaned_content(monkeypatch) -> None:
    articles = [{"title": "A", "url": "https://example.com/a"}]

    def fake_call_diffbot_article(**kwargs):
        return {"objects": [{"text": " First paragraph. \n\n Second paragraph. "}]}

    monkeypatch.setattr(diffbot_client, "call_diffbot_article", fake_call_diffbot_article)

    extracted = await diffbot_client.extract_articles_with_diffbot(
        articles,
        token="token",
        concurrency=1,
        max_retries=0,
    )

    assert extracted[0]["cleaned_content"] == "First paragraph.\nSecond paragraph."
    assert extracted[0]["cleaned_content_length"] == len("First paragraph.\nSecond paragraph.")


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
