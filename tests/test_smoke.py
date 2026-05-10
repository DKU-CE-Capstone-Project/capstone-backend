from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_with_mocked_news() -> None:
    """End-to-end: hits /analyze with mocked NewsAPI fixture and no Gemini key.
    Without an LLM key the summarizer/expander fall back to deterministic outputs,
    so we can still assert structure."""
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
