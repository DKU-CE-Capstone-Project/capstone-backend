import json

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import diffbot_client, naver_categories, news_fetcher, report_generator
from app.agents.naver_client import NaverNewsPage
from app.api.v1 import reports
from app.config import settings
from app.main import app

URL = "https://n.news.naver.com/article/001/123"


@pytest.fixture
def detail_flow(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "diffbot_token", "test")

    async def fetch(*args, **kwargs):
        return NaverNewsPage(
            [
                {
                    "url": "https://example.com/article",
                    "naver_url": URL,
                    "title": "HBM 투자",
                    "description": "네이버 설명",
                    "summary": "네이버 설명",
                    "source": "example.com",
                    "published_at": "2026-09-18T00:00:00Z",
                    "_news_provider": "naver",
                }
            ],
            1,
        )

    async def categories(url):
        return ["경제"]

    monkeypatch.setattr(news_fetcher, "fetch_naver_page", fetch)
    monkeypatch.setattr(naver_categories, "fetch_categories", categories)
    monkeypatch.setattr(
        naver_categories, "page_image", lambda url: "https://example.com/photo.jpg"
    )


def test_diffbot_only_on_report_and_details_remain_description_only(monkeypatch, detail_flow):
    calls, reports_seen = [], []

    def extract(**kwargs):
        calls.append(kwargs["url"])
        return {
            "objects": [
                {
                    "text": "전체 본문\n마지막 근거 문장",
                    "meta": {"og": {"og:image": "https://example.com/detail.jpg"}},
                }
            ]
        }

    async def generate(center, related):
        reports_seen.append(center["cleaned_content"])
        return {
            "title": "리포트",
            "summary": "요약",
            "event_analysis": "분석",
            "market_impact": "영향",
            "risk_factors": [],
        }

    monkeypatch.setattr(diffbot_client, "call_diffbot_article", extract)
    monkeypatch.setattr(reports, "generate_report", generate)
    client = TestClient(app)
    result = client.get("/api/v1/news/search?q=HBM").json()
    card = result["news_cards"][0]
    nid = card["news_id"]
    assert calls == []
    assert card["summary"] == "네이버 설명"
    assert card["source_url"] == URL
    assert "cleaned_content" not in store.news_cache[nid]
    detail = client.get(f"/api/v1/news/{nid}/source").json()
    assert detail["description"] == "네이버 설명"
    assert detail["original_body"] == ""
    assert calls == reports_seen == []
    assert card["description"] == "네이버 설명"
    response = client.post("/api/v1/reports", json={"news_id": nid})
    assert response.status_code == 201
    assert reports_seen == ["전체 본문\n마지막 근거 문장"]
    assert calls == [URL]
    # Full body stays internal even after report generation.
    detail = client.get(f"/api/v1/news/{nid}/source").json()
    assert detail["description"] == "네이버 설명" and detail["original_body"] == ""
    assert client.post("/api/v1/reports", json={"news_id": nid}).status_code == 201
    assert calls == [URL]


def test_failed_report_extraction_does_not_block_description(monkeypatch, detail_flow):
    def fail(**kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(diffbot_client, "call_diffbot_article", fail)
    client = TestClient(app)
    nid = client.get("/api/v1/news/search?q=HBM").json()["news_cards"][0]["news_id"]
    assert client.get(f"/api/v1/news/{nid}/source").status_code == 200
    assert client.post("/api/v1/reports", json={"news_id": nid}).status_code == 502


async def test_report_prompt_includes_body_not_only_description(monkeypatch):
    prompts = []

    async def generate(prompt):
        prompts.append(prompt)
        return json.dumps({"title": "보고서", "summary": "요약"})

    monkeypatch.setattr(report_generator, "generate", generate)
    await report_generator.generate_report(
        {
            "title": "뉴스",
            "description": "짧은 설명",
            "cleaned_content": "본문에만 있는 사실과 마지막 문장",
        },
        [],
    )
    assert "본문에만 있는 사실과 마지막 문장" in prompts[0]
