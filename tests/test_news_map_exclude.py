"""exclude_ids: child expansion and re-search never return articles already on the map."""
import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import llm
from app.api.v1 import news
from app.main import app


@pytest.fixture
def client(monkeypatch):
    articles = [
        {
            "news_id": str(i), "title": f"AI GPU 공급 계약 product{i}",
            "description": f"새 GPU 공급 계약의 검색 설명 {i}. " * 20,
            "url": f"https://publisher.test/news/{i}",
            "source": "publisher.test", "published_at": "2026-09-30T09:00:00Z",
            "keywords": ["AI", "GPU"], "categories": ["반도체"],
            "_search_keyword": "AI",
        }
        for i in range(10)
    ]
    store.news_cache.update({a["news_id"]: a for a in articles})

    async def embed(text, **kwargs):
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)

    async def cached_search(keyword):
        return articles

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news, "_fetch_and_cache", cached_search)
    return TestClient(app)


def related_ids(client, **params):
    response = client.get("/api/v1/news/0/related", params=params)
    assert response.status_code == 200
    return response.json()


def test_excluded_articles_are_never_returned_and_are_counted(client):
    first = related_ids(client)
    shown = [a["news_id"] for a in first["related_news"]]
    assert len(shown) == 3 and first["selection"]["excluded"] == 0

    again = related_ids(client, exclude_ids=",".join(shown))
    ids = [a["news_id"] for a in again["related_news"]]
    assert len(ids) == 3 and not set(ids) & set(shown)
    assert again["selection"]["excluded"] == 3


def test_same_exclusions_give_the_same_result(client):
    a = related_ids(client, exclude_ids="1,2")
    b = related_ids(client, exclude_ids="2,1,2")
    assert [x["news_id"] for x in a["related_news"]] == [x["news_id"] for x in b["related_news"]]


def test_exhausted_candidates_are_a_normal_empty_result(client):
    everything = ",".join(str(i) for i in range(1, 10))
    data = related_ids(client, exclude_ids=everything)
    assert data["related_news"] == []
    assert data["selection"]["returned"] == 0 and data["selection"]["status"] in {"insufficient", "partial"}


def test_center_and_unknown_ids_are_ignored(client):
    data = related_ids(client, exclude_ids="0,missing,,")
    assert len(data["related_news"]) == 3 and data["selection"]["excluded"] == 0


def test_too_many_exclusions_are_rejected(client):
    too_many = ",".join(f"x{i}" for i in range(101))
    assert client.get("/api/v1/news/0/related", params={"exclude_ids": too_many}).status_code == 422
