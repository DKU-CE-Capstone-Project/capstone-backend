"""Article details survive every news response, with one graph/related policy."""
import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import llm
from app.api.v1 import news
from app.main import app


@pytest.fixture
def news_client(monkeypatch):
    articles = [
        {
            "news_id": str(i), "title": f"AI GPU 공급 계약 product{i}",
            "description": f"새 GPU 공급 계약의 검색 설명 {i}. " * 20,
            "summary": "Generated summary should not replace the description",
            "url": f"https://publisher.test/news/{i}",
            "naver_url": f"https://n.news.naver.com/mnews/article/214/{i}",
            "source": "publisher.test", "published_at": "2026-09-30T09:00:00Z",
            "thumbnail_url": f"https://publisher.test/images/news-{i}.jpg",
            "keywords": ["AI", "GPU"], "categories": ["반도체"],
            "_search_keyword": "AI",
        }
        for i in range(7)
    ]
    store.news_cache.update({a["news_id"]: a for a in articles})

    async def embed(text, **kwargs):
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)

    async def cached_search(keyword):
        return articles

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news, "_fetch_and_cache", cached_search)
    return TestClient(app), articles


def test_search_related_and_graph_return_complete_matching_cards(news_client):
    client, articles = news_client
    search = client.get("/api/v1/news/search?q=AI").json()["news_cards"]
    related = client.get("/api/v1/news/0/related?tier=PAID&limit=6").json()["related_news"]
    graph = client.get("/api/v1/news/0/graph?tier=PAID&limit=6").json()
    assert graph["center_node"] == graph["nodes"][0]
    assert graph["center_node"]["relevance_score"] is None
    assert graph["center_node"]["distance"] == 0
    assert len(related) == len(graph["edges"]) == 6
    for card in [*search, *related, *graph["nodes"]]:
        original = articles[int(card["news_id"])]
        assert card["title"] == original["title"]
        assert card["description"] == card["summary"] == original["description"]
        assert card["source_name"] == "publisher.test"
        assert card["source_url"] == original["naver_url"]
        assert card["published_at"] == original["published_at"]
        assert card["thumbnail_url"] == original["thumbnail_url"]
        assert card["keywords"] == ["AI", "GPU"]
        assert card["categories"] == ["반도체"]
    for item, node, edge in zip(related, graph["nodes"][1:], graph["edges"]):
        assert item == {key: value for key, value in node.items() if key != "is_center"}
        assert edge["source"] == "0" and edge["target"] == item["news_id"]
        assert edge["distance"] == item["distance"] == 1


@pytest.mark.parametrize("tier", [None, "FREE", "BASIC", "PAID"])
@pytest.mark.parametrize("limit", [1, 6, 50])
def test_both_endpoints_apply_tier_limit_and_score_visibility(news_client, tier, limit):
    client, _ = news_client
    params = {"limit": limit, "include_score": "true"}
    if tier is not None:
        params["tier"] = tier
    graph = client.get("/api/v1/news/0/graph", params=params)
    related = client.get("/api/v1/news/0/related", params=params)
    assert graph.status_code == related.status_code == 200
    nodes, items = graph.json()["nodes"][1:], related.json()["related_news"]
    assert len(nodes) == len(items) == min(limit, 6 if tier == "PAID" else 3)
    assert [n["news_id"] for n in nodes] == [n["news_id"] for n in items]
    assert [n["relevance_score"] for n in nodes] == [n["relevance_score"] for n in items]
    assert all((n["relevance_score"] is not None) == (tier == "PAID") for n in nodes)


def test_graph_uses_requested_threshold_and_rejects_invalid_policy(news_client):
    client, _ = news_client
    for endpoint in ("graph", "related"):
        response = client.get(f"/api/v1/news/0/{endpoint}?min_relevance=1&tier=PAID")
        assert response.status_code == 200
        data = response.json()
        assert (data["nodes"][1:] if endpoint == "graph" else data["related_news"]) == []
        for query in ("tier=INVALID", "limit=51", "min_relevance=1.1"):
            assert client.get(f"/api/v1/news/0/{endpoint}?{query}").status_code == 422


def test_related_response_is_documented_in_openapi(news_client):
    client, _ = news_client
    spec = client.get("/openapi.json").json()
    response = spec["paths"]["/api/v1/news/{news_id}/related"]["get"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/RelatedResponse")
