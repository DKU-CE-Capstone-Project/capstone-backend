"""GET /keywords/related: search-article keyword frequency, recommended fill-in, no LLM."""
import pytest
from fastapi.testclient import TestClient

from app.agents.naver_client import NaverNewsError
from app.api.v1 import keywords as keywords_api
from app.main import app
from app.schemas import NewsCard, SearchResponse


def card(i, keywords):
    return NewsCard(news_id=str(i), title=f"t{i}", summary="", thumbnail_url="", source_name="s",
                    published_at="2026-09-30T09:00:00Z", keywords=keywords)


@pytest.fixture
def search(monkeypatch):
    def install(cards=None, error=None):
        async def fake(*, q, page, size, sort):
            assert (page, size, sort) == (1, 20, "relevance")
            if error:
                raise error
            return SearchResponse(news_cards=cards or [], total_count=len(cards or []))
        monkeypatch.setattr(keywords_api, "search_news", fake)
    return install


def test_frequent_article_keywords_come_first_and_query_is_excluded(search):
    search([
        card(0, ["원유", "호르무즈", "정유"]),
        card(1, ["호르무즈", "현대차", "해운"]),
        card(2, ["호르무즈", "정유", "현대자동차"]),
        card(3, ["단독 키워드"]),
    ])
    data = TestClient(app).get("/api/v1/keywords/related", params={"q": " 원유 ", "limit": 5}).json()
    assert data["query"] == "원유" and data["article_total"] == 4
    picked = [(k["keyword"], k["article_count"], k["source"]) for k in data["keywords"]]
    assert picked[:3] == [("호르무즈", 3, "articles"), ("정유", 2, "articles"), ("현대자동차", 2, "articles")]
    assert all(source == "recommended" and count == 0 for _, count, source in picked[3:])
    assert len(picked) == 5 and "원유" not in [k for k, _, _ in picked]


def test_no_articles_returns_recommended_only(search):
    search([])
    data = TestClient(app).get("/api/v1/keywords/related", params={"q": "없는검색어"}).json()
    assert data["article_total"] == 0
    assert data["keywords"] and all(k["source"] == "recommended" for k in data["keywords"])


def test_search_failure_is_502_not_a_fallback(search):
    search(error=NaverNewsError("down"))
    assert TestClient(app).get("/api/v1/keywords/related", params={"q": "원유"}).status_code == 502


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x" * 101}, {"q": "a", "limit": 13}])
def test_invalid_input_is_422(params):
    assert TestClient(app).get("/api/v1/keywords/related", params=params).status_code == 422
