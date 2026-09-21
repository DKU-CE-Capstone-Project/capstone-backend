from fastapi.testclient import TestClient

from app import store
from app.config import settings
from app.agents import diffbot_client
from app.agents import news_fetcher
from app.agents.graph_builder import build_graph
from app.api.v1 import news as news_routes
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
        "maxrecords": 10,
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


def test_search_endpoint_preserves_gdelt_thumbnail(monkeypatch) -> None:
    async def fake_fetch_news(keyword: str, page_size: int = 10):
        return [
            {
                "title": "Nvidia thumbnail story",
                "url": "https://example.com/nvidia-thumbnail",
                "source": "example.com",
                "published_at": "2026-06-04T01:02:03Z",
                "description": "Nvidia thumbnail story",
                "thumbnail_url": "https://example.com/real-thumbnail.jpg",
            }
        ]

    monkeypatch.setattr(news_routes, "fetch_news", fake_fetch_news)

    resp = client.get("/api/v1/news/search?q=엔비디아&page=1&size=10")

    assert resp.status_code == 200
    card = resp.json()["news_cards"][0]
    assert card["thumbnail_url"] == "https://example.com/real-thumbnail.jpg"


async def test_fetch_news_uses_gdelt_list_without_diffbot(monkeypatch) -> None:
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
        "maxrecords": 10,
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


def test_source_endpoint_returns_description_without_diffbot(monkeypatch) -> None:
    news_id = "clicked-news"
    store.news_cache[news_id] = {
        "news_id": news_id,
        "title": "Clicked title",
        "url": "https://example.com/clicked",
        "source": "example.com",
        "published_at": "20260604T010203Z",
        "description": "Clicked title",
    }

    async def fake_extract_articles_with_diffbot(articles, **kwargs):
        return [{**articles[0], "cleaned_content": "Diffbot body for clicked article."}]

    monkeypatch.setattr(settings, "use_mongodb", False)
    monkeypatch.setattr(diffbot_client, "extract_articles_with_diffbot", fake_extract_articles_with_diffbot)

    resp = client.get(f"/api/v1/news/{news_id}/source")

    assert resp.status_code == 200
    body = resp.json()
    assert body["original_body"] == ""
    assert body["description"] == "Clicked title"
    assert store.news_cache[news_id]["description"] == "Clicked title"


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


# ── 쿠키 기반 세션 식별 (12주차 회의 결정) ────────────────────────────────────


def test_session_cookie_is_issued_and_reused() -> None:
    """첫 요청에 세션 쿠키가 발급되고, 이후 요청에서는 같은 세션이 유지된다."""
    from app.session import SESSION_COOKIE

    fresh = TestClient(app)
    r1 = fresh.get("/api/v1/session")
    assert r1.status_code == 200
    assert SESSION_COOKIE in r1.cookies, "세션 쿠키가 발급되지 않았다"

    sid1 = r1.json()["session_id"]
    assert sid1

    # 같은 클라이언트(=같은 브라우저)는 같은 세션을 유지한다
    r2 = fresh.get("/api/v1/session")
    assert r2.json()["session_id"] == sid1


def test_session_cookie_is_httponly() -> None:
    """세션 쿠키는 JS에서 읽을 수 없어야 한다 (XSS로 탈취 방지)."""
    from app.session import SESSION_COOKIE

    fresh = TestClient(app)
    resp = fresh.get("/api/v1/session")
    set_cookie = resp.headers.get("set-cookie", "")
    assert SESSION_COOKIE in set_cookie
    assert "HttpOnly" in set_cookie


def test_two_sessions_have_independent_mindmaps() -> None:
    """서로 다른 사용자는 서로 다른 마인드맵을 본다.

    12주차 회의: "김성민 씨랑 문주원 씨랑은 다른 마인드맵이 보여야 되잖아요"
    로그인이 없으므로 이 구분은 오직 세션 쿠키로만 이뤄진다.
    """
    store.news_cache["sess_a"] = {
        "news_id": "sess_a", "title": "A 기사", "url": "https://example.com/a",
        "description": "", "_search_keyword": "세션",
    }
    store.news_cache["sess_b"] = {
        "news_id": "sess_b", "title": "B 기사", "url": "https://example.com/b",
        "description": "", "_search_keyword": "세션",
    }

    user1 = TestClient(app)
    user2 = TestClient(app)

    # 서로 다른 세션 id를 받는다
    sid1 = user1.get("/api/v1/session").json()["session_id"]
    sid2 = user2.get("/api/v1/session").json()["session_id"]
    assert sid1 != sid2, "두 클라이언트가 같은 세션을 공유하면 사용자 구분이 안 된다"

    # user1만 노드를 펼친다
    r = user1.post("/api/v1/session/mindmap/expand", json={"news_id": "sess_b"})
    assert r.status_code == 200
    assert r.json()["mindmap"]["expanded_news_ids"] == ["sess_b"]

    # user2의 마인드맵은 영향을 받지 않는다
    assert user2.get("/api/v1/session").json()["mindmap"]["expanded_news_ids"] == []


def test_mindmap_expand_and_collapse_roundtrip() -> None:
    fresh = TestClient(app)
    fresh.get("/api/v1/session")

    fresh.post("/api/v1/session/mindmap/expand", json={"news_id": "n1"})
    body = fresh.post("/api/v1/session/mindmap/expand", json={"news_id": "n2"}).json()
    assert body["mindmap"]["expanded_news_ids"] == ["n1", "n2"]

    body = fresh.post("/api/v1/session/mindmap/collapse", json={"news_id": "n1"}).json()
    assert body["mindmap"]["expanded_news_ids"] == ["n2"]

    body = fresh.request("DELETE", "/api/v1/session/mindmap").json()
    assert body["mindmap"]["expanded_news_ids"] == []


def test_changing_center_resets_expansion() -> None:
    """중심 노드가 바뀌면 이전 확장 상태는 초기화된다."""
    import asyncio

    from app import session as session_store

    sid = "test-sid-reset"
    asyncio.get_event_loop_policy().new_event_loop()

    async def scenario():
        await session_store.record_center(sid, "center1")
        await session_store.expand_node(sid, "x1")
        state = await session_store.load(sid)
        assert state["mindmap"]["expanded_news_ids"] == ["x1"]

        await session_store.record_center(sid, "center2")
        state = await session_store.load(sid)
        assert state["mindmap"]["center_news_id"] == "center2"
        assert state["mindmap"]["expanded_news_ids"] == []
        await session_store.clear(sid)

    asyncio.run(scenario())



# 보류(M2 완료 기준 제외): test_graph_records_center_in_session ·
# test_expanded_node_adds_new_nodes_to_graph · test_depth_one_skips_expansion 은
# app/api/v1/news.py 의 마인드맵 확장 포팅에 딸린 테스트다. 그 포팅이
# 2026-09-19 보류 결정이라 함께 보류한다. 세션 저장소 자체(record_center 포함)는
# 위 test_changing_center_resets_expansion 이 덮는다.
