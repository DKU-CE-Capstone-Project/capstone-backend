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


def test_source_endpoint_extracts_clicked_article_with_diffbot(monkeypatch) -> None:
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
    monkeypatch.setattr(news_routes, "extract_articles_with_diffbot", fake_extract_articles_with_diffbot)

    resp = client.get(f"/api/v1/news/{news_id}/source")

    assert resp.status_code == 200
    body = resp.json()
    assert body["original_body"] == "Diffbot body for clicked article."
    assert store.news_cache[news_id]["description"] == "Diffbot body for clicked article."


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


# ── 설계 스키마 매핑 (Notion "설계 › 데이터베이스 › 구조크") ────────────────────


def test_to_news_document_matches_design_schema() -> None:
    """앱 내부 dict → 설계 `news` 문서 변환이 validator 요구사항을 만족하는지."""
    from datetime import datetime

    from app.utils import to_news_document

    doc = to_news_document({
        "news_id": "abc123",
        "title": "Nvidia earnings beat",
        "url": "https://www.example.com/news/1?utm_source=x",
        "source": "Reuters",
        "published_at": "2026-06-04T01:02:03Z",
        "description": "본문",
        "summary": "한 줄 요약",
        "thumbnail_url": "https://example.com/a.jpg",
        "_search_keyword": "nvidia",
    })

    # mongo-init/01-collections.js의 news.required 와 일치해야 한다
    for field in (
        "news_id", "url", "title", "source", "status",
        "language", "is_deleted", "collected_at", "created_at", "updated_at",
    ):
        assert field in doc, f"required 필드 누락: {field}"

    # 설계는 source를 {name, domain} 객체로 정의한다 (앱 내부는 문자열)
    assert doc["source"] == {"name": "Reuters", "domain": "www.example.com"}
    # 설계의 published_at은 date 타입 — 문자열이면 validator가 거부한다
    assert isinstance(doc["published_at"], datetime)
    assert isinstance(doc["collected_at"], datetime)
    assert doc["status"] == "collected"
    assert doc["is_deleted"] is False
    assert doc["content"] == "본문"


def test_to_news_document_tolerates_unparsable_date() -> None:
    """발행일시 형식이 깨져도 문서 생성은 계속돼야 한다 (published_at은 null 허용)."""
    from app.utils import to_news_document

    doc = to_news_document({"url": "https://example.com/x", "published_at": "언제인지 모름"})
    assert doc["published_at"] is None


def test_from_news_document_restores_app_shape() -> None:
    """설계 문서 → 앱 내부 dict 복원 (캐시 미스 폴백 경로)."""
    from app.utils import from_news_document, to_news_document

    original = {
        "news_id": "abc123",
        "title": "제목",
        "url": "https://example.com/news/1",
        "source": "Bloomberg",
        "published_at": "2026-06-04T01:02:03+00:00",
        "description": "본문",
        "_search_keyword": "반도체",
    }
    restored = from_news_document(to_news_document(original))

    assert restored["news_id"] == "abc123"
    assert restored["title"] == "제목"
    assert restored["source"] == "Bloomberg"        # 객체 → 문자열로 되돌아온다
    assert restored["description"] == "본문"
    assert restored["_search_keyword"] == "반도체"
    assert restored["published_at"].startswith("2026-06-04T01:02:03")


async def test_resolve_news_falls_back_to_mongodb(monkeypatch) -> None:
    """in-memory 캐시에 없으면 MongoDB에서 복원하고 L1 캐시를 다시 채운다.

    api 프로세스 재시작 후 GET /api/v1/news/{id}/* 가 404 나던 문제를 막는 경로.
    """
    from app import database
    from app.utils import resolve_news, to_news_document

    news_id = "restored001"
    store.news_cache.pop(news_id, None)

    stored = to_news_document({
        "news_id": news_id,
        "title": "복원된 기사",
        "url": "https://example.com/restored",
        "source": "Example",
        "published_at": "2026-06-04T01:02:03Z",
        "description": "본문",
        "_search_keyword": "복원",
    })

    async def fake_get_news(nid: str):
        return stored if nid == news_id else None

    monkeypatch.setattr(database, "get_news", fake_get_news)

    art = await resolve_news(news_id)

    assert art is not None
    assert art["title"] == "복원된 기사"
    # L1 캐시에 다시 적재되어 다음 조회는 DB를 타지 않는다
    assert store.news_cache[news_id]["title"] == "복원된 기사"


async def test_resolve_news_returns_none_when_absent(monkeypatch) -> None:
    from app import database
    from app.utils import resolve_news

    async def fake_get_news(nid: str):
        return None

    monkeypatch.setattr(database, "get_news", fake_get_news)
    store.news_cache.pop("nope", None)

    assert await resolve_news("nope") is None


def test_relations_endpoint_persists_to_news_relations(monkeypatch) -> None:
    """연관도 점수를 계산하면 설계 news_relations 스키마로 저장돼야 한다.

    12주차 회의에서 "릴레이션까지는 저장하자"(매 요청 재계산 방지)고 결정된 항목.
    """
    from app import database

    src = {
        "news_id": "rel_src",
        "title": "Nvidia HBM supply expands for AI servers",
        "description": "Nvidia HBM supply",
        "url": "https://example.com/rel-src",
    }
    tgt = {
        "news_id": "rel_tgt",
        "title": "HBM supply shortage hits AI servers",
        "description": "HBM supply shortage",
        "url": "https://example.com/rel-tgt",
    }
    store.news_cache["rel_src"] = src
    store.news_cache["rel_tgt"] = tgt

    saved: list[dict] = []

    async def fake_save_relation(doc):
        saved.append(doc)

    monkeypatch.setattr(database, "save_relation", fake_save_relation)

    resp = client.get(
        "/api/v1/news/rel_src/relations?target_news_ids=rel_tgt&tier=PAID"
    )
    assert resp.status_code == 200

    assert len(saved) == 1, "연관도 저장이 호출되지 않았다"
    doc = saved[0]
    # mongo-init/01-collections.js의 news_relations.required 와 일치해야 한다
    for field in ("source_news_id", "target_news_id", "relation", "created_at", "updated_at"):
        assert field in doc, f"required 필드 누락: {field}"
    assert doc["source_news_id"] == "rel_src"
    assert doc["target_news_id"] == "rel_tgt"
    # relation.type은 설계의 6종 enum 중 하나여야 한다
    assert doc["relation"]["type"] in {
        "same_topic", "cause_effect", "same_company",
        "same_industry", "opposite_view", "follow_up",
    }
    assert doc["relation"]["score"] > 0
    assert isinstance(doc["shared_keywords"], list)


def test_relations_endpoint_skips_unrelated_pair(monkeypatch) -> None:
    """겹치는 토큰이 없으면 관계로 저장하지 않는다."""
    from app import database

    store.news_cache["norel_a"] = {
        "news_id": "norel_a", "title": "Nvidia earnings", "description": "",
        "url": "https://example.com/a",
    }
    store.news_cache["norel_b"] = {
        "news_id": "norel_b", "title": "제주 감귤 작황", "description": "",
        "url": "https://example.com/b",
    }

    saved: list[dict] = []

    async def fake_save_relation(doc):
        saved.append(doc)

    monkeypatch.setattr(database, "save_relation", fake_save_relation)

    resp = client.get(
        "/api/v1/news/norel_a/relations?target_news_ids=norel_b&tier=PAID"
    )
    assert resp.status_code == 200
    assert saved == []


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


def test_graph_records_center_in_session(monkeypatch) -> None:
    """그래프를 조회하면 세션에 중심 노드가 기록된다."""
    monkeypatch.setattr(settings, "use_mongodb", False)

    for nid, title in [("g_center", "반도체 공급망"), ("g_rel1", "반도체 수출"), ("g_rel2", "반도체 investment"), ("g_rel3", "반도체 memory")]:
        store.news_cache[nid] = {
            "news_id": nid, "title": title, "url": f"https://example.com/{nid}",
            "description": "", "_search_keyword": "그래프세션",
        }

    fresh = TestClient(app)
    resp = fresh.get("/api/v1/news/g_center/graph?limit=5")
    assert resp.status_code == 200

    state = fresh.get("/api/v1/session").json()
    assert state["mindmap"]["center_news_id"] == "g_center"
    assert "g_center" in state["viewed_news_ids"]


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


def test_expanded_node_adds_new_nodes_to_graph(monkeypatch) -> None:
    """확장한 노드의 이웃이 실제로 그래프에 추가돼야 한다.

    회귀 방지: _related_articles는 '같은 검색어로 수집된 기사'를 캐시 순서대로
    돌려주기 때문에, 그대로 쓰면 기본 그래프가 이미 가져간 상위 N개와 겹쳐
    확장이 아무 노드도 추가하지 못한다. 확장 노드 기준으로 재정렬해야 한다.
    """
    monkeypatch.setattr(settings, "use_mongodb", False)

    # 같은 검색어 풀에 기사를 충분히 넣어, limit이 작으면 일부가 화면 밖에 남게 한다
    titles = {
        "exp_center": "Tariff package shakes markets",
        "exp_a": "Tariff package detail revealed",
        "exp_b": "Markets slide on tariff news",
        "exp_c": "Oil prices climb amid tension",
        "exp_d": "Semiconductor exports under review",
        "exp_e": "Won weakens against dollar",
    }
    for nid, title in titles.items():
        store.news_cache[nid] = {
            "news_id": nid, "title": title, "description": title,
            "url": f"https://example.com/{nid}", "_search_keyword": "확장테스트",
        }

    c = TestClient(app)
    base = c.get("/api/v1/news/exp_center/graph?limit=2").json()
    base_ids = {n["news_id"] for n in base["nodes"]}
    assert len(base_ids) == 3, "중심 + 관련 2개"

    # 화면에 이미 있는 노드 하나를 펼친다
    target = next(nid for nid in base_ids if nid != "exp_center")
    c.post("/api/v1/session/mindmap/expand", json={"news_id": target})

    after = c.get("/api/v1/news/exp_center/graph?limit=2").json()
    after_ids = {n["news_id"] for n in after["nodes"]}

    assert len(after_ids) > len(base_ids), "확장했는데 노드가 늘지 않았다"
    expanded_edges = [e for e in after["edges"] if e["relation_type"] == "expanded"]
    assert expanded_edges, "expanded 엣지가 없다"
    assert all(e["source"] == target for e in expanded_edges)


def test_depth_one_skips_expansion() -> None:
    """depth=1이면 중심 + 1홉만 보여주고 확장은 적용하지 않는다."""
    c = TestClient(app)
    store.news_cache["d1_center"] = {
        "news_id": "d1_center", "title": "중심", "description": "",
        "url": "https://example.com/d1c", "_search_keyword": "깊이",
    }
    for i in range(4):
        store.news_cache[f"d1_{i}"] = {
            "news_id": f"d1_{i}", "title": f"중심 관련 {i}", "description": "",
            "url": f"https://example.com/d1-{i}", "_search_keyword": "깊이",
        }

    c.get("/api/v1/news/d1_center/graph?limit=2")
    c.post("/api/v1/session/mindmap/expand", json={"news_id": "d1_0"})

    g = c.get("/api/v1/news/d1_center/graph?limit=2&depth=1").json()
    assert not [e for e in g["edges"] if e["relation_type"] == "expanded"]
