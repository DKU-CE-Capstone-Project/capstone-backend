"""Opt-in local MongoDB integration with mocked Gemini, never developer/production DB."""
import os
import socket
import uuid
from urllib.parse import urlsplit

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from app import database, store
from app.agents import article_embeddings as embeddings
from app.agents import llm, news_fetcher, news_map, related_candidates
from app.api.v1 import news
from app.config import settings
from app.utils import cache_articles

_REAL_CONNECT = socket.socket.connect


@pytest.fixture
async def local_mongo(monkeypatch):
    uri = os.getenv("NEWS_MAP_TEST_MONGODB_URI", "")
    if not uri:
        pytest.skip("Set NEWS_MAP_TEST_MONGODB_URI for the isolated local test instance")
    address = urlsplit(uri)
    assert address.scheme == "mongodb" and address.hostname == "127.0.0.1"
    assert address.port == 27028 and address.username is None and address.password is None

    def local_only(sock, target):
        assert target[:2] == ("127.0.0.1", 27028), "Only the isolated test MongoDB is allowed"
        return _REAL_CONNECT(sock, target)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=3000)
    name = "econmind_news_map_test_" + uuid.uuid4().hex
    db = client[name]
    await client.admin.command("ping")
    await db.create_collection(database.NEWS, validator={"$jsonSchema": {
        "bsonType": "object", "required": ["title", "url", "source", "news_id"],
        "properties": {"source": {"bsonType": "object"}, "embedding": {"bsonType": "array"}},
    }})
    await db[database.NEWS].create_index("url", unique=True)
    await db[database.NEWS].create_index("news_id", unique=True)
    monkeypatch.setattr(database, "_get_db", lambda: db)
    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "mongodb_required", True)
    try:
        yield db
    finally:
        assert name.startswith("econmind_news_map_test_")
        await client.drop_database(name)
        client.close()


async def test_local_mongo_reuse_refresh_and_candidate_reload(local_mongo, monkeypatch):
    calls = []
    async def embed(text, **kwargs):
        calls.append((text, kwargs))
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)
    monkeypatch.setattr(llm, "embed", embed)
    raw = {"title": "AI GPU 발표", "description": "새 GPU 제품 출시", "url": "https://example.test/first", "_search_keyword": "AI"}
    article = (await cache_articles([raw]))[0]
    await embeddings.article_vector(article)
    assert len(calls) == 2  # Separate RAG + map purposes.
    saved = await local_mongo[database.NEWS].find_one({"news_id": article["news_id"]})
    assert saved["news_map_embedding"]["metadata"]["task_type"] == "SEMANTIC_SIMILARITY"
    assert saved["embedding_metadata"]["purpose"] == "rag"

    store.news_cache.clear()
    embeddings._cache.clear()
    loaded = await store.get_news(article["news_id"])
    await embeddings.article_vector(loaded)
    embeddings._cache.clear()
    # Fresh search dict without internal fields must also read the persisted slot.
    await embeddings.article_vector({**raw, "news_id": article["news_id"]})
    await cache_articles([raw])
    assert len(calls) == 2  # Real Mongo round trip, including repeated article save.
    persisted = await database.news_candidates("AI", 40)
    assert len(persisted) == 1 and persisted[0]["news_id"] == article["news_id"]

    changed = {**raw, "description": "새 AI GPU 공급 계약"}
    article = (await cache_articles([changed]))[0]
    await embeddings.article_vector(article)
    assert len(calls) == 4
    monkeypatch.setattr(settings, "news_map_embedding_dimensions", 128)
    await embeddings.article_vector(article)
    assert len(calls) == 5
    stored = await local_mongo[database.NEWS].find_one({"news_id": article["news_id"]})
    assert len(stored["news_map_embedding"]["values"]) == 128
    assert len(stored["embedding"]) == 768


async def test_local_mongo_pool_keeps_search_rank_and_selection_after_restart(local_mongo, monkeypatch):
    async def embed(text, **kwargs):
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)

    async def forbidden_fetch(*args, **kwargs):
        pytest.fail("Persisted candidate pool should avoid a new external news search")

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news_fetcher, "fetch_search_page", forbidden_fetch)
    session = "2026-10-01T00:00:00+00:00"
    # IDs deliberately disagree with search rank; the pool must follow rank.
    articles = await cache_articles([{"title": f"AI GPU 소식 {i}", "description": f"제품 출시 정보 {i}",
                                      "news_id": f"z{9 - i}", "url": f"https://example.test/{i}",
                                      "published_at": "2026-09-30T09:00:00Z", "_search_keyword": "AI",
                                      "_search_rank": i + 1, "_search_end": 20, "_searched_at": session}
                                     for i in range(5)])
    center_id = articles[0]["news_id"]
    store.news_cache.clear()
    persisted = await database.news_candidates("AI", 40)
    assert [a["news_id"] for a in persisted] == ["z9", "z8", "z7", "z6", "z5"]
    assert persisted[0]["_search_rank"] == 1 and persisted[0]["_searched_at"] == session
    assert "embedding" not in persisted[0] or persisted[0]["embedding"] == []
    before = await news_map.build_news_map(await store.get_news(center_id), target=10)
    store.news_cache.clear()
    embeddings._cache.clear()
    after = await news_map.build_news_map(await store.get_news(center_id), target=10)
    assert [g.representative.article["news_id"] for g in after.groups] == \
        [g.representative.article["news_id"] for g in before.groups]
    free = await news.get_related(center_id, limit=10, min_relevance=0, include_score=False, tier="FREE", expand=True)
    graph = await news.get_graph(center_id, depth=3, limit=10, include_distance=True,
                                 min_relevance=0, include_score=False, tier="FREE", expand=True)
    assert [a.news_id for a in free.related_news] == [a.news_id for a in graph.nodes[1:]]
    assert len(free.related_news) <= 3 and free.selection == graph.selection


async def test_local_mongo_expansion_cards_and_map_vectors_survive_restart(local_mongo, monkeypatch):
    from app.agents.naver_client import NaverNewsPage
    from tests.test_news_map_expansion import angle, cached, repeat

    calls, searches = [], []

    async def embed(text, **kwargs):
        calls.append((text, kwargs))
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)

    async def fetch(query, *, start, size):
        searches.append((query, start))
        found = [{**angle("forecast"), "_search_rank": start}]
        return NaverNewsPage(articles=found, total=1, start=start, end=start + size - 1)

    async def passthrough(articles):
        return articles

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news_fetcher, "fetch_search_page", fetch)
    monkeypatch.setattr(news_fetcher, "inspect_search_articles", passthrough)
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 1)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mock_news", False)
    from tests.test_news_map_diversity import OIL_CENTER, oil
    await cache_articles([cached(oil("center", *OIL_CENTER), 1), cached(repeat(1), 2)])
    before = await news_map.build_news_map(await store.get_news("center"), target=1)
    assert [g.representative.article["news_id"] for g in before.groups] == ["forecast"]
    assert len(searches) == 1 and searches[0][0] != "국제유가" and searches[0][1] == 1
    saved = await local_mongo[database.NEWS].find_one({"news_id": "forecast"})
    assert "embedding" not in saved  # No extra RAG generation during expansion.
    assert saved["news_map_embedding"]["metadata"]["purpose"] == "news_map"
    assert saved["_search_keyword"] == searches[0][0] and saved["_search_rank"] == 1
    count = len(calls)
    store.news_cache.clear()
    embeddings._cache.clear()
    related_candidates._cache.clear()
    after = await news_map.build_news_map(await store.get_news("center"), target=1)
    assert [g.representative.article["news_id"] for g in after.groups] == ["forecast"]
    assert after.groups[0].representative.article["description"] == angle("forecast")["description"]
    # The saved card and map vector are reused after the restart: no new vectors.
    assert len(calls) == count and after.stats["embedding_calls"] == 0
