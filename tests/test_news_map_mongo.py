"""Opt-in local MongoDB integration with mocked Gemini, never developer/production DB."""
import os
import socket
import uuid
from urllib.parse import urlsplit

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from app import database, store
from app.agents import article_embeddings as embeddings
from app.agents import llm
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


async def test_local_mongo_endpoints_pool_has_same_order_after_restart(local_mongo, monkeypatch):
    async def embed(text, **kwargs):
        return [1.0] + [0.0] * (kwargs.get("dimensions", 768) - 1)
    async def forbidden_fetch(*args, **kwargs):
        pytest.fail("Persisted candidate pool should avoid a new external news search")
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news, "fetch_news", forbidden_fetch)
    articles = await cache_articles([{"title": f"AI GPU 소식 {i}", "description": f"제품 출시 {i}",
                                    "url": f"https://example.test/{i}", "_search_keyword": "AI"} for i in range(5)])
    center_id = articles[0]["news_id"]
    center = await store.get_news(center_id)
    before = await news._selected_related(center, 10)
    store.news_cache.clear()
    embeddings._cache.clear()
    after = await news._selected_related(await store.get_news(center_id), 10)
    assert [r.article["news_id"] for r in after] == [r.article["news_id"] for r in before]
    free = await news.get_related(center_id, limit=10, min_relevance=0, include_score=False, tier="FREE")
    graph = await news.get_graph(center_id, depth=3, limit=10, include_distance=True)
    assert [a["news_id"] for a in free["related_news"]] == [a.news_id for a in graph.nodes[1:4]]
    assert len(graph.nodes) == 5 and len(free["related_news"]) == 3
