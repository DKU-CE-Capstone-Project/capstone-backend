import asyncio

import pytest
from fastapi.testclient import TestClient
from motor import motor_asyncio

from app import database
from app import session as session_store
from app.agents import article_metadata
from app.config import settings
from app.main import app


@pytest.mark.parametrize("connected", [True, False])
def test_readiness_uses_actual_ping(monkeypatch, connected):
    monkeypatch.setattr(settings, "use_mongodb", True)
    async def ping(): return connected
    monkeypatch.setattr(database, "ping", ping)
    response = TestClient(app).get("/ready")
    assert response.status_code == (200 if connected else 503)


def test_readiness_requires_shared_session_store(monkeypatch):
    monkeypatch.setattr(settings, "session_store_required", True)
    monkeypatch.setattr(settings, "use_mongodb", False)

    async def unavailable():
        return False

    monkeypatch.setattr(session_store, "ping", unavailable)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["redis"] == "unavailable"


def test_readiness_recovers_after_startup_mongodb_ping_failure(monkeypatch):
    class Admin:
        available = False
        calls = 0

        async def command(self, _command):
            self.calls += 1
            if not self.available:
                raise ConnectionError("temporary MongoDB outage")
            return {"ok": 1}

    class Database:
        def __init__(self, admin):
            self.admin = admin
            self.created_indexes = []

        async def command(self, command):
            return await self.admin.command(command)

        def __getitem__(self, _name):
            return self

        async def create_index(self, _keys, **options):
            self.created_indexes.append(options["name"])

    class Client:
        def __init__(self, admin):
            self.admin = admin

        def __getitem__(self, _name):
            return db

        def close(self):
            pass

    admin = Admin()
    db = Database(admin)
    vector_index_calls = []

    async def ensure_vector_index():
        vector_index_calls.append(1)

    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "session_store_required", False)
    monkeypatch.setattr(settings, "mongodb_uri", "mongodb://127.0.0.1:27017")
    monkeypatch.setattr(database, "_client", Client(admin))
    monkeypatch.setattr(database, "_db", db)
    monkeypatch.setattr(database, "_indexes_checked_for", None)
    monkeypatch.setattr(motor_asyncio, "AsyncIOMotorClient", lambda *_args, **_kwargs: Client(admin))
    monkeypatch.setattr(database, "ensure_vector_index", ensure_vector_index)

    asyncio.run(database.ensure_indexes())
    assert db.created_indexes == []
    assert TestClient(app).get("/ready").status_code == 503
    admin.available = True
    assert TestClient(app).get("/ready").status_code == 200
    assert set(db.created_indexes) == {
        "uniq_news_url", "uniq_news_news_id", "idx_news_search_keyword",
        "uniq_reports_report_id", "uniq_strategies_strategy_id",
    }
    assert vector_index_calls == [1]
    assert TestClient(app).get("/ready").status_code == 200
    assert len(db.created_indexes) == 5
    assert vector_index_calls == [1]
    assert admin.calls >= 2


def test_mongodb_client_creation_retries_after_transient_failure(monkeypatch):
    class Database:
        pass

    class Client:
        def __init__(self, db):
            self.db = db

        def __getitem__(self, _name):
            return self.db

    db = Database()
    attempts = []
    clock = [2000.0]

    def connect(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("temporary constructor failure")
        return Client(db)

    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "mongodb_uri", "mongodb://127.0.0.1:27017")
    monkeypatch.setattr(database, "_client", None)
    monkeypatch.setattr(database, "_db", None)
    monkeypatch.setattr(database, "_retry_at", 0.0)
    monkeypatch.setattr(motor_asyncio, "AsyncIOMotorClient", connect)
    monkeypatch.setattr(database.time, "monotonic", lambda: clock[0])

    assert database._get_db() is None
    assert database._get_db() is None
    assert len(attempts) == 1
    clock[0] += 5.1
    assert database._get_db() is db
    assert len(attempts) == 2


async def test_required_database_write_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(settings, "mongodb_required", True)
    monkeypatch.setattr(database, "_get_db", lambda: None)
    with pytest.raises(database.DatabasePersistenceError):
        await database.save_news({"url": "https://example.com/test"})


async def test_optional_database_write_keeps_local_mode_available(monkeypatch):
    monkeypatch.setattr(settings, "mongodb_required", False)
    monkeypatch.setattr(database, "_get_db", lambda: None)
    await database.save_news({"url": "https://example.com/test"})


async def test_batch_deadline_covers_all_metadata_calls(monkeypatch, metadata_ai):
    monkeypatch.setattr(settings, "metadata_batch_timeout_seconds", .01)
    async def slow(prompt):
        await asyncio.sleep(10)
        return ""
    monkeypatch.setattr(article_metadata, "generate", slow)
    result = await article_metadata.enrich_articles([{"title": f"HBM 생산 {i}"} for i in range(10)])
    assert len(result) == 10
    assert all(a["metadata_extraction"]["reason"] == "batch_timeout" for a in result)
