import asyncio

import pytest
from fastapi.testclient import TestClient

from app import database
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


async def test_required_database_write_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(settings, "mongodb_required", True)
    monkeypatch.setattr(database, "_get_db", lambda: None)
    with pytest.raises(database.DatabasePersistenceError):
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
