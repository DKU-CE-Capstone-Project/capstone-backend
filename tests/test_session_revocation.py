"""A late v1 mutation cannot revive the cookie revoked through v2."""
import asyncio

import httpx
import pytest

from app import session as sessions
from app import store
from app.config import settings
from app.main import app
from app.v2 import state


@pytest.mark.asyncio
async def test_inflight_legacy_mutation_cannot_restore_revoked_result_access(monkeypatch):
    monkeypatch.setattr(settings, "v2_state_memory", True)
    monkeypatch.setattr(settings, "session_cookie_secure", False)
    state._memory.clear()
    before_save, proceed = asyncio.Event(), asyncio.Event()
    original = sessions.save

    async def delayed_save(sid, data, *, _create=False):
        if not _create:
            before_save.set()
            await proceed.wait()
        return await original(sid, data, _create=_create)

    monkeypatch.setattr(sessions, "save", delayed_save)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/api/v2/session")).status_code == 200
        revoked_sid = client.cookies.get(sessions.SESSION_COOKIE)
        store.report_cache["owned-fixture"] = {
            "owner_sid": revoked_sid, "report_id": "owned-fixture", "title": "fixture",
            "summary": "fixture", "event_analysis": "fixture", "market_impact": "fixture",
            "related_stocks": [], "evidence_news": [], "risk_factors": [],
            "created_at": "2026-10-11T00:00:00Z",
        }
        assert (await client.get("/api/v1/reports/owned-fixture")).status_code == 200
        pending = asyncio.create_task(client.post("/api/v1/session/mindmap/expand", json={"news_id": "private"}))
        await before_save.wait()
        assert (await client.delete("/api/v2/session")).status_code == 204
        proceed.set()
        assert (await pending).status_code == 410
        assert await sessions._existing(revoked_sid) is None
        headers = {"Cookie": f"{sessions.SESSION_COOKIE}={revoked_sid}"}
        assert (await client.get("/api/v2/maps/old-map", headers=headers)).status_code == 410
        assert (await client.get("/api/v1/reports/owned-fixture", headers=headers)).status_code == 404
        assert await sessions._existing(revoked_sid) is None
    state._memory.clear()


@pytest.mark.asyncio
async def test_only_issuance_can_create_identity():
    with pytest.raises(sessions.SessionExpired):
        await sessions.save("never-issued", {"sid": "never-issued"})
    sid = await sessions.issue("127.0.0.1")
    original = await sessions.load(sid)
    with pytest.raises(sessions.SessionStoreUnavailable):
        await sessions.save(sid, {"sid": sid, "overwritten": True}, _create=True)
    assert await sessions.load(sid) == original
