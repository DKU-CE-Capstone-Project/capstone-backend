"""Quota-denial timing and HTTP policy regressions."""

import asyncio
import math
import shutil
import socket
from types import SimpleNamespace

import pytest
import redis.asyncio as aioredis
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import usage_limits
from app.config import settings
from app.main import app

pytest_plugins = ("tests.test_v2_maps",)

_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


@pytest.mark.asyncio
async def test_memory_denial_uses_longest_blocked_ttl_without_partial_charge(monkeypatch):
    monkeypatch.setattr(usage_limits, "time", SimpleNamespace(time=lambda: 1000.2))
    usage_limits._memory.update({
        "quota:search:global": (1, 1002.5),
        "quota:map:session:peer": (1, 1011.1),
        "quota:map:client:192.0.2.1": (1, 1007.1),
    })
    before = usage_limits._memory.copy()

    with pytest.raises(HTTPException) as caught:
        await usage_limits.check_quotas(
            usage_limits.QuotaRequest("peer", "search", per_session=2, global_limit=1),
            usage_limits.QuotaRequest("peer", "map", per_session=1, global_limit=2,
                                      client_host="192.0.2.1", per_client=1),
        )

    assert caught.value.status_code == 429
    assert caught.value.headers == {"Retry-After": "11"}
    assert usage_limits._memory == before


@pytest.fixture
async def local_quota_redis(monkeypatch):
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server is required for the Lua quota regression")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def loopback_only(sock, address):
        if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", port):
            raise AssertionError("Only the isolated quota Redis may be contacted")
        return _ORIGINAL_SOCKET_CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setattr(settings, "session_store_required", True)
    process = await asyncio.create_subprocess_exec(
        binary, "--bind", "127.0.0.1", "--port", str(port), "--save", "",
        "--appendonly", "no",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    client = aioredis.Redis(host="127.0.0.1", port=port, decode_responses=True)
    monkeypatch.setattr(usage_limits, "_client", client)
    try:
        for _ in range(40):
            if process.returncode is not None:
                pytest.fail("isolated redis-server exited during startup")
            try:
                await client.ping()
                break
            except aioredis.ConnectionError:
                await asyncio.sleep(0.05)
        else:
            pytest.fail("isolated redis-server did not become ready")
        yield client
    finally:
        await client.aclose()
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=3)


@pytest.mark.asyncio
async def test_redis_lua_denial_reports_longest_ttl_and_charges_nothing(local_quota_redis):
    client = local_quota_redis
    await client.psetex("quota:search:global", 2200, "1")
    await client.psetex("quota:map:session:peer", 7200, "1")
    search = usage_limits.QuotaRequest("peer", "search", per_session=1, global_limit=1)
    map_request = usage_limits.QuotaRequest("peer", "map", per_session=1, global_limit=1)
    before_longest = math.ceil(await client.pttl("quota:map:session:peer") / 1000)

    with pytest.raises(HTTPException) as caught:
        await usage_limits.check_quotas(search, map_request)

    after_longest = math.ceil(await client.pttl("quota:map:session:peer") / 1000)
    assert caught.value.status_code == 429
    assert after_longest <= int(caught.value.headers["Retry-After"]) <= before_longest
    assert int(caught.value.headers["Retry-After"]) > 3  # Longer than the search bucket.
    assert await client.mget(
        "quota:search:global", "quota:search:session:peer",
        "quota:map:global", "quota:map:session:peer",
    ) == ["1", None, None, "1"]

    # The same Lua reservation admits only one of two concurrent requests.
    await client.delete("quota:search:global", "quota:map:session:peer")
    outcomes = await asyncio.gather(
        usage_limits.check_quotas(search, map_request),
        usage_limits.check_quotas(search, map_request),
        return_exceptions=True,
    )
    assert sum(outcome is None for outcome in outcomes) == 1
    rejected = [outcome for outcome in outcomes if isinstance(outcome, HTTPException)]
    assert len(rejected) == 1 and rejected[0].status_code == 429
    assert await client.mget(
        "quota:search:global", "quota:search:session:peer",
        "quota:map:global", "quota:map:session:peer",
    ) == ["1", "1", "1", "1"]


@pytest.mark.asyncio
async def test_redis_lua_repairs_missing_expiry_before_denial(local_quota_redis):
    client = local_quota_redis
    await client.set("quota:search:global", "1")

    with pytest.raises(HTTPException) as caught:
        await usage_limits.check_quotas(
            usage_limits.QuotaRequest("peer", "search", per_session=1, global_limit=1),
        )

    assert caught.value.headers["Retry-After"] == "3600"
    assert 0 < await client.pttl("quota:search:global") <= 3_600_000
    assert await client.get("quota:search:session:peer") is None


def test_v1_429_retains_retry_after_from_session_middleware(monkeypatch):
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 0)
    response = TestClient(app).get("/api/v1/session")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3600"
    assert response.json()["detail"]


def test_v1_public_429_retains_retry_after_from_route(monkeypatch):
    monkeypatch.setattr(settings, "news_map_client_hourly_limit", 0)
    response = TestClient(app).get("/api/v1/news/missing/graph")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3600"


def test_v2_early_429_exposes_retry_after_to_allowed_origin(monkeypatch):
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 0)
    response = TestClient(app).get(
        "/api/v2/session", headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert response.headers["retry-after"] == "3600"
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_v2_map_denial_is_atomic_and_exposes_retry_after(fixture, monkeypatch):
    client, _, selection_calls, _ = fixture
    monkeypatch.setattr(settings, "news_map_global_hourly_limit", 1)
    usage_limits._memory["quota:news_map:global"] = (1, usage_limits.time.time() + 120)

    denied = client.post(
        "/api/v2/maps", json={"query": "검색"},
        headers={"Origin": "http://localhost:5173"},
    )

    assert denied.status_code == 429
    assert denied.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert 0 < int(denied.headers["retry-after"]) <= 120
    assert "retry-after" in denied.headers["access-control-expose-headers"].lower()
    assert denied.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "quota:news_search:global" not in usage_limits._memory
    assert selection_calls == []

    usage_limits._memory.pop("quota:news_map:global")
    admitted = client.post("/api/v2/maps", json={"query": "검색"})
    assert admitted.status_code == 201
    assert usage_limits._memory["quota:news_search:global"][0] == 1
    assert usage_limits._memory["quota:news_map:global"][0] == 1


def test_v2_stored_news_reads_do_not_spend_provider_quota(fixture, monkeypatch):
    client, _, _, _ = fixture
    monkeypatch.setattr(settings, "news_search_client_hourly_limit", 1)
    created = client.post("/api/v2/maps", json={"query": "검색"})
    assert created.status_code == 201
    news_id = created.json()["nodes"][0]["news_id"]
    initial_charges = usage_limits._memory["quota:news_search:global"][0]
    assert initial_charges == 1
    assert client.post("/api/v2/maps", json={"query": "다른 검색"}).status_code == 429

    for _ in range(3):
        cached = client.get(f"/api/v2/news/{news_id}")
        assert cached.status_code == 200
        assert cached.json()["news_id"] == news_id
    assert usage_limits._memory["quota:news_search:global"][0] == initial_charges
