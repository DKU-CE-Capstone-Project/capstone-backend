"""Opt-in tests against isolated loopback MongoDB/Redis/NATS with AI fixtures.

ECONMIND_V2_LOCAL_SERVICES=1 ECONMIND_V2_SCHEMA_FILE=/path/to/deploy/scripts/lib/newsmap-v2-schema.json \
  .venv/bin/python -m pytest -q tests/test_v2_reports_local_services.py
"""
import asyncio
import json
import os
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from nats.js.api import AckPolicy, ConsumerConfig
from starlette.requests import Request

from app import database, store, usage_limits
from app import session as base_session
from app.api.v2 import report_jobs as api
from app.config import settings
from app.v2 import persistence, report_worker, reports, state
from app.v2.errors import V2Error

_socket_connect = socket.socket.connect
pytestmark = pytest.mark.skipif(os.getenv("ECONMIND_V2_LOCAL_SERVICES") != "1", reason="Explicit isolated local-service opt-in required")


@pytest.fixture
async def local_services(monkeypatch):
    schema_file = os.getenv("ECONMIND_V2_SCHEMA_FILE")
    if not schema_file:
        pytest.skip("ECONMIND_V2_SCHEMA_FILE must identify the canonical deployment validator JSON")
    spec = json.loads(Path(schema_file).read_text())

    def loopback_only(sock, address):
        if not isinstance(address, tuple) or address[0] != "127.0.0.1" or address[1] not in (27218, 6388, 4233):
            raise AssertionError("Only isolated fixture services are permitted")
        return _socket_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    db_name = "econmind_v2_report_test_" + uuid.uuid4().hex
    for key, value in {"mongodb_uri": "mongodb://127.0.0.1:27218", "mongodb_db_name": db_name, "use_mongodb": True, "mongodb_required": True, "redis_url": "redis://127.0.0.1:6388/15", "nats_url": "nats://127.0.0.1:4233", "session_store_required": True, "v2_state_memory": False, "prototype_tier": "PAID", "paid_demo_enabled": True}.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(database, "_client", None)
    monkeypatch.setattr(database, "_db", None)
    monkeypatch.setattr(database, "_disabled", False)
    monkeypatch.setattr(base_session, "_redis", None)
    monkeypatch.setattr(base_session, "_redis_disabled", False)
    monkeypatch.setattr(usage_limits, "_client", None)
    fixture_id = uuid.uuid4().hex
    async def isolated_quota(owner_sid, action, **kwargs):
        return await usage_limits.check_quota(owner_sid, f"{action}_{fixture_id}", **kwargs)

    monkeypatch.setattr(base_session, "check_quota", isolated_quota)
    monkeypatch.setattr(api, "check_quota", isolated_quota)
    monkeypatch.setattr(reports, "check_quota", isolated_quota)
    from app.api.v2 import maps as map_api
    monkeypatch.setattr(map_api, "check_quota", isolated_quota)
    monkeypatch.setattr(report_worker, "STREAM", "reports_v2_test_" + fixture_id)
    monkeypatch.setattr(report_worker, "SUBJECT", "reports.generate.v2.fixture." + fixture_id)
    monkeypatch.setattr(report_worker, "DURABLE", "report-fixture-" + fixture_id)
    db = database._get_db()
    await db.command("ping")
    await db.create_collection("news")
    for collection in ("news_bodies", "reports", "strategies"):
        await db.create_collection(collection, validator=spec["validators"][collection], validationLevel="strict", validationAction="error")
    await persistence.ready()
    sessions = []
    calls = {"extract": 0, "analysis": 0, "strategy": 0}

    async def extract(articles, **kwargs):
        calls["extract"] += 1
        return [{**articles[0], "cleaned_content": "실제 외부 API 대신 넣은 로컬 테스트 본문 근거입니다. " * 30}]

    async def generate(prompt):
        calls["analysis"] += 1
        return json.dumps({"title": "로컬 fixture 분석", "event_summary": "확인한 기사 사건", "event_analysis": "본문으로 확인한 분석", "market_impact": "해석한 시장 영향", "stock_impacts": [], "risk_factors": ["변동성"]})

    async def strategy(*args, **kwargs):
        calls["strategy"] += 1
        return {"strategy_summary": "로컬 fixture 관망", "risk_warning": "실제 성과가 아닙니다.", "strategy_items": [{"ticker": None, "stock_name": "관련 업종", "action": "watch", "reason": "추가 근거 확인"}]}

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", extract)
    monkeypatch.setattr(reports.llm, "generate", generate)
    monkeypatch.setattr(reports, "generate_strategy", strategy)
    try:
        yield db, sessions, calls
    finally:
        for sid in sessions:
            await state.clear_session(sid)
            await base_session.clear(sid)
            keys = [key async for key in state._client().scan_iter(match=f"v2:*:{sid}:*")]
            if keys:
                await state._client().delete(*keys)
        quota_keys = [key async for key in state._client().scan_iter(match=f"quota:*{fixture_id}*")]
        if quota_keys:
            await state._client().delete(*quota_keys)
        existing = report_worker._connections.pop(asyncio.get_running_loop(), None)
        if existing:
            await existing[2].delete_stream(report_worker.STREAM)
            await existing[1].close()
        current_client = database._client
        await current_client.drop_database(db_name)
        current_client.close()
        database._db = None
        database._client = None
        if base_session._redis:
            await base_session._redis.aclose()
        if usage_limits._client:
            await usage_limits._client.aclose()
        current_state = state._clients.pop(asyncio.get_running_loop(), None)
        if current_state:
            await current_state[1].aclose()
        persistence._indexed.discard(("mongodb://127.0.0.1:27218", db_name))


def request(sid):
    result = Request({"type": "http", "method": "POST", "path": "/api/v2/report-jobs", "headers": [], "client": ("127.0.0.1", 1234)})
    result.state.session_id = sid
    return result


async def fixture_map(local_services):
    _db, sessions, _calls = local_services
    sid = await base_session.issue("127.0.0.1")
    sessions.append(sid)
    session = await state.session(sid)
    news_id = "fixture00001"
    article = {"news_id": news_id, "title": "로컬 경제 기사", "description": "검색 설명", "source": "로컬 fixture 매체", "url": "https://fixture.invalid/article", "published_at": "2026-10-11T00:00:00Z"}
    await database.save_news(database.news_doc_from_article(article))
    store.news_cache[news_id] = article
    map_id = str(uuid.uuid4())
    map_doc = {"map_id": map_id, "revision": 1, "nodes": [{"news_id": news_id}], "selected_news_ids": [], "report_panel_open": False, "active_report_job_id": None, "last_result": None, "expires_at": session["expires_at"]}
    await state.put(state.key(sid, "map", map_id), map_doc, settings.session_ttl_seconds)
    await state.register_resource(sid, "map", map_id, session["expires_at"])
    return sid, map_id, news_id


async def submit(sid, map_id, news_id, key=None, body=None):
    current = await state.resource(sid, "map", map_id)
    body = body or api.ReportJobCreate(map_id=map_id, expected_revision=current["revision"], client_request_id=uuid.uuid4(), news_ids=[news_id])
    return await api.create_job(body, request(sid), key or str(uuid.uuid4()))


async def consume_once():
    js = await report_worker.get_jetstream()
    sub = await js.pull_subscribe(report_worker.SUBJECT, stream=report_worker.STREAM, durable=report_worker.DURABLE,
                                  config=ConsumerConfig(durable_name=report_worker.DURABLE, ack_policy=AckPolicy.EXPLICIT, ack_wait=60, max_ack_pending=1))
    messages = await sub.fetch(1, timeout=5)
    message = messages[0]
    payload = json.loads(message.data)
    outcome = await reports.process_job(payload["session_id"], payload["job_id"], message)
    await message.ack_sync()
    return outcome, payload


async def test_real_admission_durable_consumer_bson_and_restart_cache(local_services):
    db, sessions, calls = local_services
    sid, map_id, news_id = await fixture_map(local_services)
    admitted = await submit(sid, map_id, news_id)
    job = await reports.get_job(sid, admitted["job_id"])
    assert job["dispatch_state"] == "published"
    assert admitted["job_id"] in await state.keys(sid, "report-job")
    assert (await state.resource(sid, "map", map_id))["active_report_job_id"] == admitted["job_id"]
    assert (await consume_once())[0] == "done"
    completed = await reports.get_job(sid, admitted["job_id"])
    assert completed["status"] == "completed"
    report_id = completed["result"]["report_id"]
    saved = await db.reports.find_one({"report_id": report_id})
    assert isinstance(saved["created_at"], datetime) and isinstance(saved["used_evidence"][0]["published_at"], datetime)
    assert await db.reports.count_documents({"created_at": {"$type": "date"}}) == 1
    assert (await db.strategies.find_one({}))["report_id"] == report_id
    original_body = await db.news_bodies.find_one({})
    assert (await db.news.find_one({"news_id": news_id}))["body_ref"]["body_id"] == original_body["body_id"]
    with pytest.raises(V2Error):
        await persistence.save_body({**original_body, "content": "immutable ID cannot overwrite", "content_hash": reports.text_hash("immutable ID cannot overwrite")})
    assert (await db.news_bodies.find_one({"body_id": original_body["body_id"]}))["content"] == original_body["content"]
    # Recreate database client and clear all API/process caches. Redis/MongoDB
    # remain authoritative; no in-process result index survives this boundary.
    database._client.close()
    database._client = database._db = None
    store.news_cache.clear()
    store.report_cache.clear()
    store.report_index.clear()
    persistence._memory["reports"].clear()
    admitted2 = await submit(sid, map_id, news_id)
    await consume_once()
    completed2 = await reports.get_job(sid, admitted2["job_id"])
    assert completed2["result"]["report_id"] == report_id
    assert completed2["result"]["report_reused"] and completed2["result"]["strategy_reused"]
    assert calls == {"extract": 1, "analysis": 1, "strategy": 1}
    other_sid = await base_session.issue("127.0.0.1")
    sessions.append(other_sid)
    with pytest.raises(V2Error) as caught:
        await api.get_report(report_id, request(other_sid))
    assert caught.value.status == 404
    exposed = await api.get_report(report_id, request(sid))
    assert "input_excerpt" not in exposed["used_evidence"][0]


async def test_real_publish_failure_idempotency_and_queued_cancel(local_services, monkeypatch):
    _db, _sessions, calls = local_services
    sid, map_id, news_id = await fixture_map(local_services)
    real_publish = report_worker.publish
    body = api.ReportJobCreate(map_id=map_id, expected_revision=1, client_request_id=uuid.uuid4(), news_ids=[news_id])
    idem_key = str(uuid.uuid4())

    async def reject(*args):
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "fixture publish failure")

    monkeypatch.setattr(report_worker, "publish", reject)
    with pytest.raises(V2Error):
        await submit(sid, map_id, news_id, body=body, key=idem_key)
    replay = await state.get(f"v2:idem:{sid}:report-jobs:{idem_key}")
    assert replay["job_id"] in await state.keys(sid, "report-job")
    assert (await state.resource(sid, "map", map_id))["revision"] == 2
    monkeypatch.setattr(report_worker, "publish", real_publish)
    retried = await submit(sid, map_id, news_id, body=body, key=idem_key)
    assert retried["job_id"] == replay["job_id"]
    cancelled = await api.cancel_job(retried["job_id"], request(sid))
    assert cancelled["status"] == "cancelled"
    await consume_once()
    assert await database._get_db().reports.count_documents({}) == 0 and calls["analysis"] == 0
    assert (await state.resource(sid, "map", map_id))["active_report_job_id"] is None


async def test_real_stale_map_and_completion_revision_race(local_services, monkeypatch):
    sid, map_id, news_id = await fixture_map(local_services)
    admitted = await submit(sid, map_id, news_id)
    original = reports.llm.generate

    async def change_revision(prompt):
        current = await state.resource(sid, "map", map_id)
        revision = current["revision"]
        current.update(revision=revision + 1, report_panel_open=False)
        # User activity can extend the live session/physical map TTL while the
        # JSON snapshot still holds an old expires_at. Finishing must refresh it.
        current["expires_at"] = state.iso(time.time() - 60)
        await state.cas(state.key(sid, "map", map_id), revision, current, settings.session_ttl_seconds)
        return await original(prompt)

    monkeypatch.setattr(reports.llm, "generate", change_revision)
    await consume_once()
    current = await state.resource(sid, "map", map_id)
    assert current["report_panel_open"] is False and current["revision"] == 4
    assert await state._client().ttl(state.key(sid, "map", map_id)) > 86300
    assert (await reports.get_job(sid, admitted["job_id"]))["status"] == "completed"
    # A new active job detaches this worker's result. The immutable body can
    # persist, while stale analysis cannot replace the current map result.
    admitted2 = await submit(sid, map_id, news_id)
    current = await state.resource(sid, "map", map_id)
    current["active_report_job_id"] = "newer-job"
    await state.put(state.key(sid, "map", map_id), current, settings.session_ttl_seconds)
    await consume_once()
    assert (await reports.get_job(sid, admitted2["job_id"]))["status"] == "cancelled"
    assert (await state.resource(sid, "map", map_id))["active_report_job_id"] == "newer-job"


async def test_real_atomic_admission_revision_conflict_writes_nothing(local_services, monkeypatch):
    sid, map_id, news_id = await fixture_map(local_services)
    original = api._admit

    async def race(sid, map_doc, job, idem_key, request_hash):
        current = await state.resource(sid, "map", map_id)
        revision = current["revision"]
        current["revision"] += 1
        await state.cas(state.key(sid, "map", map_id), revision, current, settings.session_ttl_seconds)
        await original(sid, map_doc, job, idem_key, request_hash)

    monkeypatch.setattr(api, "_admit", race)
    with pytest.raises(V2Error) as caught:
        await submit(sid, map_id, news_id)
    assert caught.value.code == "MAP_REVISION_CONFLICT"
    assert await state.keys(sid, "report-job") == []
    redis_keys = [key async for key in state._client().scan_iter(match=f"v2:report-job:{sid}:*")]
    assert redis_keys == []


async def test_http_end_to_end_session_map_async_report_and_isolation(local_services, monkeypatch):
    from app.agents.naver_client import NaverNewsPage
    from app.agents.news_map import NewsMapResult
    from app.main import app
    from app.v2 import maps

    _db, sessions, _calls = local_services
    article = {"news_id": "fixturehttp1", "title": "경제 HTTP fixture 기사", "description": "선택 본문과 구별되는 검색 설명", "source": "fixture 매체", "url": "https://fixture.invalid/http", "published_at": "2026-10-11T00:00:00Z"}

    async def search(query, start=1, size=20):
        return NaverNewsPage(articles=[article], start=start, end=start, total=1)

    async def inspect(articles):
        return articles

    async def select(center, pool, target, excluded, context):
        return NewsMapResult(items=[], status="insufficient", reason="no_eligible_related", target=target)

    monkeypatch.setattr(maps.news_fetcher, "fetch_search_page", search)
    monkeypatch.setattr(maps.news_fetcher, "inspect_search_articles", inspect)
    monkeypatch.setattr(maps, "_select", select)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        session_response = await client.get("/api/v2/session")
        assert session_response.status_code == 200
        assert session_response.json()["resumed"] is False and "sid" not in session_response.json()
        assert "HttpOnly" in session_response.headers["set-cookie"]
        sid = client.cookies.get(base_session.SESSION_COOKIE)
        sessions.append(sid)
        assert (await client.get("/api/v2/session")).json()["resumed"] is True
        map_response = await client.post("/api/v2/maps", json={"query": "경제"})
        assert map_response.status_code == 201, map_response.text
        snapshot = map_response.json()
        assert "candidate_articles" not in snapshot and snapshot["nodes"][0]["news"]["description"] == article["description"]
        map_id = snapshot["map_id"]
        selected = await client.patch(f"/api/v2/maps/{map_id}", json={"expected_revision": snapshot["revision"], "selected_news_ids": [article["news_id"]]})
        assert selected.status_code == 200
        payload = {"map_id": map_id, "expected_revision": selected.json()["revision"], "client_request_id": str(uuid.uuid4()), "news_ids": [article["news_id"]]}
        key = str(uuid.uuid4())
        admitted = await client.post("/api/v2/report-jobs", json=payload, headers={"Idempotency-Key": key})
        assert admitted.status_code == 202, admitted.text
        assert "dispatch_state" not in admitted.json() and "request" not in admitted.json()
        replay = await client.post("/api/v2/report-jobs", json=payload, headers={"Idempotency-Key": key})
        assert replay.status_code == 202 and replay.json()["job_id"] == admitted.json()["job_id"]
        await consume_once()
        job_response = await client.get("/api/v2/report-jobs/" + admitted.json()["job_id"])
        assert job_response.status_code == 200 and job_response.json()["status"] == "completed"
        result = job_response.json()["result"]
        report_response = await client.get("/api/v2/reports/" + result["report_id"])
        assert report_response.status_code == 200, report_response.text
        assert report_response.json()["coverage"]["used_count"] == 1
        assert "input_excerpt" not in report_response.json()["used_evidence"][0]
        strategy_response = await client.get("/api/v2/strategies/" + result["strategy_id"])
        assert strategy_response.status_code == 200 and strategy_response.json()["report_id"] == result["report_id"]
        restored = await client.get(f"/api/v2/maps/{map_id}")
        assert restored.status_code == 200 and restored.json()["last_result"] == {"report_id": result["report_id"], "strategy_id": result["strategy_id"]}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as other:
            await other.get("/api/v2/session")
            sessions.append(other.cookies.get(base_session.SESSION_COOKIE))
            denied = await other.get("/api/v2/reports/" + result["report_id"])
            assert denied.status_code == 404 and denied.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
        await client.delete("/api/v2/session")
        expired = await client.get("/api/v2/reports/" + result["report_id"])
        assert expired.status_code == 410 and expired.json()["error"]["code"] == "SESSION_EXPIRED"


async def test_real_refresh_poll_reconciles_unpublished_job(local_services, monkeypatch):
    sid, map_id, news_id = await fixture_map(local_services)
    real_publish = report_worker.publish

    async def reject(*args):
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "temporary fixture queue failure")

    monkeypatch.setattr(report_worker, "publish", reject)
    with pytest.raises(V2Error):
        await submit(sid, map_id, news_id)
    job_id = (await state.resource(sid, "map", map_id))["active_report_job_id"]
    current = await state.resource(sid, "report-job", job_id)
    current["last_dispatch_attempt"] = 0
    await reports._save_job(sid, current)
    old_expiry = (await state.session(sid))["expires_at"]
    monkeypatch.setattr(report_worker, "publish", real_publish)
    recovered = await api.get_job(job_id, request(sid))
    assert recovered["job_id"] == job_id and await state.keys(sid, "report-job") == [job_id]
    assert (await state.session(sid))["expires_at"] == old_expiry
    await consume_once()
    assert (await reports.get_job(sid, job_id))["status"] == "completed"
