"""Regression tests for ownership, session rotation, and admission controls."""

from fastapi.testclient import TestClient

from app import session as session_store
from app import store, usage_limits
from app.api.v1 import jobs, reports, strategies
from app.config import settings
from app.main import app


def test_unknown_sid_is_replaced_and_reset_rotates() -> None:
    client = TestClient(app)
    client.cookies.set("econmind_sid", "a" * 32, domain="testserver.local", path="/")
    first = client.get("/api/v1/session")
    assert first.status_code == 200
    issued = client.cookies.get("econmind_sid")
    assert issued != "a" * 32
    assert "session_id" not in first.json()

    client.post("/api/v1/session/mindmap/expand", json={"news_id": "private"})
    assert client.get("/api/v1/session").json()["mindmap"]["expanded_news_ids"] == ["private"]
    client.delete("/api/v1/session")
    rotated = client.cookies.get("econmind_sid")
    assert rotated != issued
    assert client.get("/api/v1/session").json()["mindmap"]["expanded_news_ids"] == []

    old_client = TestClient(app)
    old_client.cookies.set("econmind_sid", issued, domain="testserver.local", path="/")
    old_client.get("/api/v1/session")
    assert old_client.cookies.get("econmind_sid") != issued


def test_session_store_outage_fails_closed_without_affecting_health(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_store_required", True)
    client = TestClient(app)
    assert client.get("/api/v1/session").status_code == 503
    assert client.get("/health").status_code == 200


def test_cookie_free_jobs_cannot_mint_unbounded_sessions(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 3)
    monkeypatch.setattr(settings, "session_issue_global_hourly_limit", 100)
    accepted = []

    async def save(job_id, data):
        accepted.append(job_id)

    async def publish(_payload):
        return None

    monkeypatch.setattr(jobs, "set_job", save)
    monkeypatch.setattr(jobs, "publish_job", publish)
    client = TestClient(app, client=("198.51.100.10", 50000))
    statuses = []
    for index in range(250):
        client.cookies.clear()
        statuses.append(client.post(
            "/jobs", json={"keyword": "반도체"},
            headers={"X-Forwarded-For": f"192.0.2.{index % 250 + 1}"},
        ).status_code)

    assert statuses == [202] * 3 + [429] * 247
    assert len(accepted) == 3
    assert len(session_store._memory_store) == 3


def test_rotated_sessions_share_a_client_work_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 20)
    monkeypatch.setattr(settings, "session_issue_global_hourly_limit", 100)
    monkeypatch.setattr(settings, "jobs_client_hourly_limit", 2)

    async def save(_job_id, _data):
        return None

    async def publish(_payload):
        return None

    monkeypatch.setattr(jobs, "set_job", save)
    monkeypatch.setattr(jobs, "publish_job", publish)
    client = TestClient(app, client=("198.51.100.10", 50000))
    statuses = []
    for _ in range(5):
        client.cookies.clear()
        statuses.append(client.post("/jobs", json={"keyword": "반도체"}).status_code)
    assert statuses == [202, 202, 429, 429, 429]
    assert usage_limits._memory["quota:jobs:global"][0] == 2
    assert usage_limits._memory["quota:jobs:client:198.51.100.10"][0] == 2

    # A separate connected client can still use its own allowance.
    other = TestClient(app, client=("198.51.100.11", 50000))
    assert other.post("/jobs", json={"keyword": "반도체"}).status_code == 202


def test_session_rotation_is_bounded_and_existing_cookie_still_works(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 2)
    monkeypatch.setattr(settings, "session_issue_global_hourly_limit", 2)
    client = TestClient(app, client=("2001:db8::1", 50000))
    assert client.get("/api/v1/session").status_code == 200
    first_sid = client.cookies.get("econmind_sid")
    assert client.delete("/api/v1/session").status_code == 200
    rotated_sid = client.cookies.get("econmind_sid")
    assert rotated_sid != first_sid
    assert client.get("/api/v1/session").status_code == 200
    assert client.delete("/api/v1/session").status_code == 200
    assert client.cookies.get("econmind_sid") is None
    assert rotated_sid not in session_store._memory_store
    assert client.get("/api/v1/session").status_code == 429


def test_session_issuance_global_limit_applies_across_peers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 2)
    monkeypatch.setattr(settings, "session_issue_global_hourly_limit", 2)
    for host in ("198.51.100.1", "198.51.100.2"):
        assert TestClient(app, client=(host, 50000)).get("/api/v1/session").status_code == 200
    assert TestClient(app, client=("198.51.100.3", 50000)).get("/api/v1/session").status_code == 429


def test_ipv6_privacy_addresses_share_a_client_bucket() -> None:
    assert usage_limits.client_key("2001:db8:1234:abcd::1") == usage_limits.client_key("2001:db8:1234:abcd::2")
    assert usage_limits.client_key("::ffff:192.0.2.1") == usage_limits.client_key("192.0.2.1")


def test_public_news_budget_groups_ipv6_privacy_addresses(monkeypatch) -> None:
    monkeypatch.setattr(settings, "news_map_client_hourly_limit", 1)
    monkeypatch.setattr(settings, "news_map_global_hourly_limit", 100)
    first = TestClient(app, client=("2001:db8:1234:abcd::1", 50000))
    second = TestClient(app, client=("2001:db8:1234:abcd::2", 50000))
    assert first.get("/api/v1/news/missing/graph").status_code == 404
    assert second.get("/api/v1/news/missing/graph").status_code == 429


def test_invalid_cookie_and_empty_reset_consume_only_new_issuance(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_issue_client_hourly_limit", 2)
    monkeypatch.setattr(settings, "session_issue_global_hourly_limit", 2)
    client = TestClient(app, client=("198.51.100.10", 50000))
    assert client.delete("/api/v1/session").status_code == 200
    assert len(session_store._memory_store) == 1

    client.cookies.set("econmind_sid", "a" * 32, domain="testserver.local", path="/")
    assert client.get("/api/v1/session").status_code == 200
    assert len(session_store._memory_store) == 2
    client.cookies.clear()
    assert client.get("/api/v1/session").status_code == 429
    assert len(session_store._memory_store) == 2


def test_paid_demo_disabled_by_default_policy(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paid_demo_enabled", False)
    assert TestClient(app).get("/api/v1/news/missing/related?tier=PAID").status_code == 403


def test_reports_and_strategies_stay_with_creating_session(monkeypatch) -> None:
    store.news_cache["news-1"] = {
        "news_id": "news-1", "title": "기사", "url": "https://example.com/article",
        "cleaned_content": "기사 본문", "content_source_url": "https://example.com/article",
    }

    async def generate_report(*_args):
        return {"title": "보고서", "summary": "요약", "event_analysis": "분석",
                "market_impact": "영향", "risk_factors": []}

    async def generate_strategy(**_kwargs):
        return {"expected_return": 1.0, "risk": "low", "strategy_summary": "요약", "strategy_items": []}

    monkeypatch.setattr(reports, "generate_report", generate_report)
    monkeypatch.setattr(strategies, "generate_strategy", generate_strategy)
    owner = TestClient(app)
    other = TestClient(app)
    first = owner.post("/api/v1/reports", json={"news_id": "news-1", "ticker_symbols": ["AAA"]})
    assert first.status_code == 201
    first_id = first.json()["report_id"]
    assert owner.get(f"/api/v1/reports/{first_id}").json()["related_stocks"] == ["AAA"]
    assert other.get(f"/api/v1/reports/{first_id}").status_code == 404
    assert other.post("/api/v1/strategies", json={"report_id": first_id}).status_code == 404

    # Same public article, different caller or ticker selection must not reuse the report.
    second = other.post("/api/v1/reports", json={"news_id": "news-1", "ticker_symbols": ["BBB"]})
    assert second.status_code == 201
    assert second.json()["report_id"] != first_id
    assert other.get(f"/api/v1/reports/{second.json()['report_id']}").json()["related_stocks"] == ["BBB"]
    third = owner.post("/api/v1/reports", json={"news_id": "news-1", "ticker_symbols": ["CCC"]})
    assert third.json()["report_id"] != first_id

    strategy = owner.post("/api/v1/strategies", json={"report_id": first_id, "risk_level": "low"})
    assert strategy.status_code == 201
    strategy_id = strategy.json()["strategy_id"]
    assert other.get(f"/api/v1/strategies/{strategy_id}").status_code == 404
    assert owner.get(f"/api/v1/strategies/{strategy_id}").status_code == 200


def test_job_result_requires_owner_and_submission_is_limited(monkeypatch) -> None:
    records = {}

    async def save(job_id, data):
        records[job_id] = data

    async def read(job_id):
        return records.get(job_id)

    async def publish(_payload):
        return None

    monkeypatch.setattr(jobs, "set_job", save)
    monkeypatch.setattr(jobs, "get_job", read)
    monkeypatch.setattr(jobs, "publish_job", publish)
    owner = TestClient(app)
    other = TestClient(app)
    first = owner.post("/jobs", json={"keyword": "반도체"})
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    assert "owner_sid" not in owner.get(f"/jobs/{job_id}").json()
    assert other.get(f"/jobs/{job_id}").status_code == 404
    assert owner.post("/jobs", json={"keyword": "x" * 101}).status_code == 422
    for _ in range(9):
        assert owner.post("/jobs", json={"keyword": "반도체"}).status_code == 202
    assert owner.post("/jobs", json={"keyword": "반도체"}).status_code == 429


def test_public_news_routes_share_search_budget_and_map_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "news_search_client_hourly_limit", 2)
    monkeypatch.setattr(settings, "news_search_global_hourly_limit", 100)
    monkeypatch.setattr(settings, "news_map_client_hourly_limit", 1)
    monkeypatch.setattr(settings, "news_map_global_hourly_limit", 100)
    client = TestClient(app)

    # Admission occurs before provider work, even when the requested ID is missing.
    assert client.get("/api/v1/news/missing/source").status_code == 404
    assert client.get("/api/v1/news/missing/source").status_code == 404
    assert client.get("/api/v1/news/search?q=반도체").status_code == 429
    assert client.get("/api/v1/news/missing/graph").status_code == 404
    assert client.get("/api/v1/news/missing/related").status_code == 429


def test_public_news_work_fails_closed_when_required_quota_store_is_down(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_store_required", True)
    client = TestClient(app)
    assert client.get("/api/v1/news/search?q=반도체").status_code == 503
    assert client.get("/api/v1/news/missing/graph").status_code == 503
    assert client.get("/health").status_code == 200


def test_foreign_origin_cannot_mutate_cookie_session() -> None:
    client = TestClient(app)
    path = "/api/v1/session/mindmap/expand"
    payload = {"news_id": "private"}
    denied = client.post(path, json=payload, headers={"Origin": "https://foreign.example"})
    assert denied.status_code == 403
    assert client.get("/api/v1/session").json()["mindmap"]["expanded_news_ids"] == []

    allowed = client.post(path, json=payload, headers={"Origin": "http://localhost:5173"})
    assert allowed.status_code == 200
    assert client.get("/api/v1/session").json()["mindmap"]["expanded_news_ids"] == ["private"]
