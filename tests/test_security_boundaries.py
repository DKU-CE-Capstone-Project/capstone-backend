"""Regression tests for ownership, session rotation, and admission controls."""

from fastapi.testclient import TestClient

from app import store
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
