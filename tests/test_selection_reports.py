"""POST /reports news_ids: async selection reports with stage, evidence status and reuse (docs/10 § 3.5·3.6)."""
import json

import pytest
from fastapi.testclient import TestClient

from app import report_jobs, store
from app.agents import naver_categories, report_generator
from app.main import app

ARTICLES = {
    nid: {
        "news_id": nid, "title": f"{name} 실적 개선 기대 {nid}", "description": f"{name} 관련 설명 {nid}",
        "url": f"https://publisher.test/{nid}", "source": "publisher.test",
        "published_at": "2026-10-01T09:00:00Z", "_search_keyword": "반도체",
    }
    for nid, name in (("a", "삼성전자"), ("b", "SK하이닉스"), ("c", "한미반도체"))
}

GOOD = {
    "title": "메모리 업황 리포트", "summary": "요약", "event_analysis": "분석", "market_impact": "영향",
    "risk_factors": ["환율"],
    "stock_impacts": [
        {"name": "삼성전자", "ticker": "005930", "direction": "up", "action": "buy", "comment": "수요 회복"},
        {"name": "SK하이닉스", "ticker": "SK하이닉스", "direction": "mixed", "action": "teleport", "comment": ""},
        {"name": "가상전자", "ticker": "999999", "direction": "up", "action": "buy", "comment": "근거 없음"},
        {"name": "한미반도체", "direction": "sideways", "action": "hold"},
    ],
    "strategy": {"stance": "분할 관심", "rationale": "근거", "watchlist": ["삼성전자", "가상전자"],
                 "risk_warning": "변동성"},
}


@pytest.fixture
def flow(monkeypatch):
    monkeypatch.setattr(report_jobs, "_redis_disabled", True)
    report_jobs.reset_memory()
    store.news_cache.update({k: dict(v) for k, v in ARTICLES.items()})
    calls = {"extract": [], "prompts": []}
    behaviour = {"bodies": {"a": "삼성전자 본문", "b": "SK하이닉스 본문", "c": ""}, "llm": json.dumps(GOOD)}

    async def extract(articles, **kwargs):
        article = articles[0]
        calls["extract"].append(article["news_id"])
        body = behaviour["bodies"].get(article["news_id"], "")
        if not body:
            return [article]
        return [{**article, "cleaned_content": body, "content_source_url": article["url"]}]

    async def generate(prompt):
        calls["prompts"].append(prompt)
        return behaviour["llm"]

    monkeypatch.setattr(report_jobs, "extract_articles_with_diffbot", extract)
    monkeypatch.setattr(report_generator, "generate", generate)
    yield TestClient(app), calls, behaviour
    report_jobs.reset_memory()


def create(client, ids, expected=202):
    response = client.post("/api/v1/reports", json={"news_ids": ids})
    assert response.status_code == expected, response.text
    return response.json()


def test_selection_report_completes_with_evidence_status_and_grounded_stocks(flow):
    client, calls, _ = flow
    created = create(client, ["a", "b", "c", "a"])
    assert created["status"] == "pending" and created["report_id"].startswith("rpt_")

    report = client.get(f"/api/v1/reports/{created['report_id']}").json()
    assert (report["status"], report["stage"], report["progress"], report["error"]) == ("completed", "done", None, None)
    assert report["requested_news_ids"] == ["a", "b", "c"]
    assert [(e["news_id"], e["body_status"]) for e in report["evidence_news"]] == [
        ("a", "extracted"), ("b", "extracted"), ("c", "description_only")]
    assert all(e["source_url"] and e["published_at"] for e in report["evidence_news"])
    # Only companies named in the evidence, with allowed values; unknown actions become watch.
    assert report["stock_impacts"] == [
        {"name": "삼성전자", "ticker": "005930", "direction": "up", "action": "buy", "comment": "수요 회복"},
        {"name": "SK하이닉스", "ticker": "", "direction": "mixed", "action": "watch", "comment": ""},
    ]
    assert report["strategy"] == {"stance": "분할 관심", "rationale": "근거", "watchlist": ["삼성전자"],
                                  "risk_warning": "변동성"}
    assert report["is_fallback"] is False and report["summary"] == "요약"
    assert sorted(calls["extract"]) == ["a", "b", "c"]
    assert "삼성전자 본문" in calls["prompts"][0] and "(본문 없음" in calls["prompts"][0]


def test_same_article_set_in_any_order_reuses_the_completed_report(flow):
    client, calls, _ = flow
    first = create(client, ["a", "b"])
    again = create(client, ["b", "a"], expected=200)
    assert again == {**again, "report_id": first["report_id"], "status": "completed"}
    assert len(calls["prompts"]) == 1
    assert create(client, ["a", "c"])["report_id"] != first["report_id"]


def test_all_extractions_failing_is_a_failed_report_without_internal_details(flow):
    client, calls, behaviour = flow
    behaviour["bodies"] = {}
    report = client.get(f"/api/v1/reports/{create(client, ['a', 'b'])['report_id']}").json()
    assert (report["status"], report["stage"]) == ("failed", "extracting")
    assert report["error"]["code"] == "body_extraction_failed"
    assert report["summary"] == "" and calls["prompts"] == []


def test_llm_failure_is_flagged_and_never_reused(flow):
    client, _, behaviour = flow
    behaviour["llm"] = ""
    first = create(client, ["a"])
    report = client.get(f"/api/v1/reports/{first['report_id']}").json()
    assert report["status"] == "completed" and report["is_fallback"] is True
    assert report["stock_impacts"] == [] and report["strategy"] is None
    behaviour["llm"] = json.dumps(GOOD)
    second = create(client, ["a"])
    assert second["report_id"] != first["report_id"]
    assert client.get(f"/api/v1/reports/{second['report_id']}").json()["is_fallback"] is False


def test_political_evidence_blocks_the_report(flow, monkeypatch):
    client, calls, _ = flow
    store.news_cache["b"]["naver_url"] = "https://n.news.naver.com/article/001/2"

    async def categories(url):
        return ["정치"]

    monkeypatch.setattr(naver_categories, "fetch_categories", categories)
    report = client.get(f"/api/v1/reports/{create(client, ['a', 'b'])['report_id']}").json()
    assert report["status"] == "failed" and report["error"]["code"] == "category_blocked"
    assert calls["extract"] == []


async def test_in_progress_state_has_stage_and_no_content(flow):
    client, _, _ = flow
    client.get("/api/v1/session")
    state = await report_jobs.create(["a", "b"], session_id=client.cookies.get("econmind_sid"), key="k1")
    state.update(status="processing", stage="extracting", progress={"done": 1, "total": 2})
    await report_jobs._save_state(state)
    report = client.get(f"/api/v1/reports/{state['report_id']}").json()
    assert (report["status"], report["stage"], report["progress"]) == ("processing", "extracting", {"done": 1, "total": 2})
    assert report["requested_news_ids"] == ["a", "b"] and report["summary"] == "" and report["stock_impacts"] == []


def test_selection_reports_are_owned_and_not_reused_across_sessions(flow):
    owner, calls, _ = flow
    other = TestClient(app)
    first = create(owner, ["a", "b"])
    assert other.get(f"/api/v1/reports/{first['report_id']}").status_code == 404
    second = create(other, ["a", "b"])
    assert second["report_id"] != first["report_id"] and len(calls["prompts"]) == 2
    assert owner.get(f"/api/v1/reports/{second['report_id']}").status_code == 404


async def test_pending_and_failed_selection_states_are_private(flow):
    owner, _, _ = flow
    owner.get("/api/v1/session")
    state = await report_jobs.create(["a"], session_id=owner.cookies.get("econmind_sid"), key="private")
    other = TestClient(app)
    path = f"/api/v1/reports/{state['report_id']}"
    assert other.get(path).status_code == 404
    state.update(status="failed", error={"code": "internal", "message": "실패"})
    await report_jobs._save_state(state)
    assert other.get(path).status_code == 404
    assert owner.get(path).json()["status"] == "failed"


def test_selection_generation_shares_the_report_quota_and_reuse_is_free(flow):
    client, _, _ = flow
    first = create(client, ["a"])
    for ids in (["b"], ["c"], ["a", "b"], ["a", "c"]):
        create(client, ids)
    assert client.post("/api/v1/reports", json={"news_ids": ["b", "c"]}).status_code == 429
    assert create(client, ["a"], expected=200)["report_id"] == first["report_id"]


async def test_concurrent_jobs_reserve_only_one_slot_per_session(flow):
    import asyncio

    from fastapi import HTTPException

    results = await asyncio.gather(
        report_jobs.create(["a"], session_id="same", key="one"),
        report_jobs.create(["b"], session_id="same", key="two"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 1
    refused = next(result for result in results if isinstance(result, HTTPException))
    assert refused.status_code == 409
    state = next(result for result in results if isinstance(result, dict))
    await report_jobs._set("report_active:same", "newer", 60)
    await report_jobs._release("report_active:same", state["report_id"])
    assert await report_jobs._get("report_active:same") == "newer"


async def test_report_job_store_requires_shared_storage_when_configured(flow, monkeypatch):
    from app.config import settings
    from app.session import SessionStoreUnavailable

    monkeypatch.setattr(settings, "session_store_required", True)
    with pytest.raises(SessionStoreUnavailable):
        await report_jobs.create(["a"], session_id="owner", key="key")


async def test_second_report_in_the_same_session_is_409_while_one_runs(flow):
    client, _, _ = flow
    client.get("/api/v1/session")
    sid = client.cookies.get("econmind_sid")
    running = await report_jobs.create(["c"], session_id=sid, key="other")
    response = client.post("/api/v1/reports", json={"news_ids": ["a"]})
    assert response.status_code == 409
    assert response.json()["detail"]["report_id"] == running["report_id"]


@pytest.mark.parametrize("payload, status", [
    ({"news_ids": ["a", "missing"]}, 404),
    ({"news_ids": []}, 422),
    ({"news_ids": ["a", "b", "c", "d", "e", "f"]}, 422),
    ({}, 422),
])
def test_invalid_selection_requests(flow, payload, status):
    client, _, _ = flow
    assert client.post("/api/v1/reports", json=payload).status_code == status


def test_unknown_report_is_404(flow):
    client, _, _ = flow
    assert client.get("/api/v1/reports/rpt_missing").status_code == 404


def test_schema_v2_document_round_trips_to_the_api_shape():
    from app import database
    result = {"report_id": "rpt_x", "summary": "s", "event_analysis": "e", "market_impact": "m",
              "risk_factors": ["r"], "evidence_news": [{"news_id": "a"}], "requested_news_ids": ["a"],
              "reuse_key": "k", "is_fallback": False, "title": "t",
              "created_at": "2026-10-04T08:00:00+00:00", "updated_at": "2026-10-04T08:00:05+00:00"}
    doc = database.report_doc_from_result(result)
    assert doc["schema_version"] == 2 and doc["source_news_ids"] == ["a"] and doc["reuse_key"] == "k"
    assert doc["sections"] == {"summary": "s", "event_analysis": "e", "market_impact": "m", "risk_factors": ["r"]}
    assert "summary" not in doc and doc["evidence"] == [{"news_id": "a"}]
    back = database.report_from_doc(doc)
    assert back["summary"] == "s" and back["evidence_news"] == [{"news_id": "a"}]
    assert back["requested_news_ids"] == ["a"] and back["created_at"].endswith("Z")
    legacy = {"report_id": "old", "summary": "x"}
    assert database.report_from_doc(legacy) is legacy
