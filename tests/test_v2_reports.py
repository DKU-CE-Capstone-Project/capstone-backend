"""Offline report pipeline regressions: actual job states, evidence and security."""
import asyncio
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime

import pytest
from starlette.requests import Request

from app import session as base_session
from app import store
from app.agents import strategy_generator
from app.api.v2 import report_jobs as api
from app.config import settings
from app.v2 import persistence, report_worker, reports, state
from app.v2.errors import V2Error


@pytest.fixture(autouse=True)
def v2_offline(monkeypatch):
    monkeypatch.setattr(settings, "v2_state_memory", True)
    monkeypatch.setattr(settings, "prototype_tier", "PAID")
    state._memory.clear()
    for collection in persistence._memory.values():
        collection.clear()


@pytest.fixture
def pipeline(monkeypatch):
    calls = {"extract": [], "analyze": [], "strategy": []}

    async def extract(articles, **kwargs):
        calls["extract"].append(articles[0]["news_id"])
        return [{**articles[0], "cleaned_content": "실제 추출 본문 근거입니다. " * 30}]

    async def generate(prompt):
        payload = json.loads(prompt.rsplit("\n", 1)[-1])
        calls["analyze"].append(payload)
        ids = [e["news_id"] for e in payload["used_evidence"]]
        return json.dumps({"title": "검증용 경제 리포트", "event_summary": "본문으로 분석한 사건", "event_analysis": "실제 원문 근거의 분석", "market_impact": "시장에 대한 영향", "stock_impacts": [{"ticker": None, "name": "반도체 업종", "impact": "본문에서 확인한 투자 영향", "direction": "unknown", "evidence_news_ids": ids}], "risk_factors": ["불확실성"]})

    async def strategy(report, risk_level="medium", period="short"):
        calls["strategy"].append((report, risk_level, period))
        return {"strategy_summary": "단기 관망 전략", "risk_warning": "투자 결과를 보장하지 않습니다.", "strategy_items": [{"ticker": None, "stock_name": "반도체 업종", "action": "watch", "reason": "불확실성 확인"}]}

    async def publish(*args):
        return None

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", extract)
    monkeypatch.setattr(reports.llm, "generate", generate)
    monkeypatch.setattr(reports, "generate_strategy", strategy)
    monkeypatch.setattr(report_worker, "publish", publish)
    return calls


def request(sid):
    result = Request({"type": "http", "method": "POST", "path": "/api/v2/report-jobs", "headers": [], "client": ("127.0.0.1", 1234)})
    result.state.session_id = sid
    return result


async def fixture_map(count=1):
    sid = await base_session.issue("127.0.0.1")
    session = await state.session(sid)
    map_id = str(uuid.uuid4())
    ids = [f"article-{i}" for i in range(count)]
    for news_id in ids:
        store.news_cache[news_id] = {"news_id": news_id, "title": f"경제 뉴스 {news_id}", "description": "검색 설명", "source": "테스트 매체", "url": f"https://fixture.invalid/{news_id}", "published_at": "2026-10-11T00:00:00Z", "cleaned_content": "검증되지 않은 레거시 본문", "content_source_url": f"https://fixture.invalid/{news_id}"}
    map_doc = {"map_id": map_id, "revision": 1, "nodes": [{"news_id": nid} for nid in ids], "selected_news_ids": [], "report_panel_open": False, "active_report_job_id": None, "last_result": None, "expires_at": session["expires_at"]}
    await state.put(state.key(sid, "map", map_id), map_doc, settings.session_ttl_seconds)
    await state.register_resource(sid, "map", map_id, session["expires_at"])
    return sid, map_id, ids


async def submit(sid, map_id, ids, *, key=None, body=None, options=None):
    map_doc = await state.resource(sid, "map", map_id)
    payload = body or api.ReportJobCreate(map_id=map_id, expected_revision=map_doc["revision"], client_request_id=uuid.uuid4(), news_ids=ids, strategy_options=options or {})
    return await api.create_job(payload, request(sid), key or str(uuid.uuid4()))


async def finish(sid, job):
    assert await reports.process_job(sid, job["job_id"]) == "done"
    return await reports.get_job(sid, job["job_id"])


async def test_success_provenance_and_access_boundary(pipeline):
    sid, map_id, ids = await fixture_map(2)
    admitted = await submit(sid, map_id, ids)
    assert admitted["status"] == "queued" and admitted["accepted_map_revision"] == 2
    assert "session_epoch" not in admitted and "request" not in admitted
    job = await finish(sid, admitted)
    assert job["status"] == "completed"
    assert job["progress"] == {"requested": 2, "processed": 2, "succeeded": 2, "failed": 0}
    report = await persistence.report(job["result"]["report_id"])
    assert report["verification"]["status"] == "unavailable"
    assert isinstance(report["created_at"], datetime)
    assert all(isinstance(e["published_at"], datetime) for e in report["used_evidence"])
    assert report["used_evidence"][0]["input_excerpt"] != "검증되지 않은 레거시 본문"
    exposed = await api.get_report(report["report_id"], request(sid))
    assert "input_excerpt" not in exposed["used_evidence"][0]
    assert "generation_key" not in exposed and exposed["created_at"].endswith("Z")
    strategy = await api.get_strategy(job["result"]["strategy_id"], request(sid))
    assert strategy["report_id"] == report["report_id"]
    assert "expected_return" not in strategy and strategy["strategy_items"][0]["action"] == "watch"
    other_sid = await base_session.issue("127.0.0.1")
    with pytest.raises(V2Error, match="찾을 수 없는") as caught:
        await api.get_report(report["report_id"], request(other_sid))
    assert caught.value.status == 404
    final_map = await state.resource(sid, "map", map_id)
    assert final_map["active_report_job_id"] is None and final_map["revision"] == 3
    assert final_map["last_result"]["report_id"] == report["report_id"]


async def test_reordered_set_reuses_persistent_result_and_options_only_change_strategy(pipeline):
    sid, map_id, ids = await fixture_map(2)
    first = await finish(sid, await submit(sid, map_id, ids))
    store.report_cache.clear()
    store.report_index.clear()
    second = await finish(sid, await submit(sid, map_id, ids[::-1]))
    assert second["result"]["report_id"] == first["result"]["report_id"]
    assert second["result"]["report_reused"] and second["result"]["strategy_reused"]
    third = await finish(sid, await submit(sid, map_id, ids, options={"risk_level": "low", "period": "long"}))
    assert third["result"]["report_id"] == first["result"]["report_id"]
    assert third["result"]["strategy_id"] != first["result"]["strategy_id"]
    assert len(pipeline["analyze"]) == 1 and len(pipeline["extract"]) == 2


async def test_partial_extraction_does_not_enter_prompt_or_complete_cache(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map(2)
    original = reports.diffbot_client.extract_articles_with_diffbot

    async def partial(articles, **kwargs):
        return [] if articles[0]["news_id"] == ids[1] else await original(articles, **kwargs)

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", partial)
    job = await finish(sid, await submit(sid, map_id, ids))
    assert job["status"] == "completed" and job["warnings"] == ["PARTIAL_EVIDENCE"]
    report = await persistence.report(job["result"]["report_id"])
    assert report["coverage"] == {"requested_count": 2, "used_count": 1, "failed_count": 1}
    assert report["failed_evidence"][0]["news_id"] == ids[1]
    assert report["reuse_eligible"] is False
    assert [e["news_id"] for e in pipeline["analyze"][0]["used_evidence"]] == [ids[0]]
    assert await persistence.reusable_report(report["generation_key"]) is None


async def test_no_usable_evidence_and_no_model_call(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()

    async def empty(*args, **kwargs):
        return []

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", empty)
    job = await finish(sid, await submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "NO_USABLE_EVIDENCE"
    assert job["result"] is None and pipeline["analyze"] == []
    assert persistence._memory["reports"] == {}


@pytest.mark.parametrize("raw", ["", "not json", '{"title":"incomplete"}'])
async def test_generation_failures_never_save_fake_results(pipeline, monkeypatch, raw):
    sid, map_id, ids = await fixture_map()

    async def invalid(*args):
        return raw

    monkeypatch.setattr(reports.llm, "generate", invalid)
    job = await finish(sid, await submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["result"] is None
    assert persistence._memory["reports"] == {}


async def test_stock_impacts_must_reference_successful_selected_evidence(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    original = reports.llm.generate

    async def invalid(prompt):
        parsed = json.loads(await original(prompt))
        parsed["stock_impacts"][0]["evidence_news_ids"] = ["unselected"]
        return json.dumps(parsed)

    monkeypatch.setattr(reports.llm, "generate", invalid)
    assert (await finish(sid, await submit(sid, map_id, ids)))["status"] == "failed"
    assert persistence._memory["reports"] == {}


async def test_strategy_failure_preserves_report_and_retry_generates_only_strategy(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    original = reports.generate_strategy

    async def fail(*args, **kwargs):
        raise ValueError("provider unavailable")

    monkeypatch.setattr(reports, "generate_strategy", fail)
    first = await finish(sid, await submit(sid, map_id, ids))
    assert first["status"] == "completed" and first["warnings"] == ["STRATEGY_GENERATION_FAILED"]
    assert first["result"]["strategy_id"] is None
    monkeypatch.setattr(reports, "generate_strategy", original)
    second = await finish(sid, await submit(sid, map_id, ids))
    assert second["result"]["report_id"] == first["result"]["report_id"]
    assert second["result"]["strategy_id"] and len(pipeline["analyze"]) == 1


async def test_idempotent_publish_failure_retry_and_conflicting_input(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    calls = []

    async def fail_once(sid, map_id, job_id):
        calls.append(job_id)
        if len(calls) == 1:
            raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "queue unavailable", True)

    monkeypatch.setattr(report_worker, "publish", fail_once)
    body = api.ReportJobCreate(map_id=map_id, expected_revision=1, client_request_id=uuid.uuid4(), news_ids=ids)
    key = str(uuid.uuid4())
    with pytest.raises(V2Error) as caught:
        await submit(sid, map_id, ids, body=body, key=key)
    assert caught.value.status == 503
    assert (await state.resource(sid, "map", map_id))["revision"] == 2
    second = await submit(sid, map_id, ids, body=body, key=key)
    assert calls == [second["job_id"], second["job_id"]]
    assert (await state.resource(sid, "map", map_id))["revision"] == 2
    changed = body.model_copy(update={"language": "en"})
    with pytest.raises(V2Error) as conflict:
        await submit(sid, map_id, ids, body=changed, key=key)
    assert conflict.value.code == "IDEMPOTENCY_CONFLICT"


async def test_free_limit_and_unselected_article_are_rejected_without_job(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "prototype_tier", "FREE")
    sid, map_id, ids = await fixture_map(2)
    with pytest.raises(V2Error) as caught:
        await submit(sid, map_id, ids)
    assert caught.value.code == "PLAN_ARTICLE_LIMIT_EXCEEDED"
    assert await state.keys(sid, "report-job") == []
    with pytest.raises(V2Error) as caught:
        await submit(sid, map_id, ["other"])
    assert caught.value.code == "INVALID_REQUEST"


async def test_queued_cancel_never_processes(pipeline):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    cancelled = await api.cancel_job(admitted["job_id"], request(sid))
    assert cancelled["status"] == "cancelled"
    assert await reports.process_job(sid, admitted["job_id"]) == "done"
    assert pipeline["extract"] == [] and persistence._memory["reports"] == {}


@pytest.mark.parametrize("mode", ["cancel", "detach", "expire"])
async def test_stage_boundary_cancel_stale_map_or_expired_session_blocks_result(pipeline, monkeypatch, mode):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    original = reports.llm.generate

    async def change_during_call(prompt):
        if mode == "cancel":
            await reports.request_cancel(sid, admitted["job_id"])
        elif mode == "detach":
            current = await state.resource(sid, "map", map_id)
            current["active_report_job_id"] = "different-job"
            await state.put(state.key(sid, "map", map_id), current, settings.session_ttl_seconds)
        else:
            await base_session.clear(sid)
        return await original(prompt)

    monkeypatch.setattr(reports.llm, "generate", change_during_call)
    assert await reports.process_job(sid, admitted["job_id"]) == "done"
    stored = await state.get(state.key(sid, "report-job", admitted["job_id"]))
    assert stored["status"] == "cancelled" and persistence._memory["reports"] == {}


async def test_duplicate_delivery_does_not_run_and_expired_lease_recovers(pipeline):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    job = await reports.get_job(sid, admitted["job_id"])
    job.update(worker_token="old-worker", lease_until=time.time() + 60, status="running")
    await reports._save_job(sid, job)
    assert await reports.process_job(sid, job["job_id"]) == "busy" and pipeline["extract"] == []
    job["lease_until"] = time.time() - 1
    await reports._save_job(sid, job)
    recovered = await finish(sid, admitted)
    assert recovered["status"] == "completed" and recovered["worker_retries"] == 1


async def test_persisted_pipeline_start_limit_survives_shutdown_and_duplicate(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    calls = []

    async def interrupted(*args):
        calls.append(args)
        raise asyncio.CancelledError()

    monkeypatch.setattr(reports, "_pipeline", interrupted)
    monkeypatch.setattr(settings, "v2_worker_max_attempts", 2)
    for count in (1, 2):
        with pytest.raises(asyncio.CancelledError):
            await reports.process_job(sid, admitted["job_id"])
        stored = await state.resource(sid, "report-job", admitted["job_id"])
        assert stored["worker_attempts"] == count
        # Redelivery during the current lease neither reruns nor counts.
        assert await reports.process_job(sid, admitted["job_id"]) == "busy"
        assert len(calls) == count
        stored["lease_until"] = time.time() - 1
        await reports._save_job(sid, stored)
    assert await reports.process_job(sid, admitted["job_id"]) == "done"
    stored = await reports.get_job(sid, admitted["job_id"])
    assert stored["status"] == "failed" and stored["error"]["code"] == "WORKER_LOST"
    assert stored["worker_attempts"] == 2 and len(calls) == 2
    assert (await state.resource(sid, "map", map_id))["active_report_job_id"] is None
    assert await reports.process_job(sid, admitted["job_id"]) == "done"
    assert len(calls) == 2


async def test_legacy_retry_count_cannot_reset_attempt_budget(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    job = await reports.get_job(sid, admitted["job_id"])
    job.update(worker_token="legacy-worker", worker_retries=1, lease_until=time.time() - 1, status="running")
    await reports._save_job(sid, job)
    monkeypatch.setattr(settings, "v2_worker_max_attempts", 2)
    assert await reports.process_job(sid, job["job_id"]) == "done"
    result = await reports.get_job(sid, job["job_id"])
    assert result["error"]["code"] == "WORKER_LOST"
    assert result["worker_attempts"] == 2
    assert pipeline["extract"] == [] and pipeline["analyze"] == []


async def test_worker_token_loss_cannot_save_or_link(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    original = reports.llm.generate

    async def replace_worker(prompt):
        job = await reports.get_job(sid, admitted["job_id"])
        job["worker_token"] = "replacement-worker"
        await reports._save_job(sid, job)
        return await original(prompt)

    monkeypatch.setattr(reports.llm, "generate", replace_worker)
    assert await reports.process_job(sid, admitted["job_id"]) == "busy"
    assert persistence._memory["reports"] == {}
    assert (await state.resource(sid, "map", map_id))["last_result"] is None


async def test_ten_article_input_budget_and_body_versions_immutable(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map(10)

    async def large(articles, **kwargs):
        return [{**articles[0], "cleaned_content": "본문 근거입니다. " * 2000}]

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", large)
    job = await finish(sid, await submit(sid, map_id, ids))
    report = await persistence.report(job["result"]["report_id"])
    assert len(report["used_evidence"]) == 10
    assert sum(e["input_char_count"] for e in report["used_evidence"]) <= 60000
    assert all(e["input_char_count"] <= 6000 and e["truncated"] for e in report["used_evidence"])
    original = deepcopy(next(iter(persistence._memory["news_bodies"].values())))
    with pytest.raises(ValueError, match="Immutable"):
        await persistence.save_body({**original, "content": "changed", "content_hash": reports.text_hash("changed")})
    assert persistence._memory["news_bodies"][original["body_id"]]["content"] == original["content"]


async def test_critic_invalid_response_is_unavailable_and_valid_warning_preserved(monkeypatch):
    monkeypatch.setattr(settings, "use_critic", True)

    async def malformed(prompt):
        return '{"confidence":0.99}'

    monkeypatch.setattr(reports.llm, "generate", malformed)
    assert (await reports.verify({}, [], []))["status"] == "unavailable"

    async def warning(prompt):
        return json.dumps({"grounded": False, "confidence": 0.7, "issues": ["근거 부족"], "unsupported_claims": [], "invalid_tickers": []})

    monkeypatch.setattr(reports.llm, "generate", warning)
    assert await reports.verify({}, [], []) == {"status": "warning", "notes": ["근거 부족"]}


async def test_strict_strategy_never_returns_legacy_demo_portfolio(monkeypatch):
    async def empty(prompt):
        return ""

    monkeypatch.setattr(strategy_generator, "generate", empty)
    with pytest.raises(ValueError):
        await strategy_generator.generate_strategy("본문 분석", strict=True)
    assert (await strategy_generator.generate_strategy("본문 분석"))["strategy_items"]


async def test_production_mode_never_uses_fixture_result_store(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "session_store_required", True)
    with pytest.raises(V2Error) as caught:
        await persistence.ready()
    assert caught.value.status == 503


async def test_simultaneous_sessions_reuse_result_without_sharing_jobs(pipeline, monkeypatch):
    sid1, map1, ids = await fixture_map()
    sid2, map2, _ = await fixture_map()
    first = await submit(sid1, map1, ids)
    second = await submit(sid2, map2, ids)
    original = reports.llm.generate

    async def slow(prompt):
        await asyncio.sleep(0.02)
        return await original(prompt)

    monkeypatch.setattr(reports.llm, "generate", slow)
    await asyncio.gather(reports.process_job(sid1, first["job_id"]), reports.process_job(sid2, second["job_id"]))
    result1 = await reports.get_job(sid1, first["job_id"])
    result2 = await reports.get_job(sid2, second["job_id"])
    assert first["job_id"] != second["job_id"]
    assert result1["result"]["report_id"] == result2["result"]["report_id"]
    assert len(pipeline["analyze"]) == 1


async def test_publish_payload_subject_and_message_id(monkeypatch):
    from app.v2 import report_worker
    recorded = []

    class JetStream:
        async def publish(self, subject, payload, headers):
            recorded.append((subject, json.loads(payload), headers))

    async def js():
        return JetStream()

    monkeypatch.setattr(report_worker, "get_jetstream", js)
    await report_worker.publish("session", "map", "job")
    assert recorded == [("reports.generate.v2", {"job_id": "job", "session_id": "session", "map_id": "map", "job_type": "report_v2"}, {"Nats-Msg-Id": "job"})]


async def test_refresh_poll_reconciles_failed_dispatch_without_new_job_or_ttl_touch(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()

    async def fail(*args):
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "queue unavailable")

    monkeypatch.setattr(report_worker, "publish", fail)
    with pytest.raises(V2Error):
        await submit(sid, map_id, ids)
    job_id = (await state.resource(sid, "map", map_id))["active_report_job_id"]
    stored = await state.resource(sid, "report-job", job_id)
    stored["last_dispatch_attempt"] = 0
    await reports._save_job(sid, stored)
    old_expiry = (await state.session(sid))["expires_at"]
    published = []

    async def succeed(*args):
        published.append(args)

    monkeypatch.setattr(report_worker, "publish", succeed)
    recovered = await api.get_job(job_id, request(sid))
    assert recovered["status"] == "queued" and published[0][-1] == job_id
    assert await state.keys(sid, "report-job") == [job_id]
    assert (await state.resource(sid, "map", map_id))["revision"] == 2
    assert (await state.session(sid))["expires_at"] == old_expiry


async def test_dispatch_reconciliation_deadline_is_terminal_and_clears_map(pipeline):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    stored = await state.resource(sid, "report-job", admitted["job_id"])
    stored.update(dispatch_state="failed", deadline_at=state.iso(time.time() - 1))
    await reports._save_job(sid, stored)
    assert (await reports.get_job(sid, admitted["job_id"]))["error"]["code"] == "JOB_TIMEOUT"
    assert (await state.resource(sid, "map", map_id))["active_report_job_id"] is None


async def test_worker_shutdown_is_recoverable_not_user_cancel(pipeline, monkeypatch):
    sid, map_id, ids = await fixture_map()
    admitted = await submit(sid, map_id, ids)
    entered = asyncio.Event()
    original = reports.llm.generate

    async def blocked(prompt):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(reports.llm, "generate", blocked)
    task = asyncio.create_task(reports.process_job(sid, admitted["job_id"]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stored = await state.resource(sid, "report-job", admitted["job_id"])
    assert stored["status"] == "running"
    stored["lease_until"] = time.time() - 1
    await reports._save_job(sid, stored)
    monkeypatch.setattr(reports.llm, "generate", original)
    assert (await finish(sid, admitted))["status"] == "completed"


async def test_report_storage_rejects_missing_body_reference_or_changed_excerpt(pipeline):
    sid, map_id, ids = await fixture_map()
    job = await finish(sid, await submit(sid, map_id, ids))
    original = await persistence.report(job["result"]["report_id"])
    missing = deepcopy(original)
    missing["used_evidence"][0]["body_id"] = str(uuid.uuid4())
    with pytest.raises(ValueError, match="body reference"):
        await persistence.save_report(missing)
    modified = deepcopy(original)
    modified["used_evidence"][0]["input_excerpt"] = "다른 본문"
    modified["used_evidence"][0]["input_char_count"] = len("다른 본문")
    with pytest.raises(ValueError, match="recorded body prefix"):
        await persistence.save_report(modified)


@pytest.mark.parametrize("stock_name,ticker,accepted", [
    ("외화 자산", None, True),
    ("외화 자산", "USDKRW", False),
    ("새로운 ETF", None, False),
    ("관련 없는 종목", "005930", False),
])
async def test_strategy_uses_only_report_assets_and_cannot_invent_tickers(monkeypatch, stock_name, ticker, accepted):
    report = {
        "event_summary": "환율 하락", "event_analysis": "분할 대응", "market_impact": "변동성",
        "stock_impacts": [{"name": "외화 자산", "ticker": None, "impact": "환율 변동", "direction": "mixed", "evidence_news_ids": ["selected"]}],
        "risk_factors": ["평가 손실"], "verification": {"status": "warning", "notes": ["추가 확인 필요"]},
    }
    prompts = []
    async def generate(prompt):
        prompts.append(prompt)
        return json.dumps({"strategy_summary": "분할 대응 참고", "risk_warning": "성과를 보장하지 않습니다.", "strategy_items": [{"stock_name": stock_name, "ticker": ticker, "action": "watch", "reason": "추가 확인"}]})
    monkeypatch.setattr(reports.llm, "generate", generate)
    if accepted:
        output = await reports.generate_strategy(report, risk_level="low", period="long")
        assert output["strategy_items"][0]["ticker"] is None
    else:
        with pytest.raises(ValueError, match="asset absent"):
            await reports.generate_strategy(report, risk_level="low", period="long")
    data = json.loads(prompts[0].rsplit("\n", 1)[-1])
    assert data["report"] == report
    assert data["risk_level"] == "low" and data["period"] == "long"
    assert "expected_return" not in prompts[0]


async def test_strategy_without_report_assets_can_abstain(monkeypatch):
    report = {"event_summary": "사건", "event_analysis": "분석", "market_impact": "불명", "stock_impacts": [], "risk_factors": [], "verification": {"status": "passed", "notes": []}}
    async def generate(prompt):
        return json.dumps({"strategy_summary": "추가 근거 확인", "risk_warning": "투자 성과를 보장하지 않습니다.", "strategy_items": []})
    monkeypatch.setattr(reports.llm, "generate", generate)
    assert (await reports.generate_strategy(report, risk_level="medium", period="short"))["strategy_items"] == []
