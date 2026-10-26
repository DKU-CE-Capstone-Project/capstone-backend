"""Budget policy through real v2 job transitions, with all providers stubbed."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app import session as base_session
from app import store
from app.api.v2 import report_jobs as api
from app.config import settings
from app.v2 import persistence, reports, state
from app.v2 import provider_budget as budget
from app.v2.errors import V2Error
from tests import test_v2_reports as helpers

# Reuse the existing offline fixtures locally without registering a global plugin.
pipeline = helpers.pipeline
v2_offline = helpers.v2_offline


@pytest.fixture(autouse=True)
def budget_policy(monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1000)
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 32)


def used(job_id):
    entry = state._memory.get(budget._job_key(job_id))
    return entry[1]["units"] if entry else 0


async def test_completed_result_and_body_cache_hits_make_no_new_reservations(pipeline):
    sid, map_id, ids = await helpers.fixture_map(2)
    first = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert first["status"] == "completed"
    assert used(first["job_id"]) == 4  # Two bodies, analysis, strategy; critic disabled.
    daily = deepcopy(state._memory[budget._DAILY_KEY])
    second = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert second["status"] == "completed"
    assert second["result"]["report_reused"] and second["result"]["strategy_reused"]
    assert used(second["job_id"]) == 0
    assert state._memory[budget._DAILY_KEY] == daily
    assert len(pipeline["extract"]) == 2 and len(pipeline["analyze"]) == 1
    assert len(pipeline["strategy"]) == 1
    assert await reports.process_job(sid, first["job_id"]) == "done"
    assert state._memory[budget._DAILY_KEY] == daily


async def test_cached_extraction_failure_does_not_repeat_charge_or_provider_call(pipeline, monkeypatch):
    sid, _, ids = await helpers.fixture_map()
    article = store.news_cache[ids[0]]
    calls = []

    async def empty(articles, **kwargs):
        calls.append(articles[0]["news_id"])
        return []

    monkeypatch.setattr(reports.diffbot_client, "extract_articles_with_diffbot", empty)
    with budget.job_budget("first"):
        body, failure = await reports.extract_one(article)
    assert body is None and failure["code"] == "BODY_EMPTY"
    daily = deepcopy(state._memory[budget._DAILY_KEY])
    with budget.job_budget("second"):
        repeated_body, repeated_failure = await reports.extract_one(article)
    assert repeated_body is None and repeated_failure == failure
    assert calls == [ids[0]] and used("first") == 1 and used("second") == 0
    assert state._memory[budget._DAILY_KEY] == daily
    assert await state.session(sid)


async def test_extraction_budget_denial_is_a_job_policy_error_with_no_external_work(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1)
    with budget.job_budget("previous-job"):
        await budget.reserve("body")
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "DAILY_BUDGET_EXCEEDED"
    assert job["result"] is None and used(job["job_id"]) == 0
    assert pipeline == {"extract": [], "analyze": [], "strategy": []}
    assert persistence._memory["news_bodies"] == {}
    assert persistence._memory["reports"] == {}


async def test_analysis_budget_denial_persists_retry_timing_in_job_error(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed"
    assert job["error"]["code"] == "DAILY_BUDGET_EXCEEDED" and job["error"]["retryable"]
    assert 1 <= job["error"]["details"]["retry_after_seconds"] <= 86400
    assert used(job["job_id"]) == 1
    assert len(pipeline["extract"]) == 1 and not pipeline["analyze"] and not pipeline["strategy"]
    assert persistence._memory["reports"] == {}


async def test_critic_budget_denial_cannot_become_unavailable_or_completed(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "use_critic", True)
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 2)
    verify_calls = []

    async def verify(*args):
        verify_calls.append(args)
        return {"status": "passed", "notes": []}

    monkeypatch.setattr(reports, "verify", verify)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "DAILY_BUDGET_EXCEEDED"
    assert len(pipeline["analyze"]) == 1 and verify_calls == [] and pipeline["strategy"] == []
    assert persistence._memory["reports"] == {} and used(job["job_id"]) == 2


async def test_rag_lookup_is_charged_before_report_reuse_but_cached_generations_are_not(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "use_rag", True)
    monkeypatch.setattr(settings, "use_mongodb", True)
    rag_calls = []

    async def rag(article, **kwargs):
        rag_calls.append(article["news_id"])
        return []

    monkeypatch.setattr(reports.report_generator, "retrieve_rag_articles", rag)
    sid, map_id, ids = await helpers.fixture_map()
    first = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    second = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert first["status"] == second["status"] == "completed"
    assert used(first["job_id"]) == 4 and used(second["job_id"]) == 1
    assert second["result"]["report_reused"] and second["result"]["strategy_reused"]
    assert rag_calls == [ids[0], ids[0]]
    assert len(pipeline["extract"]) == len(pipeline["analyze"]) == len(pipeline["strategy"]) == 1


async def test_rag_budget_denial_starts_no_rag_lookup_or_generation(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "use_rag", True)
    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1)
    rag_calls = []

    async def rag(article, **kwargs):
        rag_calls.append(article["news_id"])
        return []

    monkeypatch.setattr(reports.report_generator, "retrieve_rag_articles", rag)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "DAILY_BUDGET_EXCEEDED"
    assert rag_calls == [] and pipeline["analyze"] == pipeline["strategy"] == []
    assert used(job["job_id"]) == 1


async def test_strategy_job_budget_denial_preserves_saved_report_and_owner_access(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 2)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "JOB_BUDGET_EXCEEDED"
    assert job["error"]["retryable"] is False and "STRATEGY_GENERATION_FAILED" not in job["warnings"]
    assert job["result"]["strategy_id"] is None and pipeline["strategy"] == []
    report_id = job["result"]["report_id"]
    assert await persistence.report(report_id)
    access_key = state.key(sid, "result-access", report_id)
    assert await state.get(access_key) is None
    exposed = await api.get_report(report_id, helpers.request(sid))
    assert exposed["report_id"] == report_id
    assert (await state.get(access_key))["result_type"] == "report"
    other = await base_session.issue("127.0.0.1")
    with pytest.raises(V2Error) as denied:
        await api.get_report(report_id, helpers.request(other))
    assert denied.value.status == 404
    final_map = await state.resource(sid, "map", map_id)
    assert final_map["active_report_job_id"] is None


async def test_redelivery_keeps_same_job_budget_and_restores_context(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 3)
    original = reports.analyze
    interrupted = False

    async def interrupt_once(evidence, rag):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise asyncio.CancelledError()
        return await original(evidence, rag)

    monkeypatch.setattr(reports, "analyze", interrupt_once)
    sid, map_id, ids = await helpers.fixture_map()
    admitted = await helpers.submit(sid, map_id, ids)
    with pytest.raises(asyncio.CancelledError):
        await reports.process_job(sid, admitted["job_id"])
    assert used(admitted["job_id"]) == 2
    retained = await reports.get_job(sid, admitted["job_id"])
    retained["lease_until"] = 0
    await reports._save_job(sid, retained)
    job = await helpers.finish(sid, admitted)
    assert job["status"] == "failed" and job["error"]["code"] == "JOB_BUDGET_EXCEEDED"
    assert job["result"]["report_id"] and job["result"]["strategy_id"] is None
    assert used(job["job_id"]) == 3
    assert len(pipeline["extract"]) == 1 and len(pipeline["analyze"]) == 1
    before = deepcopy(state._memory[budget._DAILY_KEY])
    await budget.reserve("body")  # The process_job task must not leak its context.
    assert state._memory[budget._DAILY_KEY] == before


async def test_budget_store_failure_stops_external_work_and_persists_safe_error(pipeline, monkeypatch):
    class Unavailable:
        async def eval(self, *args):
            raise ConnectionError("fixture-secret-connection-string")

    # Keep job/session persistence available while isolating a budget-ledger outage.
    monkeypatch.setattr(budget, "state", SimpleNamespace(memory_enabled=lambda: False, _client=Unavailable))
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "BUDGET_UNAVAILABLE"
    assert "fixture-secret" not in str(job)
    assert pipeline == {"extract": [], "analyze": [], "strategy": []}
    assert persistence._memory["news_bodies"] == {} and persistence._memory["reports"] == {}
