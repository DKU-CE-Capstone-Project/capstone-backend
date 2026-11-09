"""WP5 contracts through real worker/HTTP paths; WP3/WP4/WP12 are mocks here."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v2 import schemas
from app.config import settings
from app.main import app
from app.v2 import persistence, reports, state
from app.v2 import provider_budget as budget
from app.v2.agents import pipeline as agent_pipeline
from app.v2.agents import trace
from tests import test_v2_reports as helpers

v2_offline = helpers.v2_offline
baseline_pipeline = helpers.pipeline
GENERATE_STRATEGY = reports.generate_strategy


@pytest.fixture(autouse=True)
def arm_defaults(monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", False)
    monkeypatch.setattr(settings, "v2_claim_check", False)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", 0)
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1000)
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 32)


def payload(news_id="article-0"):
    return {
        "title": "모의 경제 리포트", "event_summary": "모의 사건", "event_analysis": "모의 본문 분석",
        "market_impact": "불확실성 존재", "risk_factors": ["모의 위험"],
        "stock_impacts": [{"ticker": None, "name": "반도체 업종", "impact": "본문 근거에 따른 혼합 영향",
                           "direction": "mixed", "evidence_news_ids": [news_id], "claim_ids": ["c1", "c2"]}],
        "claims": [
            {"claim_id": "c1", "text": "본문에 근거가 있다.", "kind": "fact",
             "evidence": [{"news_id": news_id, "sentence": "실제 추출 본문 근거입니다."}], "falsifier": None},
            {"claim_id": "c2", "text": "모의 검증이 반박하는 주장", "kind": "interpretation",
             "evidence": [{"news_id": news_id, "sentence": "실제 추출 본문 근거입니다."}], "falsifier": "다른 수치가 확인될 경우"},
        ],
        "debate": [{"name": "반도체 업종", "ticker": None, "bull": ["수요 개선 가능성"],
                    "bear": ["개선 지속 여부 불명"], "direction": "mixed", "verdict": "상반된 근거를 함께 고려함",
                    "evidence_news_ids": [news_id]}],
    }


@pytest.fixture
def mock_components(monkeypatch, baseline_pipeline):
    """Exercise coordination only; these do not stand in for delivered WP3/12."""
    calls = {"generate": [], "claims": [], "revision": [], "stages": []}
    original_component = agent_pipeline._component
    original_update = reports._update

    async def generate(prompt, *, purpose=None, max_output_tokens=None):
        calls["generate"].append((purpose, max_output_tokens))
        if purpose in {"bull", "bear"}:
            return json.dumps({"arguments": [{"name": "반도체 업종", "ticker": None,
                               "text": "수요 개선 가능성" if purpose == "bull" else "개선 지속 여부 불명",
                               "kind": "interpretation",
                               "evidence": [{"news_id": "article-0", "sentence": "실제 추출 본문 근거입니다."}],
                               "falsifier": "관련 통계가 반대로 확인될 경우"}]})
        if purpose in {"judge", "analysis"}:
            return json.dumps(payload(), ensure_ascii=False)
        if purpose == "critic":
            return json.dumps({"fixture_verification": True})
        if purpose == "revise":
            revised = payload()
            revised["claims"][1]["text"] = "수정된 모의 주장"
            return json.dumps(revised, ensure_ascii=False)
        raise AssertionError("Unexpected provider role")

    async def analyze_with_claims(evidence, rag, *, generate):
        result = json.loads(await generate("fixture-analysis"))
        result.pop("debate", None)
        return result

    async def verify_claims(claims, evidence, rag, *, critic):
        calls["claims"].append(deepcopy(claims))
        await critic("fixture-critic")
        result = deepcopy(claims)
        for claim in result:
            refuted = claim["text"] == "모의 검증이 반박하는 주장"
            claim.update(verdict="refuted" if refuted else "supported", sentence_found=True, reason="모의 검증")
        refuted = sum(c["verdict"] == "refuted" for c in result)
        return result, {"status": "warning" if refuted else "passed",
                        "notes": ["c2: 모의 반박"] if refuted else [],
                        "claims_summary": {"total": len(result), "supported": len(result) - refuted,
                                           "refuted": refuted, "unclear": 0, "sentence_missing": 0}}

    async def revise_once(report, verification, evidence, rag, *, rewrite, verify, max_rounds):
        calls["revision"].append({"report": deepcopy(report), "verification": deepcopy(verification), "max_rounds": max_rounds})
        initial = deepcopy(verification)
        revised = json.loads(await rewrite("fixture-rewrite"))
        revised, checked = await verify(revised)
        trace.record("revise", "adopted", "모의 문제 주장 1건 수정", elapsed_ms=1)
        revision = {"attempts": 1, "adopted": True, "initial_status": initial["status"],
                    "initial_notes": initial["notes"], "rounds": [{"round": 1, "targets": ["c2"],
                    "before": 1, "after": 0, "decision": "adopted",
                    "changes": [{"claim_id": "c2", "action": "rewritten", "before_text": "모의 검증이 반박하는 주장"}]}]}
        return revised, checked, revision

    modules = {"claims": SimpleNamespace(analyze_with_claims=analyze_with_claims, verify_claims=verify_claims),
               "revise": SimpleNamespace(revise_once=revise_once)}

    def component(name, functions):
        return modules.get(name) or original_component(name, functions)

    async def update(sid, job_id, token, **kwargs):
        if "stage" in kwargs:
            calls["stages"].append(kwargs["stage"])
        return await original_update(sid, job_id, token, **kwargs)

    monkeypatch.setattr(agent_pipeline, "_component", component)
    monkeypatch.setattr(agent_pipeline.llm, "generate", generate)
    monkeypatch.setattr(reports, "_update", update)
    return calls


async def read_http(sid, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture.invalid",
                           cookies={"econmind_sid": sid}) as client:
        result = await client.get(path)
    assert result.status_code == 200, result.text
    return result.json()


async def test_off_arm_keeps_historical_generation_and_http_shape(baseline_pipeline):
    assert reports.generation() == {
        "language": "ko", "report_type": "investment", "model": reports.model_config(),
        "prompt_version": "investment-multi-v2", "preprocessing_version": reports.EXTRACTOR_VERSION,
        "rag_version": "selected-first-v1", "input_budget_version": "equal-60000-12000-v1",
    }
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed"
    report = await persistence.report(job["result"]["report_id"])
    response = await read_http(sid, f'/api/v2/reports/{report["report_id"]}')
    assert set(response) == {"schema_version", "report_id", "title", "event_summary", "event_analysis", "market_impact",
                             "stock_impacts", "risk_factors", "requested_news_ids", "used_evidence", "failed_evidence",
                             "coverage", "rag_evidence", "verification", "created_at"}
    assert set(response["stock_impacts"][0]) == {"ticker", "name", "impact", "direction", "evidence_news_ids"}
    assert response["verification"] == {"status": "unavailable", "notes": ["검증 기능이 비활성화되어 있습니다."]}
    assert "pipeline" not in report["generation"]
    strategy = await read_http(sid, f'/api/v2/strategies/{job["result"]["strategy_id"]}')
    assert "agent_trace" not in strategy


@pytest.mark.parametrize("flags", [(True, False, 0), (False, True, 0), (False, False, 1)])
async def test_missing_wp4_fails_before_extraction_or_provider(baseline_pipeline, monkeypatch, flags):
    multi, claims, rounds = flags
    monkeypatch.setattr(settings, "v2_multi_agent", multi)
    monkeypatch.setattr(settings, "v2_claim_check", claims)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", rounds)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed"
    assert job["error"]["code"] == "MULTI_AGENT_DEPENDENCY_UNAVAILABLE"
    assert baseline_pipeline == {"extract": [], "analyze": [], "strategy": []}
    assert budget._job_key(job["job_id"]) not in state._memory
    assert not persistence._memory["reports"] and not persistence._memory["news_bodies"]


@pytest.mark.parametrize(("name", "flags"), [("claims", (False, True, 0)), ("revise", (False, False, 2))])
async def test_missing_claim_or_revision_component_is_preflighted(baseline_pipeline, monkeypatch, name, flags):
    async def available_generate(prompt, *, purpose=None, max_output_tokens=None):
        raise AssertionError("Preflight failure must make no provider call")

    original = agent_pipeline._component

    def component(component_name, functions):
        if component_name == name:
            raise agent_pipeline._missing()
        return original(component_name, functions)

    monkeypatch.setattr(reports.llm, "generate", available_generate)
    monkeypatch.setattr(agent_pipeline, "_component", component)
    monkeypatch.setattr(settings, "v2_multi_agent", flags[0])
    monkeypatch.setattr(settings, "v2_claim_check", flags[1])
    monkeypatch.setattr(settings, "v2_revise_max_rounds", flags[2])
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["error"]["code"] == "MULTI_AGENT_DEPENDENCY_UNAVAILABLE"
    assert not baseline_pipeline["extract"]


async def test_enabled_worker_http_and_reused_report_keep_separate_strategy_trace(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    monkeypatch.setattr(settings, "v2_claim_check", True)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", 2)
    monkeypatch.setattr(settings, "use_critic", True)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed", job
    report_id, strategy_id = job["result"]["report_id"], job["result"]["strategy_id"]
    report = await read_http(sid, f"/api/v2/reports/{report_id}")
    strategy = await read_http(sid, f"/api/v2/strategies/{strategy_id}")
    assert report["claims"][1]["text"] == "수정된 모의 주장"
    assert report["verification"]["claims_summary"] == {"total": 2, "supported": 2, "refuted": 0, "unclear": 0, "sentence_missing": 0}
    assert report["verification"]["revision"]["attempts"] == 1
    assert [row["agent"] for row in report["agent_trace"]] == ["bull", "bear", "judge", "critic", "critic", "revise"]
    assert all(row["agent"] != "strategy" for row in report["agent_trace"])
    assert strategy["agent_trace"] == [{"seq": 1, "agent": "strategy", "elapsed_ms": strategy["agent_trace"][0]["elapsed_ms"],
                                        "summary": "리포트 대상에 맞춘 전략 생성", "decision": "ok"}]
    assert "input_excerpt" not in report["used_evidence"][0]
    assert "extractor_version" not in report["used_evidence"][0]
    assert "generation" not in report and "generation_key" not in report
    assert state._memory[budget._job_key(job["job_id"])][1]["units"] == 8
    assert [x[0] for x in mock_components["generate"]] == ["bull", "bear", "judge", "critic", "revise", "critic"]
    assert dict(mock_components["generate"]) == {"bull": 4096, "bear": 4096, "judge": 8192, "critic": 2048, "revise": 6144}
    stages = mock_components["stages"]
    assert "debating" in stages and "judging" in stages and "revising" in stages
    assert "verifying" not in stages[stages.index("revising"):]
    assert stages.index("saving") < stages.index("strategy")
    assert mock_components["revision"][0]["max_rounds"] == 2
    assert mock_components["revision"][0]["verification"]["status"] == "warning"
    original = deepcopy(await persistence.report(report_id))
    second = await helpers.finish(sid, await helpers.submit(sid, map_id, ids, options={"period": "long", "risk_level": "low"}))
    assert second["status"] == "completed" and second["result"]["report_reused"]
    assert second["result"]["report_id"] == report_id and second["result"]["strategy_id"] != strategy_id
    assert await persistence.report(report_id) == original
    second_strategy = await read_http(sid, f'/api/v2/strategies/{second["result"]["strategy_id"]}')
    assert [row["seq"] for row in second_strategy["agent_trace"]] == [1]
    assert len(mock_components["generate"]) == 6


async def test_claims_only_arm_uses_analysis_callback_and_distinct_cache(mock_components, monkeypatch):
    baseline = reports.generation()
    monkeypatch.setattr(settings, "v2_claim_check", True)
    monkeypatch.setattr(settings, "use_critic", True)
    enabled = reports.generation()
    assert baseline != enabled and "pipeline" not in baseline
    assert enabled["pipeline"]["prompt_versions"] == {"analysis": "analysis-claims-v1", "claims_critic": "claim-critic-v1"}
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed", job
    report = await read_http(sid, f'/api/v2/reports/{job["result"]["report_id"]}')
    assert report["verification"]["status"] == "warning"
    assert [x[0] for x in mock_components["generate"]] == ["analysis", "critic"]
    assert "debate" not in report


async def test_budget_rejection_propagates_without_extra_provider_calls(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 2)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "JOB_BUDGET_EXCEEDED"
    assert [x[0] for x in mock_components["generate"]] == ["bull"]
    assert state._memory[budget._job_key(job["job_id"])][1]["units"] == 2
    assert persistence._memory["reports"] == {}


async def test_debate_without_claims_does_not_expose_claim_fields(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed", job
    report = await read_http(sid, f'/api/v2/reports/{job["result"]["report_id"]}')
    assert report["debate"]
    assert "claims" not in report
    assert all("claim_ids" not in stock for stock in report["stock_impacts"])


async def test_disabled_critic_never_calls_claim_critic_or_revision_provider(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    monkeypatch.setattr(settings, "v2_claim_check", True)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", 2)
    monkeypatch.setattr(settings, "use_critic", False)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed", job
    report = await read_http(sid, f'/api/v2/reports/{job["result"]["report_id"]}')
    assert report["verification"]["status"] == "unavailable"
    assert report["verification"]["claims_summary"]["supported"] == 0
    assert all(c["verdict"] == "unclear" for c in report["claims"])
    assert report["verification"]["revision"]["attempts"] == 0
    assert not report["verification"]["revision"]["adopted"]
    assert [x[0] for x in mock_components["generate"]] == ["bull", "bear", "judge"]
    assert mock_components["revision"] == []


@pytest.mark.parametrize("corruption", ["asset_identity", "protected_claim_changed", "protected_claim_removed"])
async def test_invalid_revision_cannot_persist_changed_assets_or_protected_claims(mock_components, monkeypatch, corruption):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    monkeypatch.setattr(settings, "v2_claim_check", True)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", 2)
    monkeypatch.setattr(settings, "use_critic", True)
    original_component = agent_pipeline._component
    revision_module = original_component("revise", ("revise_once",))

    async def invalid_revision(*args, **kwargs):
        report, checked, revision = await revision_module.revise_once(*args, **kwargs)
        if corruption == "asset_identity":
            report["stock_impacts"][0]["name"] = "다른 종목"
        elif corruption == "protected_claim_changed":
            report["claims"][0]["text"] = "supported 주장을 임의 변경"
        else:
            report["claims"].pop(0)
        return report, checked, revision

    def component(name, functions):
        return SimpleNamespace(revise_once=invalid_revision) if name == "revise" else original_component(name, functions)

    monkeypatch.setattr(agent_pipeline, "_component", component)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "failed" and job["error"]["code"] == "REPORT_GENERATION_FAILED"
    assert not persistence._memory["reports"] and not persistence._memory["strategies"]
    assert state._memory[budget._job_key(job["job_id"])][1]["units"] == 7


async def test_both_analysts_unusable_fall_back_once_without_judge(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    calls = []

    async def generate(prompt, *, purpose=None, max_output_tokens=None):
        calls.append(purpose)
        if purpose in {"bull", "bear"}:
            return "invalid json"
        assert purpose is None
        return json.dumps(payload())

    monkeypatch.setattr(reports.llm, "generate", generate)
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.finish(sid, await helpers.submit(sid, map_id, ids))
    assert job["status"] == "completed", job
    report = await read_http(sid, f'/api/v2/reports/{job["result"]["report_id"]}')
    assert calls == ["bull", "bear", None]
    assert report["debate"] == [] and "claims" not in report
    assert [(row["agent"], row["decision"]) for row in report["agent_trace"]] == [
        ("bull", "failed"), ("bear", "failed"), ("analysis", "ok"), ("critic", "warning")]
    assert "judging" not in mock_components["stages"]
    assert "analyzing" in mock_components["stages"]
    assert state._memory[budget._job_key(job["job_id"])][1]["units"] == 5


@pytest.mark.parametrize("exception", [reports.JobCancelled, reports.LeaseLost, asyncio.CancelledError])
async def test_stage_control_exception_prevents_later_provider_reservation(mock_components, monkeypatch, exception):
    monkeypatch.setattr(settings, "v2_claim_check", True)
    reservations = []

    async def on_stage(stage):
        raise exception()

    async def reserve(operation, units=1):
        reservations.append(operation)

    monkeypatch.setattr(budget, "reserve", reserve)
    with pytest.raises(exception):
        await agent_pipeline.analyze_report([], [], analyze=reports.analyze, verify=reports.verify, on_stage=on_stage)
    assert reservations == [] and mock_components["generate"] == []


async def test_cancellation_after_provider_prevents_next_stage_or_paid_call(mock_components, monkeypatch):
    monkeypatch.setattr(settings, "v2_claim_check", True)
    calls, reservations = [], []

    async def generate(prompt, *, purpose=None, max_output_tokens=None):
        calls.append(purpose)
        return json.dumps(payload())

    async def on_stage(stage):
        if calls:
            raise reports.JobCancelled()

    async def reserve(operation, units=1):
        reservations.append(operation)

    monkeypatch.setattr(reports.llm, "generate", generate)
    monkeypatch.setattr(budget, "reserve", reserve)
    with pytest.raises(reports.JobCancelled):
        await agent_pipeline.analyze_report([], [], analyze=reports.analyze, verify=reports.verify, on_stage=on_stage)
    assert calls == ["analysis"] and reservations == ["analysis"]
    assert mock_components["claims"] == [] and mock_components["revision"] == []


@pytest.mark.parametrize("stage", ["queued", "extracting", "analyzing", "debating", "judging", "verifying", "revising", "saving", "strategy", "done"])
async def test_job_dto_accepts_old_and_new_stages(baseline_pipeline, stage):
    sid, map_id, ids = await helpers.fixture_map()
    job = await helpers.submit(sid, map_id, ids)
    job["stage"] = stage
    assert schemas.ReportJob.model_validate(job).stage == stage


async def test_strategy_prompt_receives_only_legacy_verification_fields(mock_components, monkeypatch):
    prompts = []

    async def generate(prompt):
        prompts.append(json.loads(prompt.rsplit("\n", 1)[-1]))
        return json.dumps({"strategy_summary": "모의 전략", "risk_warning": "모의 한계",
                           "strategy_items": [{"target_index": 0, "action": "watch", "reason": "모의 사유"}]})

    monkeypatch.setattr(reports.llm, "generate", generate)
    report = payload()
    report["verification"] = {"status": "warning", "notes": ["fixture"], "claims_summary": {"total": 2},
                              "revision": {"attempts": 1, "fixture_secret": "do-not-forward"}}
    await GENERATE_STRATEGY(report, risk_level="low", period="short")
    assert prompts[0]["report"]["verification"] == {"status": "warning", "notes": ["fixture"]}
