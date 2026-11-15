"""Offline contracts for independent analysts, safe degradation and citations."""
import asyncio
import json
from copy import deepcopy

import pytest

from app.telemetry import _context, usage_context
from app.v2.agents import debate
from app.v2.provider_budget import ProviderBudgetError

QUOTE = "수출액은 전년 동월보다 12% 늘었다."
EVIDENCE = [{"news_id": "n1", "input_excerpt": f"10월 통계다. {QUOTE}", "title": "수출 증가"}]


def side_payload():
    return {"arguments": [{
        "name": "반도체 업종", "ticker": None, "text": "수출 증가로 수요 개선 가능성",
        "kind": "interpretation", "evidence": [{"news_id": "n1", "sentence": QUOTE}],
        "falsifier": "다음 통계가 감소로 정정되는 경우",
    }]}


def report_payload():
    return {
        "title": "수출 분석", "event_summary": "수출이 증가했다.",
        "event_analysis": "한 달 수치이며 추세는 불확실하다.",
        "market_impact": "수요 개선 가능성과 추세 불확실성이 함께 있다.",
        "stock_impacts": [{"ticker": None, "name": "반도체 업종", "impact": "개선 가능성과 불확실성",
                           "direction": "mixed", "evidence_news_ids": ["n1"], "claim_ids": ["c1"]}],
        "risk_factors": ["지속성 불확실"],
        "claims": [{"claim_id": "c1", "text": "수출액은 증가했다.", "kind": "fact",
                    "evidence": [{"news_id": "n1", "sentence": QUOTE}], "falsifier": None}],
        "debate": [{"name": "반도체 업종", "ticker": None, "bull": ["수출 증가"], "bear": ["추세 불확실"],
                    "direction": "mixed", "verdict": "양쪽 근거가 있어 방향 단정 불가", "evidence_news_ids": ["n1"]}],
    }


def generator(payload, calls=None):
    async def generate(prompt, **kwargs):
        if calls is not None:
            calls.append((json.loads(prompt.rsplit("\n", 1)[-1]), kwargs, (_context.get() or {}).get("stage")))
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return generate


async def run(*, bull=None, bear=None, judge=None, claims_enabled=True):
    return await debate.run_debate(
        deepcopy(EVIDENCE), [],
        bull_generate=bull or generator(side_payload()),
        bear_generate=bear or generator(side_payload()),
        judge_generate=judge or generator(report_payload()),
        claims_enabled=claims_enabled,
    )


async def test_normal_parallel_analysts_same_evidence_and_bounded_calls():
    entered = []
    ready = asyncio.Event()
    calls = []

    async def analyst(prompt, **kwargs):
        entered.append(kwargs["purpose"])
        calls.append((json.loads(prompt.rsplit("\n", 1)[-1]), kwargs, (_context.get() or {}).get("stage")))
        if len(entered) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 1)
        return json.dumps(side_payload())

    judge_calls = []
    report, rows = await run(bull=analyst, bear=analyst, judge=generator(report_payload(), judge_calls))
    assert set(entered) == {"bull", "bear"}
    assert [call[0]["used_evidence"] for call in calls] == [EVIDENCE, EVIDENCE]
    assert all(call[1]["max_output_tokens"] == 4096 for call in calls)
    assert {call[2] for call in calls} == {"bull", "bear"}
    assert judge_calls[0][1] == {"purpose": "judge", "max_output_tokens": 8192}
    assert judge_calls[0][2] == "judge"
    assert report["claims"][0]["verdict"] == "unclear"
    assert report["claims"][0]["sentence_found"] is True
    assert rows == report_payload()["debate"]


@pytest.mark.parametrize("side", ["bull", "bear"])
async def test_overflow_keeps_first_two_for_each_target_and_reaches_judge(side, capsys):
    template = side_payload()["arguments"][0]
    targets = [template["name"], "은행", "운수", "건설", "화학", "바이오"]
    allowed = [
        {**template, "name": name, "text": f"{name} 논거 {number}"}
        for number in (1, 2)
        for name in targets
    ]
    excess = {**template, "text": "상한을 넘은 논거"}
    calls = []

    _, rows = await run(
        **{side: generator({"arguments": [*allowed[:7], excess, *allowed[7:]]})},
        judge=generator(report_payload(), calls),
    )

    assert [item["text"] for item in calls[0][0][side]] == [item["text"] for item in allowed]
    assert rows[0][side]
    event = json.loads(capsys.readouterr().out)
    assert event["kind"] == "agent_output_normalized"
    assert event["stage"] == side
    assert event["discarded_arguments"] == 1


@pytest.mark.parametrize("failed_side", ["bull", "bear"])
async def test_one_invalid_side_continues_and_forces_failed_array_empty(failed_side, capsys):
    calls = []
    report, rows = await run(**{failed_side: generator('{"arguments":[')}, judge=generator(report_payload(), calls))
    assert report["title"] == "수출 분석"
    assert calls[0][0][failed_side] is None
    assert rows[0][failed_side] == []
    assert rows[0]["bear" if failed_side == "bull" else "bull"]
    event = json.loads(capsys.readouterr().out)
    assert event["kind"] == "agent_output_validation"
    assert event["stage"] == failed_side
    assert event["error_type"] == "invalid_json"


async def test_invalid_analyst_schema_logs_field_and_code_without_response_values(capsys):
    payload = side_payload()
    marker = "private-provider-output"
    payload["arguments"][0]["text"] = marker * 10
    job_id = "400b5319-b7f7-4bc3-b41e-be316db87e5f"

    with usage_context(job_id=job_id):
        _, rows = await run(bear=generator(payload))

    assert rows[0]["bear"] == []
    output = capsys.readouterr().out
    assert marker not in output
    event = json.loads(output)
    assert event["job_id"] == job_id
    assert event["stage"] == "bear"
    assert event["error_type"] == "invalid_argument_schema"
    assert event["validation_issues"] == [{"path": ["arguments", 0, "text"], "type": "string_too_long"}]


async def test_both_unusable_sides_signal_existing_analysis_fallback_to_caller():
    calls = []
    with pytest.raises(debate.DebateUnavailable, match="analysts_unavailable"):
        await run(bull=generator(""), bear=generator("bad json"), judge=generator(report_payload(), calls))
    assert calls == []


async def test_judge_rejects_stock_reference_outside_selected_bodies():
    payload = report_payload()
    payload["stock_impacts"][0]["evidence_news_ids"] = ["rag-only-id"]
    with pytest.raises(debate.DebateOutputError, match="invalid_judgement"):
        await run(judge=generator(payload))


async def test_truncated_judge_json_fails_without_including_provider_output():
    secret_like_marker = "private-provider-response"
    with pytest.raises(debate.DebateOutputError) as caught:
        await run(judge=generator('{"title":"' + secret_like_marker))
    assert secret_like_marker not in str(caught.value)


async def test_invalid_base_report_stays_a_failure():
    payload = report_payload()
    payload.pop("event_analysis")
    with pytest.raises(debate.DebateOutputError):
        await run(judge=generator(payload))


def many_stocks_payload(count):
    payload = report_payload()
    template = payload["stock_impacts"][0]
    payload["stock_impacts"] = [dict(template, name=f"종목{index}") for index in range(count)]
    return payload


@pytest.mark.parametrize("count", [7, 25])
async def test_judge_stock_impacts_over_limit_keep_first_six_in_order(count):
    report, rows = await run(judge=generator(many_stocks_payload(count)))
    expected = [f"종목{index}" for index in range(6)]
    assert [stock["name"] for stock in report["stock_impacts"]] == expected
    assert [row["name"] for row in rows] == expected
    assert all(stock["claim_ids"] == ["c1"] for stock in report["stock_impacts"])


async def test_judge_stock_impacts_not_a_list_stays_a_failure():
    payload = report_payload()
    payload["stock_impacts"] = {"name": "반도체 업종"}
    with pytest.raises(debate.DebateOutputError, match="invalid_judgement"):
        await run(judge=generator(payload))


def test_normalize_report_still_rejects_too_many_stock_impacts():
    with pytest.raises(debate.DebateOutputError, match="too_many_stocks"):
        debate.normalize_report(many_stocks_payload(7), EVIDENCE, claims_enabled=True)


def test_optional_claims_drop_individual_bad_fields_and_clear_untrusted_verdicts():
    payload = report_payload()
    first = payload["claims"][0]
    first.update(verdict="supported", sentence_found=True, reason="모델이 검증했다는 주장")
    unknown = deepcopy(first)
    unknown["claim_id"] = "c2"
    unknown["evidence"][0]["news_id"] = "not-selected"
    long_quote = deepcopy(first)
    long_quote["claim_id"] = "c3"
    long_quote["evidence"][0]["sentence"] = "가" * 201
    payload["claims"] += [unknown, long_quote, None, {"claim_id": "c99"}, deepcopy(first)]
    payload["stock_impacts"][0]["claim_ids"] = ["c99", "c1", "c1", "c2", 1]
    report = debate.normalize_report(payload, EVIDENCE, claims_enabled=True)
    assert [c["claim_id"] for c in report["claims"]] == ["c1", "c2"]
    assert all(c["verdict"] == "unclear" and c["reason"] is None for c in report["claims"])
    assert report["claims"][0]["sentence_found"] is True
    assert report["claims"][1]["sentence_found"] is False
    assert report["stock_impacts"][0]["claim_ids"] == ["c1", "c2"]


@pytest.mark.parametrize("claims", [None, {}, "bad", [None], []])
def test_missing_or_malformed_claims_never_break_valid_report(claims):
    payload = report_payload()
    payload["claims"] = claims
    report = debate.normalize_report(payload, EVIDENCE, claims_enabled=True)
    assert report["claims"] == []
    assert report["stock_impacts"][0]["claim_ids"] == []


def test_claim_and_claim_link_counts_are_bounded():
    payload = report_payload()
    template = payload["claims"][0]
    payload["claims"] = [{**deepcopy(template), "claim_id": f"c{i}"} for i in range(1, 14)]
    payload["stock_impacts"][0]["claim_ids"] = [f"c{i}" for i in range(1, 14)]
    report = debate.normalize_report(payload, EVIDENCE, claims_enabled=True)
    assert len(report["claims"]) == 12
    assert report["stock_impacts"][0]["claim_ids"] == ["c1", "c2", "c3"]


async def test_claims_disabled_ignores_claims_and_requests_no_claim_output():
    calls = []
    report, _ = await run(claims_enabled=False, judge=generator(report_payload(), calls))
    assert report["claims"] == []
    assert report["stock_impacts"][0]["claim_ids"] == []


def test_quote_presence_uses_normalized_whitespace_not_model_boolean():
    payload = report_payload()
    payload["claims"][0]["evidence"][0]["sentence"] = "수출액은\n전년 동월보다 12% 늘었다."
    report = debate.normalize_report(payload, EVIDENCE, claims_enabled=True)
    assert report["claims"][0]["sentence_found"] is True
    payload["claims"][0]["evidence"][0]["sentence"] = "수출이 앞으로 두 배 늘 것이다."
    report = debate.normalize_report(payload, EVIDENCE, claims_enabled=True)
    assert report["claims"][0]["sentence_found"] is False


async def test_optional_debate_display_keeps_canonical_asset_order_and_bounds():
    payload = report_payload()
    row = payload["debate"][0]
    row.update(bull=["가" * 130] * 5, bear=[None, "약세"], verdict="나" * 250,
               direction="up", evidence_news_ids=["untrusted"])
    report, rows = await run(judge=generator(payload))
    assert len(rows[0]["bull"]) == 2
    assert all(len(x) == 100 for x in rows[0]["bull"])
    assert rows[0]["bear"] == ["약세"]
    assert len(rows[0]["verdict"]) == 200
    assert rows[0]["direction"] == report["stock_impacts"][0]["direction"]
    assert rows[0]["evidence_news_ids"] == ["n1"]


async def test_generator_control_error_propagates_and_cancels_other_analyst():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking(prompt, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def denied(prompt, **kwargs):
        await entered.wait()
        raise ProviderBudgetError(429, "JOB_BUDGET_EXCEEDED", "생성 예산 초과", False)

    with pytest.raises(ProviderBudgetError):
        await run(bull=denied, bear=blocking)
    assert cancelled.is_set()


async def test_task_cancellation_cleans_up_both_analysts():
    entered = []
    cleaned = []
    ready = asyncio.Event()

    async def blocking(prompt, **kwargs):
        entered.append(kwargs["purpose"])
        if len(entered) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(kwargs["purpose"])

    task = asyncio.create_task(run(bull=blocking, bear=blocking))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(cleaned) == {"bull", "bear"}


async def test_non_response_generator_errors_are_not_hidden_as_degraded_success():
    async def broken(prompt, **kwargs):
        raise RuntimeError("control failure")

    with pytest.raises(RuntimeError, match="control failure"):
        await run(bull=broken)


async def test_bear_prompt_is_red_team_with_falsifiers_and_no_peer_arguments():
    prompts = []

    async def bear(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(side_payload())

    await run(bear=bear)
    assert "레드팀" in prompts[0]
    assert "낙관적으로 읽힐 수 있는 해석의 반례" in prompts[0]
    assert "반증 조건(falsifier)" in prompts[0]
    assert set(json.loads(prompts[0].rsplit("\n", 1)[-1])) == {"used_evidence", "rag_evidence"}


async def test_trace_contains_only_synthetic_summaries():
    from app.v2.agents.trace import collect

    with collect() as records:
        await run()
    assert {row["agent"] for row in records} == {"bull", "bear", "judge"}
    assert all(row["decision"] == "ok" for row in records)
    assert QUOTE not in json.dumps(records, ensure_ascii=False)


async def test_fenced_json_response_is_supported():
    payload = "```json\n" + json.dumps(report_payload()) + "\n```"
    report, _ = await run(judge=generator(payload))
    assert report["title"] == "수출 분석"
