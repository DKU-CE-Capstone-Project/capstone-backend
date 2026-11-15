"""Exercise the WP8 runner with the shipped agents and offline provider responses."""

import json

import pytest

from app import telemetry
from app.agents import llm
from app.config import settings
from scripts import compare_ab

QUOTE = "반도체 수출액은 전년 동월보다 12% 늘었다."
EVIDENCE = [{"news_id": "n1", "title": "반도체 수출 통계", "input_excerpt": QUOTE}]
TARGET = "반도체 업종"
BULL = "수출 증가로 수요 개선 가능성"
BEAR = ["한 달 통계만으로 추세 판단 어려움", "다음 달 지표 확인 필요"]
EXCESS = "대상별 상한을 넘은 약세 논거"
WRONG = "반도체 수출이 두 배가 되었다."
CORRECTED = "반도체 수출은 12% 증가했다."


def report_output():
    citation = {"news_id": "n1", "sentence": QUOTE}
    return {
        "title": "반도체 수출 동향",
        "event_summary": "전년 동월보다 수출 증가",
        "event_analysis": "한 달 수치로 지속 추세는 판단하기 어렵다.",
        "market_impact": "수요 개선 가능성과 불확실성이 공존한다.",
        "risk_factors": ["지속성 불확실"],
        "stock_impacts": [{
            "name": TARGET, "ticker": None, "direction": "mixed",
            "impact": "개선 가능성과 지속성 불확실", "evidence_news_ids": ["n1"],
            "claim_ids": ["c1", "c2"],
        }],
        "debate": [{
            "name": TARGET, "ticker": None, "bull": [BULL], "bear": BEAR,
            "direction": "mixed", "verdict": "다음 달 수치 확인 필요",
            "evidence_news_ids": ["n1"],
        }],
        "claims": [
            {"claim_id": claim_id, "text": text, "kind": "fact",
             "evidence": [citation], "falsifier": None}
            for claim_id, text in (("c1", "이번 통계의 대상은 반도체 수출이다."), ("c2", WRONG))
        ],
    }


@pytest.fixture
def provider_calls(monkeypatch):
    calls = []

    async def respond(prompt, *, max_output_tokens=None):
        stage = telemetry._context.get()["stage"]
        data = json.loads(prompt.rsplit("\n", 1)[-1])
        calls.append((stage, max_output_tokens, data))
        if stage in {"bull", "bear"}:
            output = {"arguments": [
                {"name": TARGET, "ticker": None, "text": text, "kind": "interpretation",
                 "evidence": [{"news_id": "n1", "sentence": QUOTE}],
                 "falsifier": "다음 달 통계가 반대로 나오는 경우"}
                for text in ([BULL] if stage == "bull" else [*BEAR, EXCESS])
            ]}
        elif stage in {"analysis", "judge"}:
            output = report_output()
        elif stage == "critic":
            if "claims" in data:
                output = {"verdicts": [
                    {"claim_id": claim["claim_id"],
                     "verdict": "refuted" if claim["text"] == WRONG else "supported",
                     "reason": "본문 수치와 대조함"}
                    for claim in data["claims"]
                ]}
            else:
                output = {"grounded": True, "confidence": 0.9, "issues": [],
                          "unsupported_claims": [], "invalid_tickers": []}
        elif stage == "revise":
            output = data["report"]
            output["claims"][1]["text"] = CORRECTED
        elif stage == "strategy":
            output = {"strategy_summary": "관망", "risk_warning": "수익을 보장하지 않습니다.",
                      "strategy_items": [{"target_index": 0, "action": "watch", "reason": "지속성 불확실"}]}
        else:
            pytest.fail(f"Unexpected provider stage: {stage}")

        with telemetry.observe("gemini", "generate", model="fixture") as observation:
            observation.response(None, usage={"prompt_token_count": 20, "candidates_token_count": 10})
        return json.dumps(output, ensure_ascii=False)

    monkeypatch.setattr(llm, "_generate_with_provider", respond)
    monkeypatch.setattr(settings, "use_critic", True)
    return calls


async def test_real_abc_runner_uses_fixed_evidence_and_adopts_claim_repair(
    tmp_path, monkeypatch, provider_calls, capsys,
):
    # Check restoration to a non-default configuration after all three arms.
    monkeypatch.setattr(settings, "v2_multi_agent", True)
    monkeypatch.setattr(settings, "v2_revise_max_rounds", 1)
    adapters = (llm.generate, llm.generate_with_usage, telemetry.emit)
    arms = ["A", "B", "C"]
    record = await compare_ab.compare(
        {"report": {"used_evidence": EVIDENCE}}, arms, tmp_path / "comparison",
        runner=compare_ab.prepare_live_runner(arms), execution_mode="live", max_calls=24,
    )

    assert [run["outcome"] for run in record["runs"]] == ["completed"] * 3
    assert [run["logical_generate_calls"] for run in record["runs"]] == [3, 5, 7]
    assert record["logical_generate_calls"] == 15
    a, b, c = (run["result"] for run in record["runs"])
    for result, count in zip((a, b, c), (3, 5, 7), strict=True):
        assert result["usage"] == {"sdk_generate_events": count, "tokens_in": count * 20,
                                   "tokens_out": count * 10, "est_cost_usd": None}
        assert result["strategy"]["strategy_items"][0]["stock_name"] == TARGET

    assert "pipeline" not in a["generation"]
    assert "debate" not in a["report"]
    assert "claims" not in b["report"]
    assert b["verification"]["revision"]["attempts"] == 0
    for result in (b, c):
        assert result["report"]["debate"][0]["bear"] == BEAR
        assert any(entry["agent"] == "bear" and entry["decision"] == "ok"
                   for entry in result["report"]["agent_trace"])
    assert c["generation"]["pipeline"]["claim_check"] is True
    assert c["generation"]["pipeline"]["prompt_versions"]["bear"] == "debate-bear-v2"
    assert c["verification"]["status"] == "passed"
    assert c["verification"]["revision"]["initial_status"] == "warning"
    assert c["verification"]["revision"]["adopted"] is True
    assert c["verification"]["revision"]["rounds"][0]["targets"] == ["c2"]
    assert [(claim["text"], claim["verdict"], claim["sentence_found"])
            for claim in c["report"]["claims"]] == [
        ("이번 통계의 대상은 반도체 수출이다.", "supported", True),
        (CORRECTED, "supported", True),
    ]
    judges = [(limit, data) for stage, limit, data in provider_calls if stage == "judge"]
    assert len(judges) == 2
    for limit, data in judges:
        assert limit == 8192
        assert data["used_evidence"] == EVIDENCE
        assert [argument["text"] for argument in data["bear"]] == BEAR
    for stage, _, data in provider_calls:
        if stage in {"analysis", "bull", "bear", "revise"}:
            assert data["used_evidence"] == EVIDENCE
        if stage == "strategy":
            assert set(data["report"]["verification"]) == {"status", "notes"}
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    normalized = [event for event in events if event["kind"] == "agent_output_normalized"]
    assert [(event["stage"], event["discarded_arguments"]) for event in normalized] == [
        ("bear", 1), ("bear", 1),
    ]
    assert (llm.generate, llm.generate_with_usage, telemetry.emit) == adapters
    assert (settings.v2_multi_agent, settings.v2_claim_check, settings.v2_revise_max_rounds) == (
        True, False, 1,
    )


async def test_real_runner_restores_adapters_after_call_ceiling(provider_calls):
    adapters = (llm.generate, llm.generate_with_usage, telemetry.emit)
    limit = compare_ab.CallLimit(2)
    runner = compare_ab.prepare_live_runner(["C"])

    with pytest.raises(compare_ab.CallLimitExceeded, match="LOGICAL_GENERATION_CALL_LIMIT"):
        await runner("C", EVIDENCE, [], {"risk_level": "medium", "period": "short"}, limit)

    assert limit.started == 2
    assert {stage for stage, _, _ in provider_calls} == {"bull", "bear"}
    assert (llm.generate, llm.generate_with_usage, telemetry.emit) == adapters
    assert (settings.v2_multi_agent, settings.v2_claim_check, settings.v2_revise_max_rounds) == (
        False, False, 0,
    )
