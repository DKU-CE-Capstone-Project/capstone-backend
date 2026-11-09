"""Offline contracts for the bounded report repair loop."""
import asyncio
import json
from copy import deepcopy

import pytest

from app.telemetry import _context
from app.v2.agents import revise, trace
from app.v2.provider_budget import ProviderBudgetError
from app.v2.reports import JobCancelled, LeaseLost

QUOTE = "수출액은 전년 동월보다 12% 늘었다."
EVIDENCE = [{"news_id": "n1", "input_excerpt": f"10월 통계다. {QUOTE}", "title": "수출 증가"}]
RAG = [{"news_id": "r1", "description": "참고 자료"}]


def claim(claim_id, verdict="supported", *, found=True, text=None):
    return {"claim_id": claim_id, "text": text or f"{claim_id} 주장", "kind": "fact",
            "evidence": [{"news_id": "n1", "sentence": QUOTE}], "falsifier": None,
            "verdict": verdict, "sentence_found": found, "reason": None}


def base_report(claims=None):
    claims = [claim("c1"), claim("c2", "refuted")] if claims is None else claims
    return {
        "title": "수출 분석", "event_summary": "수출이 늘었다.", "event_analysis": "한 달 수치다.",
        "market_impact": "수요 개선 가능성", "risk_factors": ["지속성 불확실"],
        "stock_impacts": [
            {"ticker": None, "name": "반도체", "impact": "개선", "direction": "up",
             "evidence_news_ids": ["n1"], "claim_ids": [c["claim_id"] for c in claims[:3]]},
            {"ticker": "005930", "name": "삼성전자", "impact": "개선", "direction": "mixed",
             "evidence_news_ids": ["n1"], "claim_ids": []},
        ],
        "claims": claims,
        "debate": [{"name": "반도체", "ticker": None, "bull": ["증가"], "bear": [], "direction": "up",
                    "verdict": "개선", "evidence_news_ids": ["n1"]}],
    }


def check(status="warning", notes=("c2: 근거가 다릅니다.",), **extra):
    return {"status": status, "notes": list(notes), "claims_summary": {"total": 2}, **extra}


def model_output(report, *, drop=(), edits=None):
    """What a compliant model returns: verdict fields are the verifier's job."""
    payload = deepcopy(report)
    payload["claims"] = [
        {key: row[key] for key in ("claim_id", "text", "kind", "evidence", "falsifier")}
        | (edits or {}).get(row["claim_id"], {})
        for row in payload["claims"] if row["claim_id"] not in drop
    ]
    for key in ("debate",):
        payload.pop(key, None)
    return payload


def rewriter(*outputs):
    calls = []

    async def rewrite(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs, "stage": (_context.get() or {}).get("stage")})
        output = outputs[min(len(calls) - 1, len(outputs) - 1)]
        if isinstance(output, BaseException):
            raise output
        return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)

    rewrite.calls = calls
    return rewrite


def verifier(*verdicts, notes=None):
    """Each round assigns ``verdicts[n]`` by claim_id; missing claims default to supported."""
    calls = []

    async def verify(candidate):
        calls.append(deepcopy(candidate))
        current = verdicts[min(len(calls) - 1, len(verdicts) - 1)]
        for row in candidate.get("claims", []):
            row["verdict"], row["sentence_found"] = current.get(row["claim_id"], ("supported", True))
        problems = sum(row["verdict"] == "refuted" for row in candidate.get("claims", []))
        return candidate, check("warning" if problems else "passed", [] if not problems else ["남은 문제"])

    verify.calls = calls
    return verify


async def run(report=None, verification=None, *, rewrite, verify=None, max_rounds=2):
    report = base_report() if report is None else report
    return await revise.revise_once(
        report, check() if verification is None else verification, deepcopy(EVIDENCE), deepcopy(RAG),
        rewrite=rewrite, verify=verify or verifier({}), max_rounds=max_rounds,
    )


def fixed(report):
    return model_output(report, edits={"c2": {"text": "수정된 주장"}})


async def test_improved_rewrite_is_adopted_with_contract_shape_and_call_arguments():
    report = base_report()
    rewrite = rewriter(fixed(report))
    verify = verifier({})

    revised, checked, revision = await run(report, rewrite=rewrite, verify=verify)

    assert revised["claims"][1]["text"] == "수정된 주장"
    assert revised["claims"][1]["verdict"] == "supported"
    assert revised["debate"] == report["debate"]
    assert checked["status"] == "passed"
    assert revision == {
        "attempts": 1, "adopted": True, "initial_status": "warning",
        "initial_notes": ["c2: 근거가 다릅니다."],
        "rounds": [{"round": 1, "targets": ["c2"], "before": 1, "after": 0, "decision": "adopted",
                    "changes": [{"claim_id": "c2", "action": "rewritten", "before_text": "c2 주장"}]}],
    }
    assert rewrite.calls[0]["kwargs"] == {"purpose": "revise", "max_output_tokens": 6144}
    assert rewrite.calls[0]["stage"] == "revise"
    assert revise.REVISE_PROMPT_VERSION == "revise-v1"
    sent = verify.calls[0]["claims"][1]
    assert sent["text"] == "수정된 주장" and sent["verdict"] == "unclear" and sent["reason"] is None
    data = json.loads(rewrite.calls[0]["prompt"].rsplit("\n", 1)[-1])
    assert data["used_evidence"] == EVIDENCE and data["rag_evidence"] == RAG
    assert [row["claim_id"] for row in data["problems"]] == ["c2"]
    assert "verdict" not in data["report"]["claims"][0] and "debate" not in data["report"]


async def test_model_written_verdict_is_not_trusted():
    report = base_report()
    output = fixed(report)
    output["claims"][1]["verdict"] = "supported"
    verify = verifier({"c2": ("refuted", True)})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert verify.calls[0]["claims"][1]["verdict"] == "unclear"
    assert revised == report and revision["rounds"][0]["decision"] == "kept_original"


async def test_no_improvement_keeps_original_and_stops():
    report = base_report()
    rewrite = rewriter(fixed(report))
    initial = check()

    revised, checked, revision = await run(report, initial, rewrite=rewrite,
                                           verify=verifier({"c2": ("refuted", True)}))

    assert revised == report and checked == initial
    assert len(rewrite.calls) == 1
    assert revision["adopted"] is False and revision["attempts"] == 1
    assert revision["rounds"] == [{"round": 1, "targets": ["c2"], "before": 1, "after": 1,
                                   "decision": "kept_original", "changes": []}]


@pytest.mark.parametrize("output", [
    "", "   ", "not json", "[1, 2]", ValueError("bad"), TypeError("bad"),
], ids=["empty", "blank", "invalid_json", "not_object", "value_error", "type_error"])
async def test_unusable_rewrite_keeps_original_as_failed(output):
    report = base_report()
    verify = verifier({})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert revised == report and verify.calls == []
    assert revision["attempts"] == 1 and revision["adopted"] is False
    assert revision["rounds"][0]["decision"] == "failed" and revision["rounds"][0]["after"] is None


@pytest.mark.parametrize("mutate", [
    lambda o: o.pop("title"),
    lambda o: o.pop("claims"),
    lambda o: o["claims"][1].update(kind="opinion"),
    lambda o: o["claims"][1].update(text="가" * 201),
    lambda o: o["claims"].append(dict(o["claims"][1])),
    lambda o: o["claims"].append({**o["claims"][1], "claim_id": "c3"}),
    lambda o: o["stock_impacts"][0].update(direction="sideways"),
], ids=["report_shape", "claims_missing", "kind", "text_limit", "duplicate", "new_claim", "direction"])
async def test_malformed_candidate_fields_keep_original(mutate):
    report = base_report()
    output = fixed(report)
    mutate(output)
    verify = verifier({})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert revised == report and verify.calls == []
    assert revision["rounds"][0]["decision"] == "failed"


@pytest.mark.parametrize("mutate", [
    lambda o: o["claims"].pop(0),
    lambda o: o["claims"][0].update(text="바뀐 주장"),
    lambda o: o["claims"][0].update(kind="interpretation"),
    lambda o: o["claims"][0].update(falsifier="새 반증 조건"),
    lambda o: o["claims"][0]["evidence"][0].update(sentence="10월 통계다."),
], ids=["deleted", "text", "kind", "falsifier", "evidence"])
async def test_changing_a_supported_claim_keeps_original_without_paying_for_verification(mutate):
    report = base_report()
    output = fixed(report)
    mutate(output)
    verify = verifier({})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert revised == report and verify.calls == []
    assert revision["adopted"] is False and revision["rounds"][0]["after"] is None


async def test_supported_claim_changed_by_verifier_keeps_original():
    report = base_report()
    rewrite = rewriter(fixed(report))

    async def verify(candidate):
        candidate["claims"] = [row for row in candidate["claims"] if row["claim_id"] != "c1"]
        candidate["claims"][0].update(verdict="supported", sentence_found=True)
        return candidate, check("passed", [])

    revised, _, revision = await run(report, rewrite=rewrite, verify=verify)

    assert revised == report and revision["rounds"][0]["decision"] == "kept_original"


async def test_supported_claim_downgraded_on_recheck_keeps_original():
    report = base_report()

    revised, _, revision = await run(report, rewrite=rewriter(fixed(report)),
                                     verify=verifier({"c1": ("unclear", True)}))

    assert revised == report and revision["rounds"][0]["after"] == 0
    assert revision["rounds"][0]["decision"] == "kept_original"


async def test_unchanged_target_that_rechecks_better_is_not_a_repair():
    report = base_report()

    revised, _, revision = await run(report, rewrite=rewriter(model_output(report)), verify=verifier({}))

    assert revised == report and revision["rounds"][0]["decision"] == "kept_original"


async def test_found_but_unclear_claim_is_neither_targeted_nor_editable():
    claims = [claim("c1", "unclear", found=True), claim("c2", "unclear", found=False), claim("c3", "refuted")]
    report = base_report(claims)
    good = model_output(report, drop={"c2"}, edits={"c3": {"text": "수정"}})
    bad = model_output(report, edits={"c1": {"text": "범위 밖 수정"}, "c3": {"text": "수정"}})

    adopted, _, revision = await run(report, rewrite=rewriter(good), verify=verifier({"c1": ("unclear", True)}))
    assert revision["rounds"][0]["targets"] == ["c2", "c3"]
    assert adopted["claims"][0]["text"] == "c1 주장"

    kept, _, rejected = await run(report, rewrite=rewriter(bad), verify=verifier({"c1": ("unclear", True)}))
    assert kept == report and rejected["rounds"][0]["decision"] == "failed"

    only_found = base_report([claim("c1", "unclear", found=True)])
    rewrite = rewriter(model_output(only_found))
    untouched, _, none = await run(only_found, rewrite=rewrite)
    assert untouched == only_found and rewrite.calls == [] and none["attempts"] == 0


@pytest.mark.parametrize("mutate", [
    lambda o: o["stock_impacts"].pop(),
    lambda o: o["stock_impacts"].reverse(),
    lambda o: o["stock_impacts"][1].update(name="다른 종목"),
    lambda o: o["stock_impacts"][1].update(ticker="000660"),
    lambda o: o["stock_impacts"].append(dict(o["stock_impacts"][0], name="새 종목")),
], ids=["removed", "reordered", "renamed", "ticker", "added"])
async def test_stock_target_list_change_keeps_original(mutate):
    report = base_report()
    output = fixed(report)
    mutate(output)
    verify = verifier({})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert revised == report and verify.calls == []
    assert revision["rounds"][0]["decision"] == "failed"


async def test_stock_target_list_changed_by_verifier_keeps_original():
    report = base_report()

    async def verify(candidate):
        candidate["stock_impacts"].pop()
        return candidate, check("passed", [])

    revised, _, revision = await run(report, rewrite=rewriter(fixed(report)), verify=verify)

    assert revised == report and revision["rounds"][0]["decision"] == "failed"


@pytest.mark.parametrize("mutate", [
    lambda o: o["stock_impacts"][0].update(evidence_news_ids=["unknown"]),
    lambda o: o["stock_impacts"][0].update(evidence_news_ids=["r1"]),
    lambda o: o["claims"][1]["evidence"][0].update(news_id="unknown"),
], ids=["stock", "rag_is_not_used_evidence", "claim"])
async def test_citation_violation_keeps_original(mutate):
    report = base_report()
    output = fixed(report)
    mutate(output)
    verify = verifier({})

    revised, _, revision = await run(report, rewrite=rewriter(output), verify=verify)

    assert revised == report and verify.calls == []
    assert revision["rounds"][0]["decision"] == "failed" and revision["rounds"][0]["after"] is None


async def test_two_round_limit_and_second_round_starts_from_first_adopted():
    report = base_report([claim("c1"), claim("c2", "refuted"), claim("c3", "refuted"), claim("c4", "refuted")])
    first = model_output(report, edits={"c2": {"text": "1차 수정"}})
    second = model_output(report, edits={"c2": {"text": "1차 수정"}, "c3": {"text": "2차 수정"}})
    third = model_output(report, edits={"c2": {"text": "1차 수정"}, "c3": {"text": "2차 수정"},
                                        "c4": {"text": "3차 수정"}})
    rewrite = rewriter(first, second, third)
    verify = verifier({"c3": ("refuted", True), "c4": ("refuted", True)}, {"c4": ("refuted", True)})

    revised, _, revision = await run(report, rewrite=rewrite, verify=verify, max_rounds=2)

    assert len(rewrite.calls) == 2 and revision["attempts"] == 2 and revision["adopted"] is True
    assert [(r["round"], r["before"], r["after"], r["decision"]) for r in revision["rounds"]] == [
        (1, 3, 2, "adopted"), (2, 2, 1, "adopted")]
    assert [row["text"] for row in revised["claims"]] == ["c1 주장", "1차 수정", "2차 수정", "c4 주장"]
    second_prompt = json.loads(rewrite.calls[1]["prompt"].rsplit("\n", 1)[-1])
    assert second_prompt["report"]["claims"][1]["text"] == "1차 수정"
    assert revision["rounds"][1]["targets"] == ["c3", "c4"]


async def test_failed_second_round_keeps_first_adopted_version():
    report = base_report([claim("c1"), claim("c2", "refuted"), claim("c3", "refuted")])
    first = model_output(report, edits={"c2": {"text": "1차 수정"}})
    verify = verifier({"c3": ("refuted", True)})

    revised, checked, revision = await run(report, rewrite=rewriter(first, ""), verify=verify)

    assert revised["claims"][1]["text"] == "1차 수정" and checked["status"] == "warning"
    assert revision["adopted"] is True
    assert [r["decision"] for r in revision["rounds"]] == ["adopted", "failed"]


async def test_stops_when_first_round_clears_every_problem():
    report = base_report()
    rewrite = rewriter(fixed(report))

    _, _, revision = await run(report, rewrite=rewrite)

    assert len(rewrite.calls) == 1 and revision["attempts"] == 1


@pytest.mark.parametrize(("report", "verification", "rounds"), [
    (base_report([claim("c1")]), check("passed", []), 2),
    (base_report([]), check("unavailable", ["검증할 주장을 만들지 못했습니다."]), 2),
    (base_report(), check("unavailable", ["확인 불가"]), 2),
    (base_report(), check(), 0),
], ids=["all_supported", "zero_claims", "unavailable", "disabled"])
async def test_no_targets_makes_no_calls(report, verification, rounds):
    rewrite = rewriter("{}")
    verify = verifier({})

    revised, checked, revision = await run(report, verification, rewrite=rewrite, verify=verify, max_rounds=rounds)

    assert revised == report and checked == verification
    assert rewrite.calls == [] and verify.calls == []
    assert revision == {"attempts": 0, "adopted": False, "initial_status": verification["status"],
                        "initial_notes": verification["notes"], "rounds": []}


async def test_rounds_above_contract_limit_are_capped():
    report = base_report([claim("c1", "refuted"), claim("c2", "refuted"), claim("c3", "refuted")])
    outputs = [model_output(report, edits={"c1": {"text": "a"}}),
               model_output(report, edits={"c1": {"text": "a"}, "c2": {"text": "b"}}),
               model_output(report, edits={"c1": {"text": "a"}, "c2": {"text": "b"}, "c3": {"text": "c"}})]
    rewrite = rewriter(*outputs)
    verify = verifier({"c2": ("refuted", True), "c3": ("refuted", True)}, {"c3": ("refuted", True)})

    _, _, revision = await run(report, rewrite=rewrite, verify=verify, max_rounds=5)

    assert revision["attempts"] == 2 and len(rewrite.calls) == 2


def notes_report():
    report = base_report([])
    for stock in report["stock_impacts"]:
        stock["claim_ids"] = []
    return report


def notes_verification(notes):
    return {"status": "warning", "notes": notes}


async def test_notes_mode_targets_truncated_notes_without_changes():
    report = notes_report()
    long_note = "가" * 130
    output = {key: deepcopy(report[key]) for key in report if key not in ("claims", "debate")}
    output["event_analysis"] = "근거에 맞춘 분석"
    rewrite = rewriter(output)

    async def verify(candidate):
        return candidate, {"status": "passed", "notes": []}

    revised, checked, revision = await run(report, notes_verification([long_note, "짧은 문제"]),
                                           rewrite=rewrite, verify=verify)

    assert revised["event_analysis"] == "근거에 맞춘 분석"
    assert revised["claims"] == [] and revised["debate"] == report["debate"]
    assert checked == {"status": "passed", "notes": []}
    assert revision["rounds"] == [{"round": 1, "targets": ["가" * 120, "짧은 문제"], "before": 2, "after": 0,
                                   "decision": "adopted", "changes": []}]
    data = json.loads(rewrite.calls[0]["prompt"].rsplit("\n", 1)[-1])
    assert data["problems"] == [long_note, "짧은 문제"]
    assert "claims" not in data["report"]


async def test_notes_mode_keeps_original_without_improvement():
    report = notes_report()
    output = {key: deepcopy(report[key]) for key in report if key not in ("claims", "debate")}
    verification = notes_verification(["문제"])

    async def verify(candidate):
        return candidate, {"status": "warning", "notes": ["여전히 문제"]}

    revised, checked, revision = await run(report, verification, rewrite=rewriter(output), verify=verify)

    assert revised == report and checked == verification
    assert revision["rounds"][0]["after"] == 1 and revision["rounds"][0]["decision"] == "kept_original"


async def test_notes_mode_echoed_report_is_not_a_repair():
    report = notes_report()
    output = {key: deepcopy(report[key]) for key in report if key not in ("claims", "debate")}

    async def verify(candidate):
        return candidate, {"status": "warning", "notes": ["한 건만 지적"]}

    revised, _, revision = await run(report, notes_verification(["문제 1", "문제 2"]),
                                     rewrite=rewriter(output), verify=verify)

    assert revised == report and revision["adopted"] is False
    assert revision["rounds"][0]["after"] == 1 and revision["rounds"][0]["decision"] == "kept_original"


async def test_removed_claim_is_dropped_from_claim_ids_and_recorded():
    report = base_report([claim("c1"), claim("c2", "unclear", found=False), claim("c3")])
    output = model_output(report, drop={"c2"})
    output["stock_impacts"][1]["claim_ids"] = ["c2"]

    revised, _, revision = await run(report, rewrite=rewriter(output))

    assert [row["claim_id"] for row in revised["claims"]] == ["c1", "c3"]
    assert revised["stock_impacts"][0]["claim_ids"] == ["c1", "c3"]
    assert revised["stock_impacts"][1]["claim_ids"] == []
    assert [(s["name"], s["ticker"]) for s in revised["stock_impacts"]] == [("반도체", None), ("삼성전자", "005930")]
    assert revision["rounds"][0]["changes"] == [{"claim_id": "c2", "action": "removed", "before_text": "c2 주장"}]


async def test_debate_row_tracks_revised_direction():
    report = base_report()
    output = fixed(report)
    output["stock_impacts"][0].update(direction="mixed")

    revised, _, _ = await run(report, rewrite=rewriter(output))

    assert revised["debate"][0]["direction"] == "mixed"
    assert report["debate"][0]["direction"] == "up"


async def test_deleting_every_claim_is_not_a_verified_improvement():
    report = base_report([claim("c1", "refuted")])

    async def verify(candidate):
        return candidate, {"status": "unavailable", "notes": ["검증할 주장을 만들지 못했습니다."]}

    revised, _, revision = await run(report, rewrite=rewriter(model_output(report, drop={"c1"})), verify=verify)

    assert revised == report and revision["rounds"][0]["decision"] == "failed"


@pytest.mark.parametrize("result", [
    ValueError("bad"),
    ("not", "enough", "values"),
    None,
    ({"claims": []}, {"status": "passed", "notes": []}),
], ids=["raises", "wrong_arity", "none", "missing_stocks"])
async def test_unusable_verification_keeps_original_as_failed(result):
    report = base_report()

    async def verify(candidate):
        if isinstance(result, Exception):
            raise result
        return result

    revised, _, revision = await run(report, rewrite=rewriter(fixed(report)), verify=verify)

    assert revised == report and revision["rounds"][0]["decision"] == "failed"
    assert revision["rounds"][0]["after"] is None


async def test_recheck_without_verdicts_keeps_original_as_failed():
    report = base_report()

    async def verify(candidate):
        for row in candidate["claims"]:
            row.pop("verdict")
        return candidate, check("passed", [])

    revised, _, revision = await run(report, rewrite=rewriter(fixed(report)), verify=verify)

    assert revised == report and revision["rounds"][0]["decision"] == "failed"


CONTROL_ERRORS = [
    ProviderBudgetError(429, "JOB_BUDGET_EXCEEDED", "budget", False),
    JobCancelled(), LeaseLost(), asyncio.CancelledError(),
]


@pytest.mark.parametrize("error", CONTROL_ERRORS, ids=lambda e: type(e).__name__)
async def test_control_errors_propagate_from_rewrite(error):
    with pytest.raises(type(error)):
        await run(rewrite=rewriter(error))


@pytest.mark.parametrize("error", CONTROL_ERRORS, ids=lambda e: type(e).__name__)
async def test_control_errors_propagate_from_verify(error):
    async def verify(candidate):
        raise error

    report = base_report()
    with pytest.raises(type(error)):
        await run(report, rewrite=rewriter(fixed(report)), verify=verify)


@pytest.mark.parametrize("output_kind", ["adopted", "failed"])
async def test_inputs_are_not_mutated(output_kind):
    report, verification = base_report(), check()
    evidence, rag = deepcopy(EVIDENCE), deepcopy(RAG)
    snapshots = deepcopy((report, verification, evidence, rag))
    rewrite = rewriter(fixed(report) if output_kind == "adopted" else "")

    revised, checked, _ = await revise.revise_once(
        report, verification, evidence, rag, rewrite=rewrite, verify=verifier({}), max_rounds=2)

    assert (report, verification, evidence, rag) == snapshots
    assert revised is not report and checked is not verification
    revised["claims"][0]["text"] = "결과 수정"
    revised["stock_impacts"][0]["claim_ids"].append("x")
    assert (report, verification, evidence, rag) == snapshots


async def test_trace_records_each_round_without_model_text():
    report = base_report([claim("c1", "refuted"), claim("c2", "refuted")])
    secret = "모델원문비밀"
    first = model_output(report, edits={"c1": {"text": "수정"}})
    first["title"] = secret
    verify = verifier({"c2": ("refuted", True)})

    with trace.collect() as entries:
        await run(report, rewrite=rewriter(first, f"{secret} 깨진 응답"), verify=verify)

    assert [(row["agent"], row["decision"]) for row in entries] == [("revise", "adopted"), ("revise", "failed")]
    assert all(len(row["summary"]) <= 120 and secret not in row["summary"] for row in entries)
    assert all(isinstance(row["elapsed_ms"], int) for row in entries)


async def test_trace_failure_does_not_affect_result(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("trace down")

    monkeypatch.setattr(trace, "record", broken)
    report = base_report()

    revised, _, revision = await run(report, rewrite=rewriter(fixed(report)))

    assert revised["claims"][1]["text"] == "수정된 주장" and revision["adopted"] is True


async def test_failed_rewrite_leaves_no_model_text_in_logs(caplog):
    secret = "모델원문비밀"
    report = base_report()

    with caplog.at_level("DEBUG"):
        await run(report, rewrite=rewriter(f"{secret} {{깨진 응답"))

    assert secret not in caplog.text
