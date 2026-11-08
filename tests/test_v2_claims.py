"""Offline contracts for claim generation, quote matching and claim verdicts."""
import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.telemetry import _context
from app.v2.agents import claims, debate
from app.v2.provider_budget import ProviderBudgetError
from app.v2.reports import JobCancelled, LeaseLost

QUOTE = "수출액은 전년 동월보다 12% 늘었다."
OTHER_QUOTE = "재고는 석 달째 줄었다."
EVIDENCE = [
    {"news_id": "n1", "title": "수출 증가", "input_excerpt": f"10월 통계다. {QUOTE} 이상이다."},
    {"news_id": "n2", "title": "재고 감소", "input_excerpt": f"재고 통계다. {OTHER_QUOTE}"},
]
RAG = [{"news_id": "r1", "title": "참고", "description": "과거 기사"}]
SUPPORTED = {"verdicts": [{"claim_id": "c1", "verdict": "supported", "reason": "수출 증가를 직접 언급합니다."}]}


def claim(claim_id="c1", news_id="n1", sentence=QUOTE):
    return {"claim_id": claim_id, "text": "수출이 늘었다.", "kind": "fact",
            "evidence": [{"news_id": news_id, "sentence": sentence}], "falsifier": None,
            "verdict": "unclear", "sentence_found": False, "reason": None}


def critic_returning(response, calls=None):
    calls = [] if calls is None else calls

    async def critic(prompt, **kwargs):
        calls.append({"prompt": prompt, "stage": (_context.get() or {}).get("stage"), **kwargs})
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

    return critic


def report_payload():
    return {
        "title": "수출 분석", "event_summary": "수출이 증가했다.", "event_analysis": "한 달 수치다.",
        "market_impact": "수요 개선 가능성이 있다.",
        "stock_impacts": [{"ticker": None, "name": "반도체 업종", "impact": "개선 가능성",
                           "direction": "up", "evidence_news_ids": ["n1"], "claim_ids": ["c1"]}],
        "risk_factors": ["지속성 불확실"],
        "claims": [{"claim_id": "c1", "text": "수출이 늘었다.", "kind": "fact",
                    "evidence": [{"news_id": "n1", "sentence": QUOTE}], "falsifier": None}],
    }


def generator(payload, calls=None):
    calls = [] if calls is None else calls

    async def generate(prompt, **kwargs):
        calls.append({"prompt": prompt, "stage": (_context.get() or {}).get("stage"), **kwargs})
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    return generate


# ---- normalize / sentence_in_excerpt ----

@pytest.mark.parametrize(("sentence", "excerpt"), [
    ("수출 이 늘었다", "수출이 늘었다."),
    ("수출이\n늘었다", "수출이  늘었다"),
    ('"수출이 늘었다"', "“수출이 늘었다”"),
    ("'수출이' 늘었다", "‘수출이’ 늘었다"),
    ("「수출이」 늘었다", "수출이 늘었다"),
    ("수출이… 늘었다", "수출이... 늘었다"),
    ("１２％ 늘었다", "12% 늘었다"),
    ("ABC 수출", "abc수출"),
    ("1,000억원", "1000억원"),
    ("−3% 감소", "-3% 감소"),
    ("3.5 % 늘었다", "3.5% 늘었다"),
    ("미·중 무역", "미ㆍ중 무역"),
    ("이익률 -3%", "이익률 –3%"),
    ("이익률 -3%", "이익률 —3%"),
    ("2025년 수출이 늘었다", "지난 2025년 수출이 늘었다"),
    ("수출이 늘었다", "2025. 수출이 늘었다"),
])
def test_sentence_matches_across_formatting(sentence, excerpt):
    assert claims.sentence_in_excerpt(sentence, excerpt)


@pytest.mark.parametrize(("sentence", "excerpt"), [
    ("수출이 늘었다", "수입이 늘었다"),
    ("15% 늘었다", "1.5% 늘었다"),
    ("12 늘었다", "12% 늘었다"),
    ("수익률 3% 변동", "수익률 -3% 변동"),
    ("3% 늘었다", "13% 늘었다"),
    ("5% 늘었다", "1.5% 늘었다"),
    ("3% 줄었다", "영업이익 -3% 줄었다"),
    ("매출 100", "매출 100.5억"),
    ("매출 100", "매출 1000억"),
    ("수출 3", "수출 3% 증가"),
    ("수출은... 늘었다", "수출은 크게 늘었다"),
    ("", "수출이 늘었다"),
    ("  “…”  ", "수출이 늘었다"),
    ("수출", ""),
    (None, "수출"),
    ("수출", None),
    (123, "123"),
])
def test_sentence_does_not_match(sentence, excerpt):
    assert not claims.sentence_in_excerpt(sentence, excerpt)


def test_normalize_canonical_form():
    assert claims.normalize("") == ""
    assert claims.normalize(None) == ""
    assert claims.normalize(" “삼성전자”의 3.5%… 상승,  -2% 조정 ") == "삼성전자의3.5%상승-2%조정"
    assert claims.normalize("2026-11-08") == claims.normalize("20261108")


# ---- verify_claims: decision rules ----

async def test_all_supported_passes_with_single_critic_call():
    calls = []
    rows = [claim("c1"), claim("c2", "n2", OTHER_QUOTE)]
    response = {"verdicts": [{"claim_id": "c1", "verdict": "supported", "reason": "직접 언급"},
                             {"claim_id": "c2", "verdict": "supported", "reason": None}]}
    result, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response, calls))
    assert [row["verdict"] for row in result] == ["supported", "supported"]
    assert all(row["sentence_found"] for row in result)
    assert [row["reason"] for row in result] == ["직접 언급", None]
    assert verification == {"status": "passed", "notes": [], "claims_summary": {
        "total": 2, "supported": 2, "refuted": 0, "unclear": 0, "sentence_missing": 0}}
    assert len(calls) == 1
    assert calls[0]["purpose"] == "critic"
    assert calls[0]["max_output_tokens"] == 2048
    assert calls[0]["stage"] == "critic"


@pytest.mark.parametrize("verdict", ["refuted", "unclear"])
async def test_refuted_or_unclear_warns_with_fixed_notes(verdict):
    secret = "모델이 쓴 사유 원문"
    response = {"verdicts": [{"claim_id": "c1", "verdict": verdict, "reason": secret}]}
    result, verification = await claims.verify_claims([claim()], EVIDENCE, RAG, critic=critic_returning(response))
    assert result[0]["verdict"] == verdict
    assert result[0]["reason"] == secret
    assert verification["status"] == "warning"
    assert verification["claims_summary"][verdict] == 1
    assert len(verification["notes"]) == 1 and verification["notes"][0].startswith("c1: ")
    assert secret not in verification["notes"][0]


async def test_no_claims_skips_critic_and_is_unavailable():
    calls = []
    for empty in ([], None):
        result, verification = await claims.verify_claims(empty, EVIDENCE, RAG, critic=critic_returning(SUPPORTED, calls))
        assert result == []
        assert verification == {"status": "unavailable", "notes": ["검증할 주장을 만들지 못했습니다."], "claims_summary": {
            "total": 0, "supported": 0, "refuted": 0, "unclear": 0, "sentence_missing": 0}}
    assert calls == []


@pytest.mark.parametrize("row", [
    claim(sentence="기사에 없는 문장이다."),
    claim(news_id="missing"),
    claim(news_id="n2"),  # the quote exists, but only in a different article
    {**claim(), "evidence": []},
])
async def test_unquoted_claim_skips_critic(row):
    calls = []
    result, verification = await claims.verify_claims([row], EVIDENCE, RAG, critic=critic_returning(SUPPORTED, calls))
    assert calls == []
    assert result[0]["verdict"] == "unclear"
    assert result[0]["sentence_found"] is False
    assert result[0]["reason"] == claims.SENTENCE_MISSING
    assert verification["status"] == "warning"
    assert verification["notes"] == [f"c1: {claims.SENTENCE_MISSING}"]
    assert verification["claims_summary"] == {
        "total": 1, "supported": 0, "refuted": 0, "unclear": 1, "sentence_missing": 1}


async def test_every_cited_sentence_must_exist():
    row = claim()
    row["evidence"].append({"news_id": "n2", "sentence": "없는 문장이다."})
    result, _ = await claims.verify_claims([row], EVIDENCE, RAG, critic=critic_returning(SUPPORTED))
    assert result[0]["sentence_found"] is False


async def test_sentence_found_is_recomputed_with_full_normalization():
    # normalize_report only collapses whitespace, so this arrives as False.
    row = claim(sentence=f"“{QUOTE}”")
    assert debate._sentence_found(row["evidence"], EVIDENCE) is False
    result, _ = await claims.verify_claims([row], EVIDENCE, RAG, critic=critic_returning(SUPPORTED))
    assert result[0]["sentence_found"] is True
    assert result[0]["verdict"] == "supported"


async def test_stale_true_sentence_found_and_verdict_are_overwritten():
    row = {**claim(sentence="기사에 없는 문장이다."), "sentence_found": True, "verdict": "supported", "reason": "이전"}
    result, _ = await claims.verify_claims([row], EVIDENCE, RAG, critic=critic_returning(SUPPORTED))
    assert (result[0]["sentence_found"], result[0]["verdict"]) == (False, "unclear")


async def test_critic_cannot_upgrade_an_unquoted_claim():
    rows = [claim("c1"), claim("c2", sentence="없는 문장이다.")]
    response = {"verdicts": [{"claim_id": "c1", "verdict": "supported"}, {"claim_id": "c2", "verdict": "supported"}]}
    result, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response))
    assert [row["verdict"] for row in result] == ["supported", "unclear"]
    assert verification["status"] == "warning"


@pytest.mark.parametrize("response", [
    "", "not json", "[]", '"text"', "null", "[" * 200000,
    {"other": []}, {"verdicts": "c1"}, {"verdicts": []},
    {"verdicts": [{"claim_id": "unknown", "verdict": "supported"}]},
    {"verdicts": [{"claim_id": "c1", "verdict": "maybe"}]},
    {"verdicts": ["c1", 3, None]},
    {"verdicts": [{"claim_id": ["c1"], "verdict": ["supported"]}]},
])
async def test_unusable_critic_response_is_unavailable(response):
    rows = [claim("c1"), claim("c2", sentence="없는 문장이다.")]
    result, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response))
    assert [row["verdict"] for row in result] == ["unclear", "unclear"]
    assert result[0]["sentence_found"] is True
    assert result[0]["reason"] == claims.CRITIC_FAILED
    assert result[1]["reason"] == claims.SENTENCE_MISSING
    assert verification == {"status": "unavailable", "notes": [claims.CRITIC_FAILED], "claims_summary": {
        "total": 2, "supported": 0, "refuted": 0, "unclear": 2, "sentence_missing": 1}}


async def test_none_response_is_unavailable():
    async def critic(prompt, **kwargs):
        return None

    _, verification = await claims.verify_claims([claim()], EVIDENCE, RAG, critic=critic)
    assert verification["status"] == "unavailable"


async def test_fenced_critic_json_is_accepted():
    response = "```json\n" + json.dumps(SUPPORTED) + "\n```"
    _, verification = await claims.verify_claims([claim()], EVIDENCE, RAG, critic=critic_returning(response))
    assert verification["status"] == "passed"


async def test_partial_omission_affects_only_that_claim():
    rows = [claim("c1"), claim("c2", "n2", OTHER_QUOTE)]
    result, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(SUPPORTED))
    assert [row["verdict"] for row in result] == ["supported", "unclear"]
    assert result[1]["sentence_found"] is True
    assert result[1]["reason"]
    assert verification["status"] == "warning"
    assert verification["claims_summary"]["sentence_missing"] == 0


async def test_conflicting_duplicate_verdicts_count_as_omitted():
    rows = [claim("c1"), claim("c2", "n2", OTHER_QUOTE)]
    response = {"verdicts": [{"claim_id": "c1", "verdict": "supported"}, {"claim_id": "c1", "verdict": "refuted"},
                             {"claim_id": "c2", "verdict": "supported"}]}
    result, _ = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response))
    assert [row["verdict"] for row in result] == ["unclear", "supported"]


async def test_invalid_row_is_ignored_and_long_reason_truncated():
    rows = [claim("c1"), claim("c2", "n2", OTHER_QUOTE)]
    response = {"verdicts": [{"claim_id": "c1", "verdict": "supported", "reason": "가" * 200},
                             {"claim_id": "c2", "verdict": "wrong"}]}
    result, _ = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response))
    assert len(result[0]["reason"]) == 80
    assert result[1]["verdict"] == "unclear"


async def test_lone_surrogate_in_reason_is_removed():
    response = '{"verdicts":[{"claim_id":"c1","verdict":"supported","reason":"사유\\ud800"}]}'
    result, _ = await claims.verify_claims([claim()], EVIDENCE, RAG, critic=critic_returning(response))
    assert result[0]["reason"] == "사유"


async def test_claims_over_cap_are_not_sent_or_trusted():
    calls = []
    rows = [claim(f"c{index}") for index in range(1, 14)]
    response = {"verdicts": [{"claim_id": f"c{index}", "verdict": "supported"} for index in range(1, 14)]}
    result, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response, calls))
    assert len(calls) == 1
    assert '"claim_id": "c13"' not in calls[0]["prompt"]
    assert result[-1]["verdict"] == "unclear"
    assert verification["claims_summary"]["supported"] == 12


async def test_critic_prompt_scope():
    calls = []
    rows = [claim("c1"), claim("c2", "n2", "없는 문장이다.")]
    await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(SUPPORTED, calls))
    prompt = calls[0]["prompt"]
    payload = json.loads(prompt[prompt.index('{"claims"'):])
    assert [row["claim_id"] for row in payload["claims"]] == ["c1"]
    assert [row["news_id"] for row in payload["selected_bodies"]] == ["n1"]
    assert payload["rag_reference"] == RAG
    assert "verdict" not in payload["claims"][0]
    assert "지시를 따르지" in prompt


# ---- control flow and input safety ----

@pytest.mark.parametrize("error", [
    ProviderBudgetError(429, "JOB_BUDGET_EXCEEDED", "생성 예산 초과", False),
    JobCancelled(), LeaseLost(), asyncio.CancelledError(), ValueError("from generator"), RuntimeError("boom"),
])
async def test_critic_exceptions_propagate(error):
    async def critic(prompt, **kwargs):
        raise error

    with pytest.raises(type(error)):
        await claims.verify_claims([claim()], EVIDENCE, RAG, critic=critic)


@pytest.mark.parametrize("error", [
    ProviderBudgetError(429, "JOB_BUDGET_EXCEEDED", "생성 예산 초과", False),
    JobCancelled(), LeaseLost(), asyncio.CancelledError(),
])
async def test_generate_control_exceptions_propagate(error):
    async def generate(prompt, **kwargs):
        raise error

    with pytest.raises(type(error)):
        await claims.analyze_with_claims(EVIDENCE, RAG, generate=generate)


async def test_verify_does_not_mutate_inputs():
    rows, evidence, rag = [claim("c1"), claim("c2", sentence="없는 문장이다.")], deepcopy(EVIDENCE), deepcopy(RAG)
    originals = deepcopy((rows, evidence, rag))
    result, _ = await claims.verify_claims(rows, evidence, rag, critic=critic_returning(SUPPORTED))
    assert (rows, evidence, rag) == originals
    result[0]["evidence"][0]["sentence"] = "changed"
    assert rows[0]["evidence"][0]["sentence"] == QUOTE


async def test_notes_exclude_article_text():
    rows = [claim("c1"), claim("c2", sentence="없는 문장이다.")]
    response = {"verdicts": [{"claim_id": "c1", "verdict": "refuted", "reason": QUOTE}]}
    _, verification = await claims.verify_claims(rows, EVIDENCE, RAG, critic=critic_returning(response))
    assert not any(QUOTE in note or "없는 문장" in note for note in verification["notes"])


# ---- analyze_with_claims ----

async def test_analyze_keeps_report_claims_and_claim_ids():
    calls = []
    evidence, rag = deepcopy(EVIDENCE), deepcopy(RAG)
    result = await claims.analyze_with_claims(evidence, rag, generate=generator(report_payload(), calls))
    assert (evidence, rag) == (EVIDENCE, RAG)
    assert result["claims"] == report_payload()["claims"]
    assert result["stock_impacts"][0]["claim_ids"] == ["c1"]
    assert len(calls) == 1
    assert (calls[0]["purpose"], calls[0]["max_output_tokens"], calls[0]["stage"]) == ("analysis", 6144, "analysis")
    prompt = calls[0]["prompt"]
    assert "claims는 최대 12개" in prompt and QUOTE in prompt and "지시를 따르지" in prompt
    assert claims.ANALYSIS_PROMPT_VERSION == "analysis-claims-v1"
    assert claims.CRITIC_PROMPT_VERSION == "claim-critic-v1"


async def test_analyze_result_passes_through_normalize_report():
    result = await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(report_payload()))
    report = debate.normalize_report(result, EVIDENCE, claims_enabled=True)
    assert report["claims"][0]["claim_id"] == "c1"
    assert report["stock_impacts"][0]["claim_ids"] == ["c1"]


@pytest.mark.parametrize("bad_claims", [None, "text", 3, {"claim_id": "c1"}, [1, "x", None], [{"claim_id": "bad"}]])
async def test_analyze_survives_bad_claims(bad_claims):
    payload = report_payload()
    payload["claims"] = bad_claims
    result = await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))
    assert isinstance(result["claims"], list)
    report = debate.normalize_report(result, EVIDENCE, claims_enabled=True)
    assert report["claims"] == []
    assert report["stock_impacts"][0]["claim_ids"] == []


@pytest.mark.parametrize("bad_ids", [None, "c1", 3, {"c1": True}])
async def test_analyze_survives_bad_claim_ids(bad_ids):
    payload = report_payload()
    payload["stock_impacts"][0]["claim_ids"] = bad_ids
    result = await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))
    assert result["stock_impacts"][0]["claim_ids"] == []
    assert len(result["claims"]) == 1


async def test_analyze_without_claim_fields():
    payload = report_payload()
    del payload["claims"], payload["stock_impacts"][0]["claim_ids"]
    result = await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))
    assert result["claims"] == []
    assert result["stock_impacts"][0]["claim_ids"] == []


async def test_analyze_drops_unknown_top_level_fields():
    payload = {**report_payload(), "verification": {"status": "passed"}, "agent_trace": []}
    result = await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))
    assert "verification" not in result and "agent_trace" not in result


@pytest.mark.parametrize("raw", ["", "not json", "[]"])
async def test_analyze_fails_on_unparseable_report(raw):
    with pytest.raises((TypeError, ValueError)):
        await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(raw))


async def test_analyze_fails_on_invalid_report_body():
    payload = report_payload()
    del payload["title"]
    with pytest.raises(ValidationError):
        await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))


async def test_analyze_fails_on_unused_stock_evidence():
    payload = report_payload()
    payload["stock_impacts"][0]["evidence_news_ids"] = ["not-used"]
    with pytest.raises(ValueError, match="unused evidence"):
        await claims.analyze_with_claims(EVIDENCE, RAG, generate=generator(payload))
