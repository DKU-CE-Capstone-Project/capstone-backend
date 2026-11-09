"""WP5 coordination with explicit, preflighted WP3/WP4/WP12 dependencies."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from copy import deepcopy

from app.agents import llm
from app.config import settings
from app.telemetry import usage_context
from app.v2 import provider_budget
from app.v2.agents import trace
from app.v2.errors import V2Error

OUTPUT_LIMITS = {"bull": 4096, "bear": 4096, "judge": 8192,
                 "analysis": 6144, "revise": 6144, "critic": 2048}
# Version the analyst and judge normalization with the prompts used for report reuse.
PROMPT_VERSIONS = {"bull": "debate-bull-v2", "bear": "debate-bear-v2",
                   "judge": "debate-judge-v2", "analysis": "analysis-claims-v1",
                   "claims_critic": "claim-critic-v1", "revise": "revise-v1"}


def enabled() -> bool:
    return settings.v2_multi_agent or settings.v2_claim_check or settings.v2_revise_max_rounds > 0


def configuration() -> dict | None:
    """A keeps its exact historical generation/cache key."""
    if not enabled():
        return None
    versions, limits = {}, {}
    if settings.v2_multi_agent:
        for name in ("bull", "bear", "judge"):
            versions[name], limits[name] = PROMPT_VERSIONS[name], OUTPUT_LIMITS[name]
        versions["analysis_fallback"] = (PROMPT_VERSIONS["analysis"]
                                         if settings.v2_claim_check else "investment-multi-v2")
    elif settings.v2_claim_check:
        versions["analysis"] = PROMPT_VERSIONS["analysis"]
        limits["analysis"] = OUTPUT_LIMITS["analysis"]
    if settings.v2_claim_check:
        versions["claims_critic"] = PROMPT_VERSIONS["claims_critic"]
        limits["critic"] = OUTPUT_LIMITS["critic"]
    if settings.v2_revise_max_rounds:
        versions["revise"], limits["revise"] = PROMPT_VERSIONS["revise"], OUTPUT_LIMITS["revise"]
    return {"multi_agent": settings.v2_multi_agent,
            "claim_check": settings.v2_claim_check,
            "revise_max_rounds": settings.v2_revise_max_rounds,
            "prompt_versions": versions, "max_output_tokens": limits}


def _missing() -> V2Error:
    return V2Error(503, "MULTI_AGENT_DEPENDENCY_UNAVAILABLE",
                   "선택한 분석 기능의 선행 모듈을 사용할 수 없습니다.", False)


def _component(name: str, functions: tuple[str, ...]):
    try:
        module = importlib.import_module(f"app.v2.agents.{name}")
    except ImportError:
        raise _missing() from None
    if any(not callable(getattr(module, function, None)) for function in functions):
        raise _missing()
    return module


def ensure_dependencies(*, multi_agent: bool, claim_check: bool, max_rounds: int) -> None:
    if type(max_rounds) is not int or not 0 <= max_rounds <= 2:
        raise ValueError("Invalid revision limit")
    if not (multi_agent or claim_check or max_rounds):
        return
    # Do not silently use the old 1024/2048-token provider path for an 8192-token
    # judge. WP4 belongs to Kimgeon; enabling before it lands fails before costs.
    parameters = inspect.signature(llm.generate).parameters
    kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if not kwargs and not {"purpose", "max_output_tokens"}.issubset(parameters):
        raise _missing()
    if multi_agent:
        _component("debate", ("run_debate",))
    if claim_check:
        _component("claims", ("analyze_with_claims", "verify_claims"))
    if max_rounds:
        _component("revise", ("revise_once",))


async def analyze_report(evidence, rag, *, analyze, verify, on_stage=None):
    """Return (report, verification); no queue/DB ownership inside agents.

    Revision verify callback: async verify(candidate) -> (candidate, verification).
    Generator callbacks: async generate(prompt, **kwargs) -> text.
    """
    multi, claims_on, rounds = (settings.v2_multi_agent, settings.v2_claim_check,
                               settings.v2_revise_max_rounds)
    ensure_dependencies(multi_agent=multi, claim_check=claims_on, max_rounds=rounds)

    async def checkpoint(stage):
        if on_stage is not None:
            await on_stage(stage)

    def generator(purpose):
        async def generate(prompt, **kwargs):
            def cancelled():
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise asyncio.CancelledError
            stage = "debating" if purpose in {"bull", "bear"} else {
                "judge": "judging", "critic": "verifying", "analysis": "analyzing",
                "revise": "revising"}[purpose]
            # Reverification remains inside revising, per WP0.
            if revising[0]:
                stage = "revising"
            cancelled()
            await checkpoint(stage)
            await provider_budget.reserve(purpose)
            cancelled()
            with usage_context(stage=purpose):
                raw = await llm.generate(prompt, purpose=purpose,
                                         max_output_tokens=OUTPUT_LIMITS[purpose])
            await checkpoint(stage)
            cancelled()
            return raw
        return generate

    revising = [False]
    claims_module = _component("claims", ("analyze_with_claims", "verify_claims")) if claims_on else None

    async def base_analysis():
        started = time.monotonic()
        await checkpoint("analyzing")
        if claims_on:
            result = await claims_module.analyze_with_claims(evidence, rag, generate=generator("analysis"))
        else:
            await provider_budget.reserve("analysis")
            result = await analyze(evidence, rag)
        if multi or claims_on:
            from app.v2.agents.debate import normalize_report
            # The legacy fallback may return up to20 targets. Only new arms
            # reduce that draft to their six-target contract.
            result = deepcopy(result)
            result["stock_impacts"] = result["stock_impacts"][:6]
            result = normalize_report(result, evidence, claims_enabled=claims_on)
        trace.record("analysis", "ok", "기존 분석 경로로 리포트 생성",
                     elapsed_ms=int((time.monotonic() - started) * 1000))
        return result

    async def verification(candidate):
        started = time.monotonic()
        await checkpoint("revising" if revising[0] else "verifying")
        candidate = deepcopy(candidate)
        if not settings.use_critic:
            if claims_on:
                for claim in candidate.get("claims", []):
                    claim.update(verdict="unclear", reason="검증 기능이 비활성화되어 있습니다.")
            checked = {"status": "unavailable", "notes": ["검증 기능이 비활성화되어 있습니다."]}
            if claims_on:
                claims = candidate.get("claims", [])
                checked["claims_summary"] = {
                    "total": len(claims), "supported": 0, "refuted": 0,
                    "unclear": len(claims),
                    "sentence_missing": sum(not claim["sentence_found"] for claim in claims)}
        elif claims_on:
            checked_claims, checked = await claims_module.verify_claims(
                candidate.get("claims", []), evidence, rag, critic=generator("critic"))
            candidate["claims"] = checked_claims
        else:
            if settings.use_critic:
                await provider_budget.reserve("critic")
            checked = await verify(candidate, evidence, rag)
        trace.record("critic", "ok" if checked["status"] == "passed" else "warning",
                     "리포트 근거 검증 완료" if checked["status"] != "unavailable" else "근거 검증 결과 이용 불가",
                     elapsed_ms=int((time.monotonic() - started) * 1000))
        return candidate, checked

    with trace.collect() as entries:
        if multi:
            from app.v2.agents.debate import DebateUnavailable, run_debate
            await checkpoint("debating")
            try:
                report, debate = await run_debate(
                    evidence, rag, bull_generate=generator("bull"),
                    bear_generate=generator("bear"), judge_generate=generator("judge"),
                    claims_enabled=claims_on)
            except DebateUnavailable:
                report, debate = await base_analysis(), []
            report["debate"] = debate
        else:
            report = await base_analysis()
        report, checked = await verification(report)
        if rounds and checked["status"] != "unavailable":
            revision_module = _component("revise", ("revise_once",))
            revising[0] = True
            before_report, before_check = deepcopy(report), deepcopy(checked)
            report, checked, revision = await revision_module.revise_once(
                report, checked, evidence, rag, rewrite=generator("revise"),
                verify=verification, max_rounds=rounds)
            # The independent revision module owns adoption, while this seam
            # also prevents a broken adapter from changing strategy identities
            # or dropping/changing previously supported claims.
            targets = lambda value: [(row["name"], row["ticker"]) for row in value["stock_impacts"]]
            protected = {row["claim_id"]: row for row in before_report.get("claims", [])
                         if row["verdict"] == "supported"}
            candidates = {row["claim_id"]: row for row in report.get("claims", [])}
            unchanged = all(
                claim_id in candidates and all(candidates[claim_id].get(key) == claim.get(key)
                                               for key in ("text", "kind", "evidence", "falsifier"))
                for claim_id, claim in protected.items())
            if targets(report) != targets(before_report) or not unchanged:
                report, checked = before_report, before_check
                # Do not fabricate successful revision statistics from an
                # incompatible adapter. The job stops before persistence.
                raise ValueError("Invalid revision output")
            checked["revision"] = revision
        elif rounds:
            checked["revision"] = {"attempts": 0, "adopted": False,
                                   "initial_status": checked["status"],
                                   "initial_notes": list(checked["notes"]), "rounds": []}
        if not claims_on:
            report.pop("claims", None)
            for stock in report["stock_impacts"]:
                stock.pop("claim_ids", None)
        if enabled():
            report["agent_trace"] = deepcopy(entries)
        return report, checked
