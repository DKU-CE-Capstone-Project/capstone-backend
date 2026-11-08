"""Replay fixed, locally exported evidence and prepare blinded comparison sheets.

No database or article-provider connections are made. Fixture replay is the default;
provider generation requires --live and an explicit logical generation-call ceiling.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
ARM_CONFIGS = {
    "A": (False, False, 0),
    "B": (True, False, 2),
    "C": (True, True, 2),
    "B-prime": (False, False, 2),
    "C-prime": (False, True, 2),
}
# Analysis/debate, initial critic, up to two rewrite+critic rounds, strategy retry.
# Article extraction, RAG fetching, SDK retries and provider fallback are excluded.
CALL_BOUNDS = {"A": 4, "B": 10, "C": 10, "B-prime": 8, "C-prime": 8}
EVIDENCE_KEYS = {
    "news_id", "title", "source_name", "source_url", "published_at", "body_id",
    "content_hash", "extractor_version", "input_excerpt", "input_char_count", "truncated",
}
RAG_KEYS = {"news_id", "title", "source_url", "description", "description_hash"}


class ComparisonError(ValueError):
    """A safe, fixed error code; never include the supplied body or provider error."""


class ComparisonUnavailable(ComparisonError):
    pass


class CallLimitExceeded(ComparisonError):
    pass


@dataclass
class CallLimit:
    maximum: int
    started: int = 0

    def reserve(self) -> None:
        if self.started >= self.maximum:
            raise CallLimitExceeded("LOGICAL_GENERATION_CALL_LIMIT")
        self.started += 1


def _digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > 5_000_000:
        raise ComparisonError("INPUT_FILE_TOO_LARGE")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ComparisonError("INVALID_INPUT_JSON") from None
    if not isinstance(value, dict):
        raise ComparisonError("EXPECTED_INPUT_OBJECT")
    return value


def evidence_snapshot(payload: dict) -> tuple[list[dict], list[dict], dict]:
    """Require the persisted excerpts; public v2 DTOs cannot reproduce model input."""
    report = payload.get("report", payload)
    if not isinstance(report, dict):
        raise ComparisonError("EXPECTED_REPORT_OBJECT")
    raw_evidence = report.get("used_evidence")
    if not isinstance(raw_evidence, list) or not 1 <= len(raw_evidence) <= 10:
        raise ComparisonError("EXPECTED_1_TO_10_USED_EVIDENCE")
    evidence = []
    seen = set()
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise ComparisonError("INVALID_USED_EVIDENCE")
        news_id, excerpt = item.get("news_id"), item.get("input_excerpt")
        if not isinstance(news_id, str) or not 1 <= len(news_id) <= 128 or news_id in seen:
            raise ComparisonError("INVALID_OR_DUPLICATE_NEWS_ID")
        if not isinstance(excerpt, str) or not 1 <= len(excerpt) <= 12000:
            raise ComparisonError("FULL_INPUT_EXCERPT_REQUIRED")
        evidence.append(deepcopy({key: item[key] for key in EVIDENCE_KEYS if key in item}))
        seen.add(news_id)
    if sum(len(item["input_excerpt"]) for item in evidence) > 60000:
        raise ComparisonError("EVIDENCE_INPUT_LIMIT_EXCEEDED")
    raw_rag = report.get("rag_evidence", [])
    if not isinstance(raw_rag, list) or len(raw_rag) > 3:
        raise ComparisonError("INVALID_RAG_SNAPSHOT")
    rag = []
    for item in raw_rag:
        if not isinstance(item, dict) or not isinstance(item.get("news_id"), str):
            raise ComparisonError("INVALID_RAG_SNAPSHOT")
        if not isinstance(item.get("description"), str) or len(item["description"]) > 4000:
            raise ComparisonError("INVALID_RAG_SNAPSHOT")
        rag.append(deepcopy({key: item[key] for key in RAG_KEYS if key in item}))
    options = payload.get("strategy_options", {"risk_level": "medium", "period": "short"})
    if (not isinstance(options, dict) or options.get("risk_level") not in {"low", "medium", "high"}
            or options.get("period") not in {"short", "mid", "long"}):
        raise ComparisonError("INVALID_STRATEGY_OPTIONS")
    return evidence, rag, {"risk_level": options["risk_level"], "period": options["period"]}


def validate_arms(arms: list[str]) -> None:
    if not arms or len(arms) != len(set(arms)):
        raise ComparisonError("EXPECTED_UNIQUE_COMPARISON_ARMS")
    if "D" in arms:
        raise ComparisonUnavailable("D_LOOKUP_IS_OUT_OF_CONTRACT")
    if any(arm not in ARM_CONFIGS for arm in arms):
        raise ComparisonUnavailable("UNKNOWN_COMPARISON_ARM")


def fixture_runner(fixtures: dict, arms: list[str]):
    """Replay supplied synthetic/recorded outputs, never pretend to regenerate them."""
    values = fixtures.get("arms")
    if not isinstance(values, dict) or any(not isinstance(values.get(arm), dict) for arm in arms):
        raise ComparisonError("FIXTURE_OUTPUT_REQUIRED_FOR_EACH_ARM")
    for arm in arms:
        _public_result(values[arm])

    async def run(arm, evidence, rag, options, limit):
        return deepcopy(values[arm])

    return run


def _nullable(schema):
    return ("nullable", schema)


def _known_shape(value, schema):
    """Copy only known schema fields and reject nested objects in scalar slots."""
    if isinstance(schema, tuple) and schema[0] == "nullable":
        return None if value is None else _known_shape(value, schema[1])
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ComparisonError("INVALID_RUNNER_FIELD_SHAPE")
        return {key: _known_shape(value[key], child) for key, child in schema.items() if key in value}
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ComparisonError("INVALID_RUNNER_FIELD_SHAPE")
        return [_known_shape(item, schema[0]) for item in value]
    if isinstance(schema, frozenset):
        if type(value) is not str or value not in schema:
            raise ComparisonError("INVALID_RUNNER_FIELD_SHAPE")
        return value
    if schema is int:
        valid = type(value) is int and value >= 0
    elif schema is float:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    else:
        valid = type(value) is schema
    if not valid:
        raise ComparisonError("INVALID_RUNNER_FIELD_SHAPE")
    return value


_STATUS = frozenset({"passed", "warning", "unavailable"})
_DIRECTION = frozenset({"up", "down", "mixed", "unknown"})
_TRACE = {
    "seq": int, "agent": str, "elapsed_ms": int, "summary": str, "decision": str,
    "tokens_in": _nullable(int), "tokens_out": _nullable(int),
}
_CLAIM = {
    "claim_id": str, "text": str, "kind": frozenset({"fact", "interpretation"}),
    "evidence": [{"news_id": str, "sentence": str}], "falsifier": _nullable(str),
    "verdict": frozenset({"supported", "refuted", "unclear"}),
    "sentence_found": bool, "reason": _nullable(str),
}
_REPORT = {
    "title": str, "event_summary": str, "event_analysis": str, "market_impact": str,
    "stock_impacts": [{"ticker": _nullable(str), "name": str, "impact": str,
        "direction": _DIRECTION, "evidence_news_ids": [str], "claim_ids": _nullable([str])}],
    "risk_factors": [str], "claims": _nullable([_CLAIM]),
    "debate": _nullable([{"name": str, "ticker": _nullable(str), "bull": [str], "bear": [str],
        "direction": _DIRECTION, "verdict": str, "evidence_news_ids": [str]}]),
    "agent_trace": _nullable([_TRACE]),
}
_STRATEGY = {
    "strategy_summary": str, "risk_warning": str,
    "strategy_items": [{"ticker": _nullable(str), "stock_name": str,
        "action": frozenset({"buy", "hold", "sell", "watch"}), "reason": str}],
    "agent_trace": _nullable([_TRACE]),
}
_VERIFICATION = {
    "status": _STATUS, "notes": [str],
    "claims_summary": _nullable({key: int for key in (
        "total", "supported", "refuted", "unclear", "sentence_missing")}),
    "revision": _nullable({"attempts": int, "adopted": bool, "initial_status": _STATUS,
        "initial_notes": [str], "rounds": [{"round": int, "targets": [str], "before": int,
            "after": _nullable(int), "decision": str,
            "changes": [{"claim_id": str, "action": str, "before_text": str}]}]}),
}
_PROMPT_STAGES = ("analysis_fallback", "analysis", "bull", "bear", "judge", "critic", "claims_critic", "revise", "strategy")
_GENERATION = {
    "language": str, "report_type": str,
    "model": {"provider": str, "name": str, "config_hash": str},
    "prompt_version": str, "preprocessing_version": str, "rag_version": str,
    "input_budget_version": str,
    "pipeline": {"multi_agent": bool, "claim_check": bool, "revise_max_rounds": int,
        "prompt_versions": {key: str for key in _PROMPT_STAGES},
        "max_output_tokens": {key: int for key in _PROMPT_STAGES}},
}
_USAGE = {
    "sdk_generate_events": int, "tokens_in": _nullable(int), "tokens_out": _nullable(int),
    "est_cost_usd": _nullable(float),
}


def _public_result(value: dict) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("report"), dict):
        raise ComparisonError("INVALID_RUNNER_REPORT")
    report = value["report"]
    if any(key not in report for key in (
        "title", "event_summary", "event_analysis", "market_impact", "stock_impacts", "risk_factors"
    )):
        raise ComparisonError("INVALID_RUNNER_REPORT")
    verification = value.get("verification", report.get("verification"))
    if not isinstance(verification, dict) or not {"status", "notes"}.issubset(verification):
        raise ComparisonError("INVALID_RUNNER_VERIFICATION")
    strategy = value.get("strategy")
    if not isinstance(strategy, dict) or "strategy_items" not in strategy:
        raise ComparisonError("INVALID_RUNNER_STRATEGY")
    return {
        "report": _known_shape(report, _REPORT),
        "verification": _known_shape(verification, _VERIFICATION),
        "strategy": _known_shape(strategy, _STRATEGY),
        "usage": _known_shape(value.get("usage"), _nullable(_USAGE)),
        "generation": _known_shape(value.get("generation"), _nullable(_GENERATION)),
    }


def prepare_live_runner(arms: list[str]):
    """Validate the full requested matrix before the first model invocation."""
    validate_arms(arms)
    from app.agents import llm
    from app.config import settings
    from app.v2 import reports

    advanced = any(ARM_CONFIGS[arm] != ARM_CONFIGS["A"] for arm in arms)
    pipeline = None
    if advanced:
        try:
            pipeline = importlib.import_module("app.v2.agents.pipeline")
            if not callable(getattr(pipeline, "analyze_report", None)):
                raise TypeError
            for arm in arms:
                multi, claims, rounds = ARM_CONFIGS[arm]
                pipeline.ensure_dependencies(multi_agent=multi, claim_check=claims, max_rounds=rounds)
            for key in ("v2_multi_agent", "v2_claim_check", "v2_revise_max_rounds"):
                if not hasattr(settings, key):
                    raise ImportError
        except Exception:  # noqa: BLE001 — preflight exposes only a safe dependency code
            raise ComparisonUnavailable("ADVANCED_PIPELINE_DEPENDENCY_MISSING") from None

    @contextmanager
    def configure(arm):
        values = dict(zip(("v2_multi_agent", "v2_claim_check", "v2_revise_max_rounds"), ARM_CONFIGS[arm]))
        prior = {key: getattr(settings, key) for key in values if hasattr(settings, key)}
        try:
            for key in prior:
                setattr(settings, key, values[key])
            yield
        finally:
            for key, value in prior.items():
                setattr(settings, key, value)

    async def run(arm, evidence, rag, options, limit):
        # A standalone process runs arms sequentially. Never install these adapters
        # in a serving process or concurrent request handler.
        from app import telemetry

        original_generate, original_emit = llm.generate, telemetry.emit
        original_with_usage = getattr(llm, "generate_with_usage", None)
        inside_generation = ContextVar("comparison_inside_generation", default=False)
        events = []

        async def bounded_generate(*args, **kwargs):
            if not inside_generation.get():
                limit.reserve()
            token = inside_generation.set(True)
            try:
                return await original_generate(*args, **kwargs)
            finally:
                inside_generation.reset(token)

        async def bounded_with_usage(*args, **kwargs):
            if not inside_generation.get():
                limit.reserve()
            token = inside_generation.set(True)
            try:
                return await original_with_usage(*args, **kwargs)
            finally:
                inside_generation.reset(token)

        def capture(kind, **fields):
            if kind == "sdk_call" and fields.get("operation") == "generate":
                events.append({key: fields.get(key) for key in (
                    "provider", "model", "input_tokens", "output_tokens", "outcome"
                )})
            original_emit(kind, **fields)

        try:
            llm.generate, telemetry.emit = bounded_generate, capture
            if callable(original_with_usage):
                llm.generate_with_usage = bounded_with_usage
            with configure(arm):
                if arm == "A":
                    report = await reports.analyze(evidence, rag)
                    verification = await reports.verify(report, evidence, rag)
                else:
                    report, verification = await pipeline.analyze_report(
                        evidence, rag, analyze=reports.analyze, verify=reports.verify
                    )
                report["verification"] = verification
                for attempt in range(2):
                    try:
                        strategy = await reports.generate_strategy(report, **options)
                        break
                    except reports.InvalidStrategyOutput:
                        if attempt:
                            raise
                generation = reports.generation()
        finally:
            llm.generate, telemetry.emit = original_generate, original_emit
            if callable(original_with_usage):
                llm.generate_with_usage = original_with_usage
        usage = {"sdk_generate_events": len(events), "est_cost_usd": None}
        for source, target in (("input_tokens", "tokens_in"), ("output_tokens", "tokens_out")):
            measured = [event[source] for event in events]
            usage[target] = sum(measured) if measured and all(type(v) is int and v >= 0 for v in measured) else None
        return {"report": report, "verification": verification, "strategy": strategy,
                "usage": usage, "generation": generation}

    run.metadata = {key: getattr(settings, key) for key in (
        "llm_provider", "gemini_model", "claude_model", "gemini_service_tier",
        "llm_max_output_tokens", "use_critic",
    )}

    return run


def _csv_safe(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_safe(row.get(key)) for key in fields} for row in rows)


def _blind_text(result: dict) -> str:
    report, strategy = result["report"], result["strategy"]
    common = {key: report[key] for key in (
        "title", "event_summary", "event_analysis", "market_impact", "risk_factors"
    )}
    common["stock_impacts"] = [
        {key: item.get(key) for key in ("name", "ticker", "impact", "direction", "evidence_news_ids")}
        for item in report["stock_impacts"] if isinstance(item, dict)
    ]
    common["strategy"] = {key: strategy.get(key) for key in (
        "strategy_summary", "risk_warning", "strategy_items"
    )}
    return json.dumps(common, ensure_ascii=False, indent=2)


def save_outputs(directory: Path, record: dict, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    completed = [r for r in record["runs"] if r["outcome"] == "completed"]
    random.Random(seed).shuffle(completed)
    blind_rows, key_rows = [], []
    for index, run in enumerate(completed, 1):
        assessment_id = f"sample-{index:03d}"
        blind_rows.append({
            "assessment_id": assessment_id, "content": _blind_text(run["result"]),
            "grounding_0_1_2": "", "important_information_0_1_2": "", "comments": "",
        })
        key_rows.append({"assessment_id": assessment_id, "arm": run["arm"], "repeat": run["repeat"]})
    rows = []
    for run in record["runs"]:
        result = run.get("result", {})
        impacts = result.get("report", {}).get("stock_impacts", [])
        strategies = result.get("strategy", {}).get("strategy_items", [])
        usage = result.get("usage") or {}
        rows.append({
            "arm": run["arm"], "repeat": run["repeat"], "outcome": run["outcome"],
            "input_digest": record["input_digest"], "execution_mode": record["execution_mode"],
            "elapsed_ms": run["elapsed_ms"], "logical_generate_calls": run["logical_generate_calls"],
            "tokens_in": usage.get("tokens_in"), "tokens_out": usage.get("tokens_out"),
            "est_cost_usd": usage.get("est_cost_usd"),
            "verification_status": result.get("verification", {}).get("status"),
            "warning_count": len(result.get("verification", {}).get("notes", [])) if result else None,
            "up_count": sum(i.get("direction") == "up" for i in impacts) if result else None,
            "impact_count": len(impacts) if result else None,
            "buy_count": sum(i.get("action") == "buy" for i in strategies) if result else None,
            "strategy_count": len(strategies) if result else None, "error_code": run.get("error_code"),
        })
    (directory / "comparison.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(directory / "comparison.csv", rows, list(rows[0]))
    _write_csv(directory / "blind_scores.csv", blind_rows, [
        "assessment_id", "content", "grounding_0_1_2", "important_information_0_1_2", "comments",
    ])
    (directory / "answer_key.json").write_text(json.dumps({
        "input_digest": record["input_digest"], "seed": seed, "samples": key_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def compare(payload: dict, arms: list[str], directory: Path, *, runner,
                  execution_mode: str = "fixture", max_calls: int = 0,
                  repeats: int = 1, seed: int = 0) -> dict:
    validate_arms(arms)
    if execution_mode not in {"fixture", "live"} or type(repeats) is not int or not 1 <= repeats <= 20:
        raise ComparisonError("INVALID_COMPARISON_OPTIONS")
    if type(max_calls) is not int or max_calls < 0:
        raise ComparisonError("INVALID_CALL_LIMIT")
    planned = sum(CALL_BOUNDS[arm] for arm in arms) * repeats if execution_mode == "live" else 0
    if planned > max_calls:
        raise ComparisonError("CALL_LIMIT_BELOW_PLANNED_UPPER_BOUND")
    if directory.exists():
        raise FileExistsError(directory)
    evidence, rag, options = evidence_snapshot(payload)
    limit = CallLimit(max_calls)
    record = {
        "schema_version": SCHEMA_VERSION, "started_at": datetime.now(UTC).isoformat(),
        "execution_mode": execution_mode, "input_digest": _digest({
            "used_evidence": evidence, "rag_evidence": rag, "strategy_options": options,
        }), "selected_news_ids": [e["news_id"] for e in evidence],
        "planned_logical_calls_upper_bound": planned, "max_logical_calls": max_calls,
        "call_count_scope": "llm.generate entry calls; not SDK-internal retries, HTTP attempts, or bills",
        "arms": {arm: dict(zip(("multi_agent", "claim_check", "revise_max_rounds"), ARM_CONFIGS[arm])) for arm in arms},
        "runner_metadata": deepcopy(getattr(runner, "metadata", None)),
        "runs": [],
    }
    try:
        for repeat in range(1, repeats + 1):
            for arm in arms:
                started, prior_calls = time.monotonic(), limit.started
                run = {"arm": arm, "repeat": repeat, "outcome": "error"}
                try:
                    result = await runner(arm, deepcopy(evidence), deepcopy(rag), deepcopy(options), limit)
                    run.update(outcome="completed", result=_public_result(result))
                except asyncio.CancelledError:
                    run.update(outcome="interrupted", error_code="INTERRUPTED")
                    raise
                except CallLimitExceeded:
                    run["error_code"] = "LOGICAL_GENERATION_CALL_LIMIT"
                except Exception:  # noqa: BLE001 — isolate arm errors without exporting provider details
                    # Provider errors and fixture content are never copied into logs.
                    run["error_code"] = "COMPARISON_ARM_FAILED"
                finally:
                    run.update(elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                               logical_generate_calls=limit.started-prior_calls)
                    record["runs"].append(run)
                if run.get("error_code") == "LOGICAL_GENERATION_CALL_LIMIT":
                    return record
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        record["logical_generate_calls"] = limit.started
        save_outputs(directory, record, seed)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local persistence export with input_excerpt")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures", type=Path, help="Replay supplied outputs without model calls")
    mode.add_argument("--live", action="store_true", help="Allow provider generation in this standalone process")
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--max-calls", type=int, default=0, help="Logical generate calls; SDK retries excluded")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        validate_arms(args.arms)
        if not 1 <= args.repeats <= 20 or args.max_calls < 0:
            raise ComparisonError("INVALID_COMPARISON_OPTIONS")
        if args.live and args.max_calls < sum(CALL_BOUNDS[arm] for arm in args.arms) * args.repeats:
            raise ComparisonError("CALL_LIMIT_BELOW_PLANNED_UPPER_BOUND")
        payload = read_json(args.input)
        evidence_snapshot(payload)
        if args.live:
            # Script invocation must use this checkout, not another editable install.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        runner = prepare_live_runner(args.arms) if args.live else fixture_runner(read_json(args.fixtures), args.arms)
        record = asyncio.run(compare(payload, args.arms, args.output_dir, runner=runner,
            execution_mode="live" if args.live else "fixture", max_calls=args.max_calls,
            repeats=args.repeats, seed=args.seed))
    except ComparisonError as exc:
        parser.exit(2, str(exc) + "\n")
    except FileExistsError:
        parser.exit(2, "OUTPUT_DIRECTORY_ALREADY_EXISTS\n")
    except OSError:
        parser.exit(2, "COMPARISON_FILE_IO_FAILED\n")
    except KeyboardInterrupt:
        return 130
    outcomes = [run["outcome"] for run in record["runs"]]
    print(json.dumps({"execution_mode": record["execution_mode"], "completed": outcomes.count("completed"),
                      "runs": len(outcomes), "logical_generate_calls": record["logical_generate_calls"]}))
    return 0 if outcomes and all(outcome == "completed" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
