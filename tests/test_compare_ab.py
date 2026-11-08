import asyncio
import csv
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from scripts import compare_ab


def snapshot():
    return {
        "report": {
            "used_evidence": [{
                "news_id": "news-1", "input_excerpt": "회사의 수출은 증가했다.",
                "title": "수출 증가", "body_id": "body-1", "content_hash": "snapshot-hash",
            }],
            "rag_evidence": [{"news_id": "rag-1", "description": "이전 분기의 참고 자료"}],
        },
        "strategy_options": {"risk_level": "medium", "period": "short"},
        "unrelated_private_metadata": "fixture-private-metadata",
    }


def result():
    return {
        "report": {
            "title": "수출 분석", "event_summary": "수출이 증가함", "event_analysis": "증가세 확인",
            "market_impact": "추세 확인 필요", "stock_impacts": [{
                "name": "수출 업종", "ticker": None, "impact": "단기 수요 기대",
                "direction": "up", "evidence_news_ids": ["news-1"], "claim_ids": ["c1"],
            }],
            "risk_factors": ["일시적 증가일 수 있음"],
            "agent_trace": [{"agent": "bull", "summary": "fixture-arm-revealing-trace"}],
        },
        "verification": {"status": "warning", "notes": ["fixture-arm-revealing-warning"]},
        "strategy": {"strategy_summary": "관망", "risk_warning": "추가 확인 필요",
                     "strategy_items": [{"stock_name": "수출 업종", "ticker": None,
                                         "action": "watch", "reason": "추가 확인 필요"}]},
        "api_key": "fixture-never-export-key",
    }


async def test_fixture_replay_isolated_snapshots_and_blinding(tmp_path):
    payload = snapshot()
    before = deepcopy(payload)
    arms = ["A", "B", "C", "B-prime", "C-prime"]
    directory = tmp_path / "comparison"
    run = compare_ab.fixture_runner({"arms": {arm: result() for arm in arms}}, arms)
    seen = []

    async def runner(arm, evidence, rag, options, limit):
        seen.append((deepcopy(evidence), deepcopy(rag), deepcopy(options)))
        value = await run(arm, evidence, rag, options, limit)
        evidence[0]["input_excerpt"] = "mutated only inside this run"
        rag.clear()
        return value

    record = await compare_ab.compare(payload, arms, directory, runner=runner, repeats=2, seed=19)
    assert payload == before
    assert all(e[0]["input_excerpt"] == "회사의 수출은 증가했다." and len(r) == 1 for e, r, _ in seen)
    assert len(record["runs"]) == 10
    assert record["logical_generate_calls"] == 0
    assert record["execution_mode"] == "fixture"
    saved = (directory / "comparison.json").read_text(encoding="utf-8")
    assert "fixture-never-export-key" not in saved and "fixture-private-metadata" not in saved
    assert "input_excerpt" not in saved
    with (directory / "comparison.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 10 and all(row["tokens_in"] == "" for row in rows)
    assert all(row["est_cost_usd"] == "" for row in rows)
    assert all(row["up_count"] == "1" and row["buy_count"] == "0" for row in rows)
    blind = (directory / "blind_scores.csv").read_text(encoding="utf-8-sig")
    assert "fixture-arm-revealing" not in blind and "claim_ids" not in blind
    with (directory / "blind_scores.csv").open(encoding="utf-8-sig", newline="") as file:
        scoring = list(csv.DictReader(file))
    assert all(row["grounding_0_1_2"] == "" for row in scoring)
    assert "arm" not in scoring[0] and "repeat" not in scoring[0]
    key = json.loads((directory / "answer_key.json").read_text(encoding="utf-8"))
    assert {row["assessment_id"] for row in key["samples"]} == {row["assessment_id"] for row in scoring}
    assert {row["arm"] for row in key["samples"]} == set(arms)
    with pytest.raises(FileExistsError):
        await compare_ab.compare(payload, arms, directory, runner=runner)


async def test_input_hash_changes_when_excerpt_or_rag_changes(tmp_path):
    async def runner(*args):
        return result()

    first = snapshot()
    changed = deepcopy(first)
    changed["report"]["used_evidence"][0]["input_excerpt"] += " 정정 보도"
    third = deepcopy(first)
    third["report"]["rag_evidence"][0]["description"] += " 정정"
    hashes = []
    for index, payload in enumerate((first, changed, third)):
        r = await compare_ab.compare(payload, ["A"], tmp_path / str(index), runner=runner)
        hashes.append(r["input_digest"])
    assert len(set(hashes)) == 3


async def test_live_limit_stops_before_an_excess_generation(tmp_path):
    completed_calls = []

    async def runner(arm, evidence, rag, options, limit):
        for i in range(5):
            limit.reserve()
            completed_calls.append(i)
        return result()

    record = await compare_ab.compare(snapshot(), ["A"], tmp_path / "capped", runner=runner,
                                      execution_mode="live", max_calls=4)
    assert completed_calls == [0, 1, 2, 3]
    assert record["runs"][0]["error_code"] == "LOGICAL_GENERATION_CALL_LIMIT"
    assert record["logical_generate_calls"] == 4


async def test_preflight_rejects_public_dto_and_insufficient_limit(tmp_path):
    called = False

    async def runner(*args):
        nonlocal called
        called = True
        return result()

    payload = snapshot()
    del payload["report"]["used_evidence"][0]["input_excerpt"]
    with pytest.raises(compare_ab.ComparisonError, match="FULL_INPUT_EXCERPT_REQUIRED"):
        await compare_ab.compare(payload, ["A"], tmp_path / "public", runner=runner)
    with pytest.raises(compare_ab.ComparisonError, match="CALL_LIMIT_BELOW_PLANNED"):
        await compare_ab.compare(snapshot(), ["A", "B"], tmp_path / "small", runner=runner,
                                  execution_mode="live", max_calls=13)
    assert not called and not list(tmp_path.iterdir())


@pytest.mark.parametrize("arms,error", [
    (["D"], "D_LOOKUP_IS_OUT_OF_CONTRACT"),
    (["A", "A"], "EXPECTED_UNIQUE_COMPARISON_ARMS"),
    (["Z"], "UNKNOWN_COMPARISON_ARM"),
])
def test_unavailable_or_ambiguous_arms_fail_upfront(arms, error):
    with pytest.raises(compare_ab.ComparisonError, match=error):
        compare_ab.validate_arms(arms)


async def test_error_records_omit_provider_error_and_cancellation_saves(tmp_path):
    async def failed(*args):
        raise RuntimeError("fixture-private-provider-diagnostic")

    directory = tmp_path / "failed"
    record = await compare_ab.compare(snapshot(), ["A"], directory, runner=failed)
    assert record["runs"][0]["error_code"] == "COMPARISON_ARM_FAILED"
    assert "fixture-private-provider-diagnostic" not in (directory / "comparison.json").read_text()

    async def cancelled(*args):
        raise asyncio.CancelledError

    directory = tmp_path / "cancelled"
    with pytest.raises(asyncio.CancelledError):
        await compare_ab.compare(snapshot(), ["A"], directory, runner=cancelled)
    record = json.loads((directory / "comparison.json").read_text())
    assert record["runs"][0]["outcome"] == "interrupted"


def test_fixture_cli_runs_without_importing_provider_configuration(tmp_path):
    source, fixtures = tmp_path / "source.json", tmp_path / "fixtures.json"
    source.write_text(json.dumps(snapshot()), encoding="utf-8")
    fixtures.write_text(json.dumps({"arms": {"A": result(), "B": result()}}), encoding="utf-8")
    directory = tmp_path / "cli"
    run = subprocess.run([
        sys.executable, "scripts/compare_ab.py", "--input", str(source),
        "--fixtures", str(fixtures), "--arms", "A", "B", "--output-dir", str(directory),
    ], text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"execution_mode": "fixture", "completed": 2,
                                    "runs": 2, "logical_generate_calls": 0}
    assert len(list(directory.iterdir())) == 4


def test_csv_cells_cannot_start_a_spreadsheet_formula():
    assert compare_ab._csv_safe(" =HYPERLINK(...) ").startswith("'")
    assert compare_ab._csv_safe("정상 텍스트") == "정상 텍스트"


async def test_live_a_runner_counts_calls_usage_and_restores_adapters(tmp_path, monkeypatch):
    from app import telemetry
    from app.agents import llm
    from app.config import settings

    monkeypatch.setattr(settings, "use_critic", True)
    outputs = iter([
        result()["report"],
        {"grounded": True, "confidence": .8, "issues": [], "unsupported_claims": [], "invalid_tickers": []},
        {"strategy_summary": "관망", "risk_warning": "참고용", "strategy_items": []},
    ])

    async def fake_generate(prompt):
        telemetry.emit("sdk_call", provider="gemini", operation="generate", model="fake",
                       input_tokens=20, output_tokens=10, outcome="success")
        return json.dumps(next(outputs), ensure_ascii=False)

    monkeypatch.setattr(llm, "generate", fake_generate)
    before_emit = telemetry.emit
    runner = compare_ab.prepare_live_runner(["A"])
    record = await compare_ab.compare(snapshot(), ["A"], tmp_path / "live", runner=runner,
                                      execution_mode="live", max_calls=4)
    run = record["runs"][0]
    assert run["outcome"] == "completed"
    assert run["logical_generate_calls"] == 3
    assert run["result"]["usage"]["tokens_in"] == 60
    assert run["result"]["usage"]["tokens_out"] == 30
    assert run["result"]["usage"]["est_cost_usd"] is None
    assert llm.generate is fake_generate and telemetry.emit is before_emit


def test_missing_advanced_dependencies_are_not_silently_downgraded(monkeypatch):
    def missing(name):
        raise ImportError("fixture-private-path")

    monkeypatch.setattr(compare_ab.importlib, "import_module", missing)
    with pytest.raises(compare_ab.ComparisonUnavailable) as error:
        compare_ab.prepare_live_runner(["A", "B"])
    assert str(error.value) == "ADVANCED_PIPELINE_DEPENDENCY_MISSING"


async def test_nested_generation_metadata_uses_known_schema_not_secret_key_heuristics(tmp_path):
    value = result()
    original_text = "API 키라는 단어가 포함된 정상적인 기사 해석"
    value["report"]["event_analysis"] = original_text
    value["generation"] = {
        "language": "ko", "report_type": "investment",
        "model": {"provider": "gemini", "name": "fake", "config_hash": "abc",
                  "api_key": "fixture-private-model-key"},
        "pipeline": {
            "multi_agent": True, "claim_check": True, "revise_max_rounds": 2,
            "prompt_versions": {"judge": "debate-judge-v1",
                                "transport": {"password": "fixture-private-prompt-key"}},
            "max_output_tokens": {"judge": 8192,
                                  "custom": {"config": "fixture-private-output-key"}},
            "transport": {"credentials": "fixture-private-pipeline-key"},
        },
        "extra": {"private": "fixture-private-generation-key"},
    }
    value["report"]["stock_impacts"][0]["extra"] = {"x": "fixture-private-impact-key"}
    value["verification"]["revision"] = {
        "attempts": 0, "adopted": False, "initial_status": "warning", "initial_notes": [],
        "rounds": [], "extra": {"x": "fixture-private-revision-key"},
    }
    runner = compare_ab.fixture_runner({"arms": {"C": value}}, ["C"])
    directory = tmp_path / "recursive"
    record = await compare_ab.compare(snapshot(), ["C"], directory, runner=runner)
    saved = (directory / "comparison.json").read_text(encoding="utf-8")
    assert "fixture-private-" not in saved
    exported = record["runs"][0]["result"]
    assert exported["report"]["event_analysis"] == original_text
    assert exported["generation"]["model"] == {
        "provider": "gemini", "name": "fake", "config_hash": "abc",
    }
    assert exported["generation"]["pipeline"] == {
        "multi_agent": True, "claim_check": True, "revise_max_rounds": 2,
        "prompt_versions": {"judge": "debate-judge-v1"},
        "max_output_tokens": {"judge": 8192},
    }


@pytest.mark.parametrize("path", [
    ("report", "stock_impacts", 0, "impact"),
    ("report", "claims", 0, "text"),
    ("report", "debate", 0, "bull", 0),
    ("report", "agent_trace", 0, "summary"),
    ("report", "risk_factors", 0),
    ("strategy", "strategy_items", 0, "reason"),
    ("verification", "revision", "rounds", 0, "targets", 0),
    ("generation", "model", "name"),
    ("generation", "pipeline", "prompt_versions", "judge"),
    ("generation", "pipeline", "max_output_tokens", "judge"),
    ("usage", "tokens_in"),
])
def test_nested_objects_cannot_occupy_contract_scalar_fields(path):
    value = result()
    value["report"]["claims"] = [{"claim_id": "c1", "text": "수출 증가"}]
    value["report"]["debate"] = [{"bull": ["수출 증가"]}]
    value["verification"]["revision"] = {"rounds": [{"targets": ["c1"]}]}
    value["generation"] = {"model": {"name": "fake"}, "pipeline": {
        "prompt_versions": {"judge": "v1"}, "max_output_tokens": {"judge": 8192},
    }}
    value["usage"] = {"tokens_in": 12}
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = {"nested_metadata": "fixture-private-nested"}
    with pytest.raises(compare_ab.ComparisonError, match="INVALID_RUNNER_FIELD_SHAPE"):
        compare_ab.fixture_runner({"arms": {"A": value}}, ["A"])
