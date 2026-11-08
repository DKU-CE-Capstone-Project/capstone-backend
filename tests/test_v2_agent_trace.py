"""Execution traces are bounded and never mix separate report/strategy runs."""
import asyncio

import pytest

from app.v2.agents import trace


def add(agent="analysis", decision="ok", summary="단계 완료", **kwargs):
    trace.record(agent, decision, summary, elapsed_ms=kwargs.pop("elapsed_ms", 3), **kwargs)


def test_outside_scope_is_noop_and_does_not_validate_untrusted_values():
    trace.record(None, None, object(), elapsed_ms=-1, usage=object())
    assert trace.snapshot() == []


def test_records_sequence_summary_limit_and_only_real_usage_counts():
    with trace.collect() as entries:
        add(summary="가" * 121, usage={"service": "gemini", "model": "fixture", "tokens_in": 12, "tokens_out": 0})
        add("critic", "warning", usage={"tokens_in": None})
        assert entries == [
            {"seq": 1, "agent": "analysis", "elapsed_ms": 3, "summary": "가" * 120, "decision": "ok", "tokens_in": 12, "tokens_out": 0},
            {"seq": 2, "agent": "critic", "elapsed_ms": 3, "summary": "단계 완료", "decision": "warning"},
        ]
    assert trace.snapshot() == []
    assert len(entries) == 2


async def test_child_tasks_share_completion_order_within_execution():
    release = asyncio.Event()

    async def bull():
        await release.wait()
        add("bull", "ok")

    async def bear():
        add("bear", "failed")
        release.set()

    with trace.collect() as entries:
        await asyncio.gather(bull(), bear())
        add("judge", "ok")
    assert [(e["seq"], e["agent"]) for e in entries] == [(1, "bear"), (2, "bull"), (3, "judge")]


async def test_concurrent_executions_are_isolated():
    ready = asyncio.Event()

    async def report():
        with trace.collect() as entries:
            add("analysis", "ok", "report")
            ready.set()
            await asyncio.sleep(0)
            add("critic", "ok", "report")
        return entries

    async def strategy():
        await ready.wait()
        with trace.collect() as entries:
            await asyncio.sleep(0)
            add("strategy", "ok", "strategy")
        return entries

    report_entries, strategy_entries = await asyncio.gather(report(), strategy())
    assert [e["seq"] for e in report_entries] == [1, 2]
    assert [e["summary"] for e in report_entries] == ["report", "report"]
    assert strategy_entries == [{"seq": 1, "agent": "strategy", "elapsed_ms": 3, "summary": "strategy", "decision": "ok"}]


def test_nested_scope_and_exception_restore_parent():
    with trace.collect() as outer:
        add()
        with pytest.raises(RuntimeError), trace.collect() as inner:
            add("strategy", "failed")
            raise RuntimeError("fixture")
        add("critic", "ok")
    assert [e["agent"] for e in outer] == ["analysis", "critic"]
    assert [e["agent"] for e in inner] == ["strategy"]
    assert trace.snapshot() == []


async def test_cancellation_restores_scope_without_swallowing_cancellation():
    with pytest.raises(asyncio.CancelledError), trace.collect() as entries:
        add()
        raise asyncio.CancelledError()
    assert len(entries) == 1
    assert trace.snapshot() == []


async def test_late_child_cannot_append_after_scope_has_closed():
    release = asyncio.Event()

    async def late():
        await release.wait()
        add("bull", "ok")

    with trace.collect() as entries:
        task = asyncio.create_task(late())
        add()
    release.set()
    await task
    assert [e["agent"] for e in entries] == ["analysis"]


def test_snapshot_is_independent_and_record_count_is_bounded():
    with trace.collect() as entries:
        add()
        saved = trace.snapshot()
        saved[0]["summary"] = "changed"
        saved.append({})
        assert entries[0]["summary"] == "단계 완료"
        for _ in range(trace.MAX_RECORDS + 10):
            add()
        assert len(entries) == trace.MAX_RECORDS
        assert [e["seq"] for e in entries] == list(range(1, trace.MAX_RECORDS + 1))


@pytest.mark.parametrize(("agent", "decision"), [
    ("analysis", "ok"), ("analysis", "failed"), ("bull", "ok"), ("bull", "failed"),
    ("bear", "ok"), ("bear", "failed"), ("judge", "ok"), ("judge", "failed"),
    ("critic", "ok"), ("critic", "warning"), ("critic", "revise_requested"),
    ("revise", "adopted"), ("revise", "kept_original"), ("revise", "failed"),
    ("strategy", "ok"), ("strategy", "failed"),
])
def test_contract_role_decision_combinations(agent, decision):
    with trace.collect() as entries:
        add(agent, decision)
    assert len(entries) == 1
    assert (entries[0]["agent"], entries[0]["decision"]) == (agent, decision)


@pytest.mark.parametrize("kwargs", [
    {"agent": "unknown"}, {"decision": "warning"}, {"decision": []},
    {"summary": None}, {"elapsed_ms": -1}, {"elapsed_ms": True}, {"elapsed_ms": 2.5},
    {"usage": {"tokens_in": -1}}, {"usage": {"tokens_out": True}}, {"usage": object()},
])
def test_invalid_record_is_omitted_without_leaking_input(caplog, kwargs):
    with trace.collect() as entries:
        add(**kwargs)
        add()
    assert len(entries) == 1
    assert entries[0]["seq"] == 1
    assert "Agent execution trace record was omitted" in caplog.text


def test_storage_failure_and_logging_failure_are_harmless(monkeypatch):
    class BrokenList(list):
        def append(self, value):
            raise RuntimeError("DO_NOT_LOG_fixture_provider_response")

    def broken_log(*args, **kwargs):
        raise RuntimeError("fixture logger failure")

    with trace.collect() as entries:
        collector = trace._current.get()
        collector.entries = BrokenList()
        monkeypatch.setattr(trace.logger, "warning", broken_log)
        add(summary="DO_NOT_LOG_fixture_prompt")
        assert entries == []
        assert trace.snapshot() == []


def test_storage_failure_log_does_not_include_exception_or_summary(caplog):
    class BrokenList(list):
        def append(self, value):
            raise RuntimeError("SECRET_FIXTURE_response")

    with trace.collect():
        trace._current.get().entries = BrokenList()
        add(summary="SECRET_FIXTURE_prompt")
    assert "SECRET_FIXTURE" not in caplog.text


def test_snapshot_failure_returns_empty_without_leaking_values(monkeypatch, caplog):
    def broken_copy(value):
        raise RuntimeError("SECRET_FIXTURE_copy")

    with trace.collect():
        add()
        monkeypatch.setattr(trace, "deepcopy", broken_copy)
        assert trace.snapshot() == []
    assert "SECRET_FIXTURE" not in caplog.text
