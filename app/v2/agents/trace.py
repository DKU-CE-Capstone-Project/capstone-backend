"""Bounded, best-effort execution summaries scoped to one report or strategy.

Only application-authored summaries belong here: no prompts, article excerpts,
provider responses, URLs, or credentials. Child asyncio tasks share the active
collector so their completion order determines ``seq``. Separate ``collect``
scopes, including nested scopes, never share their record lists.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

MAX_RECORDS = 128
MAX_SUMMARY_LENGTH = 120
_DECISIONS = {
    "analysis": frozenset({"ok", "failed"}),
    "bull": frozenset({"ok", "failed"}),
    "bear": frozenset({"ok", "failed"}),
    "judge": frozenset({"ok", "failed"}),
    "critic": frozenset({"ok", "warning", "revise_requested"}),
    "revise": frozenset({"adopted", "kept_original", "failed"}),
    "strategy": frozenset({"ok", "failed"}),
}


@dataclass
class _Collector:
    entries: list[dict] = field(default_factory=list)
    active: bool = True
    lock: Lock = field(default_factory=Lock)


_current: ContextVar[_Collector | None] = ContextVar("v2_agent_trace", default=None)


def _warn() -> None:
    # Logging itself is optional. Never put the rejected value or exception in
    # the log: they can contain a prompt or an upstream provider response.
    try:
        logger.warning("Agent execution trace record was omitted")
    except Exception:  # noqa: BLE001, S110 — a failed logger cannot log itself
        pass


@contextmanager
def collect() -> Iterator[list[dict]]:
    """Collect one execution; the yielded list remains readable after exit.

    Exiting also closes inherited child contexts. A late child cannot mutate a
    report's trace after its execution scope has already completed.
    """
    collector = _Collector()
    token = _current.set(collector)
    try:
        yield collector.entries
    finally:
        with collector.lock:
            collector.active = False
        _current.reset(token)


def record(agent, decision, summary, *, elapsed_ms, usage=None) -> None:
    """Append a valid bounded record, swallowing ordinary recording failures.

    ``usage`` may contain provider metadata, but only explicitly supplied token
    counts are retained. Missing counts remain absent rather than becoming zero.
    Cancellation and other ``BaseException`` subclasses are never swallowed.
    """
    collector = _current.get()
    if collector is None:
        return
    try:
        if not isinstance(agent, str) or decision not in _DECISIONS.get(agent, ()):
            raise ValueError("Invalid trace role or decision")
        if not isinstance(summary, str):
            raise TypeError("Expected a summary string")
        if type(elapsed_ms) is not int or elapsed_ms < 0:
            raise ValueError("Expected nonnegative integer milliseconds")
        entry = {
            "agent": agent,
            "elapsed_ms": elapsed_ms,
            "summary": summary[:MAX_SUMMARY_LENGTH],
            "decision": decision,
        }
        if usage is not None:
            if not isinstance(usage, Mapping):
                raise TypeError("Expected a usage mapping")
            for key in ("tokens_in", "tokens_out"):
                value = usage.get(key)
                if value is not None:
                    if type(value) is not int or value < 0:
                        raise ValueError("Expected nonnegative integer tokens")
                    entry[key] = value
        with collector.lock:
            if not collector.active:
                return
            if len(collector.entries) >= MAX_RECORDS:
                return
            collector.entries.append({"seq": len(collector.entries) + 1, **entry})
    except Exception:  # noqa: BLE001 — optional execution tracing is best effort
        _warn()


def snapshot() -> list[dict]:
    """Return an independent copy of the current trace, or [] outside a scope."""
    collector = _current.get()
    if collector is None:
        return []
    try:
        with collector.lock:
            return deepcopy(collector.entries)
    except Exception:  # noqa: BLE001 — optional execution tracing is best effort
        _warn()
        return []
