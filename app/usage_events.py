"""Provider usage events: one structured log line per provider call.

The admin usage container reads these lines; the format follows the
``usage_event`` contract in econmind-docs 23 §10.3. Events never carry user
identity, prompts, article bodies, URLs or credentials. ``trace_id`` is the v2
report job ID when the call runs inside a job, otherwise null.

Recording must never break the provider call it describes, so every failure
here is swallowed with a single warning.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from app.config import settings

logger = logging.getLogger("econmind.usage")
# The API configures the "econmind" logger in app.main; the v2 worker process
# never imports it, and INFO lines would otherwise be dropped there.
logger.setLevel(logging.INFO)


def _ensure_handler() -> None:
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
        logger.addHandler(handler)


Service = Literal["gemini", "claude", "decision", "diffbot", "naver"]


def _current_trace_id() -> str | None:
    # Imported lazily: provider_budget pulls in v2 state, which legacy paths
    # must not need just to log a provider call.
    from app.v2 import provider_budget

    return provider_budget.current_job_id()


def _estimate_cost(service: str, tokens_in: int | None, tokens_out: int | None, calls: int) -> float | None:
    """Price table values come from settings; 0 means "unknown", not "free"."""
    if service == "gemini":
        price_in = settings.usage_price_gemini_input_per_mtok
        price_out = settings.usage_price_gemini_output_per_mtok
        if not (price_in or price_out) or tokens_in is None or tokens_out is None:
            return None
        return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 8)
    per_call = {"diffbot": settings.usage_price_diffbot_per_call,
                "decision": settings.usage_price_decision_per_call}.get(service, 0.0)
    if not per_call:
        return None
    return round(calls * per_call, 8)


def emit(
    service: Service,
    operation: str | None,
    *,
    started: float,
    status: str,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    calls: int = 1,
    key_slot: int | None = None,
    trace_id: str | None = None,
) -> None:
    """Record one provider call. ``started`` is a ``time.monotonic()`` value."""
    if not settings.usage_events_enabled:
        return
    try:
        event: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_id": trace_id if trace_id is not None else _current_trace_id(),
            "service": service,
            "operation": operation or "unspecified",
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "calls": calls,
            "latency_ms": max(0, int((time.monotonic() - started) * 1000)),
            "status": status,
            "est_cost_usd": _estimate_cost(service, tokens_in, tokens_out, calls),
        }
        if key_slot is not None:
            event["key_slot"] = key_slot
        _ensure_handler()
        logger.info("[usage] %s", json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    except Exception as exc:  # noqa: BLE001 — usage recording is best effort
        logger.warning("[usage] event dropped: %s", type(exc).__name__)
