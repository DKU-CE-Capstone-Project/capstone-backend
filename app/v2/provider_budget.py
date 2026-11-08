"""Shared v2 report workload credits, not provider bills or HTTP-attempt counts.

Each explicitly instrumented body/RAG/generation operation consumes credits
before it starts. Unknown outcomes are never refunded. The daily ledger uses
Redis time and a Korean calendar day; the job ledger survives day changes and
worker restarts. Direct legacy helpers outside a job context remain unchanged.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from app.config import settings
from app.v2 import state
from app.v2.errors import V2Error

_OPERATIONS = frozenset({"body", "rag", "analysis", "critic", "strategy"})
_DAILY_KEY = "v2:provider-budget:daily"
_JOB_PREFIX = "v2:provider-budget:job:"
_KST_OFFSET = 9 * 3600
_DAY_SECONDS = 86400
_job_id: ContextVar[str | None] = ContextVar("v2_provider_budget_job", default=None)
_memory_lock = threading.Lock()
_RESERVE = """
local now = tonumber(redis.call('TIME')[1])
local day = math.floor((now + 32400) / 86400)
local reset = (day + 1) * 86400 - 32400
local daily_units = 0
local raw = redis.call('GET', KEYS[1])
if raw then
  local daily = cjson.decode(raw)
  if type(daily) ~= 'table' or type(daily.day) ~= 'number'
     or type(daily.units) ~= 'number' or daily.day % 1 ~= 0
     or daily.units < 0 or daily.units % 1 ~= 0 or daily.day > day then
    return redis.error_reply('Invalid budget ledger')
  end
  if daily.day == day then daily_units = daily.units end
end
local job_units = tonumber(redis.call('GET', KEYS[2]) or '0')
if not job_units or job_units < 0 or job_units % 1 ~= 0 then
  return redis.error_reply('Invalid budget ledger')
end
local units = tonumber(ARGV[1])
if job_units + units > tonumber(ARGV[3]) then return {2, 0} end
if daily_units + units > tonumber(ARGV[2]) then return {1, reset - now} end
redis.call('SET', KEYS[1], cjson.encode({day=day, units=daily_units + units}),
           'EX', reset - now + 86400)
redis.call('SET', KEYS[2], tostring(job_units + units), 'EX', ARGV[4])
return {0, 0}
"""


class ProviderBudgetError(V2Error):
    """A policy denial must not turn into an ordinary provider failure."""


@contextmanager
def job_budget(job_id: str):
    """Bind the durable job identity; task-local state always resets on exit."""
    if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
        raise ValueError("Expected a bounded job identity")
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)


def _job_key(job_id: str) -> str:
    # Internal IDs need no plaintext representation in an aggregate ledger.
    return _JOB_PREFIX + hashlib.sha256(job_id.encode()).hexdigest()


def _window(now: float) -> tuple[int, int]:
    day = math.floor((now + _KST_OFFSET) / _DAY_SECONDS)
    reset = (day + 1) * _DAY_SECONDS - _KST_OFFSET
    return day, max(1, math.ceil(reset - now))


def _job_ttl() -> int:
    # A short job-resource TTL must not reset an in-flight/recoverable budget.
    return settings.v2_job_ttl_seconds + settings.v2_job_timeout_seconds + 60


def _memory_reserve(job_key: str, units: int) -> tuple[int, int]:
    now = time.time()
    day, retry_after = _window(now)
    with _memory_lock:
        daily = state._memory.get(_DAILY_KEY)
        daily_units = daily[1]["units"] if daily and daily[0] > now and daily[1]["day"] == day else 0
        job = state._memory.get(job_key)
        job_units = job[1]["units"] if job and job[0] > now else 0
        if job_units + units > settings.v2_provider_job_budget_units:
            return 2, 0
        if daily_units + units > settings.v2_provider_daily_budget_units:
            return 1, retry_after
        state._memory[_DAILY_KEY] = (now + retry_after + _DAY_SECONDS,
                                     {"day": day, "units": daily_units + units})
        state._memory[job_key] = (now + _job_ttl(), {"units": job_units + units})
    return 0, 0


def _unavailable() -> ProviderBudgetError:
    return ProviderBudgetError(503, "BUDGET_UNAVAILABLE", "생성 예산을 확인할 수 없습니다.", True)


async def reserve(operation: str, units: int = 1) -> None:
    """Atomically charge shared daily and durable job ledgers before work.

    This is intentionally a no-op for helpers invoked outside ``job_budget``;
    it does not claim to govern v1, maps, SDK retries or account-wide spend.
    """
    job_id = _job_id.get()
    if job_id is None:
        return
    if operation not in _OPERATIONS:
        raise ValueError("Unknown report budget operation")
    if type(units) is not int or units < 1:
        raise ValueError("Budget units must be a positive integer")
    job_key = _job_key(job_id)
    if state.memory_enabled():
        status, retry_after = _memory_reserve(job_key, units)
    else:
        try:
            result = await state._client().eval(
                _RESERVE, 2, _DAILY_KEY, job_key, units,
                settings.v2_provider_daily_budget_units,
                settings.v2_provider_job_budget_units, _job_ttl(),
            )
            if (not isinstance(result, (list, tuple)) or len(result) != 2
                    or any(type(value) is not int for value in result)
                    or result[0] not in (0, 1, 2)
                    or (result[0] == 1 and not 1 <= result[1] <= _DAY_SECONDS)):
                raise ValueError("Invalid budget response")
            status, retry_after = result
        except Exception:  # noqa: BLE001 — all ledger/protocol failures must deny paid work
            raise _unavailable() from None
    if status == 2:
        raise ProviderBudgetError(429, "JOB_BUDGET_EXCEEDED", "작업의 생성 예산을 초과했습니다.", False)
    if status == 1:
        raise ProviderBudgetError(429, "DAILY_BUDGET_EXCEEDED", "오늘의 생성 예산을 초과했습니다.", True,
                                  {"retry_after_seconds": retry_after})
