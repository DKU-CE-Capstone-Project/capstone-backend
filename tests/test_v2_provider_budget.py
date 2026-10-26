"""Report workload-credit policy; no provider requests or production services."""
import asyncio
import copy
import json
import os
import socket
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest

from app.config import settings
from app.v2 import provider_budget as budget
from app.v2 import state

_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def isolated_budget(monkeypatch):
    monkeypatch.setattr(settings, "v2_state_memory", True)
    monkeypatch.setattr(settings, "session_store_required", False)
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1000)
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 32)
    state._memory.clear()
    yield
    state._memory.clear()


async def charged(job_id, units=1):
    with budget.job_budget(job_id):
        await budget.reserve("body", units)


async def test_absent_job_context_does_not_contact_budget_store(monkeypatch):
    monkeypatch.setattr(settings, "v2_state_memory", False)

    def forbidden():
        raise AssertionError("Legacy helper must not contact the budget store")

    monkeypatch.setattr(state, "_client", forbidden)
    await budget.reserve("body")
    assert state._memory == {}


async def test_concurrent_jobs_share_daily_budget_and_denials_do_not_create_ledgers(monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 7)
    results = await asyncio.gather(*(charged(str(i)) for i in range(30)), return_exceptions=True)
    assert sum(result is None for result in results) == 7
    errors = [result for result in results if isinstance(result, budget.ProviderBudgetError)]
    assert len(errors) == 23
    assert all(error.code == "DAILY_BUDGET_EXCEEDED" and error.retryable for error in errors)
    assert all(1 <= error.details["retry_after_seconds"] <= 86400 for error in errors)
    assert state._memory[budget._DAILY_KEY][1]["units"] == 7
    assert len([key for key in state._memory if key.startswith(budget._JOB_PREFIX)]) == 7


async def test_job_budget_rejects_without_charging_daily_or_extending_ttl(monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 2)
    await charged("same-job", 2)
    before = copy.deepcopy(state._memory)
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("same-job")
    assert (error.value.status, error.value.code, error.value.retryable) == (429, "JOB_BUDGET_EXCEEDED", False)
    assert state._memory == before


async def test_daily_denial_does_not_charge_job_or_extend_ttl(monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 2)
    await charged("first", 2)
    before = copy.deepcopy(state._memory)
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("other")
    assert error.value.code == "DAILY_BUDGET_EXCEEDED"
    assert state._memory == before


async def test_kst_midnight_resets_daily_but_not_same_job_budget(monkeypatch):
    now = datetime(2026, 10, 26, 14, 59, 59, tzinfo=UTC).timestamp() + 0.5
    monkeypatch.setattr(budget.time, "time", lambda: now)
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 1)
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 2)
    await charged("same-job")
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("other")
    assert error.value.details == {"retry_after_seconds": 1}
    first_day = state._memory[budget._DAILY_KEY][1]["day"]
    now += 1
    await charged("same-job")
    assert state._memory[budget._DAILY_KEY][1] == {"day": first_day + 1, "units": 1}
    assert state._memory[budget._job_key("same-job")][1]["units"] == 2
    with pytest.raises(budget.ProviderBudgetError) as exhausted:
        await charged("same-job")
    assert exhausted.value.code == "JOB_BUDGET_EXCEEDED"
    assert state._memory[budget._DAILY_KEY][1]["units"] == 1


async def test_context_nesting_and_exception_reset_do_not_rebind_another_job():
    with budget.job_budget("outer"):
        await budget.reserve("analysis")
        with pytest.raises(RuntimeError), budget.job_budget("inner"):
            await budget.reserve("critic")
            raise RuntimeError("fixture failure")
        await budget.reserve("strategy")
    before = copy.deepcopy(state._memory)
    await budget.reserve("body")
    assert state._memory == before
    assert state._memory[budget._job_key("outer")][1]["units"] == 2
    assert state._memory[budget._job_key("inner")][1]["units"] == 1


async def test_failed_or_cancelled_work_does_not_refund_reservation():
    with pytest.raises(asyncio.CancelledError), budget.job_budget("cancelled"):
        await budget.reserve("body")
        raise asyncio.CancelledError()
    assert state._memory[budget._job_key("cancelled")][1]["units"] == 1
    assert state._memory[budget._DAILY_KEY][1]["units"] == 1


@pytest.mark.parametrize("units", [0, -1, True, 1.5])
async def test_invalid_reservation_cannot_reduce_or_bypass_ledger(units):
    with budget.job_budget("job"), pytest.raises(ValueError):
        await budget.reserve("body", units)
    assert state._memory == {}


async def test_unknown_operations_cannot_create_unmetered_alternative_bucket():
    with budget.job_budget("job"), pytest.raises(ValueError):
        await budget.reserve("caller-controlled-scope")
    assert state._memory == {}


async def test_job_retention_outlives_resource_ttl_and_processing_deadline(monkeypatch):
    now = 1_800_000_000.0
    monkeypatch.setattr(budget.time, "time", lambda: now)
    await charged("job")
    expires = state._memory[budget._job_key("job")][0]
    assert expires >= now + settings.v2_job_ttl_seconds + settings.v2_job_timeout_seconds


class FakeRedis:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.args = None

    async def eval(self, *args):
        self.args = args
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize("result, expected", [([0, 0], None), ([1, 15], "DAILY_BUDGET_EXCEEDED"),
                                            ([2, 0], "JOB_BUDGET_EXCEEDED"), ([9, 0], "BUDGET_UNAVAILABLE"),
                                            ([1, 0], "BUDGET_UNAVAILABLE"), (None, "BUDGET_UNAVAILABLE")])
async def test_redis_protocol_uses_shared_atomic_reservation(monkeypatch, result, expected):
    monkeypatch.setattr(settings, "v2_state_memory", False)
    fake = FakeRedis(result)
    monkeypatch.setattr(state, "_client", lambda: fake)
    if expected:
        with pytest.raises(budget.ProviderBudgetError) as error:
            await charged("job", 2)
        assert error.value.code == expected
    else:
        await charged("job", 2)
    assert fake.args[1:] == (2, budget._DAILY_KEY, budget._job_key("job"), 2, 1000, 32, budget._job_ttl())
    assert "redis.call('TIME')" in fake.args[0]
    assert state._memory == {}


async def test_required_storage_cannot_fall_back_to_local_budget(monkeypatch):
    monkeypatch.setattr(settings, "session_store_required", True)
    fake = FakeRedis(error=ConnectionError("fixture-private-connection-string"))
    monkeypatch.setattr(state, "_client", lambda: fake)
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("job")
    assert (error.value.status, error.value.code) == (503, "BUDGET_UNAVAILABLE")
    assert "fixture-private" not in str(error.value.envelope())
    assert state._memory == {}


@pytest.fixture
async def real_budget_redis(monkeypatch):
    url = os.environ.get("ECONMIND_V2_TEST_REDIS_URL")
    if not url:
        pytest.skip("Explicit loopback Redis required")
    parsed = urlsplit(url)
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.scheme != "redis" or parsed.username or parsed.password:
        pytest.fail("Only an isolated loopback Redis without credentials is accepted")
    port = parsed.port or 6379

    def local_connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1") or address[1] != port:
            raise AssertionError("Only isolated loopback Redis may be contacted")
        return _CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(settings, "v2_state_memory", False)
    monkeypatch.setattr(settings, "redis_url", url)
    prefix = "v2:budget-fixture:" + uuid.uuid4().hex
    monkeypatch.setattr(budget, "_DAILY_KEY", prefix + ":daily")
    monkeypatch.setattr(budget, "_JOB_PREFIX", prefix + ":job:")
    client = state._client()
    await client.ping()
    try:
        yield client
    finally:
        keys = [key async for key in client.scan_iter(match=prefix + ":*")]
        if keys:
            await client.delete(*keys)
        current = state._clients.pop(asyncio.get_running_loop(), None)
        if current:
            await current[1].aclose()


async def test_real_redis_concurrent_requests_cannot_overspend(real_budget_redis, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_daily_budget_units", 7)
    results = await asyncio.gather(*(charged(str(i)) for i in range(30)), return_exceptions=True)
    assert sum(result is None for result in results) == 7
    errors = [result for result in results if isinstance(result, budget.ProviderBudgetError)]
    assert len(errors) == 23 and all(error.code == "DAILY_BUDGET_EXCEEDED" for error in errors)
    assert json.loads(await real_budget_redis.get(budget._DAILY_KEY))["units"] == 7
    keys = [key async for key in real_budget_redis.scan_iter(match=budget._JOB_PREFIX + "*")]
    assert len(keys) == 7


async def test_real_redis_job_ledger_survives_daily_reset_without_partial_charge(real_budget_redis, monkeypatch):
    monkeypatch.setattr(settings, "v2_provider_job_budget_units", 3)
    await charged("job", 2)
    now = (await real_budget_redis.time())[0]
    day, _ = budget._window(now)
    await real_budget_redis.set(budget._DAILY_KEY, json.dumps({"day": day - 1, "units": 1000}), ex=60)
    await charged("job")
    before = await real_budget_redis.get(budget._DAILY_KEY)
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("job")
    assert error.value.code == "JOB_BUDGET_EXCEEDED"
    assert await real_budget_redis.get(budget._DAILY_KEY) == before
    assert json.loads(before)["units"] == 1
    assert int(await real_budget_redis.get(budget._job_key("job"))) == 3
    assert await real_budget_redis.ttl(budget._job_key("job")) >= settings.v2_job_ttl_seconds


async def test_real_redis_success_repairs_missing_expiry_and_denial_preserves_ledger(real_budget_redis):
    now = (await real_budget_redis.time())[0]
    day, _ = budget._window(now)
    await real_budget_redis.set(budget._DAILY_KEY, json.dumps({"day": day, "units": 1}))
    await real_budget_redis.set(budget._job_key("job"), "31")
    assert await real_budget_redis.ttl(budget._DAILY_KEY) == -1
    await charged("job")
    assert await real_budget_redis.ttl(budget._DAILY_KEY) > 0
    assert await real_budget_redis.ttl(budget._job_key("job")) > 0
    daily = await real_budget_redis.get(budget._DAILY_KEY)
    with pytest.raises(budget.ProviderBudgetError) as error:
        await charged("job")
    assert error.value.code == "JOB_BUDGET_EXCEEDED"
    assert await real_budget_redis.get(budget._DAILY_KEY) == daily
    assert int(await real_budget_redis.get(budget._job_key("job"))) == 32
