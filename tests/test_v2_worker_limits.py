"""Shared durable limits and recoverable delivery, with no provider requests."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nats.js.api import AckPolicy, ConsumerConfig, RetentionPolicy, StreamConfig
from nats.js.errors import NotFoundError

from app.config import settings
from app.v2 import report_worker


class JetStream:
    def __init__(self, config=None):
        self.config = config
        self.updates = []

    async def consumer_info(self, stream, durable):
        assert (stream, durable) == (report_worker.STREAM, report_worker.DURABLE)
        if self.config is None:
            raise NotFoundError()
        return SimpleNamespace(config=deepcopy(self.config))

    async def add_consumer(self, stream, *, config):
        assert stream == report_worker.STREAM
        self.updates.append(deepcopy(config))
        self.config = deepcopy(config)


def config(**changes):
    return ConsumerConfig(name=report_worker.DURABLE, durable_name=report_worker.DURABLE,
                          filter_subject=report_worker.SUBJECT, ack_policy=AckPolicy.EXPLICIT,
                          ack_wait=60, max_deliver=3, max_ack_pending=1, **changes)


@pytest.mark.parametrize("changes", [{"max_age": 0}, {"max_age": 86401}, {"retention": RetentionPolicy.LIMITS}])
async def test_stream_without_effective_transport_expiry_is_rejected(changes):
    values = {"subjects": [report_worker.SUBJECT], "max_age": 86400, "retention": RetentionPolicy.WORK_QUEUE, **changes}
    js = SimpleNamespace(stream_info=AsyncMock(return_value=SimpleNamespace(config=StreamConfig(**values))))
    with pytest.raises(ValueError, match="bounded"):
        await report_worker.ensure_stream(js)


async def test_new_short_lived_stream_has_compatible_duplicate_window(monkeypatch):
    monkeypatch.setattr(settings, "v2_job_ttl_seconds", 60)
    added = []

    async def add_stream(*, config):
        added.append(config)
        return SimpleNamespace(config=config)

    js = SimpleNamespace(stream_info=AsyncMock(side_effect=NotFoundError()), add_stream=add_stream)
    await report_worker.ensure_stream(js)
    assert added[0].max_age == 60 and added[0].duplicate_window == 60


async def test_new_consumer_has_shared_fleet_cap_and_recoverable_transport(monkeypatch):
    monkeypatch.setattr(settings, "v2_worker_max_concurrency", 2)
    js = JetStream()
    await report_worker.ensure_consumer(js)
    assert js.config.max_ack_pending == 2
    assert js.config.max_deliver == -1 and js.config.ack_wait == 60
    assert js.config.filter_subject == report_worker.SUBJECT


async def test_existing_durable_is_updated_without_resetting_other_properties(monkeypatch):
    monkeypatch.setattr(settings, "v2_worker_max_concurrency", 1)
    original = config(max_waiting=128, description="existing durable")
    original.max_ack_pending = 50
    original.ack_wait = 10
    js = JetStream(original)
    await report_worker.ensure_consumer(js)
    assert js.config.max_ack_pending == 1 and js.config.max_deliver == -1
    assert js.config.max_waiting == 128 and js.config.description == "existing durable"
    assert len(js.updates) == 1
    await report_worker.ensure_consumer(js)
    assert len(js.updates) == 1


@pytest.mark.parametrize("changes", [
    {"filter_subject": "other"}, {"deliver_subject": "push"},
    {"ack_policy": AckPolicy.NONE}, {"filter_subjects": [report_worker.SUBJECT, "other"]},
])
async def test_conflicting_durable_fails_without_recreation(changes):
    original = config()
    for key, value in changes.items():
        setattr(original, key, value)
    js = JetStream(original)
    with pytest.raises(ValueError, match="conflicting"):
        await report_worker.ensure_consumer(js)
    assert not js.updates


async def test_server_limits_are_verified_after_update():
    js = JetStream(config())
    js.add_consumer = AsyncMock()  # Server/permission did not apply requested limits.
    with pytest.raises(ValueError, match="not applied"):
        await report_worker.ensure_consumer(js)


def message(data=None):
    return SimpleNamespace(data=data if data is not None else json.dumps({
        "job_type": "report_v2", "session_id": "session", "job_id": "job",
    }).encode(), ack=AsyncMock(), nak=AsyncMock(), term=AsyncMock(), in_progress=AsyncMock())


async def test_busy_duplicate_waits_on_same_delivery_then_acknowledges(monkeypatch):
    process = AsyncMock(side_effect=["busy", "busy", "done"])
    sleep = AsyncMock()
    monkeypatch.setattr(report_worker, "process_job", process)
    monkeypatch.setattr(report_worker.asyncio, "sleep", sleep)
    msg = message()
    await report_worker.consume_message(msg)
    assert process.await_count == 3 and msg.in_progress.await_count == 2
    assert sleep.await_count == 2 and all(call.args == (5,) for call in sleep.await_args_list)
    msg.ack.assert_awaited_once()
    msg.nak.assert_not_awaited()


async def test_dependency_retry_never_acknowledges_or_logs_exception_text(monkeypatch, capsys):
    monkeypatch.setattr(report_worker, "process_job", AsyncMock(side_effect=RuntimeError("secret-token")))
    msg = message()
    await report_worker.consume_message(msg)
    msg.nak.assert_awaited_once_with(delay=20)
    msg.ack.assert_not_awaited()
    assert "secret-token" not in capsys.readouterr().out


@pytest.mark.parametrize("data", [b"[1]", b"bad-json", b"\xff", b'{"job_type":"report_v2","job_id":true,"session_id":"a"}'])
async def test_invalid_queue_envelopes_are_terminated_without_processing(data, monkeypatch):
    process = AsyncMock()
    monkeypatch.setattr(report_worker, "process_job", process)
    msg = message(data)
    await report_worker.consume_message(msg)
    msg.term.assert_awaited_once()
    process.assert_not_awaited()


async def test_worker_shutdown_keeps_delivery_recoverable(monkeypatch):
    monkeypatch.setattr(report_worker, "process_job", AsyncMock(side_effect=asyncio.CancelledError()))
    msg = message()
    with pytest.raises(asyncio.CancelledError):
        await report_worker.consume_message(msg)
    msg.ack.assert_not_awaited()
    msg.nak.assert_not_awaited()
