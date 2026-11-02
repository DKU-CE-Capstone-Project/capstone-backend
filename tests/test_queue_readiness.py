import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from nats.js.api import RetentionPolicy, StreamConfig

from app.config import settings
from app.main import app
from app.v2 import queue_health


def client(monkeypatch, config=None, error=None):
    probe = AsyncMock(return_value=SimpleNamespace(config=config or StreamConfig(
        subjects=['reports.generate.v2'], retention=RetentionPolicy.WORK_QUEUE, max_age=86400)))
    if error:
        probe.side_effect = error
    nc = SimpleNamespace(connect=AsyncMock(), close=AsyncMock(),
                         jetstream=Mock(return_value=SimpleNamespace(stream_info=probe)))
    monkeypatch.setattr(queue_health, 'Client', lambda: nc)
    return nc, probe


async def test_probe_only_reads_stream_and_closes_connection(monkeypatch):
    nc, probe = client(monkeypatch)
    assert await queue_health.ping()
    probe.assert_awaited_once_with('reports_v2')
    nc.connect.assert_awaited_once_with(settings.nats_url, connect_timeout=1,
                                      allow_reconnect=False, max_reconnect_attempts=0,
                                      error_cb=queue_health._ignore_error)
    nc.close.assert_awaited_once()


@pytest.mark.parametrize('config', [
    StreamConfig(subjects=['wrong'], retention=RetentionPolicy.WORK_QUEUE, max_age=86400),
    StreamConfig(subjects=['reports.generate.v2'], retention=RetentionPolicy.LIMITS, max_age=86400),
    StreamConfig(subjects=['reports.generate.v2'], retention=RetentionPolicy.WORK_QUEUE, max_age=0),
    StreamConfig(subjects=['reports.generate.v2'], retention=RetentionPolicy.WORK_QUEUE, max_age=86401),
])
async def test_probe_rejects_incompatible_stream_without_mutation(monkeypatch, config):
    nc, _ = client(monkeypatch, config)
    assert not await queue_health.ping()
    nc.close.assert_awaited_once()


async def test_probe_hides_dependency_error_and_closes(monkeypatch):
    nc, _ = client(monkeypatch, error=RuntimeError('unit-test-private-uri'))
    assert not await queue_health.ping()
    nc.close.assert_awaited_once()


async def test_probe_has_bounded_connect_and_cancellation_cleanup(monkeypatch):
    nc, _ = client(monkeypatch)
    async def blocked(*a, **kw):
        await asyncio.Event().wait()
    nc.connect.side_effect = blocked
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(queue_health.ping(), 0.02)
    nc.close.assert_awaited_once()


async def test_internal_probe_deadline_returns_false(monkeypatch):
    nc, _ = client(monkeypatch)
    async def blocked(*a, **kw):
        await asyncio.Event().wait()
    nc.connect.side_effect = blocked
    assert not await asyncio.wait_for(queue_health.ping(), 3)
    nc.close.assert_awaited_once()


@pytest.mark.parametrize('required,available,status,value', [
    (False, False, 200, 'off'), (True, True, 200, 'on'), (True, False, 503, 'unavailable'),
])
def test_ready_queue_policy(monkeypatch, required, available, status, value):
    monkeypatch.setattr(settings, 'nats_required', required)
    probe = AsyncMock(return_value=available)
    monkeypatch.setattr(queue_health, 'ping', probe)
    response = TestClient(app).get('/ready')
    assert response.status_code == status
    assert response.json()['nats'] == value
    assert probe.await_count == int(required)
