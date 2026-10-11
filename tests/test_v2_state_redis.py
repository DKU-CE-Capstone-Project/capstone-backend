"""Opt-in real Redis tests. Only loopback URLs and random sessions are allowed."""
import asyncio
import os
import socket
import time
import uuid
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from app import session as base_session
from app import usage_limits
from app.config import settings
from app.v2 import state
from app.v2.errors import V2Error

_CONNECT = socket.socket.connect


@pytest_asyncio.fixture
async def real_redis(monkeypatch):
    url = os.environ.get('ECONMIND_V2_TEST_REDIS_URL')
    if not url:
        pytest.skip('Explicit loopback Redis required')
    parsed = urlsplit(url)
    if parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.scheme != 'redis':
        pytest.fail('Only loopback Redis URLs are accepted')
    port = parsed.port or 6379

    def local_connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1') or address[1] != port:
            raise AssertionError('Only the isolated local Redis may be contacted')
        return _CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, 'connect', local_connect)
    monkeypatch.setattr(settings, 'session_store_required', True)
    monkeypatch.setattr(settings, 'v2_state_memory', False)
    monkeypatch.setattr(settings, 'redis_url', url)
    monkeypatch.setattr(base_session, '_redis_disabled', False)
    monkeypatch.setattr(base_session, '_redis', None)
    monkeypatch.setattr(usage_limits, '_client', None)
    fixture_id = uuid.uuid4().hex

    async def isolated_quota(owner, action, **kwargs):
        return await usage_limits.check_quota(owner, f'{action}_{fixture_id}', **kwargs)

    monkeypatch.setattr(base_session, 'check_quota', isolated_quota)
    sid = await base_session.issue('127.0.0.1')
    await state.session(sid)
    try:
        yield sid
    finally:
        await state.clear_session(sid)
        await base_session.clear(sid)
        async for key in state._client().scan_iter(match=f'v2:*{sid}*'):
            await state._client().delete(key)
        async for key in state._client().scan_iter(match=f'quota:*{fixture_id}*'):
            await state._client().delete(key)
        if base_session._redis:
            await base_session._redis.aclose()
        if usage_limits._client:
            await usage_limits._client.aclose()
        current = state._clients.pop(asyncio.get_running_loop(), None)
        if current:
            await current[1].aclose()


@pytest.mark.asyncio
async def test_real_cas_reserves_one_revision_and_checks_parent(real_redis):
    sid = real_redis
    key = state.key(sid, 'map', 'cas-fixture')
    await state.put(key, {'revision': 1}, 120)
    results = await asyncio.gather(state.cas(key, 1, {'revision': 2, 'chosen': 'first'}, 120),
                                   state.cas(key, 1, {'revision': 2, 'chosen': 'second'}, 120),
                                   return_exceptions=True)
    assert sum(isinstance(r, dict) for r in results) == 1
    error = next(r for r in results if isinstance(r, V2Error))
    assert error.status == 409
    await base_session.clear(sid)
    with pytest.raises(V2Error) as failed:
        await state.cas(key, 2, {'revision': 3}, 120)
    assert failed.value.code == 'SESSION_EXPIRED'
    assert (await state.get(key))['revision'] == 2


@pytest.mark.asyncio
async def test_real_tombstones_and_result_access_are_session_owned(real_redis):
    sid = real_redis
    await state.register_resource(sid, 'map', 'expired', state.iso(time.time() - 1))
    with pytest.raises(V2Error) as failed:
        await state.resource(sid, 'map', 'expired')
    assert failed.value.status == 410
    with pytest.raises(V2Error) as failed:
        await state.resource(sid, 'map', 'unknown')
    assert failed.value.status == 404
    await state.grant_result(sid, 'fixture-result', 'report')
    await state.authorize_result(sid, 'fixture-result', 'report')
    with pytest.raises(V2Error) as failed:
        await state.authorize_result(sid, 'fixture-result', 'strategy')
    assert failed.value.status == 404


@pytest.mark.asyncio
async def test_polling_does_not_extend_parent_but_user_activity_does(real_redis):
    sid = real_redis
    before = await state.session(sid)
    base_ttl = await state._client().ttl(f'session:{sid}')
    await state.session(sid)
    assert (await state.session(sid))['expires_at'] == before['expires_at']
    assert await state._client().ttl(f'session:{sid}') <= base_ttl
    touched = await state.session(sid, touch=True)
    assert state.epoch(touched['expires_at']) >= state.epoch(before['expires_at'])


@pytest.mark.asyncio
async def test_lease_loss_never_releases_another_owners_lock(real_redis):
    sid = real_redis
    lock_key = f'v2:fixture-lock:{sid}'
    with pytest.raises(V2Error) as failed:
        async with state.mutex(lock_key, ttl=5) as lease:
            await state._client().set(lock_key, 'replacement-owner', ex=20)
            await lease.assert_owned()
    assert failed.value.code == 'LEASE_LOST'
    assert await state._client().get(lock_key) == 'replacement-owner'


@pytest.mark.asyncio
async def test_touch_cannot_resurrect_identity_deleted_during_refresh(real_redis, monkeypatch):
    sid = real_redis
    before_save, proceed = asyncio.Event(), asyncio.Event()
    original = state._save_live_session

    async def pause(sid, data, touch=False):
        if touch:
            before_save.set()
            await proceed.wait()
        return await original(sid, data, touch)

    monkeypatch.setattr(state, '_save_live_session', pause)
    refresh = asyncio.create_task(state.session(sid, touch=True))
    await before_save.wait()
    await base_session.clear(sid)
    await state.clear_session(sid)
    proceed.set()
    with pytest.raises(V2Error) as failed:
        await refresh
    assert failed.value.code == 'SESSION_EXPIRED'
    assert await base_session._existing(sid) is None
    assert await state.get(f'v2:session:{sid}') is None


@pytest.mark.asyncio
async def test_legacy_save_cannot_resurrect_revoked_cookie(real_redis, monkeypatch):
    sid = real_redis
    before_save, proceed = asyncio.Event(), asyncio.Event()
    original = base_session.save

    async def pause(sid, data, *, _create=False):
        if not _create:
            before_save.set()
            await proceed.wait()
        return await original(sid, data, _create=_create)

    monkeypatch.setattr(base_session, 'save', pause)
    pending = asyncio.create_task(base_session.expand_node(sid, 'private'))
    await before_save.wait()
    await base_session.clear(sid)
    await state.clear_session(sid)
    proceed.set()
    with pytest.raises(base_session.SessionExpired):
        await pending
    assert await base_session._existing(sid) is None
