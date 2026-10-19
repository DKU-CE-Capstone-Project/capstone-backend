"""Redis state with CAS revisions, session TTLs and token-owned leases.

The memory implementation is explicitly enabled only for isolated fixtures.
Redis failure never changes the storage backend.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from datetime import UTC, datetime
from weakref import WeakKeyDictionary

import redis.asyncio as redis

from app import session as base_session
from app.config import settings
from app.v2.errors import V2Error

_memory: dict[str, tuple[float, dict]] = {}
_locks: WeakKeyDictionary = WeakKeyDictionary()
_clients: WeakKeyDictionary = WeakKeyDictionary()
_CAS = """
if redis.call('EXISTS', KEYS[2]) == 0 then return -2 end
local old = redis.call('GET', KEYS[1])
if not old then return -1 end
local revision = tonumber(cjson.decode(old).revision)
if revision ~= tonumber(ARGV[1]) then return revision end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 0
"""
_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
_SAVE_SESSION = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if ARGV[3] == '1' then redis.call('EXPIRE', KEYS[1], ARGV[4]) end
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
return 1
"""
_DELETE_MAP = """
if redis.call('EXISTS', KEYS[1]) == 0 then return -2 end
local raw = redis.call('GET', KEYS[2])
local ttl = redis.call('PTTL', KEYS[2])
if not raw or ttl <= 0 then return -2 end
if redis.call('EXISTS', KEYS[3]) == 0 then return -1 end
local session = cjson.decode(raw)
local maps = session.resource_index and session.resource_index.map
if not maps or maps[ARGV[1]] == nil or maps[ARGV[1]] == cjson.null then return -1 end
maps[ARGV[1]] = cjson.null
redis.call('SET', KEYS[2], cjson.encode(session), 'PX', ttl)
redis.call('DEL', KEYS[3])
return 0
"""


def iso(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(epoch if epoch is not None else time.time(), UTC).isoformat()


def epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def memory_enabled() -> bool:
    return settings.v2_state_memory and not settings.session_store_required


def key(sid: str, kind: str, resource_id: str) -> str:
    return f"v2:{kind}:{sid}:{resource_id}"


def _client():
    loop = asyncio.get_running_loop()
    current = _clients.get(loop)
    if current is None or current[0] != settings.redis_url:
        current = (settings.redis_url, redis.from_url(settings.redis_url, decode_responses=True,
                                                    socket_connect_timeout=3, socket_timeout=3))
        _clients[loop] = current
    return current[1]


def unavailable() -> V2Error:
    return V2Error(503, "DEPENDENCY_UNAVAILABLE", "공유 저장소를 사용할 수 없습니다.", True)


async def get(storage_key: str) -> dict | None:
    if memory_enabled():
        entry = _memory.get(storage_key)
        if entry and entry[0] > time.time():
            return deepcopy(entry[1])
        _memory.pop(storage_key, None)
        return None
    try:
        raw = await _client().get(storage_key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        raise unavailable() from exc


async def put(storage_key: str, value: dict, ttl: float) -> None:
    ttl = max(1, int(ttl))
    if memory_enabled():
        _memory[storage_key] = (time.time() + ttl, deepcopy(value))
        return
    try:
        await _client().set(storage_key, json.dumps(value, ensure_ascii=False), ex=ttl)
    except Exception as exc:
        raise unavailable() from exc


async def delete(storage_key: str) -> None:
    if memory_enabled():
        _memory.pop(storage_key, None)
        return
    try:
        await _client().delete(storage_key)
    except Exception as exc:
        raise unavailable() from exc


async def _expire(storage_key: str, ttl: int) -> None:
    if memory_enabled():
        entry = _memory.get(storage_key)
        if entry and entry[0] > time.time():
            _memory[storage_key] = (time.time() + ttl, entry[1])
        return
    try:
        await _client().expire(storage_key, ttl)
    except Exception as exc:
        raise unavailable() from exc


class Lease:
    def __init__(self, storage_key: str, token: str, ttl: int):
        self.key, self.token, self.ttl = storage_key, token, ttl
        self.lost = False

    async def assert_owned(self) -> None:
        if self.lost:
            raise V2Error(503, "LEASE_LOST", "처리 권한이 만료되었습니다.", True)
        if not memory_enabled():
            try:
                if await _client().get(self.key) != self.token:
                    self.lost = True
            except Exception as exc:
                raise unavailable() from exc
            if self.lost:
                raise V2Error(503, "LEASE_LOST", "처리 권한이 만료되었습니다.", True)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(min(15, max(0.1, self.ttl / 4)))
            try:
                if not await _client().eval(_RENEW, 1, self.key, self.token, self.ttl):
                    self.lost = True
                    return
            except Exception:  # noqa: BLE001 -- losing Redis means losing the lease
                self.lost = True
                return


@asynccontextmanager
async def mutex(storage_key: str, ttl: int = 60):
    if memory_enabled():
        locks = _locks.setdefault(asyncio.get_running_loop(), {})
        lock = locks.setdefault(storage_key, asyncio.Lock())
        async with lock:
            yield Lease(storage_key, str(uuid.uuid4()), ttl)
        return
    token = str(uuid.uuid4())
    started = time.monotonic()
    try:
        while not await _client().set(storage_key, token, nx=True, ex=ttl):
            if time.monotonic() - started > settings.v2_job_timeout_seconds:
                raise V2Error(503, "LOCK_TIMEOUT", "처리 대기 시간이 초과되었습니다.", True)
            await asyncio.sleep(0.1)
    except V2Error:
        raise
    except Exception as exc:
        raise unavailable() from exc
    lease = Lease(storage_key, token, ttl)
    heartbeat = asyncio.create_task(lease.heartbeat())
    try:
        yield lease
        await lease.assert_owned()
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        with suppress(Exception):
            await _client().eval(_RELEASE, 1, storage_key, token)


async def cas(storage_key: str, expected_revision: int, value: dict, ttl: float) -> dict:
    sid = storage_key.split(':', 3)[2]
    if memory_enabled():
        async with mutex(f"cas:{storage_key}"):
            if await base_session._existing(sid) is None:
                raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
            old = await get(storage_key)
            if old is None:
                raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")
            if old['revision'] != expected_revision:
                raise V2Error(409, "MAP_REVISION_CONFLICT", "다른 요청에서 탐색이 변경되었습니다.",
                              details={"revision": old['revision']})
            await put(storage_key, value, ttl)
    else:
        try:
            result = await _client().eval(_CAS, 2, storage_key, f"session:{sid}", expected_revision,
                                          json.dumps(value, ensure_ascii=False), max(1, int(ttl)))
        except Exception as exc:
            raise unavailable() from exc
        if result == -2:
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        if result == -1:
            raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")
        if result:
            raise V2Error(409, "MAP_REVISION_CONFLICT", "다른 요청에서 탐색이 변경되었습니다.",
                          details={"revision": result})
    return deepcopy(value)


async def session(sid: str, touch: bool = False) -> dict:
    async with mutex(f"v2:index-lock:{sid}"):
        try:
            base = await base_session._existing(sid)
        except base_session.SessionStoreUnavailable as exc:
            raise unavailable() from exc
        if base is None:
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        storage_key = f"v2:session:{sid}"
        data = await get(storage_key)
        if data is None:
            if memory_enabled():
                expires = base_session._memory_store.get(sid, (time.time(), None))[0]
            else:
                try:
                    ttl = await _client().ttl(f"session:{sid}")
                except Exception as exc:
                    raise unavailable() from exc
                expires = time.time() + max(0, ttl)
            data = {"schema_version": 2, "session_epoch": str(uuid.uuid4()), "created_at": iso(),
                    "expires_at": iso(expires), "resource_index": {},
                    "prototype_tier": "PAID" if settings.prototype_tier == "PAID"
                    and settings.paid_demo_enabled else "FREE"}
        if epoch(data['expires_at']) <= time.time():
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        if touch:
            data['expires_at'] = iso(time.time() + settings.session_ttl_seconds)
            for kind, resources in data['resource_index'].items():
                if kind not in ('map', 'result-access'):
                    continue
                for resource_id, expires_at in resources.items():
                    if expires_at is None:  # Explicitly deleted map: retain 410 without renewing a missing key.
                        continue
                    await _expire(key(sid, kind, resource_id), settings.session_ttl_seconds)
                    resources[resource_id] = data['expires_at']
        await _save_live_session(sid, data, touch)
        return data


async def _save_live_session(sid: str, data: dict, touch: bool = False) -> None:
    """Refresh an existing identity atomically; never issue an old SID again."""
    ttl = max(1, int(epoch(data['expires_at']) - time.time()))
    if memory_enabled():
        entry = base_session._memory_store.get(sid)
        if not entry or entry[0] <= time.time():
            raise V2Error(410, 'SESSION_EXPIRED', '세션이 만료되었습니다.')
        if touch:
            base_session._memory_store[sid] = (time.time() + settings.session_ttl_seconds, entry[1])
        _memory[f'v2:session:{sid}'] = (time.time() + ttl, deepcopy(data))
        return
    try:
        saved = await _client().eval(_SAVE_SESSION, 2, f'session:{sid}', f'v2:session:{sid}',
                                    json.dumps(data, ensure_ascii=False), ttl,
                                    '1' if touch else '0', settings.session_ttl_seconds)
    except Exception as exc:
        raise unavailable() from exc
    if not saved:
        raise V2Error(410, 'SESSION_EXPIRED', '세션이 만료되었습니다.')


async def keys(sid: str, kind: str) -> list[str]:
    resources = (await session(sid)).get('resource_index', {}).get(kind, {})
    return [rid for rid, expires_at in resources.items() if expires_at is not None]


async def register_resource(sid: str, kind: str, resource_id: str, expires_at: str) -> None:
    await session(sid)
    async with mutex(f"v2:index-lock:{sid}"):
        data = await get(f"v2:session:{sid}")
        if data is None:
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        data['resource_index'].setdefault(kind, {})[resource_id] = expires_at
        await _save_live_session(sid, data)


async def tombstone_map(sid: str, map_id: str) -> None:
    """Delete a map while retaining its 410 marker without a live-resource renewal."""
    map_key = key(sid, 'map', map_id)
    async with mutex(f"v2:index-lock:{sid}"):
        if memory_enabled():
            # Match memory CAS users so a concurrent revision write cannot
            # recreate the map after deletion.
            async with mutex(f"cas:{map_key}"):
                if await base_session._existing(sid) is None:
                    raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
                data = await get(f"v2:session:{sid}")
                if data is None:
                    raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
                if not await get(map_key) or data['resource_index'].get('map', {}).get(map_id) is None:
                    raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")
                await delete(map_key)
                data['resource_index']['map'][map_id] = None
                await _save_live_session(sid, data)
            return
        try:
            status = await _client().eval(_DELETE_MAP, 3, f"session:{sid}",
                                          f"v2:session:{sid}", map_key, map_id)
        except Exception as exc:
            raise unavailable() from exc
        if status == -2:
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        if status == -1:
            raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")


async def resource(sid: str, kind: str, resource_id: str) -> dict:
    data = await session(sid)
    value = await get(key(sid, kind, resource_id))
    if value is not None:
        return value
    if resource_id in data['resource_index'].get(kind, {}):
        raise V2Error(410, "RESOURCE_EXPIRED", "이 자원이 만료되었습니다.")
    raise V2Error(404, "RESOURCE_NOT_FOUND", "찾을 수 없는 자원입니다.")


async def grant_result(sid: str, result_id: str, kind: str) -> None:
    data = await session(sid)
    await put(key(sid, 'result-access', result_id), {'result_type': kind},
              epoch(data['expires_at']) - time.time())
    await register_resource(sid, 'result-access', result_id, data['expires_at'])


async def authorize_result(sid: str, result_id: str, kind: str) -> None:
    data = await session(sid)
    access = await get(key(sid, 'result-access', result_id))
    if access and access.get('result_type') == kind:
        return
    field = 'strategy_id' if kind in ('strategy', 'strategies') else 'report_id'
    for resource_kind in ('map', 'report-job'):
        for rid, expires_at in data['resource_index'].get(resource_kind, {}).items():
            if expires_at is None:
                continue
            owned = await get(key(sid, resource_kind, rid))
            result = (owned or {}).get('last_result' if resource_kind == 'map' else 'result') or {}
            if result.get(field) == result_id:
                await grant_result(sid, result_id, kind)
                return
    raise V2Error(404, "RESOURCE_NOT_FOUND", "찾을 수 없는 자원입니다.")


async def clear_session(sid: str) -> None:
    data = await get(f"v2:session:{sid}") or {}
    for kind, resources in data.get('resource_index', {}).items():
        for rid, expires_at in resources.items():
            if expires_at is None:
                continue
            await delete(key(sid, kind, rid))
    await delete(f"v2:session:{sid}")
