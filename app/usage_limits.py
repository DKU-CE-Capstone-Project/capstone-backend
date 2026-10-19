"""Shared, atomic admission limits for provider-backed requests.

Global, session, and optional connected-client buckets are checked in one
reservation. Local-memory mode is only for explicitly isolated dev.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from ipaddress import IPv6Address, ip_address, ip_network

import redis.asyncio as aioredis
from fastapi import HTTPException

from app.config import settings

_client: aioredis.Redis | None = None
_memory: dict[str, tuple[int, float]] = {}
_WINDOW_SECONDS = 3600
_LIMIT_MESSAGE = "요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
_RESERVE = """
local retry_ms = 0
for i = 1, #KEYS do
  if tonumber(redis.call('GET', KEYS[i]) or '0') >= tonumber(ARGV[i]) then
    local ttl = redis.call('PTTL', KEYS[i])
    if ttl == -1 then
      redis.call('EXPIRE', KEYS[i], tonumber(ARGV[#KEYS + 1]))
      ttl = tonumber(ARGV[#KEYS + 1]) * 1000
    elseif ttl < 1 then
      ttl = 1
    end
    if ttl > retry_ms then retry_ms = ttl end
  end
end
if retry_ms > 0 then return -math.ceil(retry_ms / 1000) end
for i = 1, #KEYS do
  local count = redis.call('INCR', KEYS[i])
  if count == 1 then redis.call('EXPIRE', KEYS[i], tonumber(ARGV[#KEYS + 1])) end
end
return 1
"""


@dataclass(frozen=True)
class QuotaRequest:
    owner_sid: str
    action: str
    per_session: int
    global_limit: int
    client_host: str | None = None
    per_client: int | None = None


def client_key(host: str | None) -> str:
    """Canonicalize the connected client; invalid values share one bucket."""
    try:
        address = ip_address(host or "")
    except ValueError:
        return "unknown"
    if isinstance(address, IPv6Address):
        if address.ipv4_mapped:
            address = address.ipv4_mapped
        else:
            # IPv6 privacy addresses can rotate within one delegated subnet.
            return str(ip_network((address, 64), strict=False))
    return str(address)


async def check_quota(
    owner_sid: str,
    action: str,
    *,
    per_session: int,
    global_limit: int,
    client_host: str | None = None,
    per_client: int | None = None,
) -> None:
    """Reserve one unit before work begins; fail closed if Redis is required."""
    await check_quotas(QuotaRequest(owner_sid, action, per_session, global_limit, client_host, per_client))


async def check_quotas(*requests: QuotaRequest) -> None:
    """Reserve all requested action budgets together, without partial charges."""
    global _client
    limits: list[tuple[str, int]] = []
    for request in requests:
        limits.extend([
            (f"quota:{request.action}:global", request.global_limit),
            (f"quota:{request.action}:session:{request.owner_sid}", request.per_session),
        ])
        if request.per_client is not None:
            limits.append((f"quota:{request.action}:client:{client_key(request.client_host)}", request.per_client))
    if not limits:
        return
    if not settings.session_store_required:
        now = time.time()
        retry_after = 0
        for key, limit in limits:
            count, expires = _memory.get(key, (0, now + _WINDOW_SECONDS))
            if expires <= now:
                count, expires = 0, now + _WINDOW_SECONDS
            if count >= limit:
                retry_after = max(retry_after, math.ceil(expires - now))
        if retry_after:
            raise HTTPException(status_code=429, detail=_LIMIT_MESSAGE,
                                headers={"Retry-After": str(retry_after)})
        for key, _ in limits:
            count, expires = _memory.get(key, (0, now + _WINDOW_SECONDS))
            _memory[key] = (count + 1, expires if expires > now else now + _WINDOW_SECONDS)
        return

    try:
        if _client is None:
            _client = aioredis.from_url(settings.redis_url, decode_responses=True)
        result = await _client.eval(
            _RESERVE, len(limits),
            *(key for key, _ in limits),
            *(limit for _, limit in limits),
            _WINDOW_SECONDS,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="요청 한도를 확인할 수 없습니다. 잠시 후 다시 시도해 주세요.") from exc
    if result != 1:
        raise HTTPException(status_code=429, detail=_LIMIT_MESSAGE,
                            headers={"Retry-After": str(max(1, -int(result)))})
