"""Shared, atomic admission limits for provider-backed requests.

Global, session, and optional connected-client buckets are checked in one
reservation. Local-memory mode is only for explicitly isolated dev.
"""
from __future__ import annotations

import time
from ipaddress import IPv6Address, ip_address, ip_network

import redis.asyncio as aioredis
from fastapi import HTTPException

from app.config import settings

_client: aioredis.Redis | None = None
_memory: dict[str, tuple[int, float]] = {}
_RESERVE = """
for i = 1, #KEYS do
  if tonumber(redis.call('GET', KEYS[i]) or '0') >= tonumber(ARGV[i]) then
    return 0
  end
end
for i = 1, #KEYS do
  local count = redis.call('INCR', KEYS[i])
  if count == 1 then redis.call('EXPIRE', KEYS[i], tonumber(ARGV[#KEYS + 1])) end
end
return 1
"""


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
    global _client
    global_key = f"quota:{action}:global"
    owner_key = f"quota:{action}:session:{owner_sid}"
    limits = [(global_key, global_limit), (owner_key, per_session)]
    if per_client is not None:
        limits.append((f"quota:{action}:client:{client_key(client_host)}", per_client))
    if not settings.session_store_required:
        now = time.time()
        for key, limit in limits:
            count, expires = _memory.get(key, (0, now + 3600))
            if expires <= now:
                count, expires = 0, now + 3600
            if count >= limit:
                raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
        for key, _ in limits:
            count, expires = _memory.get(key, (0, now + 3600))
            _memory[key] = (count + 1, expires if expires > now else now + 3600)
        return

    try:
        if _client is None:
            _client = aioredis.from_url(settings.redis_url, decode_responses=True)
        allowed = await _client.eval(
            _RESERVE, len(limits),
            *(key for key, _ in limits),
            *(limit for _, limit in limits),
            3600,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="요청 한도를 확인할 수 없습니다. 잠시 후 다시 시도해 주세요.") from exc
    if not allowed:
        raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
