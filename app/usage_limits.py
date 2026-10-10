"""Shared, atomic admission limits for provider-backed requests.

The global bucket prevents rotating anonymous sessions from bypassing the
capacity ceiling. The per-session bucket keeps a single browser from using
the whole allowance. Local-memory mode is only for explicitly isolated dev.
"""
from __future__ import annotations

import time

import redis.asyncio as aioredis
from fastapi import HTTPException

from app.config import settings

_client: aioredis.Redis | None = None
_memory: dict[str, tuple[int, float]] = {}
_RESERVE = """
local global_count = tonumber(redis.call('GET', KEYS[1]) or '0')
local owner_count = tonumber(redis.call('GET', KEYS[2]) or '0')
if global_count >= tonumber(ARGV[1]) or owner_count >= tonumber(ARGV[2]) then
  return 0
end
for i = 1, 2 do
  local count = redis.call('INCR', KEYS[i])
  if count == 1 then redis.call('EXPIRE', KEYS[i], tonumber(ARGV[3])) end
end
return 1
"""


async def check_quota(owner_sid: str, action: str, *, per_session: int, global_limit: int) -> None:
    """Reserve one unit before work begins; fail closed if Redis is required."""
    global _client
    global_key = f"quota:{action}:global"
    owner_key = f"quota:{action}:session:{owner_sid}"
    if not settings.session_store_required:
        now = time.time()
        for key, limit in ((global_key, global_limit), (owner_key, per_session)):
            count, expires = _memory.get(key, (0, now + 3600))
            if expires <= now:
                count, expires = 0, now + 3600
            if count >= limit:
                raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
        for key in (global_key, owner_key):
            count, expires = _memory.get(key, (0, now + 3600))
            _memory[key] = (count + 1, expires if expires > now else now + 3600)
        return

    try:
        if _client is None:
            _client = aioredis.from_url(settings.redis_url, decode_responses=True)
        allowed = await _client.eval(_RESERVE, 2, global_key, owner_key, global_limit, per_session, 3600)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="요청 한도를 확인할 수 없습니다. 잠시 후 다시 시도해 주세요.") from exc
    if not allowed:
        raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
