"""
Redis 기반 job 결과 저장소 (멀티팟 공유 상태).

api 팟이 발행한 job을 worker 팟이 처리한 뒤 결과를 여기에 기록하고,
api 팟이 GET /jobs/{id}로 조회한다. 여러 worker 팟이 공유하므로
in-memory dict가 아니라 Redis를 사용한다.
(발표 자료의 MongoDB/Vector Search는 RAG 후속 단계 — MVP는 경량 Redis.)
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

_client: aioredis.Redis | None = None


def _redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def set_job(job_id: str, data: dict[str, Any], ttl: int = 3600) -> None:
    """job 상태/결과를 저장한다 (TTL 1시간)."""
    await _redis().set(f"job:{job_id}", json.dumps(data, ensure_ascii=False), ex=ttl)


async def get_job(job_id: str) -> dict[str, Any] | None:
    """job 상태/결과를 조회한다. 없으면 None."""
    raw = await _redis().get(f"job:{job_id}")
    return json.loads(raw) if raw else None
