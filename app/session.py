"""
쿠키 기반 세션 식별.

12주차 회의 결정 사항:
  - 로그인 기능을 빼기로 했으므로, 사용자를 구분할 수단이 없다.
    ("김성민 씨랑 문주원 씨랑은 다른 마인드맵이 보여야 되잖아요")
  - 계정 대신 쿠키/세션으로 식별한다.
  - 마인드맵은 DB에 저장하지 않는다("mindmaps: DB 구현 안함 - 쿠키 사용").
    세션이 지워지면 데이터도 함께 사라져야 한다.

그래서 세션 상태를 Redis에 TTL과 함께 둔다:
  - TTL 만료 = 세션 소멸 → "세션이 지워지면 데이터를 지운다"가 그대로 구현된다
  - 이미 스택에 있는 컴포넌트라 새 의존성이 늘지 않는다 (app/job_store.py와 동일)
  - api를 여러 개로 띄워도 세션이 공유된다 (in-memory dict면 팟마다 갈린다)

Redis가 없거나 죽어 있으면 프로세스 로컬 dict로 폴백한다. 단일 인스턴스에서는
동작하지만 재시작 시 세션이 사라지므로, 운영에서는 Redis를 붙이는 것을 전제로 한다.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.config import settings

SESSION_COOKIE = "econmind_sid"
_KEY_PREFIX = "session:"

# Redis 불가 시 폴백 저장소: sid -> (만료 epoch, payload)
_memory_store: dict[str, tuple[float, dict[str, Any]]] = {}

_redis = None
_redis_disabled = False


def new_session_id() -> str:
    return uuid.uuid4().hex


def _client():
    """Redis 클라이언트 lazy 초기화. 실패 시 None (폴백 사용)."""
    global _redis, _redis_disabled
    if _redis_disabled:
        return None
    if _redis is None:
        try:
            import redis.asyncio as aioredis

            _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[session] Redis 초기화 실패 → in-memory 폴백: {type(exc).__name__}: {exc}")
            _redis_disabled = True
            return None
    return _redis


def _empty(sid: str) -> dict[str, Any]:
    return {
        "sid": sid,
        "created_at": time.time(),
        # 마인드맵 상태: 사용자가 무엇을 중심으로 보고 무엇을 펼쳤는지
        "mindmap": {
            "center_news_id": "",
            "expanded_news_ids": [],
            "query": "",
        },
        "viewed_news_ids": [],
    }


def _prune_memory() -> None:
    now = time.time()
    for sid in [s for s, (exp, _) in _memory_store.items() if exp < now]:
        _memory_store.pop(sid, None)


async def load(sid: str) -> dict[str, Any]:
    """세션 상태를 읽는다. 없으면 빈 상태를 돌려준다(생성하지는 않음)."""
    if not sid:
        return _empty("")

    client = _client()
    if client is not None:
        try:
            raw = await client.get(f"{_KEY_PREFIX}{sid}")
            if raw:
                return json.loads(raw)
            return _empty(sid)
        except Exception as exc:  # noqa: BLE001
            print(f"[session] Redis get 실패 → in-memory 폴백: {type(exc).__name__}: {exc}")

    _prune_memory()
    entry = _memory_store.get(sid)
    return entry[1] if entry else _empty(sid)


async def save(sid: str, data: dict[str, Any]) -> None:
    """세션 상태를 저장하고 TTL을 갱신한다(접근할 때마다 만료가 뒤로 밀린다)."""
    if not sid:
        return
    ttl = settings.session_ttl_seconds

    client = _client()
    if client is not None:
        try:
            await client.set(
                f"{_KEY_PREFIX}{sid}",
                json.dumps(data, ensure_ascii=False),
                ex=ttl,
            )
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[session] Redis set 실패 → in-memory 폴백: {type(exc).__name__}: {exc}")

    _prune_memory()
    _memory_store[sid] = (time.time() + ttl, data)


async def clear(sid: str) -> None:
    """세션을 삭제한다. 마인드맵 등 세션 종속 데이터가 함께 사라진다."""
    if not sid:
        return
    client = _client()
    if client is not None:
        try:
            await client.delete(f"{_KEY_PREFIX}{sid}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[session] Redis delete 실패 → in-memory 폴백: {type(exc).__name__}: {exc}")
    _memory_store.pop(sid, None)


# ── 마인드맵 상태 조작 ────────────────────────────────────────────────────────


async def record_center(sid: str, news_id: str, query: str = "") -> dict[str, Any]:
    """마인드맵 중심 노드를 기록한다. 중심이 바뀌면 확장 상태를 초기화한다."""
    data = await load(sid)
    mindmap = data.setdefault("mindmap", {})
    if mindmap.get("center_news_id") != news_id:
        mindmap["center_news_id"] = news_id
        mindmap["expanded_news_ids"] = []
    if query:
        mindmap["query"] = query

    viewed = data.setdefault("viewed_news_ids", [])
    if news_id and news_id not in viewed:
        viewed.append(news_id)
        del viewed[:-50]  # 최근 50개만 유지

    await save(sid, data)
    return data


async def expand_node(sid: str, news_id: str) -> dict[str, Any]:
    """사용자가 노드를 펼쳤음을 기록한다 (마인드맵 확장 기능).

    12주차 회의에서 "확장 기능은 아직 안 들어가 있다"고 확인된 부분이다.
    확장 결과 자체는 DB에 저장하지 않고, 어떤 노드를 펼쳤는지만 세션에 남긴다.
    """
    data = await load(sid)
    mindmap = data.setdefault("mindmap", {})
    expanded = mindmap.setdefault("expanded_news_ids", [])
    if news_id and news_id not in expanded:
        expanded.append(news_id)
        del expanded[:-20]  # 확장 노드는 20개까지
    await save(sid, data)
    return data


async def collapse_node(sid: str, news_id: str) -> dict[str, Any]:
    data = await load(sid)
    mindmap = data.setdefault("mindmap", {})
    expanded = mindmap.setdefault("expanded_news_ids", [])
    if news_id in expanded:
        expanded.remove(news_id)
    await save(sid, data)
    return data


async def reset_mindmap(sid: str) -> dict[str, Any]:
    data = await load(sid)
    data["mindmap"] = {"center_news_id": "", "expanded_news_ids": [], "query": ""}
    await save(sid, data)
    return data
