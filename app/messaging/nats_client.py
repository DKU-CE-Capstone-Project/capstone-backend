"""
NATS JetStream 연결·발행·소비 헬퍼.

발표 자료의 "NATS = 이벤트 버스 / 큐"를 구현한다.
속보 burst 시 분석 작업을 JetStream 스트림에 버퍼링하여
API 실패로 전파하지 않고, worker가 자기 속도로 소비하도록 한다.
"""
from __future__ import annotations

import nats

from app.config import settings

# 스트림/주제/durable consumer 이름 — KEDA ScaledObject와 반드시 일치해야 함
STREAM_NAME = "analysis"
SUBJECT = "analysis.jobs"
DURABLE = "analysis-workers"

# api 프로세스용 공유 연결 (lazy singleton)
_nc = None
_js = None


async def connect():
    """NATS 서버에 연결한다. 재연결 무제한."""
    return await nats.connect(
        settings.nats_url,
        max_reconnect_attempts=-1,
        reconnect_time_wait=2,
    )


async def ensure_stream(js) -> None:
    """analysis 스트림을 멱등하게 생성한다 (이미 있으면 무시)."""
    try:
        await js.add_stream(name=STREAM_NAME, subjects=[SUBJECT])
    except Exception:
        # 이미 존재하거나 동시 생성 — 무시
        pass


async def get_jetstream():
    """api 프로세스에서 재사용할 JetStream 컨텍스트를 반환한다."""
    global _nc, _js
    if _js is None:
        _nc = await connect()
        _js = _nc.jetstream()
        await ensure_stream(_js)
    return _js


async def publish_job(payload: bytes) -> None:
    """분석 job을 JetStream에 발행한다."""
    js = await get_jetstream()
    await js.publish(SUBJECT, payload)
