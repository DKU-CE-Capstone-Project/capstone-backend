"""Separate JetStream v2 reports worker: python -m app.v2.report_worker."""
from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from weakref import WeakKeyDictionary

from nats.js.api import AckPolicy, ConsumerConfig, RetentionPolicy, StreamConfig
from nats.js.errors import NotFoundError

from app.config import settings
from app.messaging.nats_client import connect
from app.v2 import persistence
from app.v2.errors import V2Error
from app.v2.reports import process_job

STREAM = "reports_v2"
SUBJECT = "reports.generate.v2"
DURABLE = "reports-v2-workers"
_connections: WeakKeyDictionary = WeakKeyDictionary()


async def ensure_stream(js) -> None:
    try:
        info = await js.stream_info(STREAM)
    except NotFoundError:
        try:
            await js.add_stream(config=StreamConfig(name=STREAM, subjects=[SUBJECT], retention=RetentionPolicy.WORK_QUEUE, max_age=settings.v2_job_ttl_seconds, duplicate_window=120))
            return
        except Exception:  # noqa: BLE001 — a concurrent stream creation must verify the actual config
            info = await js.stream_info(STREAM)
    if info.config.subjects != [SUBJECT]:
        raise ValueError("V2 reports stream has conflicting subjects")


async def get_jetstream():
    loop = asyncio.get_running_loop()
    existing = _connections.get(loop)
    if existing and existing[0] == settings.nats_url and not existing[1].is_closed:
        return existing[2]
    nc = await connect()
    js = nc.jetstream()
    try:
        await ensure_stream(js)
    except Exception:
        await nc.close()
        raise
    _connections[loop] = (settings.nats_url, nc, js)
    return js


async def publish(sid: str, map_id: str, job_id: str) -> None:
    """Return only after the server acknowledged durable admission."""
    try:
        js = await get_jetstream()
        await js.publish(SUBJECT, json.dumps({"job_id": job_id, "session_id": sid, "map_id": map_id, "job_type": "report_v2"}).encode(), headers={"Nats-Msg-Id": job_id})
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 작업 큐를 사용할 수 없습니다. 같은 요청으로 다시 시도해 주세요.", True) from exc


async def main() -> None:
    await persistence.ready()
    await persistence.ensure_indexes()
    nc = await connect()
    js = nc.jetstream()
    await ensure_stream(js)
    config = ConsumerConfig(durable_name=DURABLE, ack_policy=AckPolicy.EXPLICIT, ack_wait=60, max_deliver=3, max_ack_pending=1)
    sub = await js.pull_subscribe(SUBJECT, durable=DURABLE, stream=STREAM, config=config)
    print(f"[v2-worker] subscribed subject={SUBJECT} durable={DURABLE}")
    try:
        while True:
            try:
                messages = await sub.fetch(1, timeout=5)
            except TimeoutError:
                continue
            for message in messages:
                try:
                    payload = json.loads(message.data.decode())
                    if payload.get("job_type") != "report_v2" or not payload.get("session_id") or not payload.get("job_id"):
                        await message.term()
                        continue
                    status = await process_job(payload["session_id"], payload["job_id"], message)
                    if status == "busy":
                        await message.nak(delay=20)
                    else:
                        await message.ack()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — dependency failure is retried by the durable queue
                    print(f"[v2-worker] retry type={type(exc).__name__}")
                    with suppress(Exception):
                        await message.nak(delay=20)
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
