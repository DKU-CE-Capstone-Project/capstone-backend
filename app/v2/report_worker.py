"""Separate JetStream v2 reports worker: python -m app.v2.report_worker."""
from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import replace
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
            info = await js.add_stream(config=StreamConfig(name=STREAM, subjects=[SUBJECT], retention=RetentionPolicy.WORK_QUEUE,
                                                          max_age=settings.v2_job_ttl_seconds, duplicate_window=min(120, settings.v2_job_ttl_seconds)))
        except Exception:  # noqa: BLE001 — a concurrent stream creation must verify the actual config
            info = await js.stream_info(STREAM)
    if info.config.subjects != [SUBJECT]:
        raise ValueError("V2 reports stream has conflicting subjects")
    if info.config.retention != RetentionPolicy.WORK_QUEUE or not 0 < info.config.max_age <= settings.v2_job_ttl_seconds:
        raise ValueError("V2 reports stream must retain only bounded work-queue messages")


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


async def ensure_consumer(js) -> None:
    """Reconcile the shared durable without resetting its delivery cursor.

    nats-py pull_subscribe(config=...) does not update an existing durable.
    Transport retries remain recoverable until stream expiry; the persisted job
    claim limits actual pipeline starts separately from queue redelivery.
    """
    desired = ConsumerConfig(durable_name=DURABLE, name=DURABLE, filter_subject=SUBJECT,
                             ack_policy=AckPolicy.EXPLICIT, ack_wait=60, max_deliver=-1,
                             max_ack_pending=settings.v2_worker_max_concurrency)
    try:
        info = await js.consumer_info(STREAM, DURABLE)
    except NotFoundError:
        # A concurrent creation is safe only after verifying actual policy below.
        with suppress(Exception):
            await js.add_consumer(STREAM, config=desired)
    else:
        actual = info.config
        if actual.deliver_subject or actual.ack_policy != AckPolicy.EXPLICIT or actual.filter_subject != SUBJECT or actual.filter_subjects:
            raise ValueError("V2 reports consumer has conflicting delivery configuration")
        if (actual.max_ack_pending, actual.max_deliver, actual.ack_wait) != (desired.max_ack_pending, desired.max_deliver, desired.ack_wait):
            await js.add_consumer(STREAM, config=replace(actual, max_ack_pending=desired.max_ack_pending,
                                                        max_deliver=desired.max_deliver, ack_wait=desired.ack_wait))
    actual = (await js.consumer_info(STREAM, DURABLE)).config
    if (actual.deliver_subject or actual.ack_policy != desired.ack_policy or actual.filter_subject != SUBJECT
            or actual.filter_subjects or actual.max_ack_pending != desired.max_ack_pending
            or actual.max_deliver != desired.max_deliver or actual.ack_wait != desired.ack_wait):
        raise ValueError("V2 reports consumer limits were not applied")


async def publish(sid: str, map_id: str, job_id: str) -> None:
    """Return only after the server acknowledged durable admission."""
    try:
        js = await get_jetstream()
        await js.publish(SUBJECT, json.dumps({"job_id": job_id, "session_id": sid, "map_id": map_id, "job_type": "report_v2"}).encode(), headers={"Nats-Msg-Id": job_id})
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 작업 큐를 사용할 수 없습니다. 같은 요청으로 다시 시도해 주세요.", True) from exc


async def consume_message(message) -> None:
    """Hold busy duplicates without spending another transport delivery."""
    try:
        try:
            payload = json.loads(message.data.decode())
        except (UnicodeError, json.JSONDecodeError):
            await message.term()
            return
        if (not isinstance(payload, dict) or payload.get("job_type") != "report_v2"
                or not all(isinstance(payload.get(key), str) and payload[key] for key in ("session_id", "job_id"))):
            await message.term()
            return
        while True:
            status = await process_job(payload["session_id"], payload["job_id"], message)
            if status != "busy":
                await message.ack()
                return
            # Another owner may finish or lose its 60-second Redis lease.
            # Keep this pending message alive while waiting, including on an
            # old consumer's last delivery during the rolling upgrade.
            await message.in_progress()
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — retry dependency repair without logging credentials
        print(f"[v2-worker] retry type={type(exc).__name__}")
        with suppress(Exception):
            await message.nak(delay=20)


async def main() -> None:
    await persistence.ready()
    await persistence.ensure_indexes()
    nc = await connect()
    try:
        js = nc.jetstream()
        await ensure_stream(js)
        await ensure_consumer(js)
        sub = await js.pull_subscribe_bind(durable=DURABLE, stream=STREAM)
        print(f"[v2-worker] subscribed subject={SUBJECT} durable={DURABLE}")
        while True:
            try:
                messages = await sub.fetch(1, timeout=5)
            except TimeoutError:
                continue
            for message in messages:
                await consume_message(message)
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
