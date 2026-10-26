"""Opt in with ECONMIND_V2_WORKER_NATS_TEST=1 and isolated NATS on 127.0.0.1:44233."""
import os
import socket
import uuid

import nats
import pytest
from nats.js.api import AckPolicy, ConsumerConfig

from app.config import settings
from app.v2 import report_worker

_socket_connect = socket.socket.connect
pytestmark = pytest.mark.skipif(os.getenv("ECONMIND_V2_WORKER_NATS_TEST") != "1", reason="Explicit isolated NATS opt-in required")


@pytest.fixture
async def queue(monkeypatch):
    def loopback_only(sock, address):
        if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", 44233):
            raise AssertionError("Only the isolated NATS test port is permitted")
        return _socket_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    fixture_id = uuid.uuid4().hex
    monkeypatch.setattr(report_worker, "STREAM", "worker_limits_" + fixture_id)
    monkeypatch.setattr(report_worker, "SUBJECT", "worker.limits." + fixture_id)
    monkeypatch.setattr(report_worker, "DURABLE", "workers-" + fixture_id)
    monkeypatch.setattr(settings, "v2_worker_max_concurrency", 1)
    first = await nats.connect("nats://127.0.0.1:44233")
    second = await nats.connect("nats://127.0.0.1:44233")
    js1, js2 = first.jetstream(), second.jetstream()
    try:
        await report_worker.ensure_stream(js1)
        # An old deployed durable must be updated without dropping queued work.
        await js1.add_consumer(report_worker.STREAM, config=ConsumerConfig(
            name=report_worker.DURABLE, durable_name=report_worker.DURABLE,
            filter_subject=report_worker.SUBJECT, ack_policy=AckPolicy.EXPLICIT,
            ack_wait=10, max_deliver=3, max_ack_pending=20, max_waiting=128,
        ))
        await js1.publish(report_worker.SUBJECT, b"one")
        await js1.publish(report_worker.SUBJECT, b"two")
        await report_worker.ensure_consumer(js1)
        actual = (await js1.consumer_info(report_worker.STREAM, report_worker.DURABLE)).config
        assert actual.max_ack_pending == 1 and actual.max_deliver == -1
        assert actual.ack_wait == 60 and actual.max_waiting == 128
        yield js1, js2
    finally:
        await js1.delete_stream(report_worker.STREAM)
        await first.close()
        await second.close()


async def test_two_connections_share_capacity_and_transport_can_recover_after_three_retries(queue):
    js1, js2 = queue
    sub1 = await js1.pull_subscribe_bind(stream=report_worker.STREAM, durable=report_worker.DURABLE)
    sub2 = await js2.pull_subscribe_bind(stream=report_worker.STREAM, durable=report_worker.DURABLE)
    msg = (await sub1.fetch(1, timeout=1))[0]
    assert msg.data == b"one"
    with pytest.raises(TimeoutError):
        await sub2.fetch(1, timeout=0.1)
    # Transport failure does not discard the durable message at delivery three.
    for count in range(2, 6):
        await msg.nak()
        msg = (await sub2.fetch(1, timeout=1))[0]
        assert msg.data == b"one" and msg.metadata.num_delivered == count
    await msg.ack_sync()
    msg2 = (await sub1.fetch(1, timeout=1))[0]
    assert msg2.data == b"two"
    await msg2.ack_sync()
    info = await js1.consumer_info(report_worker.STREAM, report_worker.DURABLE)
    assert info.num_ack_pending == 0
