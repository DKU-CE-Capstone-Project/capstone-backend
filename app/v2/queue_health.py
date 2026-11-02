"""Bounded read-only queue probe; it does not assert worker liveness."""
import asyncio
from contextlib import suppress

from nats.aio.client import Client

from app.config import settings
from app.v2.report_worker import STREAM, validate_stream


async def _ignore_error(error) -> None:
    # nats-py's default error callback logs transport exceptions. Readiness only
    # exposes a boolean, never credentials or connection details from exceptions.
    pass


async def ping() -> bool:
    nc = Client()
    try:
        async with asyncio.timeout(2):
            await nc.connect(settings.nats_url, connect_timeout=1,
                             allow_reconnect=False, max_reconnect_attempts=0,
                             error_cb=_ignore_error)
            info = await nc.jetstream(timeout=1).stream_info(STREAM)
            validate_stream(info)
        return True
    except Exception:  # noqa: BLE001 — public readiness must not expose connection credentials
        return False
    finally:
        with suppress(Exception):
            await asyncio.wait_for(nc.close(), timeout=0.5)
