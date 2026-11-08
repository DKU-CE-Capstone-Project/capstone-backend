"""Best-effort embedding token measurement, without storing input text."""

import asyncio
import math
import time

COUNT_TIMEOUT_SECONDS = 2.0
MAX_COUNT_REQUESTS = 31
# gemini-embedding-001 reserves eight of its 2,048 positions internally.
TEXT_TOKEN_LIMIT = 2040


def _get(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _tokens(value):
    if type(value) is int and value >= 0:
        return value
    if (
        type(value) is float
        and math.isfinite(value)
        and value >= 0
        and value.is_integer()
    ):
        return int(value)
    return None


async def record_embedding_usage(event, response, *, client, model, texts):
    """Prefer response usage; otherwise count the exact, already-sent input.

    Role-free Content objects avoid the SDK's extra user-role token. Split large
    groups to detect individual inputs that may have been silently truncated.
    Counting is optional, bounded, never retried and never an inference event.
    """
    usage = _get(response, "usage_metadata")
    event.response(response, usage=usage)
    event.fields["embedding_items"] = len(texts)
    if event.fields.get("input_tokens") is not None:
        event.fields["input_token_source"] = "response"
        return
    embeddings = _get(response, "embeddings") or []
    statistics = [
        _tokens(_get(_get(e, "statistics"), "token_count")) for e in embeddings
    ]
    if len(statistics) == len(texts) and all(n is not None for n in statistics):
        event.fields.update(
            input_tokens=sum(statistics), input_token_source="statistics"
        )
        return
    if model != "gemini-embedding-001":
        return

    from google.genai import types

    started, requests = time.monotonic(), 0
    semaphore = asyncio.Semaphore(4)

    async def count(group):
        nonlocal requests
        async with semaphore:
            if requests >= MAX_COUNT_REQUESTS:
                raise ValueError("count request limit")
            requests += 1
            result = await client.aio.models.count_tokens(
                model=model,
                contents=[
                    types.Content(parts=[types.Part(text=text)]) for text in group
                ],
                config=types.CountTokensConfig(
                    http_options=types.HttpOptions(
                        timeout=1500, retry_options=types.HttpRetryOptions(attempts=1)
                    )
                ),
            )
        total = _tokens(_get(result, "total_tokens"))
        if total is None:
            raise ValueError("missing token count")
        if total <= TEXT_TOKEN_LIMIT:
            return total, False
        if len(group) == 1:
            return total, True
        midpoint = len(group) // 2
        tasks = [
            asyncio.create_task(count(part))
            for part in (group[:midpoint], group[midpoint:])
        ]
        try:
            children = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if sum(n for n, _ in children) != total:
            raise ValueError("inconsistent grouped count")
        return total, any(truncated for _, truncated in children)

    try:
        total, may_truncate = await asyncio.wait_for(
            count(texts), COUNT_TIMEOUT_SECONDS
        )
        event.fields.update(
            counted_input_tokens=total,
            input_token_source="count_tokens",
            token_count_status="possible_truncation" if may_truncate else "ok",
        )
    except TimeoutError:
        event.fields["token_count_status"] = "timeout"
    except Exception:  # noqa: BLE001 — optional telemetry must preserve embeddings.
        event.fields["token_count_status"] = "unavailable"
    finally:
        event.fields.update(
            token_count_requests=requests,
            token_count_duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
