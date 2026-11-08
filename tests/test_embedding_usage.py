"""Token accounting must not change embeddings or expose their input."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import embedding_usage, telemetry
from app.agents import llm
from app.config import settings


def client_with_counter(counter):
    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(count_tokens=counter))
    )


async def measure(response, texts, counter):
    with telemetry.observe(
        "gemini", "embed_batch", model="gemini-embedding-001"
    ) as event:
        await embedding_usage.record_embedding_usage(
            event,
            response,
            client=client_with_counter(counter),
            model="gemini-embedding-001",
            texts=texts,
        )
    return event.fields


@pytest.mark.parametrize("source", ["usage", "statistics"])
async def test_response_usage_precedes_optional_token_counting(source):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=17)
        if source == "usage"
        else None,
        embeddings=[SimpleNamespace(statistics=SimpleNamespace(token_count=17.0))],
    )
    counter = AsyncMock()
    result = await measure(response, ["private input"], counter)
    assert result["input_tokens"] == 17
    assert result["input_token_source"] == (
        "response" if source == "usage" else "statistics"
    )
    counter.assert_not_called()


async def test_exact_sent_text_is_counted_without_role_or_extra_inference(
    monkeypatch, capsys
):
    monkeypatch.setattr(settings, "google_api_key", "secret-key")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    inputs = ["  private one  ", "private two", "private three"]
    embeddings, counted = [], []

    async def embed_content(**kwargs):
        texts = kwargs["contents"]
        texts = texts if isinstance(texts, list) else [texts]
        embeddings.append(texts)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in texts]
        )

    async def count_tokens(**kwargs):
        contents = kwargs["contents"]
        assert all(content.role is None for content in contents)
        assert kwargs["model"] == "gemini-embedding-001"
        assert kwargs["config"].http_options.retry_options.attempts == 1
        counted.append([content.parts[0].text for content in contents])
        return SimpleNamespace(total_tokens=len(contents) * 7)

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                embed_content=embed_content,
                count_tokens=count_tokens,
            )
        )
    )
    monkeypatch.setattr(llm, "_client", lambda: client)
    assert await llm.embed_batch(inputs, dimensions=2, batch_size=2) == [[0.1, 0.2]] * 3
    assert await llm.embed("x" * 8001, dimensions=2) == [0.1, 0.2]
    assert (
        counted
        == embeddings
        == [["private one", "private two"], ["private three"], ["x" * 8000]]
    )
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3 and all(row["kind"] == "sdk_call" for row in rows)
    assert [row["counted_input_tokens"] for row in rows] == [14, 7, 7]
    assert all(
        row["input_tokens"] is None and row["token_count_status"] == "ok"
        for row in rows
    )
    assert "private" not in json.dumps(rows) and "secret-key" not in json.dumps(rows)


async def test_large_group_splits_to_check_each_input_for_silent_truncation():
    calls = []

    async def count_tokens(**kwargs):
        texts = [content.parts[0].text for content in kwargs["contents"]]
        calls.append(texts)
        return SimpleNamespace(total_tokens=sum(int(text) for text in texts))

    result = await measure(SimpleNamespace(), ["1000", "1000", "1000"], count_tokens)
    assert (
        result["counted_input_tokens"] == 3000 and result["token_count_status"] == "ok"
    )
    assert result["token_count_requests"] == len(calls) == 3
    result = await measure(SimpleNamespace(), ["2041", "1"], count_tokens)
    assert result["token_count_status"] == "possible_truncation"


@pytest.mark.parametrize("result", [None, True, -1, 1.5, float("inf")])
async def test_invalid_counts_stay_missing(result):
    fields = await measure(
        SimpleNamespace(metadata=SimpleNamespace(billable_character_count=99)),
        ["private"],
        AsyncMock(return_value=SimpleNamespace(total_tokens=result)),
    )
    assert fields["token_count_status"] == "unavailable"
    assert fields["input_tokens"] is None and "counted_input_tokens" not in fields


async def test_timeout_and_provider_failure_preserve_successful_embedding(
    monkeypatch, capsys
):
    monkeypatch.setattr(settings, "google_api_key", "fixture")
    monkeypatch.setattr(embedding_usage, "COUNT_TIMEOUT_SECONDS", 0.01)

    async def slow(**kwargs):
        await asyncio.sleep(10)

    model = SimpleNamespace(
        embed_content=AsyncMock(
            return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5])])
        ),
        count_tokens=slow,
    )
    monkeypatch.setattr(
        llm, "_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=model))
    )
    assert await llm.embed("private") == [0.5]
    model.count_tokens = AsyncMock(side_effect=RuntimeError("private provider error"))
    assert await llm.embed_batch(["private"], dimensions=1) == [[0.5]]
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["token_count_status"] for row in rows] == ["timeout", "unavailable"]
    assert all(row["outcome"] == "success" for row in rows)
    assert "private" not in json.dumps(rows)


async def test_grouped_failure_cancels_remaining_optional_requests():
    cancelled = asyncio.Event()

    async def count_tokens(**kwargs):
        texts = [content.parts[0].text for content in kwargs["contents"]]
        if len(texts) == 2:
            return SimpleNamespace(total_tokens=3000)
        if texts == ["failure"]:
            raise RuntimeError("private")
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    result = await measure(SimpleNamespace(), ["failure", "wait"], count_tokens)
    assert result["token_count_status"] == "unavailable"
    assert cancelled.is_set()


async def test_counting_request_limit_and_inconsistent_results_stay_unpriced(
    monkeypatch,
):
    monkeypatch.setattr(embedding_usage, "MAX_COUNT_REQUESTS", 1)
    result = await measure(
        SimpleNamespace(),
        ["a", "b"],
        AsyncMock(return_value=SimpleNamespace(total_tokens=3000)),
    )
    assert (
        result["token_count_status"] == "unavailable"
        and result["token_count_requests"] == 1
    )
    monkeypatch.setattr(embedding_usage, "MAX_COUNT_REQUESTS", 31)
    counter = AsyncMock(
        side_effect=[
            SimpleNamespace(total_tokens=3000),
            SimpleNamespace(total_tokens=1000),
            SimpleNamespace(total_tokens=1000),
        ]
    )
    result = await measure(SimpleNamespace(), ["a", "b"], counter)
    assert (
        result["token_count_status"] == "unavailable"
        and "counted_input_tokens" not in result
    )


async def test_caller_cancellation_is_not_swallowed():
    started = asyncio.Event()

    async def count_tokens(**kwargs):
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(measure(SimpleNamespace(), ["private"], count_tokens))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
