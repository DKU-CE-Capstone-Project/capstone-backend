"""Usage contracts: observed calls only, isolated correlation, safe status reads."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import database, telemetry
from app.agents import decision_client, diffbot_client, llm
from app.config import settings
from app.main import app
from app.v2 import report_worker


def events(capsys):
    return [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event_type": "econmind_usage"')
    ]


async def test_context_isolated_across_tasks_and_threads(capsys):
    async def work(number):
        with telemetry.usage_context(
            request_id=str(number) * 32, session_id="secret-session", stage="analysis"
        ):
            await asyncio.to_thread(
                telemetry.emit, "sdk_call", provider="gemini", prompt="secret-prompt"
            )

    await asyncio.gather(work(1), work(2))
    telemetry.emit("cache_hit", provider="gemini")
    rows = events(capsys)
    assert {r["request_id"] for r in rows[:2]} == {"1" * 32, "2" * 32}
    assert "request_id" not in rows[2]
    assert "secret" not in json.dumps(rows)
    assert len({r["event_id"] for r in rows}) == 3


async def test_sdk_usage_missing_and_cache_subset_are_preserved(monkeypatch, capsys):
    monkeypatch.setattr(settings, "google_api_key", "secret-key")
    responses = iter(
        [
            SimpleNamespace(
                text="ok",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=20,
                    cached_content_token_count=80,
                    thoughts_token_count=5,
                    total_token_count=125,
                ),
            ),
            SimpleNamespace(text="ok"),
        ]
    )

    async def generate_content(**kwargs):
        return next(responses)

    monkeypatch.setattr(
        llm,
        "_client",
        lambda: SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        ),
    )
    await llm.generate("secret-prompt")
    await llm.generate("secret-prompt")
    rows = events(capsys)
    assert len(rows) == 2 and all(r["kind"] == "sdk_call" for r in rows)
    assert rows[0]["input_tokens"] == 100 and rows[0]["cached_input_tokens"] == 80
    assert rows[0]["total_tokens"] == 125
    assert rows[1]["input_tokens"] is None
    assert "secret" not in json.dumps(rows)


async def test_sdk_timeout_and_telemetry_failure_do_not_change_product(
    monkeypatch, capsys
):
    monkeypatch.setattr(settings, "google_api_key", "fixture")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 0.01)

    async def wait(**kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(
        llm,
        "_client",
        lambda: SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=wait))
        ),
    )
    assert await llm.generate("private") == ""
    assert events(capsys)[0]["outcome"] == "cancelled"

    class Broken:
        def write(self, value):
            raise OSError("closed")

    monkeypatch.setattr(telemetry.sys, "stdout", Broken())
    telemetry.emit("sdk_call", provider="gemini")


async def test_decision_retries_and_cache_do_not_recount_tokens(monkeypatch, capsys):
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("secret-decision-key"))
    monkeypatch.setattr(
        decision_client, "normalize", lambda body, questions: {"ok": True}
    )
    monkeypatch.setattr(decision_client.asyncio, "sleep", AsyncMock())
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            429 if len(calls) == 1 else 200, json={"usage": {"input_tokens": 27}}
        )

    async with decision_client.DecisionClient(
        transport=httpx.MockTransport(handle)
    ) as client:
        assert await client.ask({}, {}, {}) == {"ok": True}
        assert await client.ask({}, {}, {}) == {"ok": True}
    rows = events(capsys)
    assert [r["kind"] for r in rows] == ["http_attempt", "http_attempt", "cache_hit"]
    assert [r["http_status"] for r in rows[:2]] == [429, 200]
    assert rows[1]["input_tokens"] == 27 and "input_tokens" not in rows[2]
    assert "secret" not in json.dumps(rows)


def test_diffbot_attempts_include_http_200_provider_errors(monkeypatch, capsys):
    responses = iter(
        [
            SimpleNamespace(status_code=429, headers={}, json=dict),
            SimpleNamespace(
                status_code=200,
                headers={},
                json=lambda: {"errorCode": 401, "error": "secret-body"},
            ),
        ]
    )
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: next(responses))
    with pytest.raises(diffbot_client.DiffbotError):
        diffbot_client.call_diffbot_article(
            "https://private.invalid",
            "secret-primary",
            token_pool=("secret-primary", "secret-secondary"),
            max_retries=0,
        )
    rows = events(capsys)
    assert len(rows) == 2 and all(r["outcome"] == "error" for r in rows)
    assert "secret" not in json.dumps(rows) and "private.invalid" not in json.dumps(
        rows
    )


async def test_queue_correlation_survives_redelivery_without_suppressing_real_calls(
    monkeypatch, capsys
):
    sent = []
    js = SimpleNamespace(
        publish=AsyncMock(side_effect=lambda *a, **kw: sent.append(kw))
    )
    monkeypatch.setattr(report_worker, "get_jetstream", AsyncMock(return_value=js))
    with telemetry.usage_context(request_id="a" * 32):
        await report_worker.publish("private-session", "map", "b" * 32)

    async def process(*args):
        telemetry.emit("sdk_call", provider="gemini")
        return "done"

    monkeypatch.setattr(report_worker, "process_job", process)
    message = SimpleNamespace(
        data=json.dumps(
            {
                "session_id": "private-session",
                "job_id": "b" * 32,
                "job_type": "report_v2",
            }
        ).encode(),
        headers=sent[0]["headers"],
        ack=AsyncMock(),
    )
    await report_worker.consume_message(message)
    await report_worker.consume_message(message)
    rows = events(capsys)
    assert len(rows) == 2 and all(r["request_id"] == "a" * 32 for r in rows)
    assert all(r["job_id"] == "b" * 32 for r in rows)
    assert len({r["run_id"] for r in rows}) == 2
    assert "private-session" not in json.dumps(rows)


@pytest.mark.parametrize("token", ["", "short", "x" * 32])
def test_admin_status_fails_closed_before_ping(monkeypatch, token):
    monkeypatch.setattr(settings, "admin_status_token", SecretStr(token))
    ping = AsyncMock()
    monkeypatch.setattr(database, "ping", ping)
    response = TestClient(app).get(
        "/internal/admin/status", headers={"Authorization": "Bearer incorrect"}
    )
    assert response.status_code == 403
    ping.assert_not_called()


@pytest.mark.parametrize(
    "connected, expected", [(True, "connected"), (False, "unreachable")]
)
def test_admin_status_only_pings_and_returns_connection(
    monkeypatch, connected, expected
):
    monkeypatch.setattr(settings, "admin_status_token", SecretStr("x" * 32))
    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(database, "ping", AsyncMock(return_value=connected))
    response = TestClient(app).get(
        "/internal/admin/status", headers={"Authorization": "Bearer " + "x" * 32}
    )
    assert response.status_code == 200 and response.json() == {"mongodb": expected}
    assert response.headers["cache-control"] == "no-store"
    assert not response.cookies


def test_api_request_records_template_not_query_or_cookie(capsys):
    TestClient(app).get(
        "/api/v1/keywords/recommended?secret=query",
        headers={"Cookie": "private-cookie"},
    )
    rows = events(capsys)
    assert len(rows) == 1 and rows[0]["kind"] == "api_request"
    assert "secret" not in json.dumps(rows) and "private-cookie" not in json.dumps(rows)
