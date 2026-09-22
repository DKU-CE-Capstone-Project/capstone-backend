"""Job failures must not expose provider errors through Redis or HTTP."""

import json

import httpx
from fastapi.testclient import TestClient

from app import worker
from app.agents import news_fetcher
from app.api.v1 import jobs
from app.config import settings
from app.job_store import PUBLIC_JOB_ERROR
from app.main import app


async def test_worker_failure_hides_exception(monkeypatch, capsys) -> None:
    stored = []
    secret = "FAKE_SECRET_TOKEN_123"

    async def fail(_keyword):
        raise RuntimeError(secret)

    async def record(_job_id, data):
        stored.append(data)

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(worker, "run_analysis", fail)
    monkeypatch.setattr(worker, "set_job", record)

    await worker._process({"job_id": "test-job", "keyword": "example"})

    assert stored == [{"status": "error", "keyword": "example", "error": PUBLIC_JOB_ERROR}]
    output = capsys.readouterr().out
    assert "type=RuntimeError" in output
    assert secret not in output


async def test_worker_success_keeps_result(monkeypatch) -> None:
    stored = []

    class Result:
        def model_dump_json(self):
            return json.dumps({"answer": "ok"})

    async def succeed(_keyword):
        return Result()

    async def record(_job_id, data):
        stored.append(data)

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(worker, "run_analysis", succeed)
    monkeypatch.setattr(worker, "set_job", record)

    await worker._process({"job_id": "test-job", "keyword": "example"})

    assert stored == [{"status": "done", "keyword": "example", "result": {"answer": "ok"}}]


def test_job_api_redacts_existing_error_records(monkeypatch) -> None:
    async def old_record(_job_id):
        return {"status": "error", "keyword": "example", "error": "FAKE_SECRET_TOKEN_123"}

    monkeypatch.setattr(jobs, "get_job", old_record)
    response = TestClient(app).get("/jobs/test-job")

    assert response.status_code == 200
    assert response.json() == {"status": "error", "keyword": "example", "error": PUBLIC_JOB_ERROR}


def test_job_api_preserves_success_records(monkeypatch) -> None:
    async def done_record(_job_id):
        return {"status": "done", "keyword": "example", "result": {"answer": "ok"}}

    monkeypatch.setattr(jobs, "get_job", done_record)
    response = TestClient(app).get("/jobs/test-job")

    assert response.status_code == 200
    assert response.json() == {"status": "done", "keyword": "example", "result": {"answer": "ok"}}


async def test_newsapi_http_failure_does_not_log_api_key(monkeypatch, capsys) -> None:
    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, params):
            request = httpx.Request("GET", url, params=params)
            return httpx.Response(401, request=request)

    secret = "FAKE_SECRET_TOKEN_123"
    monkeypatch.setattr(settings, "newsapi_key", secret)
    monkeypatch.setattr(news_fetcher.httpx, "AsyncClient", lambda **_kwargs: FailingClient())

    assert await news_fetcher._fetch_newsapi("semiconductor", 3) == []
    output = capsys.readouterr().out
    assert "HTTPStatusError" in output
    assert secret not in output
