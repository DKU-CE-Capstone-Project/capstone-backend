import csv
import json
import uuid

import httpx
import pytest

from scripts import measure_v2


@pytest.mark.parametrize(
    "session_limit,max_expansions,selected,stop",
    [
        (10, 6, 10, "target_reached"),
        (10, 1, 7, "expansion_limit"),
        (1, 6, 1, "session_limit"),
    ],
)
async def test_measurement_uses_expanded_map_and_saves_both_outputs(
    tmp_path,
    monkeypatch,
    session_limit,
    max_expansions,
    selected,
    stop,
):
    monkeypatch.setattr(measure_v2, "POLL_INTERVAL", 0.001)
    revision = 1
    observed = iter(["extracting", "extracting", "done"])
    count = 4
    expanded = {0}
    failed_evidence = (
        [{"news_id": "article-0", "code": "BODY_EMPTY"}] if selected > 1 else []
    )

    def snapshot():
        return {
            "map_id": "map-a",
            "revision": revision,
            "nodes": [
                {
                    "news_id": f"article-{n}",
                    "node_id": f"node-{n}",
                    "level": 0 if n == 0 else 1 if n < 4 else 2,
                    "expanded": n in expanded,
                }
                for n in range(count)
            ],
        }

    def handle(request):
        nonlocal revision, count
        path = request.url.path
        if path == "/api/v2/session":
            return httpx.Response(
                200,
                headers={"Set-Cookie": "sid=fixture-cookie-secret; HttpOnly"},
                json={
                    "prototype_tier": "PAID" if session_limit == 10 else "FREE",
                    "session_epoch": "fixture-session-epoch",
                    "capabilities": {
                        "max_report_articles": session_limit,
                        "max_map_nodes": 30,
                        "max_map_depth": 2,
                    },
                },
            )
        assert "fixture-cookie-secret" in request.headers["cookie"]
        assert request.headers["authorization"].startswith("Basic ")
        if path == "/api/v2/maps":
            assert json.loads(request.content) == {"query": "반도체"}
            return httpx.Response(201, json=snapshot())
        if path == "/api/v2/maps/map-a/branches":
            assert json.loads(request.content) == {
                "expected_revision": revision,
                "parent_node_id": f"node-{revision}",
                "mode": "expand",
            }
            expanded.add(revision)
            revision += 1
            count += 3
            return httpx.Response(200, json=snapshot())
        if path == "/api/v2/report-jobs":
            payload = json.loads(request.content)
            assert payload["expected_revision"] == revision
            assert payload["news_ids"] == [f"article-{n}" for n in range(selected)]
            uuid.UUID(request.headers["Idempotency-Key"])
            assert payload["strategy_options"] == {
                "risk_level": "medium",
                "period": "short",
            }
            return httpx.Response(
                202, json={"job_id": "job-a", "stage": "queued", "status": "queued"}
            )
        if path == "/api/v2/report-jobs/job-a":
            stage = next(observed)
            result = {
                "report_id": "report-a",
                "strategy_id": "strategy-a",
                "report_reused": False,
                "strategy_reused": False,
            }
            return httpx.Response(
                200,
                json={
                    "job_id": "job-a",
                    "stage": stage,
                    "status": "completed" if stage == "done" else "running",
                    "result": result if stage == "done" else None,
                },
            )
        if path == "/api/v2/reports/report-a":
            return httpx.Response(
                200,
                json={
                    "report_id": "report-a",
                    "title": "한글 리포트",
                    "coverage": {
                        "requested_count": selected,
                        "used_count": selected - len(failed_evidence),
                        "failed_count": len(failed_evidence),
                    },
                    "failed_evidence": failed_evidence,
                },
            )
        assert path == "/api/v2/strategies/strategy-a"
        return httpx.Response(
            200, json={"strategy_id": "strategy-a", "report_id": "report-a"}
        )

    directory = tmp_path / "measurement"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://127.0.0.1",
        auth=("fixture-user", "fixture-password"),
    ) as client:
        await measure_v2.run(
            client, query="반도체", directory=directory, max_expansions=max_expansions
        )
        with pytest.raises(FileExistsError):
            await measure_v2.run(client, query="반도체", directory=directory)

    raw = (directory / "measure_v2.json").read_text(encoding="utf-8")
    saved = json.loads(raw)
    with (directory / "measure_v2.csv").open(encoding="utf-8", newline="") as file:
        (row,) = csv.DictReader(file)
    assert row["query"] == "반도체" and row["job_id"] == "job-a"
    assert row["outcome"] == "completed"
    assert int(row["selected_articles"]) == selected
    assert int(row["selection_shortfall"]) == 10 - selected
    assert row["selection_stop"] == stop
    assert int(row["requested_count"]) == selected and int(row["failed_count"]) == len(
        failed_evidence
    )
    assert json.loads(row["failed_evidence_codes"]) == [
        item["code"] for item in failed_evidence
    ]
    assert row["report_reused"] == "False"
    assert saved["report"]["title"] == "한글 리포트"
    assert saved["strategy"]["report_id"] == "report-a"
    assert len(saved["job_observations"]) == 4
    assert [step["stage"] for step in saved["stage_changes"]] == [
        "queued",
        "extracting",
        "done",
    ]
    assert all(
        secret not in raw
        for secret in (
            "fixture-cookie-secret",
            "fixture-session-epoch",
            "fixture-password",
        )
    )


async def test_api_failure_still_saves_outputs_without_response_body(tmp_path):
    directory = tmp_path / "failed"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            503,
            json={
                "error": {
                    "code": "DEPENDENCY_UNAVAILABLE",
                    "message": "fixture-private-diagnostic",
                }
            },
        )
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1"
    ) as client:
        await measure_v2.run(client, query="환율", directory=directory)
    raw = (directory / "measure_v2.json").read_text(encoding="utf-8")
    assert json.loads(raw)["error"] == {"status": 503, "code": "DEPENDENCY_UNAVAILABLE"}
    assert "fixture-private-diagnostic" not in raw
    with (directory / "measure_v2.csv").open(encoding="utf-8", newline="") as file:
        (row,) = csv.DictReader(file)
    assert row["outcome"] == "api_error" and row["job_id"] == ""
