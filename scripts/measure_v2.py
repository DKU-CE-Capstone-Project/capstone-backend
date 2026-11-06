"""Measure one v2 report job and write one CSV row plus public JSON evidence."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

POLL_INTERVAL = 1.0
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class ApiError(Exception):
    def __init__(self, status: int, code: str | None):
        self.status = status
        self.code = code


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


async def measure(
    client: httpx.AsyncClient, record: dict, max_expansions: int, poll_timeout: float
) -> None:
    async def request(method, path, expected=200, **kwargs):
        step = {"method": method, "path": path, "started_at": utc_now()}
        record["requests"].append(step)
        started = time.monotonic()
        try:
            response = await client.request(method, path, **kwargs)
        finally:
            step["elapsed_seconds"] = round(time.monotonic() - started, 3)
        step["status"] = response.status_code
        if response.status_code != expected:
            code = None
            try:
                candidate = response.json()["error"]["code"]
                if isinstance(candidate, str) and re.fullmatch(
                    r"[A-Z][A-Z0-9_]{0,79}", candidate
                ):
                    code = candidate
            except (ValueError, KeyError, TypeError):
                pass
            step["error_code"] = code
            raise ApiError(response.status_code, code)
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Expected a v2 JSON object")
        return body

    session = await request("GET", "/api/v2/session")
    capabilities = session["capabilities"]
    record["session"] = {
        "prototype_tier": session["prototype_tier"],
        "capabilities": capabilities,
    }
    target = record["target_articles"]
    limit = min(target, capabilities["max_report_articles"])
    record["effective_target_articles"] = limit
    snapshot = await request(
        "POST", "/api/v2/maps", 201, json={"query": record["query"]}
    )
    map_id = snapshot["map_id"]
    record["map_id"] = map_id
    record["maps"].append(snapshot)
    path = f"/api/v2/maps/{map_id}/branches"
    available = list(dict.fromkeys(node["news_id"] for node in snapshot["nodes"]))
    record["available_articles"] = len(available)
    stop = "expansion_limit"

    for _ in range(max_expansions):
        if len(available) >= limit:
            break
        if len(snapshot["nodes"]) >= capabilities["max_map_nodes"]:
            stop = "map_limit"
            break
        parent = next(
            (
                node
                for node in snapshot["nodes"]
                if not node["expanded"]
                and node["level"] < capabilities["max_map_depth"]
            ),
            None,
        )
        if parent is None:
            stop = "no_expandable_nodes"
            break
        try:
            snapshot = await request(
                "POST",
                path,
                json={
                    "expected_revision": snapshot["revision"],
                    "parent_node_id": parent["node_id"],
                    "mode": "expand",
                },
            )
        except ApiError as exc:
            if exc.status != 422 or exc.code != "MAP_LIMIT_REACHED":
                raise
            stop = "map_limit"
            break
        record["maps"].append(snapshot)
        available = list(dict.fromkeys(node["news_id"] for node in snapshot["nodes"]))
        record["available_articles"] = len(available)

    if len(available) >= limit:
        stop = "session_limit" if limit < target else "target_reached"
    selected = available[:limit]
    record.update(
        selected_news_ids=selected,
        selection_shortfall=target - len(selected),
        selection_stop=stop,
    )
    if not selected:
        record["outcome"] = "no_articles"
        return

    request_id = str(uuid.uuid4())
    job_started = time.monotonic()
    # Report admission sets selected_news_ids atomically on the map.
    job = await request(
        "POST",
        "/api/v2/report-jobs",
        202,
        headers={"Idempotency-Key": request_id},
        json={
            "map_id": map_id,
            "expected_revision": snapshot["revision"],
            "client_request_id": request_id,
            "news_ids": selected,
            "strategy_options": record["strategy_options"],
        },
    )

    def observe(current_job):
        elapsed = round(time.monotonic() - job_started, 3)
        sample = {"observed_at": utc_now(), "elapsed_seconds": elapsed}
        changes = record["stage_changes"]
        if not changes or changes[-1]["stage"] != current_job["stage"]:
            changes.append(
                sample
                | {"stage": current_job["stage"], "status": current_job["status"]}
            )
        record["job_observations"].append(sample | {"job": current_job})

    observe(job)
    next_poll = time.monotonic() + POLL_INTERVAL
    try:
        async with asyncio.timeout(poll_timeout):
            while job["status"] not in TERMINAL_STATUSES:
                await asyncio.sleep(max(0, next_poll - time.monotonic()))
                next_poll = time.monotonic() + POLL_INTERVAL
                job = await request("GET", f"/api/v2/report-jobs/{job['job_id']}")
                observe(job)
    except TimeoutError:
        record["outcome"] = "poll_timeout"
        return

    record["outcome"] = job["status"]
    result = job.get("result") or {}
    if result.get("report_id"):
        record["report"] = await request(
            "GET", f"/api/v2/reports/{result['report_id']}"
        )
    if result.get("strategy_id"):
        record["strategy"] = await request(
            "GET", f"/api/v2/strategies/{result['strategy_id']}"
        )


def save_results(directory: Path, record: dict) -> None:
    final = record["job_observations"][-1] if record["job_observations"] else {}
    job = final.get("job", {})
    result = job.get("result") or {}
    report = record.get("report", {})
    coverage = report.get("coverage", {})
    error = record.get("error") or job.get("error") or {}
    row = {
        "query": record["query"],
        "outcome": record["outcome"],
        "prototype_tier": record.get("session", {}).get("prototype_tier"),
        "target_articles": record["target_articles"],
        "effective_target_articles": record.get("effective_target_articles"),
        "available_articles": record.get("available_articles"),
        "selected_articles": len(record.get("selected_news_ids", [])),
        "selection_shortfall": record.get("selection_shortfall"),
        "selection_stop": record.get("selection_stop"),
        "map_id": record.get("map_id"),
        "job_id": job.get("job_id"),
        "job_status": job.get("status"),
        "last_stage": job.get("stage"),
        "last_observed_elapsed_seconds": final.get("elapsed_seconds"),
        "report_id": result.get("report_id"),
        "strategy_id": result.get("strategy_id"),
        "requested_count": coverage.get("requested_count"),
        "used_count": coverage.get("used_count"),
        "failed_count": coverage.get("failed_count"),
        "failed_evidence_codes": json.dumps(
            [item["code"] for item in report.get("failed_evidence", [])]
        ),
        "report_reused": result.get("report_reused"),
        "strategy_reused": result.get("strategy_reused"),
        "error_code": error.get("code"),
    }
    csv_path = directory / "measure_v2.csv"
    json_path = directory / "measure_v2.json"
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        json_path.write_text(payload, encoding="utf-8")
    except OSError:
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise


async def run(
    client: httpx.AsyncClient,
    *,
    query: str,
    directory: Path,
    target_articles: int = 10,
    max_expansions: int = 6,
    poll_timeout: float = 1800,
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "started_at": utc_now(),
        "base_url": str(client.base_url),
        "outcome": "error",
        "query": query,
        "target_articles": target_articles,
        "max_expansions": max_expansions,
        "poll_interval_seconds": POLL_INTERVAL,
        "poll_timeout_seconds": poll_timeout,
        "strategy_options": {"risk_level": "medium", "period": "short"},
        "requests": [],
        "maps": [],
        "job_observations": [],
        "stage_changes": [],
    }
    try:
        await measure(client, record, max_expansions, poll_timeout)
    except ApiError as exc:
        record.update(
            outcome="api_error", error={"status": exc.status, "code": exc.code}
        )
    except httpx.RequestError as exc:
        record.update(outcome="transport_error", error={"code": type(exc).__name__})
    except (KeyError, TypeError, ValueError):
        record.update(outcome="invalid_response", error={"code": "INVALID_V2_RESPONSE"})
    except asyncio.CancelledError:
        record["outcome"] = "interrupted"
        raise
    finally:
        record["finished_at"] = utc_now()
        save_results(directory, record)
    return record


def base_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if (
            url.hostname
            and url.username is None
            and url.password is None
            and url.path in ("", "/")
            and not url.query
            and not url.fragment
            and (url.port is None or url.port > 0)
            and (
                url.scheme == "https"
                or (
                    url.scheme == "http"
                    and url.hostname in {"127.0.0.1", "::1", "localhost"}
                )
            )
        ):
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        "Use an HTTPS origin or loopback HTTP, without credentials"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "For team Basic Auth, set ECONMIND_API_USERNAME and ECONMIND_API_PASSWORD."
        ),
    )
    parser.add_argument("--query", required=True, help="News-map search query")
    parser.add_argument("--base-url", type=base_url, default="http://127.0.0.1:8026")
    parser.add_argument("--articles", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--max-expansions", type=int, default=6)
    parser.add_argument("--poll-timeout", type=int, default=1800, metavar="SECONDS")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for CSV and JSON"
    )
    args = parser.parse_args()
    query = args.query.strip()
    if not 1 <= len(query) <= 100 or args.max_expansions < 0 or args.poll_timeout < 1:
        parser.error(
            "Query must be 1–100 characters, expansions >= 0 and poll timeout >= 1"
        )
    user = os.getenv("ECONMIND_API_USERNAME")
    password = os.getenv("ECONMIND_API_PASSWORD")
    if bool(user) != bool(password):
        parser.error("Set both Basic Auth environment variables")

    async def execute():
        async with httpx.AsyncClient(
            base_url=args.base_url,
            auth=(user, password) if user and password else None,
            timeout=180,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            return await run(
                client,
                query=query,
                directory=args.output_dir,
                target_articles=args.articles,
                max_expansions=args.max_expansions,
                poll_timeout=args.poll_timeout,
            )

    try:
        record = asyncio.run(execute())
    except FileExistsError:
        parser.error("The output directory already exists; choose a new one")
    except OSError:
        parser.exit(1, "Could not write the measurement output files\n")
    except KeyboardInterrupt:
        return 130
    print(
        json.dumps(
            {"outcome": record["outcome"], "output_dir": str(args.output_dir)},
            ensure_ascii=False,
        )
    )
    return 0 if record["outcome"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
