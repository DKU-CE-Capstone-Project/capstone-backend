"""Explicit real-provider HTTP check; saves public evidence, never cookies."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx


async def verify(args) -> None:
    target = urlsplit(args.base_url)
    if target.scheme != "http" or target.hostname != "127.0.0.1":
        raise ValueError("Only the isolated loopback API may be tested")
    record = {"checked_at": datetime.now(UTC).isoformat(), "query": args.query,
              "base_url": args.base_url, "external_scope": ["NAVER search/category/image", "OpenAI Decisions"],
              "report_generation": "not_requested", "steps": []}
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180, trust_env=False) as client:
        async def request(method, path, expected=200, **kwargs):
            started = time.monotonic()
            response = await client.request(method, path, **kwargs)
            step = {"method": method, "path": path, "status": response.status_code,
                    "elapsed_ms": round((time.monotonic() - started) * 1000)}
            record["steps"].append(step)
            print(json.dumps(step, ensure_ascii=False), flush=True)
            if response.status_code != expected:
                # v2 errors are sanitized; never copy request headers/cookies.
                raise RuntimeError(f"Unexpected API status: {response.status_code}; {response.text[:400]}")
            return response

        ready = (await request("GET", "/ready")).json()
        assert ready["mongodb"] == "on" and ready["redis"] == "on"
        session_response = await request("GET", "/api/v2/session")
        assert "httponly" in session_response.headers.get("set-cookie", "").lower()
        assert not {"sid", "session_id"} & session_response.json().keys()
        record["session"] = {"httponly": True, "prototype_tier": session_response.json()["prototype_tier"]}
        created = (await request("POST", "/api/v2/maps", expected=201, json={"query": args.query})).json()
        record["initial_map"] = created
        assert created["state"] == "active" and created["nodes"], "No eligible real articles; choose another query"
        assert created["initial_selection"]["requested"] == 4
        assert created["initial_selection"]["returned"] <= 4
        assert all("fixture" not in n["news"]["title"].lower() and n["news"]["source_url"] for n in created["nodes"])
        path = f"/api/v2/maps/{created['map_id']}"
        restored = (await request("GET", path)).json()
        assert restored["nodes"] == created["nodes"] and restored["revision"] == created["revision"]
        parent = next((node for node in restored["nodes"] if node["level"] == 1), None)
        if parent is not None:
            expanded = (await request("POST", path + "/branches", json={
                "expected_revision": restored["revision"], "parent_node_id": parent["node_id"], "mode": "expand"})).json()
            branch = next(branch for branch in expanded["branches"] if branch["parent_node_id"] == parent["node_id"])
            assert branch["last_attempt"]["requested"] == 3
            assert len(branch["child_node_ids"]) <= 3
            ids = [node["news_id"] for node in expanded["nodes"]]
            assert len(ids) == len(set(ids))
            record["expanded_map"] = expanded
            record["branch_selection"] = branch["last_attempt"]
        else:
            record["branch_selection"] = {"skipped": "No eligible first-level article"}
        async with httpx.AsyncClient(base_url=args.base_url, timeout=10, trust_env=False) as other:
            assert (await other.get("/api/v2/session")).status_code == 200
            assert (await other.get(path)).status_code == 404
            await other.delete("/api/v2/session")
        record["cross_session_denied"] = True
        await request("DELETE", path, expected=204)
        await request("DELETE", "/api/v2/session", expected=204)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "passed", "initial": created["initial_selection"],
                      "branch": record["branch_selection"], "artifact": str(args.output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8026")
    parser.add_argument("--query", default="국제유가")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(verify(parser.parse_args()))


if __name__ == "__main__":
    main()
