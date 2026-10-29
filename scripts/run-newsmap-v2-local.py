"""Run real news-map API or report worker with isolated local persistence."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--tier", choices=("FREE", "PAID"), default="PAID", help="Local prototype tier (default: PAID, up to 10 selected articles)")
    parser.add_argument("--with-reports", action="store_true", help="Enable real Diffbot/Gemini reports and critic (run --worker separately)")
    parser.add_argument("--worker", action="store_true", help="Run the report worker with the same isolated report profile")
    args = parser.parse_args()
    credentials = dotenv_values(args.credentials_file)
    required = {
        "NAVER_CLIENT_ID": credentials.get("NAVER_CLIENT_ID"),
        "NAVER_CLIENT_SECRET": credentials.get("NAVER_CLIENT_SECRET"),
        "OPENAI_API_KEY": credentials.get("OPENAI_API_KEY") or credentials.get("OAI_KEY"),
        "GOOGLE_API_KEY": credentials.get("GOOGLE_API_KEY"),
    }
    reports_enabled = args.with_reports or args.worker
    if reports_enabled:
        required.update({
            "DIFFBOT_TOKEN": credentials.get("DIFFBOT_API_KEY") or credentials.get("DIFFBOT_TOKEN"),
        })
    missing = [name for name, value in required.items() if not value or not value.strip()]
    if missing:
        parser.error("Missing credentials: " + ", ".join(missing))
    # Copy only explicitly selected provider keys, never DB endpoints from .env.
    os.environ.update({
        "NEWS_PROVIDER": "naver", "NEWS_MAP_SELECTOR": "decision",
        "USE_MOCK_NEWS": "false", "DEMO_MODE": "false",
        "MONGODB_URI": "mongodb://127.0.0.1:27218",
        "MONGODB_DB_NAME": "econmind_v2_local_test",
        "USE_MONGODB": "true", "MONGODB_REQUIRED": "true",
        "REDIS_URL": "redis://127.0.0.1:6388/0",
        "NATS_URL": "nats://127.0.0.1:4233",
        "SESSION_STORE_REQUIRED": "true", "V2_STATE_MEMORY": "false",
        "SESSION_COOKIE_SECURE": "false", "SESSION_COOKIE_SAMESITE": "lax",
        "PROTOTYPE_TIER": args.tier, "PAID_DEMO_ENABLED": "true" if args.tier == "PAID" else "false",
        "V2_BODY_CONCURRENCY": "1", "V2_BODY_BATCH_TIMEOUT_SECONDS": "60", "V2_BODY_REQUEST_INTERVAL_SECONDS": "1",
        "CORS_ORIGINS": "http://127.0.0.1:5196,http://localhost:5196,http://127.0.0.1:5173,http://localhost:5173",
        "USE_LLM_METADATA": "false", "USE_LLM_SUMMARIES": "false",
        "USE_RAG": "false", "USE_CRITIC": "true" if reports_enabled else "false",
        "LLM_PROVIDER": "gemini",
        "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "DIFFBOT_TOKEN": "", "DIFFBOT_API_KEY": "",
        "DIFFBOT_TOKEN_SECONDARY": (credentials.get("DIFFBOT_TOKEN_SECONDARY") or "").strip() if reports_enabled else "",
    })
    os.environ.update({name: value.strip() for name, value in required.items()})
    sys.path.insert(0, str(ROOT))
    if args.worker:
        from app.v2.report_worker import main as worker_main

        print("[local-v2] report worker: real Diffbot/Gemini + critic; local RAG disabled")
        asyncio.run(worker_main())
        return
    import uvicorn

    print(f"[local-v2] real NAVER + Decision; loopback MongoDB/Redis/NATS; {args.tier}")
    print("[local-v2] real Diffbot/Gemini reports + critic; start --worker separately" if reports_enabled else "[local-v2] metadata uses rules; report generation is outside this profile")
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
