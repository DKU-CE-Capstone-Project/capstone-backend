"""Run real NAVER/Decision exploration with isolated local persistence."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=8026)
    args = parser.parse_args()
    credentials = dotenv_values(args.credentials_file)
    required = {
        "NAVER_CLIENT_ID": credentials.get("NAVER_CLIENT_ID"),
        "NAVER_CLIENT_SECRET": credentials.get("NAVER_CLIENT_SECRET"),
        "OPENAI_API_KEY": credentials.get("OPENAI_API_KEY") or credentials.get("OAI_KEY"),
    }
    missing = [name for name, value in required.items() if not value or not value.strip()]
    if missing:
        parser.error("Missing credentials: " + ", ".join(missing))
    # Read only the three authorized provider keys. Never inherit the credential
    # file's DB endpoints or other provider keys into this verification profile.
    os.environ.update({name: value.strip() for name, value in required.items()})
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
        "PROTOTYPE_TIER": "FREE", "PAID_DEMO_ENABLED": "false",
        "CORS_ORIGINS": "http://127.0.0.1:5196,http://localhost:5196,http://127.0.0.1:5173,http://localhost:5173",
        "USE_LLM_METADATA": "false", "USE_LLM_SUMMARIES": "false",
        "USE_RAG": "false", "USE_CRITIC": "false",
        "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "DIFFBOT_TOKEN": "", "DIFFBOT_API_KEY": "",
    })
    sys.path.insert(0, str(ROOT))
    import uvicorn

    print("[local-v2] real NAVER + Decision; loopback MongoDB/Redis/NATS; FREE")
    print("[local-v2] metadata uses rules; report generation is outside this profile")
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
