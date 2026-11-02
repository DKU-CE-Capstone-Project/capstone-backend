"""Best-effort, allowlisted usage events. No request bodies or credentials."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps

_context: ContextVar[dict | None] = ContextVar("usage_context", default=None)
_write_lock = threading.Lock()
_ID = re.compile(r"[a-fA-F0-9-]{16,64}\Z")
_FIELDS = {
    "provider",
    "model",
    "operation",
    "service_tier",
    "outcome",
    "http_status",
    "duration_ms",
    "input_tokens",
    "counted_input_tokens",
    "input_token_source",
    "token_count_status",
    "token_count_requests",
    "token_count_duration_ms",
    "embedding_items",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_tokens",
    "thought_tokens",
    "total_tokens",
    "route",
    "method",
    "attempt",
    "error_type",
}


def current_request_id() -> str:
    return (_context.get() or {}).get("request_id", "")


@contextmanager
def usage_context(**values):
    allowed = {}
    for key in ("request_id", "job_id", "run_id"):
        value = values.get(key)
        if isinstance(value, str) and _ID.fullmatch(value):
            allowed[key] = value
    if "stage" in values:
        allowed["stage"] = values["stage"]
    token = _context.set({**(_context.get() or {}), **allowed})
    try:
        yield
    finally:
        _context.reset(token)


def stage(name):
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with usage_context(stage=name):
                return await function(*args, **kwargs)

        return wrapped

    return decorate


def emit(kind: str, **fields) -> None:
    """Never make application work fail when telemetry output is unavailable."""
    try:
        event = {
            "event_type": "econmind_usage",
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": kind,
            **(_context.get() or {}),
            **{k: v for k, v in fields.items() if k in _FIELDS},
        }
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with _write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    except Exception:  # noqa: BLE001, S110 — telemetry is best effort
        pass


def _value(source, name):
    value = (
        source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    )
    return value if type(value) is int and value >= 0 else None


def token_usage(source, provider: str) -> dict:
    # Gemini prompt tokens already include cached input. Keep subsets separate.
    names = {
        "gemini": {
            "input_tokens": "prompt_token_count",
            "output_tokens": "candidates_token_count",
            "cached_input_tokens": "cached_content_token_count",
            "thought_tokens": "thoughts_token_count",
            "total_tokens": "total_token_count",
        },
        "anthropic": {
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "cached_input_tokens": "cache_read_input_tokens",
            "cache_creation_tokens": "cache_creation_input_tokens",
        },
        "openai": {
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "total_tokens": "total_tokens",
        },
    }
    return {
        key: _value(source, value) for key, value in names.get(provider, {}).items()
    }


class observe:
    """One HTTP attempt or SDK invocation, including timeout/cancellation.

    SDK internal retries are deliberately not reported as observed HTTP counts.
    """

    def __init__(self, provider, operation, *, model="", kind="sdk_call", **fields):
        self.fields = dict(
            provider=provider, operation=operation, model=model, **fields
        )
        self.kind = kind

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def response(self, response, *, usage=None):
        status = getattr(response, "status_code", None)
        self.fields.update(
            http_status=status,
            outcome="error" if status and status >= 400 else "success",
        )
        self.fields.update(token_usage(usage, self.fields["provider"]))

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.fields["outcome"] = (
                "cancelled" if issubclass(exc_type, asyncio.CancelledError) else "error"
            )
            self.fields["error_type"] = exc_type.__name__
        self.fields.setdefault("outcome", "success")
        emit(
            self.kind,
            **self.fields,
            duration_ms=round((time.monotonic() - self.started) * 1000, 2),
        )


class UsageMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(
            ("/api/", "/jobs", "/analyze")
        ):
            return await self.app(scope, receive, send)
        status = 500

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        started = time.monotonic()
        with usage_context(request_id=uuid.uuid4().hex):
            try:
                await self.app(scope, receive, capture)
            finally:
                # New FastAPI versions retain the original relative route in
                # scope["route"]; the effective context includes router prefixes.
                effective = scope.get("fastapi", {}).get("effective_route_context")
                route = getattr(effective, "path_format", None) or getattr(
                    scope.get("route"), "path_format", "unmatched"
                )
                emit(
                    "api_request",
                    route=route,
                    method=scope["method"],
                    http_status=status,
                    outcome="error" if status >= 400 else "success",
                    duration_ms=round((time.monotonic() - started) * 1000, 2),
                )
