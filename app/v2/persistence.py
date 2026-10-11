"""Immutable v2 evidence/results; production MongoDB failures are never success."""
from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo.errors import DuplicateKeyError

from app import database
from app.config import settings
from app.v2.errors import V2Error

_memory: dict[str, dict[str, dict]] = {"news_bodies": {}, "reports": {}, "strategies": {}}
_indexed: set[tuple[str, str]] = set()


def now() -> datetime:
    return datetime.now(UTC)


def iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = database._parse_dt(value)
    if isinstance(value, datetime):
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return None


def _db():
    db = database._get_db()
    if db is None:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 결과 저장소를 사용할 수 없습니다.", True)
    return db


async def ready() -> None:
    if settings.v2_state_memory and not settings.session_store_required:
        return
    db = _db()
    try:
        await asyncio.wait_for(db.command("ping"), timeout=3)
        identity = (settings.mongodb_uri, settings.mongodb_db_name)
        if identity not in _indexed:
            await ensure_indexes()
            _indexed.add(identity)
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 결과 저장소를 사용할 수 없습니다.", True) from exc


async def ensure_indexes() -> None:
    """Add required indexes; collisions must stop v2 readiness, not be ignored."""
    if settings.v2_state_memory and not settings.session_store_required:
        return
    db = _db()
    specs = [
        ("news_bodies", [("body_id", 1)], {"name": "uniq_news_bodies_body_id", "unique": True}),
        ("news_bodies", [("news_id", 1), ("source_url", 1), ("extractor_version", 1), ("extracted_at", -1)], {"name": "idx_news_bodies_latest"}),
        ("reports", [("report_id", 1)], {"name": "uniq_reports_report_id", "unique": True, "sparse": True}),
        ("reports", [("generation_key", 1)], {"name": "uniq_reports_v2_generation", "unique": True, "partialFilterExpression": {"schema_version": 2, "reuse_eligible": True}}),
        ("reports", [("requested_news_ids", 1), ("created_at", -1)], {"name": "idx_reports_requested_created"}),
        ("strategies", [("strategy_id", 1)], {"name": "uniq_strategies_strategy_id", "unique": True, "sparse": True}),
        ("strategies", [("generation_key", 1)], {"name": "uniq_strategies_v2_generation", "unique": True, "partialFilterExpression": {"schema_version": 2}}),
        ("strategies", [("report_id", 1), ("created_at", -1)], {"name": "idx_strategies_report_created"}),
    ]
    try:
        for collection, keys, options in specs:
            await db[collection].create_index(keys, **options)
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "v2 저장 인덱스 검증이 필요합니다.", False) from exc


async def _find(collection: str, query: dict, *, sort: list | None = None) -> dict | None:
    if settings.v2_state_memory and not settings.session_store_required:
        records = [doc for doc in _memory[collection].values() if all(doc.get(k) == v for k, v in query.items())]
        if sort:
            records.sort(key=lambda d: d.get(sort[0][0], now()), reverse=sort[0][1] == -1)
        return deepcopy(records[0]) if records else None
    try:
        return await _db()[collection].find_one(query, {"_id": 0}, sort=sort)
    except V2Error:
        raise
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 결과를 조회할 수 없습니다.", True) from exc


async def body(news_id: str, source_url: str, extractor_version: str) -> dict | None:
    doc = await _find("news_bodies", {"news_id": news_id, "source_url": source_url, "extractor_version": extractor_version}, sort=[("extracted_at", -1)])
    if doc:
        extracted = database._parse_dt(doc.get("extracted_at"))
        if extracted is not None:
            extracted = extracted.replace(tzinfo=UTC) if extracted.tzinfo is None else extracted
            if extracted >= now() - timedelta(seconds=settings.v2_body_cache_seconds):
                return doc
    return None


async def report(report_id: str) -> dict | None:
    return await _find("reports", {"report_id": report_id, "schema_version": 2})


async def strategy(strategy_id: str) -> dict | None:
    return await _find("strategies", {"strategy_id": strategy_id, "schema_version": 2})


async def reusable_report(generation_key: str) -> dict | None:
    return await _find("reports", {"generation_key": generation_key, "schema_version": 2, "reuse_eligible": True})


async def reusable_strategy(generation_key: str) -> dict | None:
    return await _find("strategies", {"generation_key": generation_key, "schema_version": 2})


def to_bson(doc: dict, collection: str) -> dict:
    out = deepcopy(doc)
    for key in ("created_at", "updated_at", "extracted_at"):
        if key in out:
            parsed = database._parse_dt(out[key])
            if parsed is None:
                raise ValueError(f"Invalid {key}")
            out[key] = parsed
    if collection == "reports":
        for evidence in out["used_evidence"]:
            evidence["published_at"] = database._parse_dt(evidence.get("published_at"))
    return out


async def _insert(collection: str, doc: dict, id_field: str) -> tuple[dict, bool]:
    payload = to_bson(doc, collection)
    if settings.v2_state_memory and not settings.session_store_required:
        if payload[id_field] in _memory[collection]:
            raise ValueError("Immutable result ID already exists")
        eligible = collection == "strategies" or payload.get("reuse_eligible")
        if eligible and collection != "news_bodies":
            existing = await _find(collection, {"generation_key": payload["generation_key"], "schema_version": 2, **({"reuse_eligible": True} if collection == "reports" else {})})
            if existing:
                return existing, True
        _memory[collection][payload[id_field]] = deepcopy(payload)
        return payload, False
    try:
        await _db()[collection].insert_one(deepcopy(payload))
        return payload, False
    except DuplicateKeyError as exc:
        # ID or unrelated index collisions are data integrity errors. Only the
        # generation index's collision can resolve to the same saved result.
        detail = exc.details or {}
        if detail.get("keyPattern") == {"generation_key": 1} and collection != "news_bodies":
            query = {"generation_key": payload["generation_key"], "schema_version": 2}
            if collection == "reports":
                query["reuse_eligible"] = True
            existing = await _find(collection, query)
            if existing:
                return existing, True
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "저장 결과의 식별 충돌을 확인해야 합니다.") from exc
    except V2Error:
        raise
    except Exception as exc:
        raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 결과를 저장하지 못했습니다.", True) from exc


async def save_body(doc: dict) -> dict:
    if not doc.get("content") or hashlib.sha256(doc["content"].encode()).hexdigest() != doc.get("content_hash"):
        raise ValueError("Body content hash does not match")
    payload, _ = await _insert("news_bodies", doc, "body_id")
    if not (settings.v2_state_memory and not settings.session_store_required):
        try:
            reference = {k: payload[k] for k in ("body_id", "content_hash", "source_url", "extractor_version", "extracted_at")}
            updated = await _db()["news"].update_one({"news_id": payload["news_id"]}, {"$set": {"body_ref": reference, "content": payload["content"], "content_source_url": payload["source_url"], "updated_at": now()}})
            if updated.matched_count != 1:
                raise ValueError("Body article reference missing")
        except Exception as exc:
            raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "기사 본문 참조를 저장하지 못했습니다.", True) from exc
    return payload


async def save_report(doc: dict) -> tuple[dict, bool]:
    requested = doc["requested_news_ids"]
    used = [e["news_id"] for e in doc["used_evidence"]]
    failed = [e["news_id"] for e in doc["failed_evidence"]]
    if not used or len(set(requested)) != len(requested) or len(set(used + failed)) != len(used + failed) or set(requested) != set(used + failed):
        raise ValueError("Invalid report evidence partition")
    if doc["coverage"] != {"requested_count": len(requested), "used_count": len(used), "failed_count": len(failed)}:
        raise ValueError("Invalid report coverage")
    if any(e["input_char_count"] != len(e["input_excerpt"]) for e in doc["used_evidence"]):
        raise ValueError("Invalid input evidence length")
    if any(not set(s["evidence_news_ids"]).issubset(used) for s in doc["stock_impacts"]):
        raise ValueError("Invalid stock evidence")
    if doc["reuse_eligible"] and failed:
        raise ValueError("Partial evidence is not reusable")
    for evidence in doc["used_evidence"]:
        body_doc = await _find("news_bodies", {"body_id": evidence["body_id"], "schema_version": 2})
        if not body_doc or any(body_doc.get(k) != evidence.get(k) for k in ("news_id", "content_hash", "source_url")):
            raise ValueError("Report body reference does not match")
        if not body_doc["content"].startswith(evidence["input_excerpt"]) or evidence["truncated"] != (len(evidence["input_excerpt"]) < len(body_doc["content"])):
            raise ValueError("Report excerpt is not a recorded body prefix")
    return await _insert("reports", doc, "report_id")


async def save_strategy(doc: dict) -> tuple[dict, bool]:
    parent = await report(doc["report_id"])
    if not parent or parent["generation_key"] != doc["report_generation_key"]:
        raise ValueError("Strategy report reference does not match")
    return await _insert("strategies", doc, "strategy_id")


def public_report(doc: dict) -> dict:
    fields = ("schema_version", "report_id", "title", "event_summary", "event_analysis", "market_impact", "stock_impacts", "risk_factors", "requested_news_ids", "used_evidence", "failed_evidence", "coverage", "rag_evidence", "verification", "created_at")
    result = deepcopy({key: doc[key] for key in fields})
    result["created_at"] = iso(result["created_at"])
    for evidence in result["used_evidence"]:
        evidence.pop("input_excerpt", None)
        evidence.pop("extractor_version", None)
        evidence["published_at"] = iso(evidence.get("published_at"))
    return result


def public_strategy(doc: dict) -> dict:
    fields = ("schema_version", "strategy_id", "report_id", "risk_level", "period", "strategy_summary", "risk_warning", "strategy_items", "created_at")
    result = deepcopy({key: doc[key] for key in fields})
    result["created_at"] = iso(result["created_at"])
    return result
