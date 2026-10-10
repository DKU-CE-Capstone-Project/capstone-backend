"""
MongoDB Atlas 영속화 + Atlas Vector Search.

발표 설계서의 MongoDB/Vector Search 단계 구현. 동시에 AI 에이전트의
RAG 그라운딩(의미 기반 유사 뉴스 검색)을 담당하는 '근거 계층'.

설계 원칙: settings.use_mongodb=False 또는 연결 실패 시 모든 함수가
graceful no-op/빈값을 반환 → 기존 in-memory 흐름을 절대 깨지 않는다.
(in-memory store는 L1 캐시로 유지, MongoDB는 write-through 영속화 + 벡터검색 read 경로)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.config import settings

# 컬렉션 이름 — Notion「구조크」의 최종 8개 목록을 따른다.
# (구 이름 articles 는 설계에 없던 것이라 news 로 교체했다. deploy/mongo-init 가
#  만드는 컬렉션·인덱스·validator 와 이름이 일치해야 한다.)
NEWS = "news"
NEWS_ANALYSIS = "news_analysis"
NEWS_RELATIONS = "news_relations"
REPORTS = "reports"
STRATEGIES = "strategies"
JOBS = "jobs"
SELECTIONS = "selections"  # 설계 외 — 프론트 뉴스 선택 저장용

VECTOR_INDEX = "vector_index"
# app/config.py 의 embedding_model 기본값은 gemini-embedding-001 이고,
# app/agents/llm.py 의 embed() 가 EmbedContentConfig(output_dimensionality=768) 로 요청한다.
EMBED_DIM = 768  # gemini-embedding-001, 768차원 요청

_client = None
_db = None
_disabled = False  # 연결 실패 1회 후 재시도 폭주 방지


class DatabasePersistenceError(RuntimeError):
    """Required persistence failed; no credentials/raw driver errors leave the API."""


async def ping() -> bool:
    db = _get_db()
    if db is None:
        return False
    try:
        await asyncio.wait_for(db.command("ping"), timeout=3)
        return True
    except Exception:  # noqa: BLE001 — readiness must report dependency failure
        return False


def _require_available(db) -> None:
    if settings.mongodb_required and db is None:
        raise DatabasePersistenceError("MongoDB is unavailable")


# ── 설계 스키마 매핑 ────────────────────────────────────────────────────────
# in-memory store / API 응답이 쓰는 dict 와 「구조크」의 news 문서는 형태가 다르다.
#   in-memory : {title, url, source(문자열), published_at(ISO 문자열), description, ...}
#   설계 news : {title, summary, url, source{name,domain}, published_at(date), ...}
# 응답 스키마(app/schemas.py)를 그대로 두어야 프론트(apiAdapter.ts)가 안 깨지므로,
# in-memory 형태는 손대지 않고 **Mongo 에 넣을 때만** 아래 매퍼로 변환한다.

def _parse_dt(value: Any) -> "datetime | None":
    """ISO 문자열 → BSON date 로 들어갈 datetime. 실패하면 None.

    설계가 datetime 을 요구하고 mongo-init 의 validator 가 bsonType:"date" 로 강제한다.
    문자열을 그대로 넣으면 code 121 로 거부되므로 여기서 반드시 변환한다.
    """
    if isinstance(value, datetime):
        return value
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    # GDELT seendate 원형이 남아 있는 경우: 20260604T010203Z → 2026-06-04T01:02:03Z
    if len(text) == 16 and text[8] == "T" and text.endswith("Z"):
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}T{text[9:11]}:{text[11:13]}:{text[13:15]}Z"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _split_source(art: dict[str, Any]) -> dict[str, str]:
    """설계의 source = {name, domain}. 현재 in-memory 는 문자열 하나뿐이다.

    NewsAPI 경로는 언론사명("Reuters"), GDELT 경로는 도메인("reuters.com")이 들어온다.
    도메인은 url 에서 뽑는 쪽이 확실하므로 그렇게 하고, name 은 있는 값을 쓴다.
    """
    raw = art.get("source")
    if isinstance(raw, dict):  # 이미 설계 형태면 그대로
        return {"name": raw.get("name", "") or "unknown", "domain": raw.get("domain", "")}
    domain = ""
    try:
        domain = urlparse(art.get("url") or "").hostname or ""
    except ValueError:
        domain = ""
    name = (raw or "").strip() if isinstance(raw, str) else ""
    return {"name": name or domain or "unknown", "domain": domain}


def _language_of(art: dict[str, Any]) -> str:
    """한글이 섞여 있으면 ko, 아니면 en. validator 가 ^[a-z]{2}$ 를 요구한다."""
    text = f"{art.get('title', '')}{art.get('description', '')}"
    return "ko" if any("\uac00" <= ch <= "\ud7a3" for ch in text) else "en"


def news_doc_from_article(art: dict[str, Any]) -> dict[str, Any]:
    """in-memory 기사 dict → 「구조크」news 문서."""
    now = datetime.now(timezone.utc)
    doc: dict[str, Any] = {
        "title": art.get("title", "") or "",
        "summary": art.get("summary") or art.get("description", "") or "",
        "url": art.get("url", "") or "",
        "source": _split_source(art),
        "published_at": _parse_dt(art.get("published_at")),
        "collected_at": now,
        "keywords": list(art.get("keywords") or []),
        "categories": list(art.get("categories") or []),
        # TODO(김성민): 종목 연관도 수치화 시 related_tickers 채우기
        "related_tickers": [],
        "status": "collected",
        "language": _language_of(art),
        "is_deleted": False,
        "created_at": now,
        "updated_at": now,
        # 앱 고유 필드 (validator 가 허용하도록 mongo-init 에 명시해 뒀다)
        "news_id": art.get("news_id", ""),
    }
    content = art.get("cleaned_content") or ""
    if content:
        doc["content"] = content
    if art.get("thumbnail_url"):
        doc["thumbnail_url"] = art["thumbnail_url"]
    if art.get("_search_keyword"):
        doc["_search_keyword"] = art["_search_keyword"]
        # Search-session membership keeps the original rank order after a restart.
        for key in ("_search_rank", "_search_end", "_searched_at"):
            if art.get(key):
                doc[key] = art[key]
    # Preserve extraction inputs/provenance so Mongo reloads can reuse valid results.
    for key in ("description", "title_original", "metadata_extraction", "_news_provider", "naver_url", "naver_categories", "content_source_url", "embedding", "embedding_metadata", "news_map_embedding"):
        if key in art:
            doc[key] = art[key]
    return doc


def _iso_z(value: Any) -> str:
    """BSON date → API 가 쓰는 ISO-8601 Z 문자열.

    pymongo/motor 는 BSON date 를 **naive** datetime(UTC 기준)으로 돌려준다.
    그대로 isoformat() 하면 오프셋이 없어 'Z' 가 빠지고, 프론트가 그걸 로컬시각으로
    해석해 시간이 밀린다. 메모리 경로가 내는 '...Z' 와 형식을 맞춘다.
    """
    if not isinstance(value, datetime):
        return ""
    dt = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def article_from_news_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """news 문서 → in-memory/API 형태. 읽기 경로를 붙일 때 이걸 쓴다."""
    source = doc.get("source") or {}
    return {
        "news_id": doc.get("news_id", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "source": source.get("name", "") if isinstance(source, dict) else (source or ""),
        "published_at": _iso_z(doc.get("published_at")),
        # Legacy/generated summaries remain summaries, never search descriptions.
        "description": doc.get("description", ""),
        "summary": doc.get("summary", ""),
        "thumbnail_url": doc.get("thumbnail_url", ""),
        "cleaned_content": doc.get("content", ""),
        "_search_keyword": doc.get("_search_keyword", ""),
        **{key: doc[key] for key in ("_search_rank", "_search_end", "_searched_at") if doc.get(key)},
        "title_original": doc.get("title_original", doc.get("title", "")),
        "keywords": list(doc.get("keywords") or []),
        "categories": list(doc.get("categories") or []),
        "metadata_extraction": doc.get("metadata_extraction") or {},
        "_news_provider": doc.get("_news_provider", ""),
        "naver_url": doc.get("naver_url", ""),
        "naver_categories": doc.get("naver_categories", []),
        "content_source_url": doc.get("content_source_url", ""),
        "embedding": doc.get("embedding", []),
        "embedding_metadata": doc.get("embedding_metadata"),
        "news_map_embedding": doc.get("news_map_embedding"),
    }


def _get_db():
    """motor DB 핸들 lazy 초기화. 비활성/실패 시 None."""
    global _client, _db, _disabled
    if _disabled or not settings.use_mongodb or not settings.mongodb_uri:
        return None
    if _db is not None:
        return _db
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )
        _db = _client[settings.mongodb_db_name]
        return _db
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] init failed → disabled: {type(exc).__name__}: {exc}")
        _disabled = True
        return None


def enabled() -> bool:
    return _get_db() is not None


# ── 인덱스 보장 (startup) ───────────────────────────────────────────────────

async def ensure_indexes() -> None:
    db = _get_db()
    if db is None:
        return
    # 1) 연결(ping) 실패만 치명적 → Mongo 비활성. 인덱스 실패는 비치명(레거시 데이터 충돌 등).
    try:
        await _client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        global _disabled
        print(f"[mongo] ping failed → disabled: {type(exc).__name__}: {exc}")
        _disabled = True
        return

    print("[mongo] connected")
    # 2) 인덱스는 개별 try (있으면 무시, 실패해도 계속) — upsert 동작엔 인덱스 불필요.
    #    이름·옵션을 deploy/mongo-init/03-indexes.js 와 똑같이 맞춰 뒀다.
    #    다르면 같은 키에 이름만 다른 인덱스라 IndexOptionsConflict 가 난다.
    #    (mongo-init 이 이미 만들었으면 여기선 전부 no-op)
    for coll, keys, opts in [
        (NEWS, [("url", 1)], {"name": "uniq_news_url", "unique": True}),
        (NEWS, [("news_id", 1)], {"name": "uniq_news_news_id", "unique": True, "sparse": True}),
        (NEWS, [("_search_keyword", 1), ("published_at", -1)], {"name": "idx_news_search_keyword"}),
        (REPORTS, [("report_id", 1)], {"name": "uniq_reports_report_id", "unique": True, "sparse": True}),
        (STRATEGIES, [("strategy_id", 1)], {"name": "uniq_strategies_strategy_id", "unique": True, "sparse": True}),
    ]:
        try:
            await db[coll].create_index(keys, **opts)
        except Exception as exc:  # noqa: BLE001
            print(f"[mongo] index {opts['name']} skip: {type(exc).__name__}: {str(exc)[:80]}")
    # 3) 벡터검색 인덱스 (RAG 핵심)
    await ensure_vector_index()


async def ensure_vector_index() -> None:
    """news.embedding 에 Atlas Vector Search 인덱스를 멱등 생성.

    deploy/mongo-init/04-vector-index.js 가 최초 기동 때 이미 만들지만,
    mongo-init 없이 뜬 DB(기존 볼륨 등)를 위해 여기서도 보장한다.
    """
    db = _get_db()
    if db is None:
        return
    try:
        existing = [i.get("name") async for i in db[NEWS].list_search_indexes()]
        if VECTOR_INDEX in existing:
            return
        from pymongo.operations import SearchIndexModel

        model = SearchIndexModel(
            name=VECTOR_INDEX,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": EMBED_DIM,
                        "similarity": "cosine",
                    }
                ]
            },
        )
        await db[NEWS].create_search_index(model=model)
        print(f"[mongo] vector index '{VECTOR_INDEX}' 생성 요청 (빌드 ~1분)")
    except Exception as exc:  # noqa: BLE001
        # 프로그램 생성 실패 시 Atlas UI 수동 생성으로 대체 (deploy/README 참고)
        print(f"[mongo] vector index 생성 skip: {type(exc).__name__}: {exc}")


# ── 저장(write-through) ─────────────────────────────────────────────────────

async def save_news(doc: dict[str, Any]) -> None:
    """news 문서 upsert. doc 은 news_doc_from_article() 이 만든 설계 형태여야 한다.

    중복 판정 키는 설계의 uniq_news_url 을 따라 url 이다(news_id 는 url 의 md5라 1:1).
    created_at/collected_at 은 $setOnInsert 로 최초 1회만 — 재수집할 때마다
    최초 수집 시각이 덮어써지면 안 된다.
    """
    db = _get_db()
    _require_available(db)
    if db is None or not doc.get("url"):
        return
    payload = dict(doc)
    on_insert = {k: payload.pop(k) for k in ("created_at", "collected_at") if k in payload}
    try:
        update: dict[str, Any] = {"$set": payload}
        if "embedding" not in payload:
            # An updated article with a failed refresh must not retain a stale RAG
            # vector in the database. This only affects the article being saved.
            update["$unset"] = {"embedding": "", "embedding_metadata": ""}
        if on_insert:
            update["$setOnInsert"] = on_insert
        await db[NEWS].update_one({"url": payload["url"]}, update, upsert=True)
    except Exception as exc:  # noqa: BLE001
        # validator 거부(code 121)도 여기로 온다 — 어느 필드가 걸렸는지 같이 찍는다
        if settings.mongodb_required:
            raise DatabasePersistenceError("News persistence failed") from exc
        detail = getattr(exc, "details", None) or {}
        print(f"[mongo] save_news skip: {type(exc).__name__}: {exc}")
        if detail.get("errInfo"):
            print(f"[mongo]   validator detail: {str(detail['errInfo'])[:300]}")


async def save_report(doc: dict[str, Any]) -> None:
    db = _get_db()
    _require_available(db)
    if db is None or not doc.get("report_id"):
        return
    try:
        await db[REPORTS].update_one(
            {"report_id": doc["report_id"]}, {"$set": doc}, upsert=True
        )
    except Exception as exc:  # noqa: BLE001
        if settings.mongodb_required:
            raise DatabasePersistenceError("Report persistence failed") from exc
        print(f"[mongo] save_report skip: {type(exc).__name__}: {exc}")


async def save_strategy(doc: dict[str, Any]) -> None:
    db = _get_db()
    _require_available(db)
    if db is None or not doc.get("strategy_id"):
        return
    try:
        await db[STRATEGIES].update_one(
            {"strategy_id": doc["strategy_id"]}, {"$set": doc}, upsert=True
        )
    except Exception as exc:  # noqa: BLE001
        if settings.mongodb_required:
            raise DatabasePersistenceError("Strategy persistence failed") from exc
        print(f"[mongo] save_strategy skip: {type(exc).__name__}: {exc}")


# ── 조회 ────────────────────────────────────────────────────────────────────

async def get_news(news_id: str) -> dict[str, Any] | None:
    """news_id 로 조회해 **in-memory/API 형태**로 돌려준다.

    설계 형태 그대로 주면 호출부가 source 를 dict 로 받게 되어 응답이 깨진다.
    """
    db = _get_db()
    if db is None:
        return None
    try:
        doc = await db[NEWS].find_one({"news_id": news_id}, {"_id": 0})
        return article_from_news_doc(doc) if doc else None
    except Exception:  # noqa: BLE001
        return None


async def get_news_embedding(news_id: str, purpose: str) -> dict[str, Any] | None:
    db = _get_db()
    if db is None or not news_id:
        return None
    fields = {"news_map_embedding": 1} if purpose == "news_map" else {"embedding": 1, "embedding_metadata": 1}
    try:
        doc = await db[NEWS].find_one({"news_id": news_id}, {"_id": 0, **fields})
        if not doc:
            return None
        if purpose == "news_map":
            return doc.get("news_map_embedding")
        return {"values": doc.get("embedding"), "metadata": doc.get("embedding_metadata")}
    except Exception as exc:
        if settings.mongodb_required:
            raise DatabasePersistenceError("Embedding lookup failed") from exc
        return None


async def save_news_embedding(news_id: str, purpose: str, record: dict[str, Any]) -> None:
    db = _get_db()
    _require_available(db)
    if db is None or not news_id:
        return
    payload = {"news_map_embedding": record} if purpose == "news_map" else {
        "embedding": record["values"], "embedding_metadata": record["metadata"],
    }
    try:
        await db[NEWS].update_one({"news_id": news_id}, {"$set": payload}, upsert=False)
    except Exception as exc:
        if settings.mongodb_required:
            raise DatabasePersistenceError("Embedding persistence failed") from exc
        print(f"[mongo] embedding save skip: {type(exc).__name__}")


async def news_candidates(keyword: str, limit: int) -> list[dict[str, Any]]:
    """Bounded same-search pool in search order (latest session, raw rank), after a restart too."""
    db = _get_db()
    if db is None or not keyword:
        return []
    try:
        order = [("_searched_at", -1), ("_search_rank", 1), ("published_at", -1), ("news_id", 1)]
        # Map selection needs card fields and map vectors, not bodies or RAG vectors.
        projection = {"content": 0, "embedding": 0, "embedding_metadata": 0}
        cursor = db[NEWS].find({"_search_keyword": keyword, "is_deleted": {"$ne": True}}, projection)
        cursor = cursor.sort(order).limit(limit)
        return [article_from_news_doc(doc) async for doc in cursor]
    except Exception as exc:
        if settings.mongodb_required:
            raise DatabasePersistenceError("News candidate lookup failed") from exc
        return []


# ── reports schema_version 2 (econmind-docs docs/10 § 5.1) ─────────────────────
# 선택 기사 리포트는 sections·evidence 로 묶어 저장하고, 읽을 때 API 의 평탄한 필드로 되돌린다.
# schema_version 이 없는 기존 문서(동기 경로)는 앱 형태 그대로라 변환하지 않는다.
_SECTION_FIELDS = ("summary", "event_analysis", "market_impact", "risk_factors")


def report_doc_from_result(report: dict[str, Any]) -> dict[str, Any]:
    doc = {k: v for k, v in report.items() if k not in _SECTION_FIELDS and k != "evidence_news"}
    doc["schema_version"] = 2
    doc["sections"] = {k: report.get(k, [] if k == "risk_factors" else "") for k in _SECTION_FIELDS}
    doc["evidence"] = report.get("evidence_news", [])
    doc["source_news_ids"] = report.get("requested_news_ids", [])
    doc["created_by"] = "system"
    for field in ("created_at", "updated_at"):
        if isinstance(doc.get(field), str):
            doc[field] = _parse_dt(doc[field]) or doc[field]
    return doc


def report_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    if doc.get("schema_version") != 2:
        return doc
    report = {k: v for k, v in doc.items()
              if k not in ("sections", "evidence", "source_news_ids", "schema_version", "created_by", "reuse_key")}
    report.update(doc.get("sections") or {})
    report["evidence_news"] = doc.get("evidence", [])
    report["requested_news_ids"] = doc.get("source_news_ids", [])
    for field in ("created_at", "updated_at"):
        if isinstance(report.get(field), datetime):
            report[field] = _iso_z(report[field])
    return report


async def get_report(report_id: str) -> dict[str, Any] | None:
    """report_id 로 조회. 기존 문서는 앱 형태 그대로, schema_version 2 는 평탄화해 돌려준다."""
    db = _get_db()
    if db is None:
        return None
    try:
        doc = await db[REPORTS].find_one({"report_id": report_id}, {"_id": 0})
    except Exception:  # noqa: BLE001
        return None
    return report_from_doc(doc) if doc else None


async def find_report_id_by_reuse_key(reuse_key: str) -> str | None:
    """같은 근거·옵션의 완료된(대체 결과가 아닌) 리포트 id. 재시작 후에도 재사용된다."""
    db = _get_db()
    if db is None:
        return None
    try:
        doc = await db[REPORTS].find_one({"reuse_key": reuse_key, "is_fallback": False}, {"report_id": 1})
    except Exception:  # noqa: BLE001
        return None
    return doc.get("report_id") if doc else None


async def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    """strategy_id 로 조회. reports 와 같은 이유로 변환 없음."""
    db = _get_db()
    if db is None:
        return None
    try:
        return await db[STRATEGIES].find_one({"strategy_id": strategy_id}, {"_id": 0})
    except Exception:  # noqa: BLE001
        return None


# ── 벡터검색 (RAG) ──────────────────────────────────────────────────────────

async def vector_search_articles(
    query_vec: list[float], k: int = 3, exclude_id: str = ""
) -> list[dict[str, Any]]:
    """임베딩으로 의미적으로 유사한 기사 top-k 반환 (Atlas $vectorSearch).

    반환 문서는 설계 형태다. report_generator._rag_block() 이 title/summary 만 읽고
    설계 news 에도 그 두 필드가 있으므로 그대로 쓸 수 있다.
    """
    db = _get_db()
    if db is None or not query_vec:
        return []
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 100,
                    "limit": k + 1,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "embedding": 0,
                }
            },
        ]
        results = []
        async for doc in db[NEWS].aggregate(pipeline):
            if doc.get("news_id") == exclude_id:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] vector_search skip: {type(exc).__name__}: {exc}")
        return []
