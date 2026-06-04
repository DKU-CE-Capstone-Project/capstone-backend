"""
MongoDB Atlas 영속화 + Atlas Vector Search.

발표 설계서의 MongoDB/Vector Search 단계 구현. 동시에 AI 에이전트의
RAG 그라운딩(의미 기반 유사 뉴스 검색)을 담당하는 '근거 계층'.

설계 원칙: settings.use_mongodb=False 또는 연결 실패 시 모든 함수가
graceful no-op/빈값을 반환 → 기존 in-memory 흐름을 절대 깨지 않는다.
(in-memory store는 L1 캐시로 유지, MongoDB는 write-through 영속화 + 벡터검색 read 경로)
"""
from __future__ import annotations

from typing import Any

from app.config import settings

# 컬렉션 이름
ARTICLES = "articles"
REPORTS = "reports"
STRATEGIES = "strategies"
SELECTIONS = "selections"

VECTOR_INDEX = "vector_index"
EMBED_DIM = 768  # text-embedding-004

_client = None
_db = None
_disabled = False  # 연결 실패 1회 후 재시도 폭주 방지


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
    # 2) 인덱스는 개별 try (있으면 무시, 실패해도 계속) — upsert 동작엔 인덱스 불필요
    for coll, key, uniq in [
        (ARTICLES, "news_id", True),
        (ARTICLES, "_search_keyword", False),
        (REPORTS, "report_id", True),
        (STRATEGIES, "strategy_id", True),
    ]:
        try:
            await db[coll].create_index(key, unique=uniq)
        except Exception as exc:  # noqa: BLE001
            print(f"[mongo] index {coll}.{key} skip: {type(exc).__name__}: {str(exc)[:80]}")
    # 3) 벡터검색 인덱스 (RAG 핵심)
    await ensure_vector_index()


async def ensure_vector_index() -> None:
    """articles.embedding 에 Atlas Vector Search 인덱스를 멱등 생성."""
    db = _get_db()
    if db is None:
        return
    try:
        existing = [i.get("name") async for i in db[ARTICLES].list_search_indexes()]
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
        await db[ARTICLES].create_search_index(model=model)
        print(f"[mongo] vector index '{VECTOR_INDEX}' 생성 요청 (빌드 ~1분)")
    except Exception as exc:  # noqa: BLE001
        # 프로그램 생성 실패 시 Atlas UI 수동 생성으로 대체 (deploy/README 참고)
        print(f"[mongo] vector index 생성 skip: {type(exc).__name__}: {exc}")


# ── 저장(write-through) ─────────────────────────────────────────────────────

async def save_article(doc: dict[str, Any]) -> None:
    db = _get_db()
    if db is None or not doc.get("news_id"):
        return
    try:
        await db[ARTICLES].update_one(
            {"news_id": doc["news_id"]}, {"$set": doc}, upsert=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] save_article skip: {type(exc).__name__}: {exc}")


async def save_report(doc: dict[str, Any]) -> None:
    db = _get_db()
    if db is None or not doc.get("report_id"):
        return
    try:
        await db[REPORTS].update_one(
            {"report_id": doc["report_id"]}, {"$set": doc}, upsert=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] save_report skip: {type(exc).__name__}: {exc}")


async def save_strategy(doc: dict[str, Any]) -> None:
    db = _get_db()
    if db is None or not doc.get("strategy_id"):
        return
    try:
        await db[STRATEGIES].update_one(
            {"strategy_id": doc["strategy_id"]}, {"$set": doc}, upsert=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] save_strategy skip: {type(exc).__name__}: {exc}")


# ── 조회 ────────────────────────────────────────────────────────────────────

async def get_article(news_id: str) -> dict[str, Any] | None:
    db = _get_db()
    if db is None:
        return None
    try:
        return await db[ARTICLES].find_one({"news_id": news_id}, {"_id": 0, "embedding": 0})
    except Exception:  # noqa: BLE001
        return None


# ── 벡터검색 (RAG) ──────────────────────────────────────────────────────────

async def vector_search_articles(
    query_vec: list[float], k: int = 3, exclude_id: str = ""
) -> list[dict[str, Any]]:
    """임베딩으로 의미적으로 유사한 기사 top-k 반환 (Atlas $vectorSearch)."""
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
        async for doc in db[ARTICLES].aggregate(pipeline):
            if doc.get("news_id") == exclude_id:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] vector_search skip: {type(exc).__name__}: {exc}")
        return []
