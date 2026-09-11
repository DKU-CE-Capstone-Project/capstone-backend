"""
MongoDB 영속화 + 벡터검색 (자체 호스팅).

컬렉션 이름과 문서 구조는 설계 문서를 기준으로 한다.
  Notion "설계 › 데이터베이스 › 구조크" — 최종 컬렉션 목록 8개
  https://app.notion.com/p/37114a4e8feb8080bfaafc4459d016f9

스키마 생성/검증(JSON Schema Validator)과 인덱스는 배포 레포의
`capstone-deploy/mongo-init/*.js`가 담당한다. 이 모듈은 같은 정의를 멱등하게
재적용하기만 하므로, 빈 MongoDB를 가리켜도 앱 단독으로 기동할 수 있다.

인프라: mongodb/mongodb-atlas-local 이미지(mongod + mongot).
일반 MongoDB Community 서버에는 $vectorSearch / createSearchIndex가 없다.

설계 원칙: settings.use_mongodb=False 또는 연결 실패 시 모든 함수가
graceful no-op/빈값을 반환 → 기존 in-memory 흐름을 절대 깨지 않는다.
(in-memory store는 L1 캐시, MongoDB는 write-through 영속화 + 읽기 폴백 + 벡터검색)
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings

# ── 컬렉션 (설계 "최종 컬렉션 목록" 8개) ───────────────────────────────────────
NEWS = "news"
NEWS_ANALYSIS = "news_analysis"
NEWS_RELATIONS = "news_relations"
MINDMAPS = "mindmaps"        # 미사용: 12주차 회의 결정에 따라 쿠키/세션으로 처리
REPORTS = "reports"
STRATEGIES = "strategies"
JOBS = "jobs"                # 미사용: app/job_store.py가 Redis로 구현
USERS = "users"              # 미사용: 로그인 미구현

VECTOR_INDEX = "vector_index"
EMBED_DIM = 768  # settings.embedding_model(기본 gemini-embedding-001)의 출력 차원

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

# 이름은 capstone-deploy/mongo-init/02-indexes.js와 반드시 일치시킨다.
# 같은 키에 다른 이름으로 인덱스를 만들면 MongoDB가 IndexOptionsConflict를 낸다.
_INDEX_SPECS: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = [
    (NEWS, [("news_id", 1)], {"name": "uniq_news_news_id", "unique": True}),
    # url이 빈 문자열인 예외 문서끼리 유니크 충돌하지 않도록 partial로 건다.
    (NEWS, [("url", 1)], {
        "name": "uniq_news_url",
        "unique": True,
        "partialFilterExpression": {"url": {"$gt": ""}},
    }),
    (NEWS, [("_search_keyword", 1)], {"name": "idx_news_search_keyword"}),
    (NEWS, [("published_at", -1)], {"name": "idx_news_published_at"}),
    (NEWS, [("status", 1)], {"name": "idx_news_status"}),
    (NEWS_RELATIONS, [("source_news_id", 1), ("target_news_id", 1)],
     {"name": "uniq_news_relations_pair", "unique": True}),
    (REPORTS, [("report_id", 1)], {"name": "uniq_reports_report_id", "unique": True}),
    (STRATEGIES, [("strategy_id", 1)], {"name": "uniq_strategies_strategy_id", "unique": True}),
]


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

    print(f"[mongo] connected (db={settings.mongodb_db_name})")
    # 2) 인덱스는 개별 try (있으면 무시, 실패해도 계속) — upsert 동작엔 인덱스 불필요
    for coll, keys, opts in _INDEX_SPECS:
        try:
            await db[coll].create_index(keys, **opts)
        except Exception as exc:  # noqa: BLE001
            print(f"[mongo] index {coll}.{opts['name']} skip: {type(exc).__name__}: {str(exc)[:80]}")
    # 3) 벡터검색 인덱스 (RAG 핵심)
    # mongot 기동을 기다리며 재시도하므로 최대 수십 초가 걸린다. startup에서 await하면
    # 그동안 /health가 응답하지 않아 컨테이너가 unhealthy로 떨어진다 → 백그라운드로 돌린다.
    asyncio.create_task(ensure_vector_index())


async def ensure_vector_index(attempts: int = 3, delay: float = 10.0) -> bool:
    """news.embedding 에 vectorSearch 인덱스를 멱등 생성한다.

    atlas-local 컨테이너는 mongod가 먼저 뜨고 mongot(검색 프로세스)이 뒤따라 뜬다.
    앱이 그 사이에 기동하면 인덱스 생성이 실패하는데, 이때 재시도하지 않으면
    RAG가 조용히 꺼진 채로 계속 돌아간다. 그래서 짧게 몇 번 재시도한다.

    반환: 인덱스가 존재하거나 생성 요청에 성공하면 True.
    """
    db = _get_db()
    if db is None:
        return False

    for attempt in range(1, attempts + 1):
        try:
            existing = [i.get("name") async for i in db[NEWS].list_search_indexes()]
            if VECTOR_INDEX in existing:
                return True

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
            return True
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                print(f"[mongo] vector index 시도 {attempt}/{attempts} 실패 → {delay}s 후 재시도: {last}")
                await asyncio.sleep(delay)
            else:
                # 여기까지 오면 RAG는 꺼진 채 동작한다(서비스는 계속 정상).
                # 수동 생성 절차는 capstone-deploy/README.md 참고.
                print(f"[mongo] vector index 생성 포기: {last}")
    return False


# ── 저장(write-through) ─────────────────────────────────────────────────────

async def _upsert(coll: str, key: dict[str, Any], doc: dict[str, Any]) -> None:
    db = _get_db()
    if db is None:
        return
    try:
        await db[coll].update_one(key, {"$set": doc}, upsert=True)
    except Exception as exc:  # noqa: BLE001
        # DocumentValidationFailure면 설계 스키마에서 벗어난 문서다 — 로그로 드러난다.
        print(f"[mongo] {coll} upsert skip: {type(exc).__name__}: {str(exc)[:200]}")


async def save_news(doc: dict[str, Any]) -> None:
    """news 컬렉션에 기사 1건 upsert. 문서는 설계 스키마 형태여야 한다."""
    if not doc.get("news_id"):
        return
    await _upsert(NEWS, {"news_id": doc["news_id"]}, doc)


async def save_report(doc: dict[str, Any]) -> None:
    if not doc.get("report_id"):
        return
    await _upsert(REPORTS, {"report_id": doc["report_id"]}, doc)


async def save_strategy(doc: dict[str, Any]) -> None:
    if not doc.get("strategy_id"):
        return
    await _upsert(STRATEGIES, {"strategy_id": doc["strategy_id"]}, doc)


async def save_relation(doc: dict[str, Any]) -> None:
    """news_relations에 (source, target) 쌍 1건 upsert.

    연관도 점수를 매 요청 재계산하지 않도록 캐시하는 용도.
    12주차 회의에서 "릴레이션까지는 저장하자"고 결정된 항목이다.
    """
    src, tgt = doc.get("source_news_id"), doc.get("target_news_id")
    if not src or not tgt:
        return
    await _upsert(NEWS_RELATIONS, {"source_news_id": src, "target_news_id": tgt}, doc)


# ── 조회 (in-memory 캐시 미스 시 폴백) ──────────────────────────────────────

_PROJECTION = {"_id": 0, "embedding": 0}


async def _find_one(coll: str, key: dict[str, Any]) -> dict[str, Any] | None:
    db = _get_db()
    if db is None:
        return None
    try:
        return await db[coll].find_one(key, _PROJECTION)
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] {coll} find_one skip: {type(exc).__name__}: {exc}")
        return None


async def get_news(news_id: str) -> dict[str, Any] | None:
    return await _find_one(NEWS, {"news_id": news_id})


async def get_report(report_id: str) -> dict[str, Any] | None:
    return await _find_one(REPORTS, {"report_id": report_id})


async def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    return await _find_one(STRATEGIES, {"strategy_id": strategy_id})


async def find_news_by_keyword(search_keyword: str, limit: int = 20) -> list[dict[str, Any]]:
    """같은 검색어로 수집된 기사들을 반환한다 (연관뉴스 조회의 DB 폴백)."""
    db = _get_db()
    if db is None or not search_keyword:
        return []
    try:
        cursor = db[NEWS].find({"_search_keyword": search_keyword}, _PROJECTION).limit(limit)
        return [doc async for doc in cursor]
    except Exception as exc:  # noqa: BLE001
        print(f"[mongo] find_news_by_keyword skip: {type(exc).__name__}: {exc}")
        return []


# ── 벡터검색 (RAG) ──────────────────────────────────────────────────────────

async def vector_search_news(
    query_vec: list[float], k: int = 3, exclude_id: str = ""
) -> list[dict[str, Any]]:
    """임베딩으로 의미적으로 유사한 기사 top-k 반환 ($vectorSearch)."""
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
            {"$project": _PROJECTION},
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
