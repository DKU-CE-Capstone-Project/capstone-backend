"""
선택 기사 리포트 작업 (econmind-docs docs/10 § 3.5·3.6·4·5.2).

POST /api/v1/reports 에 news_ids 가 오면 이 모듈이 작업을 만들고, 응답을 보낸 뒤
같은 API 프로세스에서 백그라운드로 처리한다(1차 실행 위치). 진행 상태는 Redis 에
TTL 과 함께 두고, 완료본은 MongoDB reports 에 저장한다.

  report_job:{report_id}     진행 상태 (TTL 1시간)
  report_active:{session_id} 세션의 진행 중 작업 (세션당 1건)
  report_reuse:{reuse_key}   같은 입력의 진행 중 작업 (중복 요청 합치기)

Redis 가 없으면 프로세스 메모리로 폴백한다(app/session.py 와 같은 방식). 단일 인스턴스에서만
일관되며 재시작 시 진행 중 작업은 사라진다 → GET 은 404.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from app import database, store
from app.agents import naver_categories
from app.agents.critic import verify_report
from app.agents.diffbot_client import extract_articles_with_diffbot
from app.agents.report_generator import (
    SELECTION_PROMPT_VERSION,
    generate_selection_report,
)
from app.config import settings

JOB_TTL = 3600
LOCK_TTL = 600
EXTRACT_CONCURRENCY = 2

# 공개 오류 문구. 내부 예외 원문은 응답·작업 상태에 넣지 않는다(SEC-07과 같은 원칙).
ERRORS = {
    "body_extraction_failed": "선택한 기사의 본문을 하나도 추출하지 못해 리포트를 생성하지 않았습니다.",
    "category_blocked": "정치·사회 기사가 근거에 포함되어 리포트를 생성하지 않았습니다.",
    "category_check_failed": "기사 분류를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.",
    "internal": "리포트를 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
}

# ── 상태 저장소 (Redis → 메모리 폴백) ────────────────────────────────────────────

_memory: dict[str, tuple[float, str]] = {}
_redis = None
_redis_disabled = False


def _client():
    global _redis, _redis_disabled
    if _redis_disabled:
        return None
    if _redis is None:
        try:
            import redis.asyncio as aioredis

            _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[report_jobs] Redis 초기화 실패 → in-memory 폴백: {type(exc).__name__}")
            _redis_disabled = True
    return _redis


def _fallback(exc: Exception) -> None:
    global _redis_disabled
    print(f"[report_jobs] Redis 오류 → in-memory 폴백: {type(exc).__name__}")
    _redis_disabled = True


async def _get(key: str) -> str | None:
    client = _client()
    if client is not None:
        try:
            return await client.get(key)
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    expires, value = _memory.get(key, (0.0, ""))
    if expires < time.time():
        _memory.pop(key, None)
        return None
    return value


async def _set(key: str, value: str, ttl: int) -> None:
    client = _client()
    if client is not None:
        try:
            await client.set(key, value, ex=ttl)
            return
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    _memory[key] = (time.time() + ttl, value)


async def _delete(key: str) -> None:
    client = _client()
    if client is not None:
        try:
            await client.delete(key)
            return
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    _memory.pop(key, None)


def reset_memory() -> None:
    """테스트용: 메모리 폴백 상태 초기화."""
    _memory.clear()


async def get_state(report_id: str) -> dict[str, Any] | None:
    raw = await _get(f"report_job:{report_id}")
    return json.loads(raw) if raw else None


async def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    await _set(f"report_job:{state['report_id']}", json.dumps(state, ensure_ascii=False), JOB_TTL)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ── 요청 처리 ────────────────────────────────────────────────────────────────


def reuse_key(news_ids: list[str], language: str, report_type: str) -> str:
    """선택 순서와 무관하게 같은 기사 집합·옵션·프롬프트 버전이면 같은 키.

    본문 버전은 요청 시점에 아직 추출 전이라 키에 넣지 않고 evidence[].content_hash 로 기록한다.
    """
    joined = "|".join(["report-v2", SELECTION_PROMPT_VERSION, language, report_type, *sorted(news_ids)])
    return hashlib.sha256(joined.encode()).hexdigest()


async def completed_report_id(key: str) -> str | None:
    """같은 입력의 완료 결과(대체 결과 제외). 메모리 → MongoDB."""
    report_id = store.report_index.get(key)
    if report_id and report_id in store.report_cache and not store.report_cache[report_id].get("is_fallback"):
        return report_id
    return await database.find_report_id_by_reuse_key(key)


async def running_report_id(key: str) -> str | None:
    report_id = await _get(f"report_reuse:{key}")
    if not report_id:
        return None
    state = await get_state(report_id)
    return report_id if state and state["status"] in ("pending", "processing") else None


async def active_report_id(session_id: str) -> str | None:
    report_id = await _get(f"report_active:{session_id}")
    if not report_id:
        return None
    state = await get_state(report_id)
    return report_id if state and state["status"] in ("pending", "processing") else None


async def create(news_ids: list[str], *, session_id: str, key: str) -> dict[str, Any]:
    report_id = f"rpt_{uuid.uuid4().hex}"
    state = {
        "report_id": report_id, "status": "pending", "stage": "queued", "progress": None,
        "requested_news_ids": news_ids, "error": None, "created_at": _now(),
    }
    await _save_state(state)
    await _set(f"report_reuse:{key}", report_id, LOCK_TTL)
    await _set(f"report_active:{session_id}", report_id, LOCK_TTL)
    return state


# ── 작업 실행 ────────────────────────────────────────────────────────────────


def _content_hash(article: dict[str, Any]) -> str:
    body = article.get("cleaned_content") or ""
    return hashlib.sha256(body.encode()).hexdigest()[:16] if body else ""


async def _extract(article: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """(기사, body_status). 이미 같은 URL 본문이 있으면 재사용한다."""
    target = article.get("naver_url") or article.get("url")
    if article.get("cleaned_content") and article.get("content_source_url") == target:
        return article, "cached"
    extracted = await extract_articles_with_diffbot([article], max_articles=1, concurrency=1, max_retries=0)
    if extracted and extracted[0].get("cleaned_content"):
        article = extracted[0]
        await database.save_news(database.news_doc_from_article(article))
        store.news_cache[article["news_id"]] = article
        return article, "extracted"
    return article, "description_only"


class _Failed(Exception):
    def __init__(self, code: str):
        self.code = code


async def _check_categories(articles: list[dict[str, Any]]) -> None:
    for article in articles:
        if not article.get("naver_url"):
            continue
        labels = await naver_categories.fetch_categories(article["naver_url"])
        if labels is None:
            raise _Failed("category_check_failed")
        if not naver_categories.allowed(labels):
            raise _Failed("category_blocked")


async def run(state: dict[str, Any], *, session_id: str, key: str, language: str, report_type: str) -> None:
    """백그라운드 실행. 어떤 실패도 상태(failed + 공개 오류 코드)로 남기고 잠금을 푼다."""
    try:
        await _run(state, key=key, language=language, report_type=report_type)
    except _Failed as failed:
        state.update(status="failed", error={"code": failed.code, "message": ERRORS[failed.code]})
        await _save_state(state)
    except Exception as exc:  # noqa: BLE001
        print(f"[report_jobs] {state['report_id']} failed: {type(exc).__name__}")
        state.update(status="failed", error={"code": "internal", "message": ERRORS["internal"]})
        await _save_state(state)
    finally:
        await _delete(f"report_reuse:{key}")
        if await _get(f"report_active:{session_id}") == state["report_id"]:
            await _delete(f"report_active:{session_id}")


async def _run(state: dict[str, Any], *, key: str, language: str, report_type: str) -> None:
    news_ids = state["requested_news_ids"]
    articles = []
    for news_id in news_ids:
        article = await store.get_news(news_id)
        if not article:
            raise _Failed("internal")
        articles.append(article)

    state.update(status="processing", stage="extracting", progress={"done": 0, "total": len(articles)})
    await _save_state(state)
    await _check_categories(articles)

    semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)
    lock = asyncio.Lock()

    async def one(article: dict[str, Any]) -> tuple[dict[str, Any], str]:
        async with semaphore:
            result = await _extract(article)
        async with lock:
            state["progress"]["done"] += 1
            await _save_state(state)
        return result

    results = await asyncio.gather(*(one(a) for a in articles))
    if not any(status != "description_only" for _, status in results):
        raise _Failed("body_extraction_failed")
    evidence_articles = [article for article, _ in results]

    state.update(stage="analyzing", progress=None)
    await _save_state(state)
    report = await generate_selection_report(evidence_articles)

    state["stage"] = "strategy"
    await _save_state(state)
    evidence_summaries = [a.get("cleaned_content") or a.get("description") or a.get("summary") or ""
                          for a in evidence_articles] + (report.get("rag_sources") or [])
    verification = None if report["is_fallback"] else await verify_report(report, evidence_summaries)

    now = _now()
    result = {
        "report_id": state["report_id"],
        "status": "completed",
        "stage": "done",
        "progress": None,
        "requested_news_ids": news_ids,
        "title": report["title"],
        "summary": report["summary"],
        "event_analysis": report["event_analysis"],
        "market_impact": report["market_impact"],
        "related_stocks": [s["name"] for s in report["stock_impacts"]],
        "evidence_news": [
            {"news_id": a.get("news_id", ""), "title": a.get("title", ""), "source_name": a.get("source", ""),
             "source_url": a.get("naver_url") or a.get("url", ""), "published_at": a.get("published_at", ""),
             "body_status": status, "content_hash": _content_hash(a)}
            for a, status in results
        ],
        "stock_impacts": report["stock_impacts"],
        "strategy": report["strategy"],
        "risk_factors": report["risk_factors"],
        "rag_sources": report.get("rag_sources", []),
        "verification": verification,
        "is_fallback": report["is_fallback"],
        "error": None,
        "report_type": report_type,
        "language": language,
        "reuse_key": key,
        "model_info": {"provider": settings.llm_provider,
                       "model": settings.claude_model if settings.llm_provider == "anthropic" else settings.gemini_model,
                       "prompt_version": SELECTION_PROMPT_VERSION},
        "created_at": state["created_at"],
        "updated_at": now,
    }
    await database.save_report(database.report_doc_from_result(result))
    store.report_cache[state["report_id"]] = result
    if not report["is_fallback"]:
        store.report_index[key] = state["report_id"]
    state.update(status="completed", stage="done", progress=None)
    await _save_state(state)
