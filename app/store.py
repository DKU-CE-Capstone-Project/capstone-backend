"""
Module-level in-memory caches.
Survive re-renders / re-imports within a single server process.
Reset on server restart (stateless by design for MVP).
"""
from __future__ import annotations

from typing import Any

# news_id (MD5[:12] of URL) → normalized article dict
news_cache: dict[str, dict[str, Any]] = {}

# report_id (UUID4 str) → report dict
report_cache: dict[str, dict[str, Any]] = {}

# strategy_id (UUID4 str) → strategy dict
strategy_cache: dict[str, dict[str, Any]] = {}

# selection_id (UUID4 str) → {news_ids, selected_news, created_at}
selection_cache: dict[str, dict[str, Any]] = {}

# ── 결과 캐싱 (LLM 재호출 방지 → 비용·지연 절감) ─────────────────────────────
# content_hash(제목+설명) → {"title_ko", "summary_ko"}  : 같은 기사 재번역/재요약 방지
summary_cache: dict[str, dict[str, str]] = {}

# content_key(news_id + 정렬된 related_ids) → report_id : 같은 뉴스 리포트 재생성 방지
report_index: dict[str, str] = {}


# ── L1(메모리) → L2(MongoDB) 조회 ───────────────────────────────────────────
# 위 dict 들은 프로세스 메모리라 API 컨테이너를 재시작하거나 --scale econmind-api=2
# 로 늘리면 전부 사라진다. 그래서 MongoDB 에 write-through 해 두고도 GET 이 404 가
# 났다(= Mongo 가 write-only 상태였다). 아래 접근자가 그 읽기 경로를 잇는다.
#
#   1. 메모리에 있으면 그대로 (기존과 동일, 추가 비용 0)
#   2. 없으면 MongoDB 조회
#   3. 찾으면 메모리에 되채워(L1 warm) 다음 요청부터 다시 빠르게
#
# use_mongodb=False 이거나 연결 실패면 database 쪽이 None 을 돌려주므로
# 결과적으로 기존 동작(메모리에 없으면 없음)과 완전히 같다.

async def get_news(news_id: str) -> dict[str, Any] | None:
    """news_id 로 기사 조회 (메모리 → MongoDB)."""
    art = news_cache.get(news_id)
    if art is not None:
        return art
    from app import database

    art = await database.get_news(news_id)
    if art is not None:
        news_cache[news_id] = art
    return art


async def get_report(report_id: str) -> dict[str, Any] | None:
    """report_id 로 리포트 조회 (메모리 → MongoDB)."""
    report = report_cache.get(report_id)
    if report is not None:
        return report
    from app import database

    report = await database.get_report(report_id)
    if report is not None:
        report_cache[report_id] = report
    return report


async def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    """strategy_id 로 전략 조회 (메모리 → MongoDB)."""
    strategy = strategy_cache.get(strategy_id)
    if strategy is not None:
        return strategy
    from app import database

    strategy = await database.get_strategy(strategy_id)
    if strategy is not None:
        strategy_cache[strategy_id] = strategy
    return strategy
