"""POST /api/v1/reports  &  GET /api/v1/reports/{report_id}"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.agents.report_generator import generate_report
from app.agents.critic import verify_report
from app import database, store
from app.schemas import ReportCreateRequest, ReportCreateResponse, ReportResponse

router = APIRouter()


def _report_key(news_id: str, related_ids: list[str]) -> str:
    """같은 뉴스+연관셋이면 동일 key → 리포트 재생성 대신 캐시 재사용."""
    joined = news_id + "|" + "|".join(sorted(related_ids))
    return hashlib.md5(joined.encode()).hexdigest()


@router.post("", response_model=ReportCreateResponse, status_code=201)
async def create_report(body: ReportCreateRequest) -> ReportCreateResponse:
    """선택한 뉴스를 기반으로 AI 투자 분석 리포트를 생성합니다.

    같은 뉴스(+연관셋)에 대한 리포트가 이미 있으면 LLM 재호출 없이 재사용합니다(캐싱).
    """
    center = store.news_cache.get(body.news_id)
    if not center:
        raise HTTPException(
            status_code=404,
            detail=f"news_id '{body.news_id}' not found. /api/v1/news/search 먼저 호출하세요.",
        )

    # ── 결과 캐싱: 동일 뉴스 리포트 재사용 ──
    cache_key = _report_key(body.news_id, body.related_news_ids)
    existing_id = store.report_index.get(cache_key)
    if existing_id and existing_id in store.report_cache:
        print(f"[report] cache hit → 재사용 {existing_id}")
        return ReportCreateResponse(
            report_id=existing_id,
            status="completed",
            created_at=store.report_cache[existing_id].get("created_at", ""),
        )

    related = [
        store.news_cache[nid]
        for nid in body.related_news_ids
        if nid in store.news_cache
    ]

    report_data = await generate_report(center, related)

    # 검증(critic) 에이전트: 근거 뉴스 대비 리포트 사실성 점검
    evidence_summaries = [center.get("summary") or center.get("description", "")] + [
        art.get("summary") or art.get("description", "") for art in related
    ] + (report_data.get("rag_sources") or [])
    verification = await verify_report(report_data, evidence_summaries)

    report_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    full_report = {
        "report_id": report_id,
        "title": report_data["title"],
        "summary": report_data["summary"],
        "event_analysis": report_data["event_analysis"],
        "market_impact": report_data["market_impact"],
        "related_stocks": body.ticker_symbols,
        "evidence_news": [
            {"news_id": body.news_id, "title": center.get("title", "")}
        ] + [
            {"news_id": nid, "title": store.news_cache[nid].get("title", "")}
            for nid in body.related_news_ids
            if nid in store.news_cache
        ],
        "risk_factors": report_data["risk_factors"],
        "rag_sources": report_data.get("rag_sources", []),
        "verification": verification,
        "created_at": now,
    }
    store.report_cache[report_id] = full_report
    store.report_index[cache_key] = report_id  # 결과 캐싱 등록
    await database.save_report(full_report)  # MongoDB write-through (use_mongodb 시)

    return ReportCreateResponse(
        report_id=report_id,
        status="completed",
        created_at=now,
    )


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str) -> ReportResponse:
    """생성된 AI 리포트 결과를 조회합니다."""
    report = store.report_cache.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"report_id '{report_id}' not found.")
    return ReportResponse(**report)
