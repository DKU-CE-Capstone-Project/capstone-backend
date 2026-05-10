"""POST /api/v1/reports  &  GET /api/v1/reports/{report_id}"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.agents.report_generator import generate_report
from app import store
from app.schemas import ReportCreateRequest, ReportCreateResponse, ReportResponse

router = APIRouter()


@router.post("", response_model=ReportCreateResponse, status_code=201)
async def create_report(body: ReportCreateRequest) -> ReportCreateResponse:
    """선택한 뉴스를 기반으로 AI 투자 분석 리포트를 생성합니다."""
    center = store.news_cache.get(body.news_id)
    if not center:
        raise HTTPException(
            status_code=404,
            detail=f"news_id '{body.news_id}' not found. /api/v1/news/search 먼저 호출하세요.",
        )

    related = [
        store.news_cache[nid]
        for nid in body.related_news_ids
        if nid in store.news_cache
    ]

    report_data = await generate_report(center, related)

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
        "created_at": now,
    }
    store.report_cache[report_id] = full_report

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
