"""POST /api/v1/reports  &  GET /api/v1/reports/{report_id}"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from app import database, report_jobs, store
from app.agents import naver_categories
from app.agents.critic import verify_report
from app.agents.diffbot_client import extract_articles_with_diffbot
from app.agents.report_generator import generate_report
from app.config import settings
from app.schemas import ReportCreateRequest, ReportCreateResponse, ReportResponse
from app.usage_limits import check_quota

router = APIRouter()


def _report_key(owner_sid: str, body: ReportCreateRequest) -> str:
    """Reuse only a report with the same owner and every response-affecting input."""
    payload = [owner_sid, body.news_id, sorted(body.related_news_ids),
               sorted(body.ticker_symbols), body.language, body.report_type]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


async def _create_selection_report(
    body: ReportCreateRequest, request: Request, background: BackgroundTasks,
) -> JSONResponse:
    """news_ids 경로: 202 + 백그라운드 생성. 같은 입력의 완료 결과는 200으로 재사용한다."""
    news_ids = body.news_ids or []
    missing = [nid for nid in news_ids if not await store.get_news(nid)]
    if missing:
        raise HTTPException(status_code=404, detail=f"news_id '{missing[0]}' not found.")

    session_id = request.state.session_id
    key = report_jobs.reuse_key(news_ids, body.language, body.report_type, owner_sid=session_id)
    done = await report_jobs.completed_report_id(key)
    if done:
        report = await store.get_report(done)
        return JSONResponse(status_code=200, content=ReportCreateResponse(
            report_id=done, status="completed", created_at=(report or {}).get("created_at", "")).model_dump())
    running = await report_jobs.running_report_id(key)
    if running:
        state = await report_jobs.get_state(running) or {}
        return JSONResponse(status_code=202, content=ReportCreateResponse(
            report_id=running, status=state.get("status", "pending"), created_at=state.get("created_at", "")
        ).model_dump())

    active = await report_jobs.active_report_id(session_id)
    if active:
        raise HTTPException(status_code=409, detail={
            "message": "진행 중인 리포트가 있습니다. 완료된 뒤 다시 요청해 주세요.", "report_id": active})

    await check_quota(session_id, "reports", per_session=5, global_limit=100)
    state = await report_jobs.create(news_ids, session_id=session_id, key=key)
    background.add_task(report_jobs.run, state, session_id=session_id, key=key,
                        language=body.language, report_type=body.report_type)
    return JSONResponse(status_code=202, content=ReportCreateResponse(
        report_id=state["report_id"], status="pending", created_at=state["created_at"]).model_dump())


@router.post(
    "", response_model=ReportCreateResponse, status_code=201,
    responses={200: {"model": ReportCreateResponse}, 202: {"model": ReportCreateResponse}, 409: {}},
)
async def create_report(
    body: ReportCreateRequest, request: Request, background: BackgroundTasks,
) -> ReportCreateResponse | JSONResponse:
    """선택한 뉴스를 기반으로 AI 투자 분석 리포트를 생성합니다.

    - news_ids(1~5): 선택 기사 비동기 생성. 202 후 GET /reports/{id}로 status·stage를 폴링합니다.
      같은 기사 집합·옵션의 완료 결과가 있으면 200으로 재사용합니다. 세션당 진행 중 작업은 1건(409).
    - news_id(+related_news_ids): 기존 동기 생성(201). 같은 뉴스(+연관셋)는 캐시를 재사용합니다.
    """
    if body.news_ids is not None:
        return await _create_selection_report(body, request, background)
    center = await store.get_news(body.news_id)
    if not center:
        raise HTTPException(
            status_code=404,
            detail=f"news_id '{body.news_id}' not found. /api/v1/news/search 먼저 호출하세요.",
        )

    # ── 결과 캐싱: 동일 뉴스 리포트 재사용 ──
    cache_key = _report_key(request.state.session_id, body)
    existing_id = store.report_index.get(cache_key)
    if existing_id and existing_id in store.report_cache:
        print(f"[report] cache hit → 재사용 {existing_id}")
        return ReportCreateResponse(
            report_id=existing_id,
            status="completed",
            created_at=store.report_cache[existing_id].get("created_at", ""),
        )

    await check_quota(request.state.session_id, "reports", per_session=5, global_limit=100)

    # 연관 뉴스도 메모리에 없으면 Mongo 폴백 — API 재시작 후에도 근거가 유지되도록.
    # (한 번만 조회해 evidence_news 에서 재사용한다)
    related: list[dict] = []
    related_ids: list[str] = []
    for nid in body.related_news_ids:
        art = await store.get_news(nid)
        if art:
            related.append(art)
            related_ids.append(nid)

    target_url = center.get("naver_url") or center.get("url")
    if center.get("naver_url"):
        labels = await naver_categories.fetch_categories(center["naver_url"])
        if labels is None:
            raise HTTPException(503, "기사 분류를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        if not naver_categories.allowed(labels):
            raise HTTPException(404, "정치·사회 기사는 제공하지 않습니다.")
    if not center.get("cleaned_content") or center.get("content_source_url") != target_url:
        extracted = await extract_articles_with_diffbot(
            [center], max_articles=1, timeout_ms=settings.diffbot_search_timeout_ms,
            concurrency=1, max_retries=0,
        )
        if not extracted or not extracted[0].get("cleaned_content"):
            raise HTTPException(502, "기사 본문을 추출하지 못해 리포트를 생성하지 않았습니다. 잠시 후 다시 시도해 주세요.")
        center = extracted[0]
        await database.save_news(database.news_doc_from_article(center))
        store.news_cache[body.news_id] = center

    report_data = await generate_report(center, related)

    # 검증(critic) 에이전트: 근거 뉴스 대비 리포트 사실성 점검
    evidence_summaries = [center.get("cleaned_content", "")] + [
        art.get("summary") or art.get("description", "") for art in related
    ] + (report_data.get("rag_sources") or [])
    verification = await verify_report(report_data, evidence_summaries)

    report_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    full_report = {
        "report_id": report_id,
        "owner_sid": request.state.session_id,
        "title": report_data["title"],
        "summary": report_data["summary"],
        "event_analysis": report_data["event_analysis"],
        "market_impact": report_data["market_impact"],
        "related_stocks": body.ticker_symbols,
        "evidence_news": [
            {"news_id": body.news_id, "title": center.get("title", "")}
        ] + [
            {"news_id": nid, "title": art.get("title", "")}
            for nid, art in zip(related_ids, related)
        ],
        "risk_factors": report_data["risk_factors"],
        "rag_sources": report_data.get("rag_sources", []),
        "is_fallback": report_data.get("is_fallback", False),
        "verification": verification,
        "created_at": now,
    }
    # MongoDB write-through (use_mongodb 시). 선택 기사 경로와 같은 schema_version 2 문서로 저장한다.
    await database.save_report(database.report_doc_from_result({
        **full_report,
        "requested_news_ids": [body.news_id, *related_ids],
        "report_type": body.report_type,
        "language": body.language,
        "updated_at": now,
    }))
    store.report_cache[report_id] = full_report
    store.report_index[cache_key] = report_id

    return ReportCreateResponse(
        report_id=report_id,
        status="completed",
        created_at=now,
    )


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str, request: Request) -> ReportResponse:
    """Only the owner may read generation status or the completed report."""
    state = await report_jobs.get_state(report_id)
    if state and state.get("owner_sid") != request.state.session_id:
        raise HTTPException(status_code=404, detail=f"report_id '{report_id}' not found.")
    if state and state["status"] != "completed":
        return ReportResponse(
            report_id=report_id, title="", summary="", event_analysis="", market_impact="",
            related_stocks=[], evidence_news=[], risk_factors=[], created_at=state.get("created_at", ""),
            status=state["status"], stage=state["stage"], progress=state.get("progress"),
            requested_news_ids=state.get("requested_news_ids", []), error=state.get("error"),
            updated_at=state.get("updated_at", ""),
        )
    report = await store.get_report(report_id)
    if not report or report.get("owner_sid") != request.state.session_id:
        raise HTTPException(status_code=404, detail=f"report_id '{report_id}' not found.")
    return ReportResponse(**report)
