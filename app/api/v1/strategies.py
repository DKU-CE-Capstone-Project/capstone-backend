"""POST /api/v1/strategies  &  GET /api/v1/strategies/{strategy_id}"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.agents.strategy_generator import generate_strategy
from app import store
from app.schemas import StrategyCreateRequest, StrategyCreateResponse, StrategyResponse

router = APIRouter()


@router.post("", response_model=StrategyCreateResponse, status_code=201)
async def create_strategy(body: StrategyCreateRequest) -> StrategyCreateResponse:
    """리포트를 바탕으로 투자 전략을 생성합니다."""
    report = store.report_cache.get(body.report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"report_id '{body.report_id}' not found. /api/v1/reports 먼저 호출하세요.",
        )

    strategy_data = await generate_strategy(
        report_summary=report.get("summary", ""),
        risk_level=body.risk_level,
        period=body.period,
    )

    strategy_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    full_strategy = {
        "strategy_id": strategy_id,
        "expected_return": strategy_data["expected_return"],
        "risk": strategy_data["risk"],
        "period": body.period,
        "strategy_summary": strategy_data["strategy_summary"],
        "strategy_items": strategy_data["strategy_items"],
        "created_at": now,
    }
    store.strategy_cache[strategy_id] = full_strategy

    return StrategyCreateResponse(
        strategy_id=strategy_id,
        status="completed",
        created_at=now,
    )


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: str) -> StrategyResponse:
    """생성된 투자 전략 결과를 조회합니다."""
    strategy = store.strategy_cache.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"strategy_id '{strategy_id}' not found.")
    return StrategyResponse(**strategy)
