from fastapi import APIRouter, Request

from app.agents.orchestrator import run_analysis
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.usage_limits import check_quota

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    await check_quota(request.state.session_id, "analyze", per_session=10, global_limit=200)
    return await run_analysis(req.keyword)
