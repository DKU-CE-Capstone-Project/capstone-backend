from fastapi import APIRouter

from app.agents.orchestrator import run_analysis
from app.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return await run_analysis(req.keyword)
