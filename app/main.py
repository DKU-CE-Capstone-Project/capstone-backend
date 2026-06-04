from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as legacy_router
from app.api.v1 import jobs, keywords, news, reports, strategies
from app.config import settings

app = FastAPI(
    title="Capstone — News Multi-Agent API",
    version="1.0.0",
    description="실시간 뉴스 기반 멀티 에이전트 투자 판단 지원 시스템",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── /api/v1 라우터 ─────────────────────────────────────────────────────────────
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(news.router,       prefix="/news",       tags=["news"])
api_v1.include_router(keywords.router,   prefix="/keywords",   tags=["keywords"])
api_v1.include_router(reports.router,    prefix="/reports",    tags=["reports"])
api_v1.include_router(strategies.router, prefix="/strategies", tags=["strategies"])

app.include_router(api_v1)

# ── 비동기 작업 큐 (NATS + Redis) — CNCF 오토스케일 데모 경로 ────────────────────
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

# ── 기존 엔드포인트 (프론트 backward compat) ────────────────────────────────────
app.include_router(legacy_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
