from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.naver_client import NaverNewsError
from app.api.routes import router as legacy_router
from app.api.v1 import jobs, keywords, news, reports, sessions, strategies
from app.config import settings
from app.database import DatabasePersistenceError
from app.session import SESSION_COOKIE, new_session_id

app = FastAPI(
    title="Capstone — News Multi-Agent API",
    version="1.0.0",
    description="실시간 뉴스 기반 멀티 에이전트 투자 판단 지원 시스템",
)


@app.exception_handler(NaverNewsError)
async def naver_error_handler(request, exc: NaverNewsError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(DatabasePersistenceError)
async def persistence_error_handler(request, exc: DatabasePersistenceError):
    return JSONResponse(status_code=503, content={"detail": "데이터를 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."})


@app.get("/ready", tags=["system"])
async def readiness():
    from app import database

    if not settings.use_mongodb:
        return JSONResponse(status_code=503 if settings.mongodb_required else 200,
                            content={"status": "error" if settings.mongodb_required else "ok", "mongodb": "off"})
    connected = await database.ping()
    return JSONResponse(status_code=200 if connected else 503,
                        content={"status": "ok" if connected else "error", "mongodb": "on" if connected else "unavailable"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    # 세션 쿠키를 주고받아야 하므로 credentials를 허용한다.
    # allow_origins=["*"]이어도 Starlette가 쿠키 요청에는 요청 Origin을 그대로 echo하므로
    # 브라우저의 "credentials + wildcard 금지" 규칙에 걸리지 않는다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 세션 쿠키 발급 ─────────────────────────────────────────────────────────────
# 로그인을 넣지 않기로 했으므로(12주차 회의) 계정 대신 쿠키로 사용자를 구분한다.
# 모든 요청에 세션 id를 붙여 두면 각 엔드포인트는 request.state.session_id만 보면 된다.
@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE) or ""
    is_new = not sid
    if is_new:
        sid = new_session_id()
    request.state.session_id = sid

    response = await call_next(request)

    if is_new:
        response.set_cookie(
            key=SESSION_COOKIE,
            value=sid,
            max_age=settings.session_ttl_seconds,
            httponly=True,          # JS에서 읽을 필요가 없다 (XSS로 세션 탈취 방지)
            samesite=settings.session_cookie_samesite,
            secure=settings.session_cookie_secure,
            path="/",
        )
    return response


# ── /api/v1 라우터 ─────────────────────────────────────────────────────────────
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(news.router,       prefix="/news",       tags=["news"])
api_v1.include_router(keywords.router,   prefix="/keywords",   tags=["keywords"])
api_v1.include_router(reports.router,    prefix="/reports",    tags=["reports"])
api_v1.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
api_v1.include_router(sessions.router,   prefix="/session",    tags=["session"])

app.include_router(api_v1)

# ── 비동기 작업 큐 (NATS + Redis) — CNCF 오토스케일 데모 경로 ────────────────────
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

# ── 기존 엔드포인트 (프론트 backward compat) ────────────────────────────────────
app.include_router(legacy_router)


@app.on_event("startup")
async def _startup() -> None:
    # MongoDB 인덱스 + 벡터검색 인덱스 보장 (use_mongodb 시, 실패해도 graceful)
    from app import database

    await database.ensure_indexes()


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    from app import database

    return {"status": "ok", "mongodb": "on" if database.enabled() else "off"}
