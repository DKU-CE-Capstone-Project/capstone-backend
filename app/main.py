import logging

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agents.naver_client import NaverNewsError
from app.api.routes import router as legacy_router
from app.api.v1 import jobs, keywords, news, reports, sessions, strategies
from app.api.v2 import maps as v2_maps
from app.api.v2 import report_jobs as v2_reports
from app.config import settings
from app.database import DatabasePersistenceError
from app.session import SESSION_COOKIE, SessionExpired, SessionStoreUnavailable, resolve
from app.v2.errors import V2Error

# News-map diagnostics are aggregate counts only; uvicorn does not configure app loggers.
_diagnostics = logging.getLogger("econmind")
if not _diagnostics.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    _diagnostics.addHandler(_handler)
    _diagnostics.setLevel(logging.INFO)

app = FastAPI(
    title="Capstone — News Multi-Agent API",
    version="1.0.0",
    description="실시간 뉴스 기반 멀티 에이전트 투자 판단 지원 시스템",
)


@app.exception_handler(V2Error)
async def v2_error_handler(request, exc: V2Error):
    return JSONResponse(status_code=exc.status, content=exc.envelope())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    if request.url.path.startswith('/api/v2/'):
        over_limit = any(error.get('type') == 'too_long' and error.get('loc')[-1:]
                         in [('news_ids',), ('selected_news_ids',)] for error in exc.errors())
        error = V2Error(422, 'ARTICLE_LIMIT_EXCEEDED' if over_limit else 'INVALID_REQUEST',
                        '기사 선택 상한은 10건입니다.' if over_limit else '요청 형식과 입력 한도를 확인해 주세요.')
        return JSONResponse(status_code=422, content=error.envelope())
    from fastapi.exception_handlers import request_validation_exception_handler
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request, exc):
    if request.url.path.startswith('/api/v2/'):
        code = {403: 'FORBIDDEN', 404: 'RESOURCE_NOT_FOUND', 429: 'RATE_LIMIT_EXCEEDED',
                503: 'DEPENDENCY_UNAVAILABLE'}.get(exc.status_code, 'INVALID_REQUEST')
        message = exc.detail if isinstance(exc.detail, str) else '요청을 처리할 수 없습니다.'
        error = V2Error(exc.status_code, code, message, exc.status_code in (429, 502, 503))
        return JSONResponse(status_code=exc.status_code, content=error.envelope(), headers=exc.headers)
    from fastapi.exception_handlers import http_exception_handler
    return await http_exception_handler(request, exc)


@app.exception_handler(NaverNewsError)
async def naver_error_handler(request, exc: NaverNewsError):
    if request.url.path.startswith('/api/v2/'):
        return await v2_error_handler(request, V2Error(exc.status_code, 'SEARCH_FAILED',
                                                     '기사 검색을 완료하지 못했습니다.', True))
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(DatabasePersistenceError)
async def persistence_error_handler(request, exc: DatabasePersistenceError):
    if request.url.path.startswith('/api/v2/'):
        return await v2_error_handler(request, V2Error(503, 'DEPENDENCY_UNAVAILABLE',
                                                     '데이터를 저장하지 못했습니다.', True))
    return JSONResponse(status_code=503, content={"detail": "데이터를 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."})


@app.exception_handler(SessionStoreUnavailable)
async def session_store_error_handler(request, exc: SessionStoreUnavailable):
    if request.url.path.startswith('/api/v2/'):
        return await v2_error_handler(request, V2Error(503, 'DEPENDENCY_UNAVAILABLE',
                                                     '세션 저장소를 사용할 수 없습니다.', True))
    return JSONResponse(status_code=503, content={"detail": "세션 저장소를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."})


@app.exception_handler(SessionExpired)
async def session_expired_handler(request, exc: SessionExpired):
    if request.url.path.startswith('/api/v2/'):
        return await v2_error_handler(request, V2Error(410, 'SESSION_EXPIRED', '세션이 만료되었습니다.'))
    return JSONResponse(status_code=410, content={"detail": "세션이 만료되었습니다."})


@app.get("/ready", tags=["system"])
async def readiness():
    from app import database
    from app import session as session_store

    redis_ready = not settings.session_store_required or await session_store.ping()

    if not settings.use_mongodb:
        ready = not settings.mongodb_required and redis_ready
        return JSONResponse(status_code=200 if ready else 503,
                            content={"status": "ok" if ready else "error", "mongodb": "off",
                                     "redis": "on" if redis_ready else "unavailable"})
    connected = await database.ping()
    ready = connected and redis_ready
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ok" if ready else "error",
                                 "mongodb": "on" if connected else "unavailable",
                                 "redis": "on" if redis_ready else "unavailable"})

# ── 세션 쿠키 발급 ─────────────────────────────────────────────────────────────
# 로그인을 넣지 않기로 했으므로(12주차 회의) 계정 대신 쿠키로 사용자를 구분한다.
# 모든 요청에 세션 id를 붙여 두면 각 엔드포인트는 request.state.session_id만 보면 된다.
@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    # Only identity-bearing endpoints need Redis-backed sessions. Keep health,
    # public news, and CORS preflights available during a session-store outage.
    path = request.url.path
    is_v2 = path.startswith('/api/v2/')
    needs_session = request.method != "OPTIONS" and (
        is_v2 or path.startswith(("/api/v1/session", "/api/v1/reports", "/api/v1/strategies", "/jobs/"))
        or path in ("/jobs", "/analyze")
    )
    if not needs_session:
        return await call_next(request)
    # CORS controls response reads, not cross-origin form submissions that
    # mutate a cookie session. Reject an explicit foreign Origin before work.
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.cors_origin_list:
            if is_v2:
                return await v2_error_handler(request, V2Error(403, 'FORBIDDEN', '허용되지 않은 요청 출처입니다.'))
            return JSONResponse(status_code=403, content={"detail": "허용되지 않은 요청 출처입니다."})
    try:
        if is_v2 and not (path == '/api/v2/session' and request.method == 'GET'):
            from app import session as session_store
            candidate = request.cookies.get(SESSION_COOKIE) or ''
            if not session_store._SID_PATTERN.fullmatch(candidate) or await session_store._existing(candidate) is None:
                return await v2_error_handler(request, V2Error(410, 'SESSION_EXPIRED', '세션이 만료되었습니다.'))
        sid, is_new = await resolve(
            request.cookies.get(SESSION_COOKIE) or "",
            request.client.host if request.client else None,
        )
    except SessionStoreUnavailable:
        if is_v2:
            return await v2_error_handler(request, V2Error(503, 'DEPENDENCY_UNAVAILABLE',
                                                         '세션 저장소를 사용할 수 없습니다.', True))
        return JSONResponse(status_code=503, content={"detail": "세션 저장소를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."})
    except HTTPException as exc:
        if is_v2:
            return await http_error_handler(request, exc)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
    request.state.session_id = sid
    request.state.session_is_new = is_new

    response = await call_next(request)

    if getattr(request.state, "session_deleted", False):
        response.delete_cookie(
            key=SESSION_COOKIE,
            path="/",
            httponly=True,
            samesite=settings.session_cookie_samesite,
            secure=settings.session_cookie_secure,
        )
    elif is_new or getattr(request.state, "session_rotated", False) or (
        is_v2 and (path == '/api/v2/session' or path.startswith('/api/v2/maps'))
        and response.status_code < 400
    ) or (
        is_v2 and request.method in ('POST', 'PATCH', 'PUT', 'DELETE') and response.status_code < 400
    ):
        response.set_cookie(
            key=SESSION_COOKIE,
            value=request.state.session_id,
            max_age=settings.session_ttl_seconds,
            httponly=True,          # JS에서 읽을 필요가 없다 (XSS로 세션 탈취 방지)
            samesite=settings.session_cookie_samesite,
            secure=settings.session_cookie_secure,
            path="/",
        )
    return response


# Keep CORS outside the session middleware so early 429/503 responses expose
# their headers to allowed browser origins as well as ordinary route responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# ── /api/v1 라우터 ─────────────────────────────────────────────────────────────
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(news.router,       prefix="/news",       tags=["news"])
api_v1.include_router(keywords.router,   prefix="/keywords",   tags=["keywords"])
api_v1.include_router(reports.router,    prefix="/reports",    tags=["reports"])
api_v1.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
api_v1.include_router(sessions.router,   prefix="/session",    tags=["session"])

app.include_router(api_v1)

api_v2 = APIRouter(prefix='/api/v2')
api_v2.include_router(v2_maps.router, tags=['v2 exploration'])
api_v2.include_router(v2_reports.router, tags=['v2 reports'])
app.include_router(api_v2)

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
