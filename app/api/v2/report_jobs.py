"""V2 asynchronous report admission, cancellation and protected result reads."""
from __future__ import annotations

import json
import time
import uuid
from datetime import timedelta
from typing import Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from app import session as base_session
from app import store
from app.api.v2.schemas import Report, ReportJob, Strategy
from app.config import settings
from app.usage_limits import check_quota
from app.v2 import persistence, reports, state
from app.v2.errors import V2Error

router = APIRouter()
_ADMIT = """
if redis.call('EXISTS', KEYS[4]) == 0 then return -2 end
local session_raw = redis.call('GET', KEYS[5])
local session_ttl = redis.call('PTTL', KEYS[5])
if not session_raw or session_ttl <= 0 then return -2 end
local session = cjson.decode(session_raw)
if session.session_epoch ~= ARGV[7] then return -2 end
local old = redis.call('GET', KEYS[1])
if not old then return -1 end
local revision = tonumber(cjson.decode(old).revision)
if revision ~= tonumber(ARGV[1]) then return revision end
if redis.call('EXISTS', KEYS[3]) == 1 then return -3 end
session.resource_index = session.resource_index or {}
session.resource_index['report-job'] = session.resource_index['report-job'] or {}
session.resource_index['report-job'][ARGV[8]] = ARGV[9]
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[5])
redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[6])
redis.call('SET', KEYS[3], ARGV[4], 'EX', ARGV[6])
redis.call('SET', KEYS[5], cjson.encode(session), 'PX', session_ttl)
return 0
"""


async def _admit(sid: str, map_doc: dict, job: dict, idem_key: str, request_hash: str) -> None:
    """Map reservation, job, replay key and resource index share one commit."""
    map_key = state.key(sid, "map", job["map_id"])
    job_key = state.key(sid, "report-job", job["job_id"])
    session_key = f"v2:session:{sid}"
    map_doc["revision"] = job["accepted_map_revision"]
    replay = {"request_hash": request_hash, "job_id": job["job_id"]}
    map_ttl = max(1, int(state.epoch(map_doc["expires_at"]) - time.time()))
    async with state.mutex(f"v2:index-lock:{sid}"):
        if state.memory_enabled():
            # Match root's CAS lock so fixture admissions cannot interleave
            # between their revision check and the four related writes.
            async with state.mutex(f"cas:{map_key}"):
                session = await state.get(session_key)
                if await base_session._existing(sid) is None or not session or session["session_epoch"] != job["session_epoch"]:
                    raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
                current = await state.get(map_key)
                if not current:
                    raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")
                if current["revision"] != job["source_map_revision"]:
                    raise V2Error(409, "MAP_REVISION_CONFLICT", "탐색 상태가 변경되었습니다.", details={"revision": current["revision"]})
                session["resource_index"].setdefault("report-job", {})[job["job_id"]] = job["expires_at"]
                await state.put(map_key, map_doc, map_ttl)
                await state.put(job_key, job, settings.v2_job_ttl_seconds)
                await state.put(idem_key, replay, settings.v2_job_ttl_seconds)
                await state.put(session_key, session, state.epoch(session["expires_at"]) - time.time())
            return
        try:
            status = await state._client().eval(_ADMIT, 5, map_key, job_key, idem_key, f"session:{sid}", session_key,
                                                job["source_map_revision"], json.dumps(map_doc, ensure_ascii=False),
                                                json.dumps(job, ensure_ascii=False), json.dumps(replay),
                                                map_ttl, settings.v2_job_ttl_seconds, job["session_epoch"], job["job_id"], job["expires_at"])
        except Exception as exc:
            raise V2Error(503, "DEPENDENCY_UNAVAILABLE", "분석 작업을 접수하지 못했습니다.", True) from exc
        if status == -2:
            raise V2Error(410, "SESSION_EXPIRED", "세션이 만료되었습니다.")
        if status == -1:
            raise V2Error(410, "RESOURCE_EXPIRED", "탐색이 만료되었습니다.")
        if status == -3:
            raise V2Error(409, "IDEMPOTENCY_CONFLICT", "같은 요청 키가 이미 접수되었습니다.")
        if status:
            raise V2Error(409, "MAP_REVISION_CONFLICT", "탐색 상태가 변경되었습니다.", details={"revision": status})


class StrategyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: Literal["low", "medium", "high"] = "medium"
    period: Literal["short", "mid", "long"] = "short"


class ReportJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    map_id: StrictStr = Field(min_length=1)
    expected_revision: StrictInt = Field(ge=1)
    client_request_id: uuid.UUID
    news_ids: list[StrictStr] = Field(min_length=1, max_length=10)
    language: Literal["ko"] = "ko"
    report_type: Literal["investment"] = "investment"
    strategy_options: StrategyOptions = Field(default_factory=StrategyOptions)

    @field_validator("news_ids")
    @classmethod
    def unique_ids(cls, value):
        if len(set(value)) != len(value) or any(not nid for nid in value):
            raise ValueError("Article IDs must be nonempty and unique")
        return value


async def _dispatch(sid: str, job: dict) -> dict:
    return await reports.dispatch_job(sid, job)


@router.post("/report-jobs", status_code=202, response_model=ReportJob)
async def create_job(body: ReportJobCreate, request: Request, idempotency_key: str = Header(alias="Idempotency-Key")):
    try:
        key_id = str(uuid.UUID(idempotency_key))
    except ValueError as exc:
        raise V2Error(422, "INVALID_REQUEST", "Idempotency-Key는 UUID여야 합니다.") from exc
    sid = request.state.session_id
    session = await state.session(sid, touch=True)
    canonical = body.model_dump(mode="json")
    request_hash = reports.digest(canonical)
    idem_key = f"v2:idem:{sid}:report-jobs:{key_id}"
    async with state.mutex(f"v2:idem-lock:{sid}:{key_id}"):
        previous = await state.get(idem_key)
        if previous:
            if previous["request_hash"] != request_hash:
                raise V2Error(409, "IDEMPOTENCY_CONFLICT", "같은 요청 키에 다른 입력을 보낼 수 없습니다.")
            existing = await reports.get_job(sid, previous["job_id"])
            return reports.public_job(await _dispatch(sid, existing))
        if session["prototype_tier"] != "PAID" and len(body.news_ids) > 1:
            raise V2Error(403, "PLAN_ARTICLE_LIMIT_EXCEEDED", "무료 세션은 기사 1건까지 분석할 수 있습니다.")
        await persistence.ready()
        from app.v2.maps import load_map
        async with state.mutex(f"v2:map-mutex:{sid}"):
            map_doc = await load_map(sid, body.map_id)
            if map_doc["revision"] != body.expected_revision:
                raise V2Error(409, "MAP_REVISION_CONFLICT", "탐색 상태가 변경되었습니다.", details={"revision": map_doc["revision"]})
            visible = {n["news_id"] for n in map_doc["nodes"]}
            if not set(body.news_ids).issubset(visible):
                raise V2Error(422, "INVALID_REQUEST", "현재 탐색에 표시된 기사만 분석할 수 있습니다.")
            for news_id in body.news_ids:
                if await store.get_news(news_id) is None:
                    raise V2Error(404, "RESOURCE_NOT_FOUND", "요청한 기사를 찾을 수 없습니다.")
            # Reservation remains outside the transaction but follows validation;
            # client/global buckets cannot be reset by creating a fresh SID.
            await check_quota(sid, "reports", per_session=5, global_limit=100, client_host=request.client.host if request.client else None, per_client=settings.reports_client_hourly_limit)
            job_id = str(uuid.uuid4())
            observed = persistence.now()
            expires_at = persistence.iso(observed + timedelta(seconds=settings.v2_job_ttl_seconds))
            job = {"job_id": job_id, "client_request_id": canonical["client_request_id"], "map_id": body.map_id, "accepted_map_revision": body.expected_revision + 1, "status": "queued", "stage": "queued", "request_news_ids": body.news_ids, "progress": {"requested": len(body.news_ids), "processed": 0, "succeeded": 0, "failed": 0}, "result": None, "warnings": [], "error": None, "created_at": persistence.iso(observed), "updated_at": persistence.iso(observed), "expires_at": expires_at, "request": canonical, "session_epoch": session["session_epoch"], "source_map_revision": body.expected_revision, "dispatch_state": "pending", "deadline_at": persistence.iso(observed + timedelta(seconds=settings.v2_job_timeout_seconds)), "worker_token": None, "lease_until": 0, "worker_retries": 0, "client_host": request.client.host if request.client else None}
            previous_job = map_doc.get("active_report_job_id")
            map_doc.update(selected_news_ids=body.news_ids, report_panel_open=True, active_report_job_id=job_id)
            await _admit(sid, map_doc, job, idem_key, request_hash)
            if previous_job:
                await reports.request_cancel(sid, previous_job, "superseded")
        return reports.public_job(await _dispatch(sid, job))


@router.get("/report-jobs/{job_id}", response_model=ReportJob)
async def get_job(job_id: str, request: Request):
    return reports.public_job(await reports.get_job(request.state.session_id, job_id))


@router.post("/report-jobs/{job_id}/cancel", response_model=ReportJob)
async def cancel_job(job_id: str, request: Request):
    sid = request.state.session_id
    await state.session(sid, touch=True)
    await reports.get_job(sid, job_id)
    return reports.public_job(await reports.cancel_job(sid, job_id))


@router.get("/reports/{report_id}", response_model=Report)
async def get_report(report_id: str, request: Request):
    await state.authorize_result(request.state.session_id, report_id, "report")
    doc = await persistence.report(report_id)
    if not doc:
        raise V2Error(404, "RESOURCE_NOT_FOUND", "리포트를 찾을 수 없습니다.")
    return persistence.public_report(doc)


@router.get("/strategies/{strategy_id}", response_model=Strategy)
async def get_strategy(strategy_id: str, request: Request):
    await state.authorize_result(request.state.session_id, strategy_id, "strategy")
    doc = await persistence.strategy(strategy_id)
    if not doc:
        raise V2Error(404, "RESOURCE_NOT_FOUND", "전략을 찾을 수 없습니다.")
    return persistence.public_strategy(doc)
