"""
POST /jobs, GET /jobs/{id} — 비동기 분석 작업 (NATS + Redis).

발표 데모의 핵심 경로:
  POST /jobs → job_id 발급 + NATS 발행(202 Accepted)
            → worker가 소비·처리 → Redis 저장
  GET /jobs/{id} → Redis에서 상태/결과 폴링

이 비동기 경로 덕분에 속보 burst가 와도 API는 즉시 202를 반환하고,
실제 처리는 KEDA가 오토스케일하는 worker 풀이 큐에서 흡수한다.
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.job_store import get_job, set_job
from app.messaging.nats_client import publish_job

router = APIRouter()


class JobCreateRequest(BaseModel):
    keyword: str


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


@router.post("", response_model=JobCreateResponse, status_code=202)
async def create_job(body: JobCreateRequest) -> JobCreateResponse:
    """분석 job을 큐에 넣고 즉시 job_id를 반환한다 (202 Accepted)."""
    job_id = str(uuid.uuid4())
    await set_job(job_id, {"status": "queued", "keyword": body.keyword})
    await publish_job(json.dumps({"job_id": job_id, "keyword": body.keyword}).encode())
    return JobCreateResponse(job_id=job_id, status="queued")


@router.get("/{job_id}")
async def read_job(job_id: str) -> dict:
    """job 상태/결과를 조회한다."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return job
