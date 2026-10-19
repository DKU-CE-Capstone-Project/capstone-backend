"""v2 session and map endpoints."""
from fastapi import APIRouter, Request, Response

from app import session as base_session
from app.api.v2 import schemas
from app.config import settings
from app.usage_limits import QuotaRequest, check_quota, check_quotas, client_key
from app.v2 import maps, state

router = APIRouter()


def _quota(request: Request, action: str) -> QuotaRequest:
    peer = client_key(request.client.host if request.client else None)
    return QuotaRequest(peer, action,
                        per_session=(settings.news_map_client_hourly_limit if action == 'news_map'
                                     else settings.news_search_client_hourly_limit),
                        global_limit=(settings.news_map_global_hourly_limit if action == 'news_map'
                                      else settings.news_search_global_hourly_limit))


async def _limit(request: Request, action: str):
    quota = _quota(request, action)
    await check_quota(quota.owner_sid, quota.action,
                      per_session=quota.per_session, global_limit=quota.global_limit)


@router.get('/session', response_model=schemas.Session)
async def read_session(request: Request):
    session = await state.session(request.state.session_id, touch=True)
    return {k: session[k] for k in ('session_epoch', 'expires_at', 'prototype_tier')} | {
        'resumed': not request.state.session_is_new, 'capabilities': maps.capabilities(session)}


@router.delete('/session', status_code=204)
async def delete_session(request: Request):
    sid = request.state.session_id
    # Revoke the identity first. A concurrent touch may only extend a live
    # key and cannot recreate it; workers reject the revoked parent session.
    await base_session.clear(sid)
    await state.clear_session(sid)
    request.state.session_deleted = True
    return Response(status_code=204)


@router.post('/maps', status_code=201, response_model=schemas.Map)
async def create_map(request: Request, body: maps.MapCreate):
    await check_quotas(_quota(request, 'news_search'), _quota(request, 'news_map'))
    return maps.public_map(await maps.create_map(request.state.session_id, body.query))


@router.get('/maps/{map_id}', response_model=schemas.Map)
async def get_map(request: Request, map_id: str):
    value = await maps.load_map(request.state.session_id, map_id)
    await state.session(request.state.session_id, touch=True)
    # Expired work cannot remain an active link; preserve already saved results.
    job_id = value.get('active_report_job_id')
    if job_id and await state.get(state.key(request.state.session_id, 'report-job', job_id)) is None:
        value['active_report_job_id'] = None
        value = await maps.commit_map(request.state.session_id, value, value['revision'])
    value['expires_at'] = (await state.session(request.state.session_id))['expires_at']
    return maps.public_map(value)


@router.patch('/maps/{map_id}', response_model=schemas.Map)
async def update_map(request: Request, map_id: str, body: maps.MapUpdate):
    await state.session(request.state.session_id, touch=True)
    return maps.public_map(await maps.update_map(request.state.session_id, map_id, body))


@router.post('/maps/{map_id}/branches', response_model=schemas.Map)
async def branch_map(request: Request, map_id: str, body: maps.BranchRequest):
    await _limit(request, 'news_map')
    await state.session(request.state.session_id, touch=True)
    return maps.public_map(await maps.branch_map(request.state.session_id, map_id, body))


@router.post('/maps/{map_id}/reset', response_model=schemas.Map)
async def reset_map(request: Request, map_id: str, body: maps.Revision):
    await state.session(request.state.session_id, touch=True)
    return maps.public_map(await maps.reset_map(request.state.session_id, map_id, body.expected_revision))


@router.delete('/maps/{map_id}', status_code=204)
async def delete_map(request: Request, map_id: str):
    await maps.delete_map(request.state.session_id, map_id)
    return Response(status_code=204)


@router.get('/news/{news_id}', response_model=schemas.NewsCard)
async def get_news(request: Request, news_id: str):
    return await maps.news_card(request.state.session_id, news_id)
