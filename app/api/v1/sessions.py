"""
GET/DELETE /api/v1/session — 쿠키 기반 세션 상태 조회·초기화
POST /api/v1/session/mindmap/expand · collapse — 마인드맵 확장 상태 기록

로그인을 넣지 않기로 했으므로 계정 대신 세션으로 사용자를 구분한다.
세션 id는 미들웨어(app/main.py)가 쿠키로 발급하며, 클라이언트가 신경 쓸 것은 없다.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app import session as session_store
from app.schemas import MindmapState, SessionResponse

router = APIRouter()


class NodeRequest(BaseModel):
    news_id: str


def _to_response(data: dict) -> SessionResponse:
    mindmap = data.get("mindmap") or {}
    return SessionResponse(
        session_id=data.get("sid", ""),
        mindmap=MindmapState(
            center_news_id=mindmap.get("center_news_id", ""),
            expanded_news_ids=mindmap.get("expanded_news_ids", []),
            query=mindmap.get("query", ""),
        ),
        viewed_news_ids=data.get("viewed_news_ids", []),
    )


@router.get("", response_model=SessionResponse)
async def read_session(request: Request) -> SessionResponse:
    """현재 세션의 마인드맵 상태를 반환합니다. 쿠키가 없으면 새로 발급됩니다."""
    data = await session_store.load(request.state.session_id)
    data["sid"] = request.state.session_id
    return _to_response(data)


@router.delete("", response_model=SessionResponse)
async def reset_session(request: Request) -> SessionResponse:
    """세션 데이터를 비웁니다. 마인드맵 상태가 함께 사라집니다."""
    sid = request.state.session_id
    await session_store.clear(sid)
    data = await session_store.load(sid)
    data["sid"] = sid
    return _to_response(data)


@router.post("/mindmap/expand", response_model=SessionResponse)
async def expand(request: Request, body: NodeRequest) -> SessionResponse:
    """마인드맵에서 노드를 펼쳤음을 기록합니다 (사용자별로 분리됩니다)."""
    data = await session_store.expand_node(request.state.session_id, body.news_id)
    return _to_response(data)


@router.post("/mindmap/collapse", response_model=SessionResponse)
async def collapse(request: Request, body: NodeRequest) -> SessionResponse:
    """펼친 노드를 접습니다."""
    data = await session_store.collapse_node(request.state.session_id, body.news_id)
    return _to_response(data)


@router.delete("/mindmap", response_model=SessionResponse)
async def reset_mindmap(request: Request) -> SessionResponse:
    """마인드맵 상태만 초기화합니다 (세션 자체는 유지)."""
    data = await session_store.reset_mindmap(request.state.session_id)
    return _to_response(data)
