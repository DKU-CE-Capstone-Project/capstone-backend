"""Generate AI investment report via Gemini (with deterministic fallback)."""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.llm import embed, generate
from app.config import settings

_PROMPT = """\
아래 뉴스를 바탕으로 투자자를 위한 분석 리포트를 한국어로 작성해줘.
반드시 다음 JSON 형식만 출력하고, 다른 텍스트는 쓰지 마.
근거(연관 뉴스·유사 과거 뉴스)에 없는 사실이나 종목은 만들어내지 말고, 근거에 기반해 분석해.

기준 뉴스 제목: {title}
기준 뉴스 요약: {summary}
연관 뉴스 요약:
{related_summaries}
{rag_block}
출력 형식:
{{
  "title": "리포트 제목 (50자 이내)",
  "summary": "핵심 요약 (200자 이내)",
  "event_analysis": "사건 분석 (300자 이내)",
  "market_impact": "시장 영향 분석 (300자 이내)",
  "risk_factors": ["리스크1", "리스크2", "리스크3"]
}}
"""


def _dummy_report(title: str) -> dict[str, Any]:
    return {
        "title": f"{title[:30]} — 투자 분석 리포트",
        "summary": "(AI 분석 준비 중) 해당 뉴스에 대한 요약을 생성하지 못했습니다.",
        "event_analysis": "(AI 분석 준비 중) 사건 분석을 생성하지 못했습니다.",
        "market_impact": "(AI 분석 준비 중) 시장 영향 분석을 생성하지 못했습니다.",
        "risk_factors": ["API 한도 초과 또는 네트워크 오류로 인해 분석이 지연되고 있습니다."],
    }


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract first JSON object from Gemini output (handles markdown fences)."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


async def retrieve_rag_articles(center: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    """벡터검색으로 의미적으로 유사한 과거 뉴스 top-k 반환 (RAG 근거).

    use_rag/use_mongodb 비활성이거나 임베딩/검색 실패 시 빈 리스트(graceful).
    """
    if not (settings.use_rag and settings.use_mongodb):
        return []
    from app import database

    text = f"{center.get('title','')} {center.get('summary') or center.get('description','')}".strip()
    vec = await embed(text)
    if not vec:
        return []
    return await database.vector_search_articles(vec, k=k, exclude_id=center.get("news_id", ""))


def _rag_block(rag_articles: list[dict[str, Any]]) -> str:
    if not rag_articles:
        return ""
    lines = "\n".join(
        f"- {a.get('title','')}: {a.get('summary') or a.get('description','')}"
        for a in rag_articles
    )
    return f"유사 과거 뉴스(벡터검색 근거):\n{lines}\n"


async def generate_report(
    center: dict[str, Any],
    related: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an investment analysis report using Gemini, grounded with RAG.

    벡터검색으로 유사 과거 뉴스를 근거로 주입(환각↓). API 오류·파싱 실패 시 dummy fallback.
    반환 dict에는 리포트 필드 + rag_sources(근거 뉴스 제목 목록)가 포함된다.
    """
    title = center.get("title", "")
    summary = center.get("summary") or center.get("description", "")

    related_summaries = "\n".join(
        f"- {art.get('summary') or art.get('description', '')}"
        for art in related[:5]
        if art.get("summary") or art.get("description")
    ) or "(연관 뉴스 없음)"

    # RAG: 의미적으로 유사한 과거 뉴스를 근거로 검색
    rag_articles = await retrieve_rag_articles(center, k=3)
    rag_sources = [a.get("title", "") for a in rag_articles if a.get("title")]
    if rag_sources:
        print(f"[report] RAG grounded with {len(rag_sources)} similar articles")

    prompt = _PROMPT.format(
        title=title,
        summary=summary,
        related_summaries=related_summaries,
        rag_block=_rag_block(rag_articles),
    )

    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None

    if not parsed:
        report = _dummy_report(title)
        report["rag_sources"] = rag_sources
        return report

    return {
        "title": parsed.get("title", f"{title[:30]} 리포트"),
        "summary": parsed.get("summary", ""),
        "event_analysis": parsed.get("event_analysis", ""),
        "market_impact": parsed.get("market_impact", ""),
        "risk_factors": parsed.get("risk_factors", []),
        "rag_sources": rag_sources,
    }
