"""Generate AI investment report via Gemini (with deterministic fallback)."""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.llm import generate

_PROMPT = """\
아래 뉴스를 바탕으로 투자자를 위한 분석 리포트를 한국어로 작성해줘.
반드시 다음 JSON 형식만 출력하고, 다른 텍스트는 쓰지 마.

기준 뉴스 제목: {title}
기준 뉴스 요약: {summary}
연관 뉴스 요약:
{related_summaries}

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


async def generate_report(
    center: dict[str, Any],
    related: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an investment analysis report using Gemini.

    Falls back to a dummy report on API error or parse failure.
    """
    title = center.get("title", "")
    summary = center.get("summary") or center.get("description", "")

    related_summaries = "\n".join(
        f"- {art.get('summary') or art.get('description', '')}"
        for art in related[:5]
        if art.get("summary") or art.get("description")
    ) or "(연관 뉴스 없음)"

    prompt = _PROMPT.format(
        title=title,
        summary=summary,
        related_summaries=related_summaries,
    )

    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None

    if not parsed:
        return _dummy_report(title)

    return {
        "title": parsed.get("title", f"{title[:30]} 리포트"),
        "summary": parsed.get("summary", ""),
        "event_analysis": parsed.get("event_analysis", ""),
        "market_impact": parsed.get("market_impact", ""),
        "risk_factors": parsed.get("risk_factors", []),
    }
