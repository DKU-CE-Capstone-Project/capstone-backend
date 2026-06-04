"""
검증(critic) 에이전트 — LLM-as-judge.

생성된 리포트가 근거 뉴스로 뒷받침되는지 점검한다. 근거 없는 주장,
존재하지 않는 종목/티커, 과장을 찾아 신뢰도 점수와 함께 보고한다.
멀티에이전트 파이프라인의 '검증 루프' 역할.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.llm import generate
from app.config import settings

_PROMPT = """\
너는 투자 리포트를 검증하는 사실확인 전문가다.
아래 '근거 뉴스'만을 기준으로 '리포트'를 검증해라.
근거에 없는 주장, 실제로 존재하지 않을 가능성이 있는 종목·티커, 과장된 표현을 찾아라.
반드시 다음 JSON 형식만 출력하고 다른 텍스트는 쓰지 마.

[근거 뉴스]
{evidence}

[리포트]
제목: {title}
요약: {summary}
사건 분석: {event_analysis}
시장 영향: {market_impact}
리스크: {risk_factors}

출력 형식:
{{
  "grounded": true 또는 false,
  "confidence": 0.0~1.0,
  "issues": ["문제점1", "문제점2"],
  "unsupported_claims": ["근거 없는 주장"],
  "invalid_tickers": ["의심되는 종목/티커"]
}}
"""


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "grounded": True,
        "confidence": None,
        "issues": [reason],
        "unsupported_claims": [],
        "invalid_tickers": [],
    }


async def verify_report(
    report: dict[str, Any],
    evidence_summaries: list[str],
) -> dict[str, Any]:
    """리포트를 근거 뉴스에 대해 검증. use_critic 비활성·오류 시 graceful skip."""
    if not settings.use_critic:
        return _skipped("검증 비활성(use_critic=false)")

    evidence = "\n".join(f"- {s}" for s in evidence_summaries if s) or "(근거 없음)"
    prompt = _PROMPT.format(
        evidence=evidence,
        title=report.get("title", ""),
        summary=report.get("summary", ""),
        event_analysis=report.get("event_analysis", ""),
        market_impact=report.get("market_impact", ""),
        risk_factors=", ".join(report.get("risk_factors", [])),
    )

    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None
    if not parsed:
        return _skipped("검증 생략(쿼터/오류)")

    return {
        "grounded": bool(parsed.get("grounded", True)),
        "confidence": parsed.get("confidence"),
        "issues": parsed.get("issues", []),
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "invalid_tickers": parsed.get("invalid_tickers", []),
    }
