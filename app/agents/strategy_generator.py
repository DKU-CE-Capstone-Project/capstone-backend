"""Generate investment strategy via Gemini (with deterministic fallback)."""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.llm import generate

_PROMPT = """\
아래 투자 리포트를 바탕으로 구체적인 투자 전략을 한국어로 작성해줘.
반드시 다음 JSON 형식만 출력하고, 다른 텍스트는 쓰지 마.

리포트 요약: {summary}
리스크 수준: {risk_level}  (low=보수적, medium=중립, high=공격적)
투자 기간: {period}  (short=단기 1~3개월, mid=중기 3~12개월, long=장기 1년 이상)

출력 형식:
{{
  "expected_return": 12.5,
  "risk": "{risk_level}",
  "strategy_summary": "전략 요약 (200자 이내)",
  "strategy_items": [
    {{
      "ticker": "005930",
      "stock_name": "삼성전자",
      "action": "buy",
      "reason": "판단 근거 (100자 이내)"
    }}
  ]
}}

action 값은 buy, hold, sell, watch 중 하나만 사용.
strategy_items는 3~5개 생성.
"""

_DUMMY_ITEMS = [
    {
        "ticker": "005930",
        "stock_name": "삼성전자",
        "action": "buy",
        "reason": "AI 서버 수요 증가로 메모리 반도체 수혜 예상",
    },
    {
        "ticker": "000660",
        "stock_name": "SK하이닉스",
        "action": "hold",
        "reason": "HBM 공급 증가로 단기 수익 실현 후 관망 권장",
    },
    {
        "ticker": "NVDA",
        "stock_name": "엔비디아",
        "action": "watch",
        "reason": "밸류에이션 부담이 있으나 AI 수요 지속 시 추가 매수 고려",
    },
]


def _dummy_strategy(risk_level: str, period: str) -> dict[str, Any]:
    return_map = {"low": 5.0, "medium": 12.5, "high": 20.0}
    return {
        "expected_return": return_map.get(risk_level, 12.5),
        "risk": risk_level,
        "strategy_summary": "(AI 분석 준비 중) 기본 포트폴리오 전략을 제공합니다.",
        "strategy_items": _DUMMY_ITEMS,
    }


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


async def generate_strategy(
    report_summary: str,
    risk_level: str = "medium",
    period: str = "short",
) -> dict[str, Any]:
    """
    Generate an investment strategy using Gemini.

    Falls back to a dummy strategy on API error or parse failure.
    """
    prompt = _PROMPT.format(
        summary=report_summary,
        risk_level=risk_level,
        period=period,
    )

    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None

    if not parsed:
        return _dummy_strategy(risk_level, period)

    items = parsed.get("strategy_items", _DUMMY_ITEMS)
    valid_actions = {"buy", "hold", "sell", "watch"}
    cleaned_items = [
        {
            "ticker": it.get("ticker", ""),
            "stock_name": it.get("stock_name", ""),
            "action": it.get("action", "watch") if it.get("action") in valid_actions else "watch",
            "reason": it.get("reason", ""),
        }
        for it in items
    ]

    return {
        "expected_return": float(parsed.get("expected_return", 10.0)),
        "risk": parsed.get("risk", risk_level),
        "strategy_summary": parsed.get("strategy_summary", ""),
        "strategy_items": cleaned_items,
    }
