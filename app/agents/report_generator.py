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
기준 뉴스 본문:
{body}
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
        body=center.get("cleaned_content", ""),
        related_summaries=related_summaries,
        rag_block=_rag_block(rag_articles),
    )

    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None

    if not parsed:
        report = _dummy_report(title)
        report["rag_sources"] = rag_sources
        report["is_fallback"] = True
        return report

    return {
        "is_fallback": False,
        "title": parsed.get("title", f"{title[:30]} 리포트"),
        "summary": parsed.get("summary", ""),
        "event_analysis": parsed.get("event_analysis", ""),
        "market_impact": parsed.get("market_impact", ""),
        "risk_factors": parsed.get("risk_factors", []),
        "rag_sources": rag_sources,
    }


# ── 선택 기사 리포트 (뉴스맵: 여러 근거 기사 · 종목 영향 · 전략 입장) ───────────────

SELECTION_PROMPT_VERSION = "selection-report-v1"
_BODY_LIMIT = 4000  # 기사당 본문 상한(문자). 5건이어도 프롬프트가 과도하게 길어지지 않게 한다.
_DIRECTIONS = {"up", "down", "mixed"}
_ACTIONS = {"buy", "hold", "sell", "watch"}

_SELECTION_PROMPT = """\
아래는 사용자가 직접 고른 뉴스 {count}건이다. 이 기사들만 근거로 투자자를 위한 분석 리포트를 한국어로 작성해줘.
반드시 다음 JSON 형식만 출력하고, 다른 텍스트는 쓰지 마.
근거 기사(와 유사 과거 뉴스)에 없는 사실이나 종목은 만들어내지 마. 종목은 근거 기사에 이름이 나온 기업만 쓴다.
direction은 기사 내용이 해당 종목에 주는 영향의 해석이며 가격 예측이 아니다. 판단할 수 없으면 그 종목을 넣지 마.

{articles}
{rag_block}
출력 형식:
{{
  "title": "리포트 제목 (50자 이내)",
  "summary": "핵심 요약 (200자 이내)",
  "event_analysis": "사건 분석 (300자 이내)",
  "market_impact": "시장 영향 분석 (300자 이내)",
  "risk_factors": ["리스크1", "리스크2", "리스크3"],
  "stock_impacts": [
    {{"name": "기사에 나온 기업명", "ticker": "알면 종목코드, 모르면 빈 문자열",
      "direction": "up|down|mixed", "action": "buy|hold|sell|watch", "comment": "한 문장 해석"}}
  ],
  "strategy": {{"stance": "전략 입장 (15자 이내)", "rationale": "근거 (150자 이내)",
               "watchlist": ["관심 종목명"], "risk_warning": "주의 사항 (100자 이내)"}}
}}
"""


def _article_block(index: int, article: dict[str, Any]) -> str:
    body = (article.get("cleaned_content") or "")[:_BODY_LIMIT]
    description = article.get("description") or article.get("summary") or ""
    return (f"[기사 {index}] {article.get('title', '')}\n"
            f"설명: {description}\n"
            f"본문: {body or '(본문 없음 — 설명만 사용)'}\n")


def _text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _stock_impacts(raw: Any, evidence_text: str) -> list[dict[str, str]]:
    """근거 기사에 이름이 나온 기업만, 허용 값만 남긴다(모델 출력은 신뢰하지 않는다)."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 40)
        direction = item.get("direction")
        if not name or name in seen or direction not in _DIRECTIONS or name not in evidence_text:
            continue
        ticker = _text(item.get("ticker"), 12)
        action = item.get("action") if item.get("action") in _ACTIONS else "watch"
        items.append({"name": name, "ticker": "" if ticker == name else ticker, "direction": direction,
                      "action": action, "comment": _text(item.get("comment"), 200)})
        seen.add(name)
    return items[:6]


def _strategy(raw: Any, evidence_text: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not _text(raw.get("stance"), 30):
        return None
    watchlist = [name for name in (_text(n, 40) for n in raw.get("watchlist") or [] if isinstance(n, str))
                 if name and name in evidence_text]
    return {"stance": _text(raw.get("stance"), 30), "rationale": _text(raw.get("rationale"), 300),
            "watchlist": list(dict.fromkeys(watchlist))[:6], "risk_warning": _text(raw.get("risk_warning"), 200)}


async def generate_selection_report(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """사용자가 고른 기사 1~5건(본문 또는 설명)을 근거로 리포트·종목 영향·전략 입장을 한 번에 생성.

    LLM 실패·파싱 실패 시 is_fallback=True 인 대체 결과(종목·전략 없음)를 돌려준다.
    """
    lead = articles[0]
    rag_articles = await retrieve_rag_articles(lead, k=3)
    chosen = {a.get("news_id") for a in articles}
    rag_articles = [a for a in rag_articles if a.get("news_id") not in chosen]
    rag_sources = [a.get("title", "") for a in rag_articles if a.get("title")]
    if rag_sources:
        print(f"[report] RAG grounded with {len(rag_sources)} similar articles")

    prompt = _SELECTION_PROMPT.format(
        count=len(articles),
        articles="\n".join(_article_block(i, a) for i, a in enumerate(articles, 1)),
        rag_block=_rag_block(rag_articles),
    )
    raw = await generate(prompt)
    parsed = _extract_json(raw) if raw else None
    if not parsed or not _text(parsed.get("summary"), 400):
        return {**_dummy_report(lead.get("title", "")), "stock_impacts": [], "strategy": None,
                "rag_sources": rag_sources, "is_fallback": True}

    evidence_text = "\n".join(
        f"{a.get('title', '')} {a.get('description') or a.get('summary') or ''} {a.get('cleaned_content') or ''}"
        for a in articles
    )
    risks = [_text(r, 200) for r in parsed.get("risk_factors") or [] if _text(r, 200)]
    return {
        "title": _text(parsed.get("title"), 80) or f"{lead.get('title', '')[:30]} 리포트",
        "summary": _text(parsed.get("summary"), 400),
        "event_analysis": _text(parsed.get("event_analysis"), 600),
        "market_impact": _text(parsed.get("market_impact"), 600),
        "risk_factors": risks[:5],
        "stock_impacts": _stock_impacts(parsed.get("stock_impacts"), evidence_text),
        "strategy": _strategy(parsed.get("strategy"), evidence_text),
        "rag_sources": rag_sources,
        "is_fallback": False,
    }
