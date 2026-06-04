import asyncio
from typing import Any

from app.agents.llm import generate
from app.config import settings

_PROMPT = (
    "다음 영문/한국어 뉴스 기사를 한국어 한 문장으로 요약해. "
    "투자 판단에 의미 있는 사실(인물, 정책, 수치, 시장 영향)에 집중하고, "
    "추측이나 사족은 빼. 50자 내외.\n\n"
    "제목: {title}\n설명: {description}\n\n요약:"
)


async def _summarize_one(article: dict[str, Any]) -> dict[str, Any]:
    description = article.get("description") or ""
    title = article.get("title") or ""

    # 무료 Gemini 쿼터 절약: use_llm_summaries=false면 LLM 호출 없이 description 사용.
    # (Gemini는 리포트/전략 생성에만 사용 → 리포트가 안정적으로 실제 생성됨)
    if settings.use_llm_summaries:
        summary = await generate(_PROMPT.format(title=title, description=description))
        if not summary:
            summary = (description or title)[:120]
    else:
        summary = (description or title)[:120]

    return {**article, "summary": summary}


async def summarize_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each article concurrently."""
    return await asyncio.gather(*(_summarize_one(a) for a in articles))
