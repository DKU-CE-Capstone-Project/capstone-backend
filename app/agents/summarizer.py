import asyncio
import hashlib
import json
import re
from typing import Any

from app import store
from app.agents.llm import generate
from app.config import settings

# 영문 뉴스를 한국어 제목 + 한국어 한 줄 요약으로 변환 (JSON 출력)
_PROMPT = (
    "다음 뉴스를 한국어로 처리해. 반드시 아래 JSON 형식만 출력하고 다른 텍스트는 쓰지 마.\n\n"
    "제목: {title}\n설명: {description}\n\n"
    '출력: {{"title_ko": "한국어로 번역한 자연스러운 제목", '
    '"summary_ko": "투자 판단에 의미 있는 사실 중심 한국어 한 줄 요약(60자 내외)"}}'
)


def _content_key(title: str, description: str) -> str:
    return hashlib.md5(f"{title}|{description}".encode()).hexdigest()


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = re.sub(r"```(?:json)?", "", raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


async def _summarize_one(article: dict[str, Any]) -> dict[str, Any]:
    title = article.get("title") or ""
    description = article.get("description") or ""

    # ── 결과 캐싱: 같은 기사면 재번역/재요약 없이 재사용 ──
    key = _content_key(title, description)
    cached = store.summary_cache.get(key)
    if cached:
        return {
            **article,
            "title": cached.get("title_ko") or title,
            "title_original": title,
            "summary": cached.get("summary_ko") or (description or title)[:120],
        }

    # LLM(Claude 우선) 사용 시: 한국어 제목 + 한국어 요약 생성
    if settings.use_llm_summaries:
        raw = await generate(_PROMPT.format(title=title, description=description[:1500]))
        parsed = _extract_json(raw)
        if parsed and (parsed.get("summary_ko") or parsed.get("title_ko")):
            title_ko = (parsed.get("title_ko") or "").strip() or title
            summary_ko = (parsed.get("summary_ko") or "").strip() or (description or title)[:120]
            store.summary_cache[key] = {"title_ko": title_ko, "summary_ko": summary_ko}
            return {**article, "title": title_ko, "title_original": title, "summary": summary_ko}

    # LLM 불가(worker/키 없음 등) → 원문 유지 + description 요약 fallback
    return {**article, "summary": (description or title)[:120], "title_original": title}


async def summarize_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """각 기사를 동시 번역·요약 (한국어). 결과는 store.summary_cache에 캐싱."""
    return await asyncio.gather(*(_summarize_one(a) for a in articles))
