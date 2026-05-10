from __future__ import annotations

import re
from typing import Any

from app.agents.llm import generate

_PROMPT = (
    "아래 뉴스 요약들을 보고, 사용자가 함께 검색하면 유용할 관련 키워드를 "
    "5~7개 한국어로 뽑아줘. 콤마(,) 한 줄로만 출력. 입력 키워드 자체는 빼고, "
    "회사/인물/사건/지표 같은 구체적 단어 위주로.\n\n"
    "입력 키워드: {keyword}\n\n"
    "요약 목록:\n{summaries}\n\n"
    "관련 키워드:"
)

# Gemini 없을 때 제외할 일반 불용어 (기사 제목에 많이 나오는 의미없는 영단어)
_STOPWORDS = {
    "news", "report", "update", "the", "and", "with", "for", "that",
    "this", "from", "have", "has", "will", "its", "are", "was", "but",
    "기사", "뉴스", "관련", "위한", "대한", "통해",
}

# 자주 등장하는 유의미 단어 → 한국어 레이블로 변환 (fallback 품질 개선)
_EN_TO_KO_LABEL: dict[str, str] = {
    "nvidia": "엔비디아",
    "samsung": "삼성전자",
    "apple": "애플",
    "tesla": "테슬라",
    "microsoft": "마이크로소프트",
    "google": "구글",
    "meta": "메타",
    "amazon": "아마존",
    "trump": "트럼프",
    "tariff": "관세",
    "semiconductor": "반도체",
    "ai": "AI",
    "gpu": "GPU",
    "hbm": "HBM",
    "chip": "반도체",
    "oil": "원유",
    "dollar": "달러",
    "rate": "금리",
    "market": "시장",
    "stock": "주식",
    "iran": "이란",
    "china": "중국",
    "korea": "한국",
    "battery": "배터리",
    "energy": "에너지",
    "defense": "방산",
    "supply": "공급망",
    "data": "데이터",
    "server": "서버",
    "power": "전력",
    "intel": "인텔",
    "amd": "AMD",
    "qualcomm": "퀄컴",
    "tsmc": "TSMC",
    "ipo": "IPO",
}


def _fallback_keywords(articles: list[dict[str, Any]], skip: str = "") -> list[str]:
    """Improved frequency-based fallback.

    - 제목 + 설명에서 토큰 추출
    - 영문 단어를 한국어 레이블로 변환 (가능한 경우)
    - 불용어·입력 키워드 제거 후 상위 7개 반환
    """
    counts: dict[str, int] = {}
    skip_lower = skip.lower()

    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')} {art.get('summary', '')}"
        for token in re.findall(r"[A-Za-z가-힣]{3,}", text):
            t = token.lower()
            if t in _STOPWORDS or t == skip_lower:
                continue
            counts[t] = counts.get(t, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])

    result: list[str] = []
    seen_labels: set[str] = set()
    for word, _ in ranked:
        label = _EN_TO_KO_LABEL.get(word, word)
        if label not in seen_labels and label.lower() != skip_lower:
            seen_labels.add(label)
            result.append(label)
        if len(result) >= 7:
            break

    return result


async def expand_keywords(keyword: str, articles: list[dict[str, Any]]) -> list[str]:
    summaries = "\n".join(f"- {a.get('summary', '')}" for a in articles if a.get("summary"))
    if not summaries:
        return _fallback_keywords(articles, skip=keyword)

    raw = await generate(_PROMPT.format(keyword=keyword, summaries=summaries))
    if not raw:
        return _fallback_keywords(articles, skip=keyword)

    parts = [p.strip().strip("·-•") for p in re.split(r"[,\n]", raw)]
    cleaned = [p for p in parts if p and p.lower() != keyword.lower()]
    return cleaned[:7] or _fallback_keywords(articles, skip=keyword)
