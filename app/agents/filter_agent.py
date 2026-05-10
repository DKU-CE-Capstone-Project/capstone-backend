import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z가-힣]+")


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "as", "is", "was", "are", "be", "by", "from", "after", "new", "what",
    "should", "that", "this",
}


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 3 and t.lower() not in _STOPWORDS
    }


def _overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dedupe_articles(
    articles: list[dict[str, Any]],
    title_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Remove duplicates by URL, then by title-content-token overlap.

    Lightweight 3-step filter from 7주차 회의록:
    URL 중복 → 제목 키워드 부분일치 → (요약 비교는 summarizer 이후 단계)

    Uses overlap coefficient (|A∩B| / min(|A|,|B|)) on length-≥3 non-stopword tokens —
    more sensitive than Jaccard for short titles where total token counts differ.
    """
    seen_urls: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_title_tokens: list[set[str]] = []

    for art in articles:
        url = (art.get("url") or "").strip()
        if not url or url in seen_urls:
            continue

        title_toks = _content_tokens(art.get("title", ""))
        if any(_overlap_coefficient(title_toks, prev) >= title_threshold for prev in kept_title_tokens):
            continue

        seen_urls.add(url)
        kept.append(art)
        kept_title_tokens.append(title_toks)

    return kept
