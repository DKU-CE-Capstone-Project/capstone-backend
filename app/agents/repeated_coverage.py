"""Conservative, grounded evidence for repeated information about one event.

This is a bounded rule vocabulary, not inferred event extraction. Missing cues
keep an article; cosine alone never supplies an entity, action, time or angle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from app.agents.article_embeddings import clean_text
from app.agents.article_metadata import ALIASES, _contains
from app.config import settings

NAMES = {
    **{k: v for k, v in ALIASES.items() if k in {
        "삼성전자", "SK하이닉스", "엔비디아", "테슬라", "현대자동차", "한국은행", "연준",
    }},
    "삼성전자": (*ALIASES["삼성전자"], "삼성"),
    "애플": ("애플", "Apple"), "LG전자": ("LG전자", "LG Electronics"),
    "구글": ("구글", "Google"), "마이크로소프트": ("마이크로소프트", "Microsoft"),
    "일론 머스크": ("일론 머스크", "일론머스크", "Elon Musk"),
    "이재용": ("이재용",), "트럼프": ("트럼프", "Trump"),
}
SUBJECTS = {
    "갤럭시": r"갤럭시|\bgalaxy\b", "아이폰": r"아이폰|\biphone\b",
    "아이패드": r"아이패드|\bipad\b", "기준금리": r"기준금리|policy rate",
    "모델Y": r"모델\s*Y|model\s*Y", "모델3": r"모델\s*3|model\s*3",
}
# Different stages/angles are deliberately distinct, even for the same product.
ACTIONS = {
    "unveil": r"공개|선보|발표|unveil|announce",
    "availability": r"출시|판매\s*(?:개시|시작)|launch|on sale",
    "contract": r"계약|수주|contract",
    "production": r"양산|생산|production",
    "investment": r"투자|증설|공장|investment",
    "earnings": r"실적|영업이익|매출|earnings",
    "comparison": r"비교|경쟁\s*제품|compare|versus|\bvs\b",
    "reaction": r"소비자|이용자|사용자|고객\s*반응|흥행|consumer|reaction",
    "market_impact": r"주가|실적\s*영향|수익성|stock price|market impact",
    "rate_cut": r"금리\s*인하|rate cut",
    "rate_hike": r"금리\s*인상|rate hike",
    "recall": r"리콜|회수|recall",
    "personnel": r"취임|선임|사임|인사|appoint|resign",
    "acquisition": r"인수|합병|acquisition|merger",
}
FACETS = {
    "comparison": ACTIONS["comparison"], "reaction": ACTIONS["reaction"],
    "impact": ACTIONS["market_impact"], "earnings": ACTIONS["earnings"],
    "review": r"리뷰|사용기|평가|review",
    "forecast": r"전망|예상|예측|forecast|outlook",
    "risk": r"결함|부작용|위험|논란|우려|defect|risk",
    "pricing": r"가격|출고가|price|pricing",
    "specification": r"사양|성능|배터리\s*용량|specification|performance",
    "follow_up": r"후속|추가\s*(?:정보|내용|설명)|처음|새롭게|업데이트",
}
MODEL = re.compile(
    r"(?<![a-z0-9])(?:[a-z]{1,12}[- ]?\d+(?:\.\d+)?[a-z0-9+-]*"
    r"(?:\s*(?:ultra|pro|max|plus|울트라|프로|플러스))?)(?![a-z0-9])"
    r"|(?:아이폰|iphone|폴드|플립)\s*\d+(?:\s*(?:pro|max|프로|플러스))?", re.IGNORECASE,
)
AMOUNT = re.compile(
    r"(?<![a-z0-9])\d[\d,]*(?:\.\d+)?\s*"
    r"(?:조\s*원|억\s*원|만\s*원|원|달러|%|퍼센트|bp|gb|tb|nm|만\s*대|대|명|건|배)(?![a-z])", re.IGNORECASE,
)
DATE = re.compile(r"(?<!\d)(?:(\d{4})\s*(?:년|[-./])\s*(\d{1,2})\s*(?:월|[-./])\s*(\d{1,2})\s*일?"
                  r"|(\d{1,2})\s*월\s*(\d{1,2})\s*일)(?!\d)")


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣%]", "", value.casefold())


def _cues(text: str, vocabulary: dict[str, str]) -> frozenset[str]:
    return frozenset(k for k, pattern in vocabulary.items() if re.search(pattern, text, re.IGNORECASE))


def _published(value: Any) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(value)
        # A timezone-less value does not establish a comparable publication time.
        return stamp.astimezone(UTC) if stamp.tzinfo else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class EventEvidence:
    title: str
    description: str
    names: frozenset[str]
    models: frozenset[str]
    subjects: frozenset[str]
    actions: frozenset[str]
    facets: frozenset[str]
    dates: frozenset[tuple[int | None, int, int]]
    periods: frozenset[str]
    amounts: frozenset[str]
    published: datetime | None


def event_evidence(article: dict[str, Any]) -> EventEvidence:
    title = clean_text(article.get("title"))[:500]
    description = clean_text(article.get("description"))[:6000]
    text = f"{title} {description}"
    names = {name for name, aliases in NAMES.items() if any(_contains(text, a) for a in aliases)}
    # Only literal names with a supplied role/organization suffix, no guessed NER.
    names.update(re.findall(r"([가-힣]{2,5})\s+(?:회장|대통령|총재|대표)", text))
    names.update(re.findall(r"[가-힣A-Za-z]{2,12}(?:은행|제약|그룹)", text))
    actions = _cues(text, ACTIONS)
    if actions - {"unveil"}:
        # Generic 'announced earnings/a contract' does not mean a product unveiling.
        actions = actions - {"unveil"}
    dates = frozenset((int(y) if y else None, int(m or mm), int(d or dd))
                      for y, m, d, mm, dd in DATE.findall(text)
                      if 1 <= int(m or mm) <= 12 and 1 <= int(d or dd) <= 31)
    periods = frozenset(_compact(p) for p in re.findall(
        r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|[1-4]\s*분기", DATE.sub(" ", text)))
    return EventEvidence(
        title, description, frozenset(names),
        frozenset(re.sub(r"[\s-]", "", m.group().casefold()) for m in MODEL.finditer(text)),
        _cues(text, SUBJECTS),
        actions, _cues(text, FACETS), dates, periods,
        frozenset(re.sub(r"[,\s]", "", m.group().casefold()).replace("퍼센트", "%")
                  for m in AMOUNT.finditer(DATE.sub(" ", text))),
        _published(article.get("published_at")),
    )


@lru_cache(maxsize=256)
def _shingles(text: str) -> frozenset[str]:
    # Character bigrams tolerate Korean spacing/particles without fabricating words.
    text = text.casefold()
    for name, aliases in {**ALIASES, **NAMES}.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(alias.casefold()) + r"(?![a-z0-9])"
            text = re.sub(pattern, name.casefold(), text)
    text = _compact(text)
    return frozenset(text[i:i + 2] for i in range(len(text) - 1))


def text_similarity(a: str, b: str) -> float:
    aa, bb = _shingles(a), _shingles(b)
    return 2 * len(aa & bb) / (len(aa) + len(bb)) if aa and bb else 0.0


def repeated_information(a: EventEvidence, b: EventEvidence, cosine: float) -> bool:
    if cosine < settings.news_map_repeat_cosine or not a.actions or a.actions != b.actions:
        return False
    # An explicit additional angle/measurement is evidence to keep the article.
    if a.facets != b.facets or a.amounts != b.amounts or a.periods != b.periods:
        return False
    if a.models != b.models or a.subjects != b.subjects or (a.names and b.names and a.names != b.names):
        return False
    model_match = bool(a.models & b.models)
    subject_match = bool(a.subjects & b.subjects)
    if not model_match and not subject_match and not (a.names & b.names):
        return False
    # Dates with different specificity may corroborate one another, never conflict.
    fully_dated = False
    if a.dates and b.dates:
        def compatible(x, y):
            return x[1:] == y[1:] and (x[0] is None or y[0] is None or x[0] == y[0])
        if not (all(any(compatible(x, y) for y in b.dates) for x in a.dates)
                and all(any(compatible(x, y) for x in a.dates) for y in b.dates)):
            return False
        fully_dated = all(d[0] is not None for d in a.dates | b.dates)
    if not fully_dated:
        # Month/day agreement does not invent a shared year. Publication proximity
        # only corroborates the supplied text; it is not assigned as an event date.
        if not a.published or not b.published:
            return False
        if abs((a.published - b.published).total_seconds()) > settings.news_map_repeat_max_hours * 3600:
            return False
    title_score = text_similarity(a.title, b.title)
    # Sparse descriptions demand very strong literal title evidence. No invented
    # action/date/angle is supplied to compensate for absent descriptions.
    if min(len(a.description), len(b.description)) < settings.news_map_repeat_description_min_chars:
        # A sparse extra description may still carry literal new information.
        # Do not discard it just because the supplied title is nearly identical.
        for short, other in ((a, b), (b, a)):
            words = _shingles(short.description)
            known = _shingles(other.title) | _shingles(other.description)
            if words and len(words - known) / len(words) > settings.news_map_repeat_novelty_ratio:
                return False
        return (model_match and len(a.title) >= 16 and len(b.title) >= 16
                and title_score >= settings.news_map_repeat_short_text_similarity)
    description_score = text_similarity(a.description, b.description)
    if not model_match and not subject_match:
        # Sharing only an organization and boilerplate is insufficient. Require
        # at least two other literal title terms plus stronger title similarity.
        def terms(t):
            return set(re.findall(r"[가-힣]{2,}|[a-z][a-z0-9]+", t.casefold()))
        common = terms(a.title) & terms(b.title)
        common = {t for t in common if not any(n.casefold() in t for n in a.names | b.names)}
        common = {t for t in common if not _cues(t, ACTIONS)
                  and t not in {"신제품", "제품", "소식", "오늘", "회장", "대표", "총재", "대통령"}}
        if len(common) < 2 or title_score < settings.news_map_repeat_text_similarity:
            return False
    return (description_score >= settings.news_map_repeat_text_similarity
            and (title_score + description_score) / 2 >= settings.news_map_repeat_text_similarity)
