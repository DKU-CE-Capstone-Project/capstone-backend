"""Grounded same-event and repeated-information judgment for news-map candidates.

Only the supplied title, description and publication time are evidence. Missing
evidence is never treated as a conflict, and nothing here assigns an event date,
cause or entity that the text does not state. A high cosine or an identical title
alone never decides a repeat, and a lexicon miss alone never decides "not a repeat":
unknown wording falls back to title/description overlap plus the same conflict checks.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from app.agents.article_metadata import ALIASES
from app.config import settings


class Relation(str, Enum):
    REPEAT = "repeat"                       # same event, mostly the same supplied information
    NEW_INFO = "same_event_new_info"        # same event with meaningful added facts or analysis
    DIFFERENT = "different_event"           # related, but another event, time or result
    INSUFFICIENT = "insufficient_evidence"  # evidence cannot confirm a repeat or added information


@dataclass(frozen=True)
class Judgment:
    relation: Relation
    reason: str

    @property
    def withheld(self) -> bool:
        """No confirmed addition, and too little evidence to confirm repetition."""
        return self.relation is Relation.INSUFFICIENT and self.reason in {
            "short_text_covered", "low_overlap", "semantic_difference",
        }


# ── text normalization ──────────────────────────────────────────────────────

_REPLACE = {"美": " 미국 ", "中": " 중국 ", "日": " 일본 ", "韓": " 한국 ", "英": " 영국 ",
            "獨": " 독일 ", "佛": " 프랑스 ", "北": " 북한 ", "↑": " 상승 ", "↓": " 하락 ",
            "⋯": "...", "s&p": "snp", "S&P": "snp", "弗": "달러"}
# Press tags and bylines, never bracketed topics such as "[국제유가]".
_TAGS = re.compile(r"[\[(【][^\])】]{0,40}=[^\])】]{0,40}[\])】]"
                   r"|[\[(【](?:단독|속보|종합\d*보?|포토|영상|사진|인터뷰|르포|그래픽|사설|칼럼)[\])】]"
                   r"|[가-힣]{2,4}\s*(?:특파원|객원기자|기자|통신원)\s*=")
# "전장 대비 0.9%" is a routine price change, not a comparison angle or a figure label.
_ROUTINE = re.compile(r"(?:전장|전월|전년|전작|전일|전주|전분기|전 거래일|작년|지난해|연초|직전)\s*(?:대비|보다|比)")


# Common newsroom abbreviations of one registered name, before any comparison.
_ABBREVIATIONS = ((re.compile(r"(?<![가-힣])한은(?=$|[^가-힣]|[이은의에가도])"), "한국은행"),)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value)) if isinstance(value, str) else ""
    text = re.sub(r"<[^>]+>", " ", text)
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    for pattern, name in _ABBREVIATIONS:
        text = pattern.sub(name, text)
    text = " ".join(_TAGS.sub(" ", text).split())
    # NAVER cuts titles/snippets mid-word; drop that possibly partial last token.
    return re.sub(r"\s*\S*\.{3,}$", "", text) if text.endswith("...") else text


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣%.]", "", text.casefold())


_JOSA = tuple(sorted(("에서는", "에서도", "에서", "에게", "으로는", "으로", "로는", "로", "까지", "부터",
                      "보다", "처럼", "만큼", "이라는", "라는", "이라고", "라고", "이며", "이고", "와",
                      "과", "의", "을", "를", "은", "는", "이", "가", "도", "만", "에", "께", "며", "엔", "서"),
                     key=len, reverse=True))
_ENDINGS = tuple(sorted(("했다", "했습니다", "한다", "합니다", "됐다", "됩니다", "된다", "하며", "하고",
                         "했고", "했으며", "이다", "였다", "있다", "없다", "됐고", "되며", "하는", "했던",
                         "되는", "됐으며", "했는데", "한다는", "했다는", "이라며", "라며", "다고", "다는",
                         "습니다", "었다", "았다", "였고", "인데", "이란"),
                        key=len, reverse=True))
_STOP = {"이날", "지난", "지난달", "지난주", "전했다", "밝혔다", "따르면", "대해", "위해", "통해",
         "이번", "오늘", "것으로", "현지시간", "현지", "시간", "관련", "말했다", "설명했다", "가운데",
         "이후", "이어", "오전", "오후", "올해", "내년", "작년", "기자", "특파원", "뉴스", "기사",
         "소식", "대비", "보다", "기준", "최대", "최소", "한때", "평균", "하루", "약", "등"}


def tokens(text: str) -> list[str]:
    """Josa/ending-stripped content tokens; numbers stay literal for overlap."""
    words = []
    for word in re.findall(r"[가-힣]+|[a-z][a-z0-9+]*|\d+(?:\.\d+)?%?", text.casefold()):
        if not word[0].isdigit():
            for ending in _ENDINGS:
                if word.endswith(ending) and len(word) > len(ending) + 1:
                    word = word[: -len(ending)]
                    break
            for josa in _JOSA:
                if word.endswith(josa) and len(word) > len(josa) + 1:
                    word = word[: -len(josa)]
                    break
        if len(word) >= 2 and word not in _STOP:
            words.append(word)
    return words


def bigrams(words: list[str] | tuple[str, ...]) -> frozenset[str]:
    grams = set()
    for word in words:
        grams.update({word} if len(word) <= 2 or word[0].isdigit() else
                     {word[i:i + 2] for i in range(len(word) - 1)})
    return frozenset(grams)


def dice(a: frozenset[str], b: frozenset[str]) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def containment(part: frozenset[str], whole: frozenset[str]) -> float:
    return len(part & whole) / len(part) if part else 0.0


# ── grounded vocabularies ───────────────────────────────────────────────────

# What the article reports on. "family:detail" keys; see _covered for generic matches.
_TARGETS = {
    "policy_rate": r"기준\s*금리|정책\s*금리|policy rate",
    "loan:mortgage": r"주택\s*담보\s*대출|주담대|mortgage",
    "loan:credit": r"신용\s*대출",
    "loan:household": r"가계\s*대출",
    "loan:business": r"기업\s*대출",
    "deposit": r"예금|수신\s*금리",
    "market_rate": r"시장\s*금리",
    "oil": r"(?<![원석])유가(?!\s*증권)|원유\s*(?:가격|값|선물)|oil price",
    "oil:brent": r"브렌트",
    "oil:wti": r"(?<![a-z])wti(?![a-z])|서부\s*텍사스",
    "oil:dubai": r"두바이유",
    "fuel": r"휘발유|휘발윳값|경윳값|경유\s*(?:값|가격|수출)|기름값",
    "oil_inventory": r"(?:원유|석유|석유\s*제품|연료|휘발유|경유|정제유)\s*재고",
    "oil_supply": r"(?:원유|석유|산유)\s*(?:공급|수출|생산|수송|운송|통과|선적)|(?:원유|석유)\s*(?:감산|증산)"
                  r"|(?:공급|수출|수송|운송|선적)\S{0,3}\s*(?:원유|석유)|산유량|opec|오펙",
    "fx": r"환율|원\s*[/·.]\s*달러|달러\s*[/·]\s*원|원화\s*(?:가치|약세|강세)",
    "fx:jpy": r"엔화|엔\s*[/·]\s*달러",
    "stocks": r"증시|주식\s*시장|코스피|코스닥|다우|나스닥|snp|3대\s*지수|주가|특징주",
    "inflation": r"물가|(?<![a-z])(?:cpi|pce)(?![a-z])|인플레이션",
    "growth": r"성장률|(?<![a-z])gdp(?![a-z])|국내\s*총생산|성장세|경제\s*성장",
    "jobs": r"고용|실업률|일자리|비농업",
    "gold": r"금값|금\s*가격|국제\s*금(?:값|시세)",
    "crypto": r"비트코인|가상\s*자산|암호\s*화폐",
    "housing": r"집값|아파트\s*값|주택\s*가격|전셋값|전세\s*가격",
    "tariff": r"관세",
}
_COUNTRIES = {"미국": "us", "미": "us", "일본": "jp", "독일": "de", "영국": "uk", "중국": "cn",
              "한국": "kr", "국내": "kr"}
_BOND = re.compile(r"(\d{1,2})\s*년\s*(?:물|만기\s*(?:미국\s*)?(?:국채|국고채))")
_BOND_GENERIC = re.compile(r"국채|국고채|장기\s*(?:금리|물)|채권\s*금리|treasury")
# Product/topic subjects; registered aliases are reused rather than redefined.
_SUBJECTS = {
    "phone": r"스마트폰|휴대폰|폴더블|갤럭시\s*[sz]\s*\d|갤s\d|아이폰",
    "tablet": r"태블릿|갤럭시\s*탭|아이패드",
    "earbuds": r"이어폰|버즈|에어팟|헤드폰",
    "tracker": r"스마트\s*태그|갤태그|에어태그|위치\s*추적",
    "watch": r"워치",
    "laptop": r"노트북|갤럭시\s*북|맥북",
    "chip": r"반도체|(?<![a-z])(?:hbm|gpu|npu)|d램|디램|낸드|메모리|파운드리|엑시노스|스냅드래곤",
    "ev": r"전기차|(?<![a-z])ev(?![a-z])",
    "battery": r"배터리|이차\s*전지",
    "dairy": r"우유|낙농|유업|음용유|분유|치즈",
    "ai": r"(?<![a-z])ai(?![a-z])|인공지능|챗gpt",
}
_PRICE_FAMILIES = {"loan", "deposit", "market_rate", "oil", "fuel", "fx", "stocks", "inflation",
                   "gold", "crypto", "housing", "bond"}
_QUANTITY_FAMILIES = {"oil_inventory", "oil_supply", "jobs"}
_PRICE_UP = r"상승|오르|오른|오를|오름|올라|올랐|급등|폭등|반등|돌파|치솟|뛰|뛴|최고(?!\s*(?:경영|책임))|강세|고공|높아|넘어|넘었|웃돌|껑충"
_PRICE_DOWN = r"하락|내리|내린|내려|내렸|떨어|급락|폭락|약세|최저|꺾|낮아|밑돌|하회|주저앉"
_QUANTITY_UP = r"증가|(?<![가-힣])늘|확대|회복|재개|최다|많아|증산|강화"
_QUANTITY_DOWN = r"감소|(?<![가-힣])줄|축소|중단|감산|감축|급감|적어|차질|약화"
_DECISIONS = {"hike": r"인상", "cut": r"인하", "hold": r"동결"}
# Different stages of a corporate/product story are different events.
_STAGES = {
    "announce": r"공개|선보|발표|출시|론칭|내놓|나왔|나온|등장|unveil|launch",
    "sale": r"판매\s*(?:개시|시작)|사전\s*(?:판매|예약|구매)|예약\s*판매",
    "price_change": r"(?:가격|출고가|값)\s*(?:인상|인하|올리|내리)",
    "contract": r"계약|수주",
    "production": r"양산|생산\s*(?:개시|시작|확대|돌입)",
    "investment": r"투자|증설|공장\s*(?:건설|착공)",
    "earnings": r"실적|영업\s*이익|매출|순이익",
    "recall": r"리콜|회수",
    "personnel": r"취임|선임|사임|내정|퇴임",
    "deal": r"인수|합병",
    "legal": r"소송|특허\s*침해|제재|기소",
    "agreement": r"타결|합의",
    "breakdown": r"결렬|무산|파행",
}
# Explicit analysis angles. Single words such as "처음" or "첫" are deliberately absent.
_FACETS = {
    "forecast": r"전망|예상|예측|관측|가능성|경고|시사",
    "impact": r"영향|여파|파장|부담|수혜|타격",
    "analysis": r"분석|진단|해석|배경|원인",
    "comparison": r"비교|대비|경쟁|versus|(?<![a-z])vs(?![a-z])",
    "reaction": r"반응|여론|소비자|이용자|투자자|평가",
    "review": r"리뷰|사용기|써보니|체험",
    "risk": r"결함|부작용|논란|리스크|위험|우려",
    "policy": r"대책|대응|조치|규제|지원책",
}
_NAMES = {
    **{name: aliases for name, aliases in ALIASES.items()
       if name in {"삼성전자", "SK하이닉스", "엔비디아", "테슬라", "현대자동차", "한국은행", "연준"}},
    "애플": ("애플", "apple"), "LG전자": ("LG전자", "LG Electronics"), "구글": ("구글", "google"),
    "마이크로소프트": ("마이크로소프트", "microsoft"), "트럼프": ("트럼프", "trump"),
    "이란": ("이란",), "사우디": ("사우디",), "중국": ("중국",), "일본은행": ("일본은행", "boj"),
}
_MODEL = re.compile(
    r"(?<![a-z0-9])(?:[a-z]{1,12}[- ]?\d+(?:\.\d+)?[a-z0-9+-]*"
    r"(?:\s*(?:ultra|pro|max|plus|울트라|프로|플러스))?)(?![a-z0-9])"
    r"|(?:아이폰|iphone|폴드|플립)\s*\d+(?:\s*(?:pro|max|프로|플러스))?", re.IGNORECASE,
)
_NOT_MODELS = re.compile(r"^(?:q[1-4]|h[12]|[a-z]2[a-z]|snp\d+)$")
# Attributive/predicate forms ("막는", "나왔다") describe; they are not new subjects.
_PREDICATE_TOKEN = re.compile(r"(?:는|던|다|며|니|면|려|러|겠|했|았|었|었나|았나|였나|건가|는가|을까|일까|할까|어본|아본)$")

# Whole tokens only: a ministry's trailing syllables must never become a person.
# These are institution shapes, not a blacklist of individual ministries.
_ROLES = r"회장|부회장|대통령|총재|부총재|대표|장관|차관|위원장|의장|원장|청장|시장|교수"
_ORGANIZATION_END = re.compile(
    r"(?:부|청|처|은행|위원회|공사|공단|협회|연구소|연구원|대학교|대학|정부|국회|법원"
    r"|시청|도청|군청|구청|그룹|전자|자동차|통신|회사|기업|센터|노조|연맹|의회|재단|병원|진흥원|회의소)$")
_PERSON = re.compile(
    rf"(?<![가-힣a-zA-Z0-9])(?P<name>[가-힣]{{2,4}})\s+"
    rf"(?:(?P<organization>(?:[가-힣a-zA-Z][가-힣a-zA-Z0-9·-]*\s+){{0,2}}"
    rf"[가-힣a-zA-Z][가-힣a-zA-Z0-9·-]*)\s+)??(?P<role>{_ROLES})"
    r"(?=$|[^가-힣]|[은는이가의도께과와을를])")
_INSTITUTION_ALIASES = frozenset(
    _compact(alias) for name, aliases in _NAMES.items()
    if name not in {"트럼프", "이란", "사우디", "중국"} for alias in aliases)
_IDENTITY_CONTEXT = {"정부", "국회", "미국", "한국", "중국", "일본", "영국", "독일", "프랑스",
                     "각국", "국내", "우리", "산업", "무역", "통상", "경제", "정치권"}
# A plausible Korean surname AND an explicit role are required. This is a
# conservative extraction cue, not proof that an arbitrary token is a person.
_KOREAN_NAME = re.compile(
    r"^(?:[김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노하곽성차주우구민진지엄채"
    r"원천방공현변함염여추도소석선설마길표명기반왕옥육]|남궁|황보|제갈|선우|독고)[가-힣]{1,3}$")
_CLAIM_PREDICATE = re.compile(
    r"(?:밝혔|말했|설명했|발표했|결정했|의결했|합의했|승인했|철회했|거부했|구성했|신설했"
    r"|시작했|중단했|확정했|체결했|지급했|완료했)(?:다|다고|으며|고)(?=$|[^가-힣])")
_REPORTING_WORD = re.compile(r"밝히|밝혔|말했|설명했|발표했|따르|알려|전했|약속|확인")
# Words identifying publication of an announcement do not describe an additional
# action/result. This is grounded by the saved public-policy and research replays.
_ANNOUNCEMENT_WORDS = {"합동", "마련", "마련한", "방안", "내용", "관련", "관계", "최근", "이번",
                       "기준", "기존", "강화", "결과", "나왔다", "전문", "실질적", "한목소리"}
_EXPLANATION_LINK = re.compile(r"때문|영향으로|에\s*따라|(?:상승|하락|급등|급락|증가|감소|변화)(?:으)?로")


def _is_organization(word: str) -> bool:
    return bool(_ORGANIZATION_END.search(word)) or _compact(word) in _INSTITUTION_ALIASES


def _people(text: str) -> frozenset[str]:
    found = set()
    for token in re.finditer(r"[가-힣a-zA-Z0-9]+", text):
        match = _PERSON.match(text, token.start())
        if not match:
            continue
        name, organization = match.group("name", "organization")
        # A role supplies person context; any intervening phrase must be an
        # institution. Ambiguous text is left unextracted, never shortened.
        if (not _KOREAN_NAME.match(name) or _is_organization(name) or name in _IDENTITY_CONTEXT or name in _STOP
                or _PREDICATE_TOKEN.search(name) or _re(rf"^(?:{_ROLES})$").match(name)):
            continue
        if organization:
            if not _is_organization(organization):
                continue
            if len(organization.split()) > 1 and not (
                    _compact(organization) in _INSTITUTION_ALIASES or all(
                        _is_organization(w) or w in _IDENTITY_CONTEXT for w in organization.split())):
                continue
        found.add(name)
    return frozenset(found)


def _organizations(text: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[가-힣a-zA-Z][가-힣a-zA-Z0-9·-]*", text)
                     if _is_organization(word))


def _ambiguous_name_lists(text: str) -> set[str]:
    """Unattributed name-shaped lists cannot establish novelty or confirmed people."""
    found = set()
    for match in re.finditer(r"(?<![가-힣])[가-힣]{2,4}(?:\s*[·,]\s*[가-힣]{2,4})+(?![가-힣])", text):
        words = re.findall(r"[가-힣]+", match.group())
        if all(_KOREAN_NAME.match(w) and not _is_organization(w) for w in words):
            found.update(words)
    return found


_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_FIGURE = re.compile(
    rf"(?<![\d.,])(?P<num>{_NUM}(?:\s*[조억만천]\s*(?:{_NUM}(?![\d.,]))?)*)\s*"
    r"(?P<unit>%\s*p|%\s*포인트|퍼센트\s*포인트|%|퍼센트|bp|달러|배럴|톤|t(?![a-z])|원(?!칙)"
    r"|배(?!럴|당)|명|건|포인트|gb|tb|nm|mm|시간)", re.IGNORECASE)
SNIPPET_FIGURE_TOLERANCE = 0.025
SCALED_FIGURE_TOLERANCE = 0.01
_UNITS = {"%p": "pp", "%포인트": "pp", "퍼센트포인트": "pp", "퍼센트": "%", "t": "톤"}
_DATE = re.compile(r"(?<![\d.])(?:(\d{4})\s*년\s*)?(?:(\d{1,2})\s*월\s*)?(\d{1,2})\s*일"
                   r"(?!\s*(?:만|간|째|연속|동안|이내|새|가량|치|내))")
_COMPILED: dict[str, re.Pattern] = {}


def _re(pattern: str) -> re.Pattern:
    if pattern not in _COMPILED:
        _COMPILED[pattern] = re.compile(pattern, re.IGNORECASE)
    return _COMPILED[pattern]


def _cues(text: str, vocabulary: dict[str, str]) -> frozenset[str]:
    return frozenset(key for key, pattern in vocabulary.items() if _re(pattern).search(text))


def _targets(text: str) -> frozenset[str]:
    if _re(_SUBJECTS["dairy"]).search(text):
        # Dairy 원유 (raw milk) is not crude oil; only explicit oil words remain.
        text = text.replace("원유", "□□")
    found = set(_cues(text, _TARGETS))
    for match in _BOND.finditer(text):
        before = text[max(0, match.start() - 8):match.start()]
        country = next((code for word, code in _COUNTRIES.items() if re.search(rf"{word}(?:의)?\s*$", before)), "?")
        found.add(f"bond:{country}:{int(match.group(1))}y")
    if _BOND_GENERIC.search(text) and not any(t.startswith("bond:") for t in found):
        found.add("bond")
    return frozenset(found)


def _covered(key: str, keys: frozenset[str]) -> bool:
    """A target is covered by the same or a compatible (more/less specific) mention."""
    if key in keys:
        return True
    family = key.split(":")[0]
    if key == family:
        return any(k.split(":")[0] == family for k in keys)
    if family == "bond":
        _, country, tenor = key.split(":")
        return "bond" in keys or any(
            k.count(":") == 2 and k.startswith("bond:") and k.split(":")[2] == tenor
            and "?" in (country, k.split(":")[1]) for k in keys)
    # A benchmark (Brent/WTI) is a detail of the same oil-price move, not a new subject.
    return family == "oil" and any(k.split(":")[0] == "oil" for k in keys)


@dataclass(frozen=True)
class Figure:
    value: float
    precision: float
    unit: str
    label: str
    scaled: bool = False  # Written with 만/억/조, where trailing digits are routinely rounded.


def _figure_value(raw: str) -> tuple[float, float]:
    total, group, precision = 0.0, 0.0, 1.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)?([조억만천]?)", raw.replace(",", "").replace(" ", "")):
        if not number and not unit:
            continue
        value = float(number) if number else 1.0
        step = 10 ** -len(number.split(".")[1]) if number and "." in number else 1.0
        if unit == "천":
            group += value * 1e3
            precision = step * 1e3
        elif unit:
            scale = {"조": 1e12, "억": 1e8, "만": 1e4}[unit]
            total += (group + value) * scale
            group, precision = 0.0, step * scale
        else:
            group += value
            precision = step
    return total + group, precision


def _figures(text: str) -> tuple[Figure, ...]:
    clean = _ROUTINE.sub(" ", text)
    found = []
    for match in _FIGURE.finditer(clean):
        unit = re.sub(r"\s", "", match.group("unit").casefold())
        value, precision = _figure_value(match.group("num"))
        before = [w for w in tokens(clean[max(0, match.start() - 24):match.start()])
                  if not w[0].isdigit() and not _re(_PRICE_UP + "|" + _PRICE_DOWN).search(w)
                  and w not in {"달러", "원", "배럴", "톤"}]
        found.append(Figure(value, precision, _UNITS.get(unit, unit), before[-1] if before else "",
                            bool(re.search(r"[조억만]", match.group("num")))))
    return tuple(found)


def _same_figure(a: Figure, b: Figure, relative: float = 0.0) -> bool:
    # Rounding is not novelty: 5.3% covers 5.304%, and "1200억달러" rounds 1209억.
    # A wider allowance is only used for labelled snippet values, where outlets quote
    # the same move slightly apart.
    if a.scaled or b.scaled:
        relative = max(relative, SCALED_FIGURE_TOLERANCE)
    allowance = max(max(a.precision, b.precision) / 2, relative * max(abs(a.value), abs(b.value)))
    return a.unit == b.unit and abs(a.value - b.value) <= allowance + 1e-9 * max(1.0, abs(a.value))


def _dates(text: str) -> tuple[tuple[int | None, int | None, int], ...]:
    found = []
    for year, month, day in _DATE.findall(text):
        if 1 <= int(day) <= 31 and (not month or 1 <= int(month) <= 12):
            found.append((int(year) if year else None, int(month) if month else None, int(day)))
    return tuple(found)


def _compatible_date(a, b) -> bool:
    return a[2] == b[2] and all(x is None or y is None or x == y for x, y in zip(a[:2], b[:2]))


def _published(value: Any) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(value)
        # A timezone-less value does not establish a comparable publication time.
        return stamp.astimezone(UTC) if stamp.tzinfo else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _polarity(text: str, up: str, down: str) -> frozenset[str]:
    return frozenset(name for name, pattern in (("up", up), ("down", down)) if _re(pattern).search(text))


@dataclass(frozen=True)
class EventEvidence:
    title: str
    description: str
    compact: str
    title_tokens: tuple[str, ...]
    title_bigrams: frozenset[str]
    description_bigrams: frozenset[str]
    all_bigrams: frozenset[str]
    title_targets: frozenset[str]
    targets: frozenset[str]
    title_subjects: frozenset[str]
    subjects: frozenset[str]
    title_models: frozenset[str]
    models: frozenset[str]
    model_lines: frozenset[tuple[str, str]]
    names: frozenset[str]
    people: frozenset[str]
    organizations: frozenset[str]
    title_stages: frozenset[str]
    title_facets: frozenset[str]
    facets: frozenset[str]
    description_facets: frozenset[str]
    price_polarity: frozenset[str]
    quantity_polarity: frozenset[str]
    decisions: frozenset[str]
    title_figures: tuple[Figure, ...]
    figures: tuple[Figure, ...]
    title_dates: tuple[tuple[int | None, int | None, int], ...]
    lead_date: tuple[int | None, int | None, int] | None
    published: datetime | None
    description_length: int


def _model_key(raw: str) -> str:
    return re.sub(r"[\s-]", "", raw.casefold())


def _models(text: str) -> frozenset[str]:
    return frozenset(m for m in map(_model_key, (x.group() for x in _MODEL.finditer(text)))
                     if not _NOT_MODELS.match(m))


def _model_lines(text: str) -> frozenset[tuple[str, str]]:
    """(model, product line named just before it), e.g. ("s12", "갤럭시탭") from "갤럭시 탭 S12"."""
    found = set()
    for match in _MODEL.finditer(text):
        model = _model_key(match.group())
        words = re.findall(r"[가-힣a-z]+", text[max(0, match.start() - 12):match.start()].casefold())[-2:]
        if words and not _NOT_MODELS.match(model):
            found.add((model, "".join(words)))
    return frozenset(found)


def event_evidence(article: dict[str, Any]) -> EventEvidence:
    title = normalize(article.get("title"))[:500]
    description = normalize(article.get("description"))[:6000]
    text = f"{title} {description}"
    title_words = tuple(tokens(title))
    names = {name for name, aliases in _NAMES.items()
             if any(re.search(r"(?<![a-z0-9])" + re.escape(a.casefold()) + r"(?![a-z0-9])", text.casefold())
                    for a in aliases)}
    people = _people(text)
    names.update(people)
    descriptions = _dates(description)
    return EventEvidence(
        title=title, description=description, compact=_compact(text), title_tokens=title_words,
        title_bigrams=bigrams(title_words), description_bigrams=bigrams(tokens(description)),
        all_bigrams=bigrams(tokens(text)),
        title_targets=_targets(title), targets=_targets(text),
        title_subjects=_cues(title, _SUBJECTS), subjects=_cues(text, _SUBJECTS),
        title_models=_models(title), models=_models(text), model_lines=_model_lines(title), names=frozenset(names),
        people=people, organizations=_organizations(text),
        title_stages=_cues(title, _STAGES),
        title_facets=_cues(_ROUTINE.sub(" ", title), _FACETS), facets=_cues(_ROUTINE.sub(" ", text), _FACETS),
        description_facets=_cues(_ROUTINE.sub(" ", description), _FACETS),
        price_polarity=_polarity(title, _PRICE_UP, _PRICE_DOWN),
        quantity_polarity=_polarity(title, _QUANTITY_UP, _QUANTITY_DOWN),
        decisions=_cues(title, _DECISIONS),
        title_figures=_figures(title), figures=_figures(text),
        title_dates=_dates(title), lead_date=descriptions[0] if descriptions else None,
        published=_published(article.get("published_at")),
        description_length=len(description),
    )


# ── comparison ──────────────────────────────────────────────────────────────

def _families(keys: frozenset[str]) -> set[str]:
    return {key.split(":")[0] for key in keys}


def _shared_title_figure(a: EventEvidence, b: EventEvidence) -> bool:
    return any(_same_figure(x, y) for x in a.title_figures for y in b.title_figures)


def _anchor(a: EventEvidence, b: EventEvidence, title_dice: float) -> bool:
    """Some literal sign that both report the same event; company names alone never count."""
    if any(_covered(t, b.title_targets) for t in a.title_targets):
        return True
    if a.title_subjects & b.title_subjects or a.title_models & b.title_models:
        return True
    shared = set(a.title_tokens) & set(b.title_tokens)
    shared -= {name.casefold() for name in a.names | b.names}
    if _shared_title_figure(a, b) and shared:
        return True
    if len(shared) >= 2 and title_dice >= 0.3:
        return True
    # Reports of the same announcement often put different parts of its supplied
    # description in their headlines. Require literal content on both sides;
    # neither the company nor the cosine alone can establish this anchor.
    def covered_title(x: EventEvidence, y: EventEvidence) -> int:
        return sum(w in y.compact for w in _subject_words(x.title_tokens)
                   if w not in _ANNOUNCEMENT_WORDS and not _is_organization(w)
                   and w not in {_compact(n) for n in x.names | y.names})

    if covered_title(a, b) >= 2 and covered_title(b, a) >= 2:
        return True
    # A substantially shared snippet can identify a release despite editorial
    # headlines. At least two non-identity content tokens are still required.
    shared_description = set(tokens(a.description)) & set(tokens(b.description))
    shared_description = {w for w in _subject_words(list(shared_description))
                          if w not in _ANNOUNCEMENT_WORDS and not _is_organization(w)}
    return len(shared_description) >= 2 and min(
        containment(a.description_bigrams, b.all_bigrams),
        containment(b.description_bigrams, a.all_bigrams),
    ) >= settings.news_map_repeat_text_similarity


def _direction_conflict(a: EventEvidence, b: EventEvidence) -> bool:
    shared = _families(a.title_targets) & _families(b.title_targets)
    for families, polarity in ((_PRICE_FAMILIES, "price_polarity"), (_QUANTITY_FAMILIES, "quantity_polarity")):
        x, y = getattr(a, polarity), getattr(b, polarity)
        # Only single-polarity titles are compared; mixed titles carry no clear claim.
        if shared & families and len(x) == len(y) == 1 and x != y:
            return True
    return "policy_rate" in shared and bool(a.decisions) and bool(b.decisions) and not a.decisions & b.decisions


def _figure_conflict(a: EventEvidence, b: EventEvidence) -> bool:
    def comparable(f: Figure, g: Figure) -> bool:
        # Differently labelled headline values ("수출 1209억달러", "반도체 600억달러")
        # are different quantities; an unlabelled value is compared with any of its unit.
        return f.unit == g.unit and (not f.label or not g.label or f.label == g.label)

    for x, y in ((a, b), (b, a)):
        for figure in x.title_figures:
            # Headline claims of the same kind must agree up to rounding.
            if any(comparable(figure, g) for g in y.title_figures) and not any(
                    _same_figure(figure, g) for g in y.figures if g.unit == figure.unit):
                return True
        for figure in x.figures:
            # In snippets, only one unambiguous value per literal label and unit on each
            # side is comparable ("수출 1209억" and "월 수출 600억" are different amounts).
            mine = [f for f in x.figures if f.unit == figure.unit and f.label == figure.label]
            same = [g for g in y.figures if g.unit == figure.unit and figure.label and g.label == figure.label]
            if len(mine) == len(same) == 1 and not _same_figure(figure, same[0], SNIPPET_FIGURE_TOLERANCE):
                return True
    return False


def _gap_hours(a: EventEvidence, b: EventEvidence) -> float | None:
    if not a.published or not b.published:
        return None
    return abs((a.published - b.published).total_seconds()) / 3600


def _date_conflict(a: EventEvidence, b: EventEvidence, gap: float | None) -> bool:
    if a.title_dates and b.title_dates and not any(
            _compatible_date(x, y) for x in a.title_dates for y in b.title_dates):
        return True
    x, y = a.lead_date, b.lead_date
    if not x or not y or _compatible_date(x, y):
        return False
    explicit = any(v is not None for v in (*x[:2], *y[:2]))
    # A local-date versus local-time day shift (30일 현지시간 / 1일) is not a new event.
    adjacent = abs(x[2] - y[2]) == 1 or (min(x[2], y[2]) == 1 and max(x[2], y[2]) >= 28)
    return explicit or not adjacent or gap is None or gap >= 12


def _subject_words(words: tuple[str, ...] | list[str]) -> list[str]:
    """Words that can name a new subject; numbers, predicates, moves, stages and angles are compared structurally."""
    return [w for w in words if not w[0].isdigit() and not _PREDICATE_TOKEN.search(w)
            and not _re(f"{_PRICE_UP}|{_PRICE_DOWN}|{_QUANTITY_UP}|{_QUANTITY_DOWN}").search(w)
            and not any(_re(p).search(w) for p in (*_STAGES.values(), *_FACETS.values()))]


def _uncovered_tokens(words: tuple[str, ...] | list[str], other: EventEvidence) -> list[str]:
    return [w for w in _subject_words(words)
            if w not in other.compact and containment(bigrams([w]), other.all_bigrams) < 0.5]


def _grounded_title_novelty(b: EventEvidence, a: EventEvidence) -> list[str]:
    """New headline words that b's own description also reports, or literal names/codes.

    Editorial headline wording ("지갑 안 닫았다", "내성") is not new information
    unless the article's supplied text actually covers it.
    """
    own = frozenset(b.description_bigrams)
    return [w for w in _information_novelty(b.title_tokens, b, a)
            if re.search(r"[a-z0-9]", w) or w in _compact(b.description) or containment(bigrams([w]), own) >= 0.5]


def _information_novelty(words: tuple[str, ...] | list[str], b: EventEvidence, a: EventEvidence) -> list[str]:
    """Entity mentions and titles identify a report; they do not assert an added fact."""
    identities = {_compact(n) for n in b.people | b.organizations | (b.names - {"이란", "사우디", "중국"})}
    identities.update(_ambiguous_name_lists(f"{b.title} {b.description}"))
    return [w for w in _uncovered_tokens(words, a)
            if _compact(w) not in identities and not _is_organization(w)
            and not _re(rf"^(?:{_ROLES})$").match(w) and w not in _ANNOUNCEMENT_WORDS]


def has_issue_connection(a: EventEvidence, b: EventEvidence, query: str) -> bool:
    """Reject rich snippets whose only literal connection is a broad topic/identity.

    This conservative gate does not infer a causal link. An explicit explanatory
    angle about the search topic can remain eligible. Short text keeps the
    existing insufficient-evidence policy; semantic relevance still precedes it.
    It is a lexical guard, not a complete judgment of usefulness.
    """
    if min(a.description_length, b.description_length) < settings.news_map_repeat_description_min_chars:
        return True
    topic = set(tokens(normalize(query)))
    identities = {_compact(n) for n in a.names | b.names | a.organizations | b.organizations}
    generic = _ANNOUNCEMENT_WORDS | {
        "올해", "지난해", "국내", "기업", "정부", "업무", "정보", "부문", "공공", "공공부문",
        "계열", "의약품", "펩타이드", "glp", "달러", "원", "확대", "발표", "공개", "추진",
    }
    shared = set(tokens(f"{a.title} {a.description}")) & set(tokens(f"{b.title} {b.description}"))
    specific = {w for w in _subject_words(list(shared))
                if w not in topic | identities | generic and not _is_organization(w)}
    if specific:
        return True
    # The supplied article must itself discuss the topic and an explanatory
    # facet. Sharing a category/metadata keyword cannot satisfy this condition.
    shared_topic = (topic and _compact(query) in a.compact and _compact(query) in b.compact) or any(
        _covered(target, b.targets) for target in a.targets)
    return bool(shared_topic and (b.description_facets - {"reaction", "policy"}
                                  or _EXPLANATION_LINK.search(b.description)))


def _attributed_addition(b: EventEvidence, a: EventEvidence) -> bool:
    """A named actor plus a separately stated proposition, never just a new name.

    Only complete supplied description clauses count. Require two uncovered
    content words besides the identity and reporting verb. This deliberately
    leaves ambiguous or truncated attributions undecided.
    """
    actors = b.people | b.organizations
    for clause in re.split(r"[.!?。]|(?<=다)\s+", b.description):
        if not _CLAIM_PREDICATE.search(clause) or not any(actor in clause for actor in actors):
            continue
        novel = [w for w in _information_novelty(tokens(clause), b, a)
                 if w not in _IDENTITY_CONTEXT and not _REPORTING_WORD.search(w)]
        if len(set(novel)) >= 2 and len(set(novel)) / max(1, len(set(tokens(clause)))) >= settings.news_map_repeat_novelty_ratio:
            return True
    return False


def _model_covered(model: str, other: EventEvidence, lines: frozenset[tuple[str, str]] = frozenset()) -> bool:
    base = re.match(r"[a-z]+\d+(?:\.\d+)?|(?:아이폰|iphone|폴드|플립)\d+", model)
    # "S12 울트라" adds no subject when the text already names S12 and its 울트라 model.
    if model in other.compact or bool(base) and base.group() in other.compact and model[base.end():].strip(
            "+") in other.compact:
        return True
    # Naming the model of a product line the other text names without any model is a
    # detail, not a new subject; differing models are a conflict handled earlier.
    return not other.models and any(m == model and line in other.compact for m, line in lines)


def _new_items(b: EventEvidence, a: EventEvidence, *, title: bool) -> list[str]:
    """Literal subjects/angles in b that a lacks; snippet subjects are too incidental to count."""
    targets = b.title_targets if title else b.targets
    items = [f"target:{t}" for t in targets if not _covered(t, a.targets)]
    if title:
        items += [f"subject:{s}" for s in b.title_subjects - a.subjects]
    items += [f"model:{m}" for m in (b.title_models if title else b.models)
              if not _model_covered(m, a, b.model_lines)]
    # A headline mentioning an angle (e.g. institutional 'evaluation') must
    # actually supply it in the description. Headline-only labels are not facts.
    facets = b.title_facets & b.description_facets if title else b.description_facets
    items += [f"facet:{f}" for f in facets - a.facets]
    # Names (people or institutions) are identity evidence, never novelty units.
    return items


def compare(a: EventEvidence, b: EventEvidence, cosine: float) -> Judgment:
    """Judge what candidate ``b`` adds for a reader who already has ``a``."""
    title_dice = dice(a.title_bigrams, b.title_bigrams)
    if not _anchor(a, b, title_dice):
        return Judgment(Relation.DIFFERENT, "no_shared_event")
    if a.models and b.models and not a.models & b.models:
        return Judgment(Relation.DIFFERENT, "model")
    if a.title_stages and b.title_stages and not a.title_stages & b.title_stages:
        return Judgment(Relation.DIFFERENT, "stage")
    if _direction_conflict(a, b):
        return Judgment(Relation.DIFFERENT, "direction")
    if _figure_conflict(a, b):
        return Judgment(Relation.DIFFERENT, "figure")
    gap = _gap_hours(a, b)
    if _date_conflict(a, b, gap):
        return Judgment(Relation.DIFFERENT, "date")
    if gap is not None and gap > settings.news_map_repeat_max_hours:
        return Judgment(Relation.DIFFERENT, "time")
    # From here on both texts describe one event; decide what b adds.
    shared = (len({t for t in a.title_targets if _covered(t, b.title_targets)})
              + len(a.title_subjects & b.title_subjects) + len(a.title_models & b.title_models)
              + _shared_title_figure(a, b))
    novel_title = _uncovered_tokens(b.title_tokens, a)
    grounded_novelty = _grounded_title_novelty(b, a)
    content = _subject_words(b.title_tokens)
    ratio = settings.news_map_repeat_novelty_ratio
    strong_cosine = settings.news_map_repeat_cosine + (1 - settings.news_map_repeat_cosine) / 3
    if shared >= 2 and cosine >= strong_cosine:
        # Strong shared structure: descriptive title wording must add more to count.
        ratio = min(1.0, ratio * 1.6)
    if _new_items(b, a, title=True) or (
            len(grounded_novelty) >= 2 and len(grounded_novelty) / max(1, len(content)) >= ratio):
        return Judgment(Relation.NEW_INFO, "title_addition")
    if b.description_length < settings.news_map_repeat_description_min_chars:
        if (a.title, a.description) == (b.title, b.description) and gap is not None \
                and cosine >= settings.news_map_repeat_cosine:
            # A verbatim reprint (same title and snippet) repeats every visible fact.
            return Judgment(Relation.REPEAT, "identical_text")
        if _information_novelty(tokens(b.description), b, a):
            return Judgment(Relation.NEW_INFO, "short_text_addition")
        if (gap is not None and cosine >= settings.news_map_repeat_cosine
                and title_dice >= settings.news_map_repeat_short_text_similarity):
            # Nothing visible is new, but a short text cannot confirm repeated information.
            return Judgment(Relation.INSUFFICIENT, "short_text_covered")
        return Judgment(Relation.INSUFFICIENT, "short_text")
    covered = containment(b.description_bigrams, a.all_bigrams)
    if len(_new_items(b, a, title=False)) >= 2 and covered < settings.news_map_repeat_text_similarity:
        return Judgment(Relation.NEW_INFO, "description_addition")
    if _attributed_addition(b, a):
        return Judgment(Relation.NEW_INFO, "description_addition")
    if cosine < settings.news_map_repeat_cosine:
        # A lower cosine alone confirms neither repetition nor an added fact.
        return Judgment(Relation.INSUFFICIENT, "semantic_difference")
    if gap is None:
        return Judgment(Relation.INSUFFICIENT, "time_unknown")
    # Every subject word of b's headline already appears in a: a one-directional
    # overlap that symmetric title similarity misses for reordered headlines.
    title_covered = len(content) >= 3 and not novel_title
    # Without a structured or grounded addition (checked above), a paraphrased headline
    # over another snippet of the same release is confirmed by a high semantic similarity.
    paraphrase = cosine >= strong_cosine
    if len(novel_title) - len(grounded_novelty) >= 2:
        # Headline words the snippet does not cover may be rhetoric or an untold new
        # fact; only a very high semantic similarity accepts them as a repeat.
        supported = paraphrase
    else:
        supported = (title_dice >= settings.news_map_repeat_title_similarity
                     or covered >= settings.news_map_repeat_text_similarity
                     or shared >= 2 or title_covered or paraphrase)
    if not supported:
        # Failure to confirm a repeat does not establish meaningful new information.
        return Judgment(Relation.INSUFFICIENT, "low_overlap")
    return Judgment(Relation.REPEAT, "repeated_information")
