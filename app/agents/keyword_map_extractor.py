"""Zero-LLM initial Keyword Map extractor (`gemini-embedding-001` batch + Kiwi + Gazetteer).

Extracts up to 8 grounded orbit nodes around a search query without generative LLM calls:
- Inner orbit (up to 4 'issue' nodes): query-projected semantic + title-Dice clusters labeled via
  clause-aware Korean NP chunking, consensus article support, and c-TF-IDF × Centroid Cosine.
- Outer orbit (up to 4 'entity' nodes): span-exclusive domain alias gazetteer + Kiwi NNP/SL
  proper nouns diversified with MMR over supporting article sets.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from kiwipiepy import Kiwi

from app.agents import llm
from app.agents.repeated_coverage import (
    _DECISIONS,
    _PRICE_DOWN,
    _PRICE_UP,
    _QUANTITY_DOWN,
    _QUANTITY_UP,
    bigrams,
    dice,
    normalize,
)
from app.agents.repeated_coverage import (
    tokens as rc_tokens,
)
from app.config import settings
from app.utils import make_news_id


class KeywordMapEmbeddingUnavailable(RuntimeError):
    """Raised when `gemini-embedding-001` batch embedding is unavailable or invalid."""


class AsyncTextEmbedder(Protocol):
    async def encode_async(self, texts: list[str]) -> np.ndarray: ...


_MAX_VEC_CACHE = 2048
_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()


def clear_embedding_cache() -> None:
    """Clear the bounded in-memory embedding cache (used by tests)."""
    _vec_cache.clear()


class GeminiBatchEmbedder:
    """Batch `gemini-embedding-001` embedder with bounded content-addressed LRU cache."""

    def __init__(
        self,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> None:
        self.model = model or settings.keyword_map_embedding_model
        self.dimensions = dimensions or settings.keyword_map_embedding_dimensions
        self.task_type = task_type or settings.keyword_map_embedding_task_type

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}:{self.dimensions}:{self.task_type}:{digest}"

    @staticmethod
    def _normalize_vec(values: list[float], dimensions: int) -> np.ndarray:
        if len(values) != dimensions:
            raise KeywordMapEmbeddingUnavailable("Unexpected embedding dimension")
        arr = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise KeywordMapEmbeddingUnavailable("Non-finite embedding values")
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12 or not math.isfinite(norm):
            raise KeywordMapEmbeddingUnavailable("Zero-norm embedding vector")
        return arr / norm

    async def encode_async(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        cleaned = [(t or "").strip() or "." for t in texts]
        keys = [self._cache_key(t) for t in cleaned]

        missing_unique: list[str] = []
        missing_key_to_text: dict[str, str] = {}
        for k, t in zip(keys, cleaned, strict=True):
            cached = _vec_cache.get(k)
            if cached is not None and cached.shape == (self.dimensions,):
                _vec_cache.move_to_end(k)
            elif k not in missing_key_to_text:
                missing_key_to_text[k] = t
                missing_unique.append(k)

        if missing_unique:
            batch_inputs = [missing_key_to_text[k] for k in missing_unique]
            raw_vectors = await llm.embed_batch(
                batch_inputs,
                model=self.model,
                dimensions=self.dimensions,
                task_type=self.task_type,
            )
            if len(raw_vectors) != len(missing_unique):
                raise KeywordMapEmbeddingUnavailable(
                    "Gemini batch embedding unavailable or incomplete"
                )
            for k, raw in zip(missing_unique, raw_vectors, strict=True):
                vec = self._normalize_vec(raw, self.dimensions)
                _vec_cache[k] = vec
                _vec_cache.move_to_end(k)
                while len(_vec_cache) > _MAX_VEC_CACHE:
                    _vec_cache.popitem(last=False)

        rows = [_vec_cache[k] for k in keys]
        return np.vstack(rows).astype(np.float32)


ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    # Korean Companies
    "삼성전자": ("삼성전자", "삼전", "삼성", "Samsung Electronics"),
    "SK하이닉스": ("SK하이닉스", "SK 하이닉스", "하이닉스", "SK hynix"),
    "한미반도체": ("한미반도체", "한미 반도체"),
    "삼성전기": ("삼성전기",),
    "리노공업": ("리노공업",),
    "이수페타시스": ("이수페타시스",),
    "DB하이텍": ("DB하이텍",),
    "솔리다임": ("솔리다임", "Solidigm"),
    "LG전자": ("LG전자", "LG Electronics"),
    "LG에너지솔루션": ("LG에너지솔루션", "LG엔솔"),
    "삼성SDI": ("삼성SDI",),
    "SK온": ("SK온",),
    "에코프로": ("에코프로", "에코프로비엠"),
    "포스코홀딩스": ("POSCO홀딩스", "포스코홀딩스"),
    "포스코플로우": ("포스코플로우",),
    "현대자동차": ("현대자동차", "현대차", "Hyundai Motor"),
    "기아": ("기아", "기아차"),
    "대한해운": ("대한해운",),
    "HMM": ("HMM", "에이치엠엠"),
    "팬오션": ("팬오션",),
    "SM그룹": ("SM그룹",),
    "한화에어로스페이스": ("한화에어로스페이스", "한화에어로"),
    "한화오션": ("한화오션",),
    "HD현대중공업": ("HD현대중공업", "현대중공업"),
    "셀트리온": ("셀트리온",),
    "삼성바이오로직스": ("삼성바이오로직스", "삼바"),
    "유한양행": ("유한양행",),
    "한미약품": ("한미약품",),
    "네이버": ("네이버", "NAVER"),
    "카카오": ("카카오", "Kakao"),
    "두산에너빌리티": ("두산에너빌리티",),
    "고려아연": ("고려아연",),
    # Global Companies
    "엔비디아": ("엔비디아", "NVIDIA", "Nvidia"),
    "TSMC": ("TSMC",),
    "ASML": ("ASML",),
    "인텔": ("인텔", "Intel"),
    "AMD": ("AMD",),
    "마이크론": ("마이크론", "Micron"),
    "브로드컴": ("브로드컴", "Broadcom"),
    "퀄컴": ("퀄컴", "Qualcomm"),
    "ARM": ("ARM", "암홀딩스"),
    "애플": ("애플", "Apple"),
    "마이크로소프트": ("마이크로소프트", "MS", "Microsoft"),
    "구글": ("구글", "알파벳", "Google", "Alphabet"),
    "아마존": ("아마존", "AWS", "Amazon"),
    "메타": ("메타", "Meta"),
    "오픈AI": ("오픈AI", "OpenAI", "챗GPT", "ChatGPT"),
    "테슬라": ("테슬라", "Tesla"),
    "BYD": ("BYD", "비야디"),
    "일라이릴리": ("일라이릴리", "일라이 릴리", "릴리", "Eli Lilly"),
    "노보노디스크": ("노보노디스크", "노보 노디스크", "Novo Nordisk"),
    "아람코": ("아람코", "사우디아람코"),
    # Institutions & Actors
    "한국은행": ("한국은행", "한은", "BOK", "금통위", "금융통화위원회"),
    "미 연준": ("미 연준", "연준", "Fed", "FOMC", "연방준비제도", "연방공개시장위원회"),
    "기획재정부": ("기획재정부", "기재부"),
    "산업통상자원부": ("산업통상자원부", "산업부"),
    "해양수산부": ("해양수산부", "해수부"),
    "과학기술정보통신부": ("과학기술정보통신부", "과기정통부", "과기부"),
    "국토교통부": ("국토교통부", "국토부"),
    "금융위원회": ("금융위원회", "금융위"),
    "금융감독원": ("금융감독원", "금감원"),
    "공정거래위원회": ("공정거래위원회", "공정위"),
    "식품의약품안전처": ("식품의약품안전처", "식약처"),
    "국민연금": ("국민연금", "국민연금공단"),
    "백악관": ("백악관", "미 행정부"),
    "미 재무부": ("미 재무부", "미국 재무부"),
    "미 상무부": ("미 상무부", "미국 상무부"),
    "유럽연합": ("유럽연합", "EU", "유럽집행위원회"),
    "OPEC+": ("OPEC+", "OPEC", "석유수출국기구"),
    "IMF": ("IMF", "국제통화기금"),
    "트럼프": ("트럼프", "도널드 트럼프"),
    "파월": ("파월", "제롬 파월"),
    "젠슨 황": ("젠슨 황", "젠슨황"),
    "이재용": ("이재용",),
    "최태원": ("최태원",),
    "이창용": ("이창용",),
    # Key Products, Technologies & Indicators
    "HBM": ("HBM", "고대역폭메모리", "고대역폭 메모리"),
    "HBM3E": ("HBM3E",),
    "HBM4": ("HBM4",),
    "D램": ("D램", "DRAM"),
    "낸드플래시": ("낸드플래시", "낸드", "NAND"),
    "파운드리": ("파운드리", "Foundry"),
    "위고비": ("위고비", "Wegovy"),
    "마운자로": ("마운자로", "젭바운드", "Mounjaro"),
    "GLP-1": ("GLP-1",),
    "원·달러 환율": ("원·달러", "원/달러", "달러·원", "원달러"),
    "엔화": ("엔화", "원·엔", "달러·엔"),
    "위안화": ("위안화",),
    "달러 인덱스": ("달러 인덱스", "달러인덱스", "DXY"),
    "브렌트유": ("브렌트유", "브렌트"),
    "WTI": ("WTI", "서부텍사스산원유", "서부 텍사스산 원유"),
    "두바이유": ("두바이유",),
    "기준금리": ("기준금리", "기준 금리"),
    "미 국채": ("미 국채", "미국 국채"),
    "소비자물가": ("소비자물가", "CPI", "인플레이션"),
    "코스피": ("코스피", "KOSPI"),
    "코스닥": ("코스닥", "KOSDAQ"),
    "나스닥": ("나스닥", "NASDAQ"),
    "필라델피아 반도체지수": ("필라델피아 반도체지수", "필라델피아반도체지수", "반도체지수"),
}

MULTI_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "삼전닉스": ("삼성전자", "SK하이닉스"),
    "삼전·닉스": ("삼성전자", "SK하이닉스"),
    "삼전·하닉": ("삼성전자", "SK하이닉스"),
}

GENERIC_ENTITY_STOPWORDS = {
    "미국", "한국", "중국", "일본", "유럽", "국내", "해외", "글로벌", "세계", "정부", "당국",
    "시장", "업계", "기업", "회사", "산업", "경제", "금융", "증권", "증시", "주가", "주식",
    "투자", "투자자", "전문가", "관계자", "연구원", "보고서", "발표", "전망", "분석", "영향",
    "우려", "기대", "가능성", "수준", "규모", "대비", "기준", "올해", "내년", "작년", "최근",
    "오늘", "오전", "오후", "서울", "부산", "명동", "여의도", "강남", "광화문", "워싱턴", "뉴욕",
    "연합뉴스", "뉴시스", "뉴스1",
    "한국경제", "매일경제", "머니투데이", "서울경제", "이데일리", "아시아경제", "헤럴드경제",
    "조선일보", "중앙일보", "동아일보", "한겨레", "경향신문", "파이낸셜뉴스", "접전지", "민주당",
    "공화당", "의회", "대통령", "장관", "회장", "대표", "선대회장", "삼전닉스", "닉스", "하닉",
}

LEADING_BRACKET_TAG_RE = re.compile(
    r"^(?:\s*(?:\[[^\]]{1,16}\]|【[^】]{1,16}】|＜[^＞]{1,16}＞|〈[^〉]{1,16}〉))+\s*"
)

SPATIAL_OR_BOUND_NOUNS = {
    "속", "앞", "뒤", "중", "간", "내", "외", "후", "전", "상", "하", "위", "아래",
    "곳", "점", "것", "수", "때", "바", "등", "각종", "만",
}

WEAK_TAIL_NOUNS = {
    "관련", "대해", "위해", "통해", "따른", "가운데", "이후", "이번", "최근", "올해", "내년",
    "모습", "상황", "분위기", "얘기", "이유", "배경", "의미", "예정", "주목", "관심", "부각",
    "설명", "참석", "방문", "포석", "수준", "규모", "기준", "입장", "과정", "차원", "일부",
    "얼마",
} | SPATIAL_OR_BOUND_NOUNS

NON_INITIAL_MODIFIER_NOUNS = {
    "규모", "수준", "가량", "정도", "대비", "이상", "이하", "미만", "초과",
    "기반", "위주", "중심", "관련", "부문",
}

BARE_PREDICATE_STARTS = {
    "상승", "하락", "급등", "급락", "반등", "폭등", "폭락", "증가", "감소", "확대", "축소",
    "중단", "재개", "철폐", "철회", "인상", "인하", "동결", "돌파", "회복", "둔화", "유지",
    "발표", "공개", "체결", "합의", "타결", "추진", "도입", "강화", "완화", "우려", "기대",
    "전망", "분석", "영향", "논란", "위기", "충격", "협업", "판매", "공격", "지원", "탈환",
    "개선", "개편", "변경", "논의", "검토", "대응", "마련", "착수", "돌입", "시동", "예고",
    "개시", "건의", "부과", "조정", "압박", "위협", "경고", "가결", "판별",
}

EVENT_HEAD_CUES = re.compile(
    rf"(?:{_PRICE_UP}|{_PRICE_DOWN}|{_QUANTITY_UP}|{_QUANTITY_DOWN}"
    r"|인상|인하|동결|양산|증설|수주|계약|타결|합의|결렬|제재|관세|소송|인수|합병"
    r"|적자|흑자|호조|부진|둔화|회복|돌파|급등|급락|반등|최고|최저|신고가"
    r"|차질|봉쇄|공습|확전|완화|강화|규제|지원|유지|철폐|철회|중단|재개|도입|패스트트랙|속도전"
    r"|침해|유출|해킹|공격|부작용|임상|승인|허가|이전|적응|징계|배포|개선|개편|협업|캠페인|가결|추모|주기"
    r"|결함|불량|제정|개정|발견|개발|출시|수출|수입)"
)


@dataclass
class ExtractedOrbitNode:
    keyword: str
    kind: Literal["center", "issue", "entity"]
    search_query: str
    article_count: int
    center_news_id: str | None
    news_ids: list[str]


@dataclass
class _PreparedCorpus:
    query_clean: str
    query_norm: str
    query_canon: str
    deduped: list[dict[str, Any]]
    texts_to_encode: list[str]


@dataclass
class _ClusterProposal:
    cluster_finalists: list[tuple[list[int], str, np.ndarray, list[tuple[str, float]]]]
    finalist_phrases: list[str]


class KeywordMapExtractor:
    """Deterministic Zero-LLM keyword map extractor over sentence vectors + Kiwi."""

    def __init__(self) -> None:
        self.kiwi = Kiwi(num_workers=1, load_typo_dict=False, load_multi_dict=False)
        self.alias_to_canonical: dict[str, str] = {}
        self.multi_alias_patterns: list[tuple[tuple[str, ...], re.Pattern[str]]] = []
        self.sorted_alias_patterns: list[tuple[str, re.Pattern[str]]] = []

        for raw_alias, targets in sorted(
            MULTI_ENTITY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            self.multi_alias_patterns.append(
                (targets, re.compile(re.escape(raw_alias), re.IGNORECASE))
            )

        raw_pairs: list[tuple[str, str]] = []
        for canonical, variants in ENTITY_ALIASES.items():
            for variant in {canonical, *variants}:
                clean_v = variant.strip()
                self.alias_to_canonical[clean_v.casefold()] = canonical
                self.alias_to_canonical[re.sub(r"\s+", "", clean_v).casefold()] = canonical
                raw_pairs.append((clean_v, canonical))
                compact = re.sub(r"\s+", "", clean_v)
                if re.fullmatch(r"[가-힣A-Za-z0-9·+-]{2,20}", compact):
                    self.kiwi.add_user_word(compact, "NNP", 10.0)

        raw_pairs.sort(key=lambda x: len(x[0]), reverse=True)
        for alias_str, canonical in raw_pairs:
            if re.fullmatch(r"[A-Za-z0-9.+-]+", alias_str):
                pat = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(alias_str) + r"(?![A-Za-z0-9])",
                    re.IGNORECASE,
                )
            else:
                pat = re.compile(re.escape(alias_str), re.IGNORECASE)
            self.sorted_alias_patterns.append((canonical, pat))

    @staticmethod
    def _first_sentence(desc: str) -> str:
        if not desc:
            return ""
        parts = re.split(r"(?<=[.!?다])\s+", desc)
        return parts[0][:140] if parts else desc[:140]

    @staticmethod
    def _direction_polarity(title: str) -> int:
        up = bool(re.search(_PRICE_UP + "|" + _QUANTITY_UP + "|" + _DECISIONS["hike"], title))
        down = bool(re.search(_PRICE_DOWN + "|" + _QUANTITY_DOWN + "|" + _DECISIONS["cut"], title))
        if up and not down:
            return 1
        if down and not up:
            return -1
        return 0

    @staticmethod
    def _build_search_query(query_clean: str, query_norm: str, label: str) -> str:
        label_norm = re.sub(r"\s+", "", label).casefold()
        if query_norm in label_norm:
            return label
        return f"{query_clean} {label}"

    def _cluster_articles(
        self,
        titles: list[str],
        raw_vecs: np.ndarray,
        query_vec: np.ndarray,
        sim_threshold: float | None = None,
        *,
        calibrate_anisotropy: bool = True,
    ) -> list[list[int]]:
        n = len(titles)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        q = query_vec.reshape(1, -1)
        proj = raw_vecs - (raw_vecs @ q.T) * q
        proj /= np.clip(np.linalg.norm(proj, axis=1, keepdims=True), 1e-9, None)
        sem_sim = proj @ proj.T

        base_th = (
            settings.keyword_map_sim_threshold
            if sim_threshold is None
            else sim_threshold
        )
        effective_th = base_th
        if calibrate_anisotropy and sim_threshold is None and n >= 2:
            iu = np.triu_indices(n, k=1)
            bg_sem_mean = float(sem_sim[iu].mean())
            effective_th = min(
                settings.keyword_map_sim_threshold_max,
                base_th + settings.keyword_map_anisotropy_gain * max(0.0, bg_sem_mean - 0.30),
            )

        t_grams = [bigrams(rc_tokens(t)) for t in titles]
        lex_sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            lex_sim[i, i] = 1.0
            for j in range(i + 1, n):
                d = dice(t_grams[i], t_grams[j])
                lex_sim[i, j] = lex_sim[j, i] = d

        hybrid_sim = 0.65 * sem_sim + 0.35 * lex_sim
        polarities = [self._direction_polarity(t) for t in titles]

        clusters: list[list[int]] = [[i] for i in range(n)]
        while len(clusters) > 1:
            best_sim = -1.0
            best_pair: tuple[int, int] | None = None
            for i in range(len(clusters)):
                ci = clusters[i]
                pol_i = {polarities[x] for x in ci if polarities[x] != 0}
                for j in range(i + 1, len(clusters)):
                    cj = clusters[j]
                    pol_j = {polarities[x] for x in cj if polarities[x] != 0}
                    if len(pol_i | pol_j) > 1:
                        avg_lex = float(lex_sim[np.ix_(ci, cj)].mean())
                        if avg_lex < 0.45:
                            continue
                    avg_sim = float(hybrid_sim[np.ix_(ci, cj)].mean())
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (i, j)

            if best_pair is None or best_sim < effective_th:
                break
            i, j = best_pair
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        clusters.sort(key=len, reverse=True)
        return clusters

    def _extract_title_phrases(
        self, title: str, query_norm: str
    ) -> list[tuple[str, list[str], bool, bool]]:
        cleaned = re.sub(r"\[[^\]]{1,12}\]|\([^\)]{1,14}\)", " ", title)
        clauses = re.split(r"(?:\.{2,}|…+|,|·\s+|/|—|--|\s-\s|[?!:;\"“”‘’])", cleaned)
        candidates: list[tuple[str, list[str], bool, bool]] = []

        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 3:
                continue
            toks = self.kiwi.tokenize(clause)
            if not toks:
                continue

            words: list[tuple[str, list[str], str, int, int]] = []
            cur_toks = [toks[0]]
            for nxt in toks[1:]:
                prev = cur_toks[-1]
                if nxt.start == prev.start + prev.len:
                    cur_toks.append(nxt)
                else:
                    words.append(self._classify_word(clause, cur_toks))
                    cur_toks = [nxt]
            words.append(self._classify_word(clause, cur_toks))

            run: list[tuple[str, list[str], str, bool]] = []
            runs: list[list[tuple[str, list[str], str, bool]]] = []
            for idx_w, (surf, lemmas, kind, _s, _e) in enumerate(words):
                prev_ok = idx_w == 0 or words[idx_w - 1][2] in ("STOP", "TERM")
                if kind == "CONTENT":
                    run.append((surf, lemmas, kind, prev_ok if not run else run[0][3]))
                elif kind == "TERM":
                    run.append((surf, lemmas, kind, prev_ok if not run else run[0][3]))
                    runs.append(run)
                    run = []
                else:
                    if run:
                        runs.append(run)
                        run = []
            if run:
                runs.append(run)

            for r in runs:
                n_w = len(r)
                for start_i in range(n_w):
                    first_surf, first_lemmas, first_kind, first_anchor = r[start_i]
                    if first_kind == "TERM":
                        continue
                    if start_i > 0 and len(first_surf) < 2:
                        continue
                    if not first_anchor and len(first_surf) < 2:
                        continue
                    if first_surf in SPATIAL_OR_BOUND_NOUNS:
                        continue
                    if first_surf in NON_INITIAL_MODIFIER_NOUNS:
                        continue
                    if (
                        first_surf in BARE_PREDICATE_STARTS
                        or (first_lemmas and first_lemmas[0] in BARE_PREDICATE_STARTS)
                    ):
                        continue
                    for length in (2, 3, 4):
                        end_i = start_i + length
                        if end_i > n_w:
                            break
                        window = r[start_i:end_i]
                        last_surf, last_lemmas, _last_kind, _ = window[-1]
                        if last_surf in WEAK_TAIL_NOUNS or (
                            last_lemmas and last_lemmas[-1] in WEAK_TAIL_NOUNS
                        ):
                            continue
                        if any(w[0] in SPATIAL_OR_BOUND_NOUNS for w in window):
                            continue

                        phrase = " ".join(w[0] for w in window)
                        compact = re.sub(r"\s+", "", phrase).casefold()
                        if not (4 <= len(phrase) <= 16) or compact == query_norm:
                            continue
                        lemmas = [lem for w in window for lem in w[1]]
                        anchored = window[0][3]
                        non_q_surface = compact.replace(query_norm, "")
                        has_event = (
                            bool(EVENT_HEAD_CUES.search(non_q_surface))
                            or window[-1][2] == "TERM"
                        )
                        candidates.append((phrase, lemmas, anchored, has_event))

            for idx_w, (surf, lemmas, kind, _s, _e) in enumerate(words):
                if (
                    kind in ("CONTENT", "TERM")
                    and 5 <= len(surf) <= 14
                    and len(lemmas) >= 2
                    and surf not in WEAK_TAIL_NOUNS
                    and surf not in NON_INITIAL_MODIFIER_NOUNS
                    and surf not in BARE_PREDICATE_STARTS
                    and re.sub(r"\s+", "", surf).casefold() != query_norm
                    and EVENT_HEAD_CUES.search(surf.replace(query_norm, ""))
                ):
                    prev_ok = idx_w == 0 or words[idx_w - 1][2] in ("STOP", "TERM")
                    candidates.append((surf, lemmas, prev_ok, True))

        return candidates

    @staticmethod
    def _classify_word(
        clause: str, word_toks: list[Any]
    ) -> tuple[str, list[str], str, int, int]:
        w_start = word_toks[0].start
        w_end = word_toks[-1].start + word_toks[-1].len
        tags = [t.tag for t in word_toks]

        if any(tag.startswith(("V", "VA", "VX", "VCP", "VCN", "MA", "IC")) for tag in tags):
            if (
                len(word_toks) >= 2
                and word_toks[0].tag in ("NNG", "NNP", "XR")
                and word_toks[1].tag in ("XSV", "XSA")
            ):
                stem_end = word_toks[0].start + word_toks[0].len
                stem = clause[w_start:stem_end].strip()
                if len(stem) >= 2:
                    return (stem, [word_toks[0].form], "TERM", w_start, stem_end)
            return ("", [], "STOP", w_start, w_end)

        noun_toks = [
            t
            for t in word_toks
            if t.tag in ("NNG", "NNP", "NNB", "NR", "SN", "SL", "XPN", "XSN")
            or (t.tag == "SW" and t.form in ("%", "·", "+", "-"))
        ]
        if not noun_toks:
            return ("", [], "STOP", w_start, w_end)

        trailing = [t for t in word_toks if t.start >= noun_toks[-1].start + noun_toks[-1].len]
        has_case_particle = any(t.tag.startswith(("J", "E")) for t in trailing)

        n_start = noun_toks[0].start
        n_end = noun_toks[-1].start + noun_toks[-1].len
        surface = clause[n_start:n_end].strip("·+- ")
        lemmas = [
            t.form for t in noun_toks if t.tag in ("NNG", "NNP", "SL", "SN", "NR") and len(t.form) >= 1
        ]
        if not surface or not lemmas:
            return ("", [], "STOP", w_start, w_end)

        if has_case_particle:
            josa_forms = {t.form for t in trailing if t.tag.startswith("J")}
            if josa_forms and josa_forms <= {"의"}:
                return (surface, lemmas, "CONTENT", n_start, n_end)
            return (surface, lemmas, "TERM", n_start, n_end)

        return (surface, lemmas, "CONTENT", n_start, n_end)

    def _scan_entities_in_text(self, text: str) -> set[str]:
        found: set[str] = set()
        occupied = [False] * len(text)

        for targets, pat in self.multi_alias_patterns:
            for match in pat.finditer(text):
                s, e = match.span()
                if any(occupied[s:e]):
                    continue
                for k in range(s, e):
                    occupied[k] = True
                for target_ent in targets:
                    found.add(target_ent)

        for canonical, pat in self.sorted_alias_patterns:
            for match in pat.finditer(text):
                s, e = match.span()
                if any(occupied[s:e]):
                    continue
                for k in range(s, e):
                    occupied[k] = True
                if canonical not in GENERIC_ENTITY_STOPWORDS:
                    found.add(canonical)

        for tok in self.kiwi.tokenize(text):
            if tok.tag in ("NNP", "SL") and len(tok.form) >= 2:
                s, e = tok.start, tok.start + tok.len
                if s < len(text) and any(occupied[s : min(e, len(text))]):
                    continue
                raw = tok.form.strip()
                canon = self.alias_to_canonical.get(raw.casefold(), raw)
                if (
                    canon not in GENERIC_ENTITY_STOPWORDS
                    and raw not in GENERIC_ENTITY_STOPWORDS
                    and not re.search(r"[\u4e00-\u9fff]", canon)
                    and not re.fullmatch(r"\d+[년월일분기차호%]*", canon)
                    and not (
                        re.fullmatch(r"[A-Za-z]+", canon)
                        and len(canon) <= 2
                        and canon not in {"EU", "AI"}
                    )
                ):
                    found.add(canon)
        return found

    def prepare_corpus(self, query: str, articles: list[dict[str, Any]]) -> _PreparedCorpus:
        query_clean = normalize(query).strip() or query.strip()
        query_norm = re.sub(r"\s+", "", query_clean).casefold()
        query_canon = self.alias_to_canonical.get(query_norm, query_clean)

        deduped: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for idx, art in enumerate(articles):
            raw_url = (art.get("url") or art.get("naver_url") or "").strip()
            dedupe_key = raw_url or f"idx-{idx}-{art.get('title', '')}"
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)
            raw_norm_title = normalize(art.get("title") or "")
            stripped_title = LEADING_BRACKET_TAG_RE.sub("", raw_norm_title).strip()
            title = stripped_title or raw_norm_title
            desc = normalize(art.get("description") or art.get("summary") or "")
            if not title:
                continue
            nid = art.get("news_id") or make_news_id(raw_url or title)
            deduped.append({**art, "news_id": nid, "_clean_title": title, "_clean_desc": desc})

        texts_to_encode = (
            [query_clean]
            + [f"{a['_clean_title']}. {self._first_sentence(a['_clean_desc'])}" for a in deduped]
            if deduped
            else []
        )
        return _PreparedCorpus(
            query_clean=query_clean,
            query_norm=query_norm,
            query_canon=query_canon,
            deduped=deduped,
            texts_to_encode=texts_to_encode,
        )

    def propose_clusters(
        self,
        corpus: _PreparedCorpus,
        query_vec: np.ndarray,
        art_vecs: np.ndarray,
    ) -> _ClusterProposal:
        deduped = corpus.deduped
        query_norm = corpus.query_norm
        titles = [a["_clean_title"] for a in deduped]
        clusters = self._cluster_articles(titles, art_vecs, query_vec)
        valid_clusters = [c for c in clusters if len(c) >= 2]

        cluster_phrase_cands: list[list[tuple[str, list[str], bool, bool, int]]] = []
        lemma_cluster_df: Counter[str] = Counter()

        for c_idxs in valid_clusters:
            c_phrases: list[tuple[str, list[str], bool, bool, int]] = []
            c_lemmas: set[str] = set()
            for idx in c_idxs:
                for phrase, lemmas, anchored, has_event in self._extract_title_phrases(
                    deduped[idx]["_clean_title"], query_norm
                ):
                    c_phrases.append((phrase, lemmas, anchored, has_event, idx))
                    c_lemmas.update(lemmas)
            for lem in c_lemmas:
                lemma_cluster_df[lem] += 1
            cluster_phrase_cands.append(c_phrases)

        num_clusters = max(1, len(valid_clusters))
        art_full_texts = [
            (a["_clean_title"] + " " + a["_clean_desc"]).casefold() for a in deduped
        ]

        cluster_finalists: list[tuple[list[int], str, np.ndarray, list[tuple[str, float]]]] = []
        finalist_phrases_set: set[str] = set()

        for c_i, c_idxs in enumerate(valid_clusters[:10]):
            c_vecs = art_vecs[c_idxs]
            centroid = c_vecs.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
            medoid_local = int(np.argmax(c_vecs @ centroid))
            medoid_id = deduped[c_idxs[medoid_local]]["news_id"]

            c_phrases = cluster_phrase_cands[c_i]
            lemma_tf = Counter(lem for _, lemmas, _, _, _ in c_phrases for lem in lemmas)
            phrase_groups: dict[str, list[tuple[str, list[str], bool, bool, int]]] = defaultdict(list)
            for item in c_phrases:
                key_p = re.sub(r"\s+", "", item[0]).casefold()
                phrase_groups[key_p].append(item)

            scored_prelim: list[tuple[str, float]] = []
            for items in phrase_groups.values():
                rep_phrase = Counter(x[0] for x in items).most_common(1)[0][0]
                lemmas = items[0][1]
                any_anchored = any(x[2] for x in items)
                any_event = any(x[3] for x in items)

                non_q_lemmas = [
                    lem
                    for lem in dict.fromkeys(lemmas)
                    if re.sub(r"\s+", "", lem).casefold() != query_norm and len(lem) >= 2
                ]
                if not non_q_lemmas:
                    continue

                support_cnt = 0
                for idx in c_idxs:
                    txt = art_full_texts[idx]
                    matched = sum(1 for lem in non_q_lemmas if lem.casefold() in txt)
                    if matched >= max(1, math.ceil(len(non_q_lemmas) * 0.75)):
                        support_cnt += 1
                coverage = support_cnt / len(c_idxs)
                if len(c_idxs) >= 3 and support_cnt < 2:
                    continue

                ctfidf = 0.0
                for lem in non_q_lemmas:
                    tf = lemma_tf[lem]
                    df = lemma_cluster_df[lem]
                    ctfidf += math.log1p(tf) * (math.log((1.0 + num_clusters) / (1.0 + df)) + 1.0)
                ctfidf /= math.sqrt(len(non_q_lemmas))

                anchor_mult = 1.15 if any_anchored else 0.82
                event_mult = 1.25 if any_event else 0.88
                len_mult = 1.10 if 5 <= len(rep_phrase) <= 13 else 0.92
                prelim = ctfidf * (0.45 + 0.55 * coverage) * anchor_mult * event_mult * len_mult
                scored_prelim.append((rep_phrase, prelim))

            scored_prelim.sort(key=lambda x: x[1], reverse=True)
            top_cands = scored_prelim[:6]
            for p_str, _ in top_cands:
                finalist_phrases_set.add(p_str)
            cluster_finalists.append((c_idxs, medoid_id, centroid, top_cands))

        return _ClusterProposal(
            cluster_finalists=cluster_finalists,
            finalist_phrases=sorted(finalist_phrases_set),
        )

    def finalize_map(
        self,
        corpus: _PreparedCorpus,
        query_vec: np.ndarray,
        art_vecs: np.ndarray,
        proposal: _ClusterProposal,
        phrase_vec_map: dict[str, np.ndarray],
        *,
        max_issues: int = 4,
        max_entities: int = 4,
    ) -> dict[str, Any]:
        deduped = corpus.deduped
        query_clean = corpus.query_clean
        query_norm = corpus.query_norm
        query_canon = corpus.query_canon

        raw_issues: list[tuple[float, ExtractedOrbitNode]] = []
        for c_idxs, medoid_id, centroid, top_cands in proposal.cluster_finalists:
            best_label = ""
            best_score = -1.0
            for p_str, prelim in top_cands:
                p_vec = phrase_vec_map.get(p_str)
                cos = float(p_vec @ centroid) if p_vec is not None else 0.5
                final_s = prelim * (max(0.1, cos) ** 2)
                if final_s > best_score:
                    best_score = final_s
                    best_label = p_str

            if not best_label:
                med_title = deduped[
                    next(i for i in c_idxs if deduped[i]["news_id"] == medoid_id)
                ]["_clean_title"]
                best_label = med_title[:14].strip()

            cluster_rank_score = len(c_idxs) * (1.0 + 0.12 * math.log1p(max(0.0, best_score)))
            raw_issues.append(
                (
                    cluster_rank_score,
                    ExtractedOrbitNode(
                        keyword=best_label,
                        kind="issue",
                        search_query=self._build_search_query(query_clean, query_norm, best_label),
                        article_count=len(c_idxs),
                        center_news_id=medoid_id,
                        news_ids=[deduped[i]["news_id"] for i in c_idxs],
                    ),
                )
            )

        raw_issues.sort(key=lambda x: x[0], reverse=True)
        selected_issues: list[ExtractedOrbitNode] = []
        for _rank_s, node in raw_issues:
            if len(selected_issues) >= max_issues:
                break
            norm_lbl = re.sub(r"\s+", "", node.keyword).casefold()
            lbl_toks = {
                t for t in rc_tokens(node.keyword) if len(t) >= 2 and t != query_norm
            }
            lbl_grams = bigrams(rc_tokens(node.keyword))
            dup = False
            for prev in selected_issues:
                prev_norm = re.sub(r"\s+", "", prev.keyword).casefold()
                prev_toks = {
                    t for t in rc_tokens(prev.keyword) if len(t) >= 2 and t != query_norm
                }
                tok_overlap = (
                    len(lbl_toks & prev_toks) / max(1, min(len(lbl_toks), len(prev_toks)))
                    if lbl_toks and prev_toks
                    else 0.0
                )
                if (
                    norm_lbl in prev_norm
                    or prev_norm in norm_lbl
                    or tok_overlap >= 0.5
                    or dice(lbl_grams, bigrams(rc_tokens(prev.keyword))) >= 0.55
                ):
                    dup = True
                    break
            if not dup:
                selected_issues.append(node)

        issue_labels_norm = {re.sub(r"\s+", "", n.keyword).casefold() for n in selected_issues}

        entity_title_hits: dict[str, set[int]] = defaultdict(set)
        entity_desc_hits: dict[str, set[int]] = defaultdict(set)

        for idx, art in enumerate(deduped):
            for ent in self._scan_entities_in_text(art["_clean_title"]):
                entity_title_hits[ent].add(idx)
            for ent in self._scan_entities_in_text(art["_clean_desc"]):
                entity_desc_hits[ent].add(idx)

        all_entities = set(entity_title_hits) | set(entity_desc_hits)
        issue_text_concat = " ".join(x.keyword.casefold() for x in selected_issues)

        entity_pool: list[tuple[float, ExtractedOrbitNode]] = []
        n_docs = len(deduped)
        min_support = 2 if n_docs >= 4 else 1
        for ent in all_entities:
            ent_norm = re.sub(r"\s+", "", ent).casefold()
            if ent == query_canon or ent_norm == query_norm:
                continue
            if ent_norm in issue_labels_norm:
                continue
            t_set = entity_title_hits[ent]
            d_set = entity_desc_hits[ent]
            idxs = sorted(t_set | d_set)
            df = len(idxs)
            if df < min_support or (n_docs >= 10 and df > int(n_docs * 0.85)):
                continue
            if ent not in ENTITY_ALIASES and not t_set and df < 3:
                continue

            idf = math.log(1.0 + n_docs / df)
            weighted_hits = 2.0 * len(t_set) + 1.0 * len(set(idxs) - t_set)
            gaz_bonus = 1.30 if ent in ENTITY_ALIASES else 0.95
            in_issue_penalty = 0.60 if ent.casefold() in issue_text_concat else 1.0
            ent_score = weighted_hits * idf * gaz_bonus * in_issue_penalty

            e_vecs = art_vecs[idxs]
            e_cent = e_vecs.mean(axis=0)
            e_cent /= max(float(np.linalg.norm(e_cent)), 1e-9)
            e_med_id = deduped[idxs[int(np.argmax(e_vecs @ e_cent))]]["news_id"]
            entity_pool.append(
                (
                    ent_score,
                    ExtractedOrbitNode(
                        keyword=ent,
                        kind="entity",
                        search_query=self._build_search_query(query_clean, query_norm, ent),
                        article_count=df,
                        center_news_id=e_med_id,
                        news_ids=[deduped[i]["news_id"] for i in idxs],
                    ),
                )
            )

        entity_pool.sort(key=lambda x: x[0], reverse=True)
        selected_entities: list[ExtractedOrbitNode] = []
        remaining = list(entity_pool)
        while remaining and len(selected_entities) < max_entities:
            filtered_rem: list[tuple[float, ExtractedOrbitNode]] = []
            for sc, cand in remaining:
                c_comp = re.sub(r"\s+", "", cand.keyword).casefold()
                c_set = set(cand.news_ids)
                is_dup = False
                for prev in selected_entities:
                    p_comp = re.sub(r"\s+", "", prev.keyword).casefold()
                    p_set = set(prev.news_ids)
                    overlap = len(c_set & p_set) / max(1, min(len(c_set), len(p_set)))
                    if (c_comp in p_comp or p_comp in c_comp) and overlap >= 0.65:
                        is_dup = True
                        break
                if not is_dup:
                    filtered_rem.append((sc, cand))
            remaining = filtered_rem
            if not remaining:
                break

            def _mmr_key(entry: tuple[float, ExtractedOrbitNode]) -> float:
                sc, cand = entry
                c_set = set(cand.news_ids)
                max_jacc = 0.0
                for prev in selected_entities:
                    p_set = set(prev.news_ids)
                    jacc = len(c_set & p_set) / max(1, len(c_set | p_set))
                    max_jacc = max(max_jacc, jacc)
                return 0.75 * sc - 0.25 * max_jacc * sc

            best_sc, best_node = max(remaining, key=_mmr_key)
            remaining.remove((best_sc, best_node))
            selected_entities.append(best_node)

        root_medoid_id = deduped[int(np.argmax(art_vecs @ query_vec))]["news_id"]
        center_node = ExtractedOrbitNode(
            keyword=query_clean,
            kind="center",
            search_query=query_clean,
            article_count=len(deduped),
            center_news_id=root_medoid_id,
            news_ids=[a["news_id"] for a in deduped],
        )
        all_nodes = [center_node, *selected_issues, *selected_entities]
        return {
            "query": query_clean,
            "total_articles": len(deduped),
            "keywords": [n.keyword for n in all_nodes],
            "nodes": [vars(n) for n in all_nodes],
        }


_extractor_lock = threading.Lock()
_extractor_instance: KeywordMapExtractor | None = None


def get_keyword_map_extractor() -> KeywordMapExtractor:
    """Return a process-wide lazy singleton Kiwi/gazetteer extractor."""
    global _extractor_instance
    with _extractor_lock:
        if _extractor_instance is None:
            _extractor_instance = KeywordMapExtractor()
        return _extractor_instance


def get_keyword_map_embedder() -> AsyncTextEmbedder:
    """Return the configured Gemini batch embedder."""
    return GeminiBatchEmbedder()


async def extract_keyword_map(
    query: str,
    articles: list[dict[str, Any]],
    *,
    embedder: AsyncTextEmbedder | None = None,
) -> dict[str, Any]:
    """Extract a grounded Keyword Map using batch `gemini-embedding-001` + local Kiwi."""
    extractor = await asyncio.to_thread(get_keyword_map_extractor)
    corpus = await asyncio.to_thread(extractor.prepare_corpus, query, articles)
    if not corpus.deduped:
        return {
            "query": corpus.query_clean,
            "total_articles": 0,
            "keywords": [],
            "nodes": [],
        }

    active_embedder = embedder or get_keyword_map_embedder()
    all_vecs = await active_embedder.encode_async(corpus.texts_to_encode)
    query_vec = all_vecs[0]
    art_vecs = all_vecs[1:]

    proposal = await asyncio.to_thread(
        extractor.propose_clusters, corpus, query_vec, art_vecs
    )

    phrase_vec_map: dict[str, np.ndarray] = {}
    if proposal.finalist_phrases:
        cand_vecs = await active_embedder.encode_async(proposal.finalist_phrases)
        phrase_vec_map = dict(zip(proposal.finalist_phrases, cand_vecs, strict=True))

    return await asyncio.to_thread(
        extractor.finalize_map,
        corpus,
        query_vec,
        art_vecs,
        proposal,
        phrase_vec_map,
    )
