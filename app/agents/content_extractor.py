"""
뉴스 본문 추출기 (DOM 정제 + trafilatura fallback).

capstone-news-logic/tavily_api/tavily_extract.py의
본문 추출 로직을 백엔드 패키지용으로 이식.

DOM 기반 추출 → 실패 시 trafilatura fallback.
도메인별 하드코딩 없이 articleBody/기사형 컨테이너를 자동 탐지합니다.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

# ── 정규식 패턴 ───────────────────────────────────────────────────────────────

_CONTENT_ATTR_RE = re.compile(
    r"(article|articlebody|article-body|news[_-]?body|body|content|contents|"
    r"view|story|text|read|post|entry)",
    re.IGNORECASE,
)
_STRONG_BODY_ATTR_RE = re.compile(
    r"(article[-_]?body|article[-_]?text|news[-_]?body|view[-_]?text|"
    r"content[-_]?body|story[-_]?body|post[-_]?content|entry[-_]?content)",
    re.IGNORECASE,
)
_NOISE_ATTR_RE = re.compile(
    r"(^|[_\-\s])(ad|ads|advert|banner|comment|reply|related|recommend|"
    r"popular|ranking|rank|share|sns|social|footer|copyright|tag|keyword|"
    r"toolbar|pagination|subscribe|newsletter|login|menu|nav|notice|photo|"
    r"image|img|caption|live_re|livere)([_\-\s]|$)|view_ad|google|"
    r"outbrain|taboola",
    re.IGNORECASE,
)
_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:Copyright|<?저작권자|\[?\s*저작권자|무단\s*전재|댓글|제보전화|"
    r"회원가입|로그인|관련기사|기자채널|다른기사|프레시안에\s*제보|"
    r"취재\(|기타\(|#|\+|·|>Please activate JavaScript|"
    r"(?:입력|수정|송고시간|기사입력)\s*\d{4})",
    re.IGNORECASE,
)
_CAPTION_LINE_RE = re.compile(
    r"^(?:[▲△].*|\[?그래픽.*|그래픽=.*|사진=.*|이미지=.*|"
    r"ChatGPT로 생성한 이미지.*|챗gpt를 이용해 제작함.*|"
    r".*\(출처=.*\)|.*제공\s*$|.*ⓒ.*)$",
    re.IGNORECASE,
)
_EMAIL_ONLY_RE = re.compile(r"^[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}$")

_BOILERPLATE_EXACT = {
    "증권 코드", "요약", "내용 컨텐츠", "관련기사", "댓글",
    "기자채널", "다른기사", "본문 글자 크기 조정", "글자 크게", "글자 작게",
}
_MIN_ARTICLE_CHARS = 250

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


# ── 텍스트 유틸 ───────────────────────────────────────────────────────────────

def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _signal_len(text: str) -> int:
    """기사성 문자(한글+영숫자) 길이."""
    return len(re.sub(r"[^0-9A-Za-z가-힣]", "", text or ""))


# ── DOM 노이즈 제거 ───────────────────────────────────────────────────────────

def _element_attrs(el: Any) -> str:
    values: list[str] = []
    for attr in ("id", "class", "role", "aria-label"):
        v = el.get(attr)
        if isinstance(v, list):
            values.extend(str(x) for x in v)
        elif v:
            values.append(str(v))
    return " ".join(values)


def _remove_noise_nodes(soup: Any) -> None:
    """광고, 댓글, 캡션, 공유 UI 등을 제거합니다."""
    for tag in list(soup([
        "script", "style", "noscript", "iframe", "svg", "canvas",
        "form", "button", "input", "select", "textarea",
        "video", "audio", "table", "figure", "figcaption",
        "header", "footer", "nav", "aside",
    ])):
        tag.decompose()

    for el in list(soup.find_all(True)):
        if el.parent is None or el.name in ("html", "body"):
            continue
        attrs = _element_attrs(el)
        if not _NOISE_ATTR_RE.search(attrs):
            continue
        itemprop = str(el.get("itemprop") or "").lower()
        if "articlebody" in itemprop:
            continue
        if el.name == "article" and _signal_len(el.get_text(" ", strip=True)) >= _MIN_ARTICLE_CHARS:
            continue
        el.decompose()


# ── 제목 집합 수집 ────────────────────────────────────────────────────────────

def _collect_title_norms(soup: Any, title: str = "") -> set[str]:
    titles = [title] if title else []
    for sel in ("h1", 'meta[property="og:title"]', 'meta[name="twitter:title"]'):
        for el in soup.select(sel):
            v = el.get("content") if el.name == "meta" else el.get_text(" ", strip=True)
            if v:
                titles.append(v)
    norms: set[str] = set()
    for v in titles:
        for part in re.split(r"[-|｜:]", v):
            part = _norm_space(part)
            if len(_compact(part)) >= 10:
                norms.add(_compact(part))
        if len(_compact(v)) >= 10:
            norms.add(_compact(v))
    return norms


# ── 라인 필터링 ───────────────────────────────────────────────────────────────

def _filter_line(line: str, title_norms: set[str]) -> str:
    line = _norm_space(line)
    if not line:
        return ""
    c = _compact(line)
    if c in title_norms or line in _BOILERPLATE_EXACT:
        return ""
    if len(c) < 8:
        return ""
    if _BOILERPLATE_LINE_RE.search(line):
        return ""
    if _EMAIL_ONLY_RE.match(line):
        return ""
    if _CAPTION_LINE_RE.match(line) and len(line) < 140:
        return ""
    return line


def _collect_lines(el: Any, title_norms: set[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for part in el.get_text("\n", strip=True).split("\n"):
        line = _filter_line(part, title_norms)
        if not line:
            continue
        key = _compact(line)
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


# ── 후보 요소 점수 산정 ───────────────────────────────────────────────────────

def _link_density(el: Any) -> float:
    tl = _signal_len(el.get_text(" ", strip=True))
    if tl == 0:
        return 1.0
    ll = sum(_signal_len(a.get_text(" ", strip=True)) for a in el.find_all("a"))
    return ll / tl


def _score(el: Any, lines: list[str]) -> int:
    text = "\n".join(lines)
    tl = _signal_len(text)
    if tl < _MIN_ARTICLE_CHARS:
        return -1
    attrs = _element_attrs(el)
    itemprop = str(el.get("itemprop") or "").lower()
    punct = sum(text.count(m) for m in ".。!?다\"'")
    score = tl + len(lines) * 80 + punct * 8
    if "articlebody" in itemprop:
        score += 5000
    if _STRONG_BODY_ATTR_RE.search(attrs):
        score += 2500
    elif _CONTENT_ATTR_RE.search(attrs):
        score += 500
    if el.name == "article":
        score += 1000
    score -= int(_link_density(el) * tl * 2)
    return score


# ── DOM 기반 추출 ─────────────────────────────────────────────────────────────

def _extract_with_dom(html: str, title: str = "") -> str:
    """BeautifulSoup + 휴리스틱으로 뉴스 본문을 추출합니다."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    soup = BeautifulSoup(html, "lxml")
    _remove_noise_nodes(soup)
    title_norms = _collect_title_norms(soup, title)

    candidates: list[tuple[int, bool, bool, str]] = []
    for el in soup.find_all(["article", "main", "section", "div"]):
        if el.parent is None:
            continue
        attrs = _element_attrs(el)
        itemprop = str(el.get("itemprop") or "").lower()
        is_candidate = (
            el.name in ("article", "main")
            or "articlebody" in itemprop
            or _CONTENT_ATTR_RE.search(attrs)
        )
        if not is_candidate:
            continue
        lines = _collect_lines(el, title_norms)
        s = _score(el, lines)
        if s > 0:
            is_semantic = "articlebody" in itemprop or bool(_STRONG_BODY_ATTR_RE.search(attrs))
            is_itemprop = "articlebody" in itemprop
            candidates.append((s, is_semantic, is_itemprop, "\n".join(lines)))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    best_score, best_semantic, best_itemprop, best_text = candidates[0]
    best_compact = _compact(best_text)

    if best_semantic and not best_itemprop:
        for _, _, _, ctext in candidates[1:]:
            cc = _compact(ctext)
            if best_compact not in cc:
                continue
            if len(ctext) <= len(best_text) * 1.25:
                return ctext

    return best_text if best_score > 0 else ""


# ── trafilatura fallback ──────────────────────────────────────────────────────

def _extract_with_trafilatura(html: str) -> str:
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return result or ""
    except Exception:
        return ""


# ── 동기 진입점 (asyncio.to_thread 전용) ─────────────────────────────────────

def _extract_sync(url: str, title: str = "") -> str:
    """URL에서 뉴스 본문을 동기적으로 추출합니다."""
    import requests
    import trafilatura

    html = ""
    try:
        resp = requests.get(
            url,
            headers=_REQUEST_HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding
        html = resp.text
    except Exception:
        html = trafilatura.fetch_url(url) or ""

    if not html:
        return ""

    body = _extract_with_dom(html, title)
    if body:
        return body
    return _extract_with_trafilatura(html)


# ── 공개 비동기 API ───────────────────────────────────────────────────────────

async def extract_article_body(url: str, title: str = "") -> str:
    """URL에서 뉴스 본문을 비동기로 추출합니다.

    DOM 기반 추출 → 실패 시 trafilatura fallback.
    예외 발생 시 빈 문자열 반환.
    """
    try:
        return await asyncio.to_thread(_extract_sync, url, title)
    except Exception as exc:
        print(f"[extractor] failed {url}: {type(exc).__name__}: {exc}")
        return ""


async def extract_articles_batch(
    articles: list[dict],
    concurrency: int = 5,
) -> list[dict]:
    """여러 기사를 병렬로 본문 추출합니다.

    Args:
        articles: [{"url", "title", ...}] 형식
        concurrency: 동시 추출 수 (기본 5)

    Returns:
        각 기사에 "cleaned_content" 필드가 추가된 리스트
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(art: dict) -> dict:
        async with semaphore:
            body = await extract_article_body(art.get("url", ""), art.get("title", ""))
            return {**art, "cleaned_content": body}

    return list(await asyncio.gather(*[_bounded(art) for art in articles]))
