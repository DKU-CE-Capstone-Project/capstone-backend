"""Tests for Zero-LLM `gemini-embedding-001` batch + Kiwi Keyword Map and /api/v1/keywords/map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents import llm, news_fetcher
from app.agents.naver_client import NaverNewsPage
from app.config import settings
from app.main import app

REPRESENTATIVES = (
    Path(__file__).resolve().parent / "fixtures" / "news_map_representatives.json"
)


def _char_ngram_vectors(texts: list[str], dimensions: int) -> list[list[float]]:
    out: list[list[float]] = []
    for raw in texts:
        vec = [0.0] * dimensions
        compact = "".join(ch for ch in raw.lower() if not ch.isspace())
        grams = [
            compact[i : i + n]
            for n in (2, 3)
            for i in range(max(0, len(compact) - n + 1))
        ]
        if not grams and compact:
            grams = [compact]
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest()
            val = int.from_bytes(digest, "little")
            slot = val % dimensions
            vec[slot] += 1.0 if (val & (1 << 31)) else -1.0
        if not any(vec):
            vec[0] = 1.0
        out.append(vec)
    return out


def _fixture_topics() -> dict:
    return json.loads(REPRESENTATIVES.read_text(encoding="utf-8"))["topics"]


def test_keyword_map_endpoint_uses_batch_gemini_embeddings_and_extracts_nodes(monkeypatch):
    topics = _fixture_topics()
    shipping_articles = topics["해운"]["articles"]
    batch_calls: list[list[str]] = []

    async def fake_embed_batch(
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int = 768,
        task_type: str | None = None,
        batch_size: int = 100,
    ) -> list[list[float]]:
        assert model == "gemini-embedding-001"
        assert task_type == "SEMANTIC_SIMILARITY"
        batch_calls.append(list(texts))
        return _char_ngram_vectors(texts, dimensions)

    async def fake_search_page(keyword: str, *, start: int, size: int) -> NaverNewsPage:
        assert keyword == "해운"
        return NaverNewsPage(
            articles=shipping_articles,
            total=len(shipping_articles),
            start=start,
            end=len(shipping_articles),
        )

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "google_api_key", "unit-test-fake-key")
    monkeypatch.setattr(llm, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(news_fetcher, "fetch_search_page", fake_search_page)

    with TestClient(app) as client:
        resp = client.get("/api/v1/keywords/map", params={"q": "해운"})

    assert resp.status_code == 200
    # Exactly 2 batch embedding calls: (1) query + articles, (2) finalist candidate phrases
    assert len(batch_calls) == 2
    assert len(batch_calls[0]) == 1 + len(shipping_articles)

    body = resp.json()
    assert body["query"] == "해운"
    assert body["total_articles"] == 5
    nodes = body["nodes"]
    assert nodes[0]["kind"] == "center"
    assert nodes[0]["keyword"] == "해운"
    assert nodes[0]["search_query"] == "해운"

    issues = [n for n in nodes if n["kind"] == "issue"]
    entities = [n for n in nodes if n["kind"] == "entity"]
    assert len(issues) == 2
    assert {n["keyword"] for n in entities} >= {"대한해운", "포스코플로우", "SM그룹"}
    assert "포스코" not in {n["keyword"] for n in entities}
    assert body["keywords"] == [n["keyword"] for n in nodes]
    for node in issues + entities:
        assert node["article_count"] >= 2
        assert node["center_news_id"] in node["news_ids"]
        assert "해운" in node["search_query"]


def test_keyword_map_separates_polarity_expands_slang_and_reuses_caches(monkeypatch):
    articles = [
        {
            "title": "[마켓뷰] 솔리다임 상장 중단... 반도체 투심 회복 기대",
            "description": "SK하이닉스 자회사 솔리다임이 상장 검토를 중단하면서 삼성전자와 SK하이닉스 투심이 개선됐다.",
            "url": "https://example.test/semi-1",
        },
        {
            "title": "“이제 삼전닉스 들어가도 되나요”...솔리다임 상장 중단에 반등",
            "description": "솔리다임 상장 중단 소식에 삼전닉스 주가가 동반 상승했다.",
            "url": "https://example.test/semi-2",
        },
        {
            "title": "증권가 \"솔리다임 상장 중단, 반도체 수급 부담 완화\"",
            "description": "SK하이닉스와 삼성전자 수급 분산 우려가 해소됐다.",
            "url": "https://example.test/semi-3",
        },
        {
            "title": "미국 반도체 급락에 필라델피아 반도체지수 6% 하락",
            "description": "엔비디아 약세 여파로 미국 반도체 업종이 급락 마감했다.",
            "url": "https://example.test/semi-4",
        },
        {
            "title": "AI 투자 불안 속 미국 반도체 급락...엔비디아 4% 약세",
            "description": "필라델피아 반도체지수가 하락하며 단기 조정 우려가 커졌다.",
            "url": "https://example.test/semi-5",
        },
        {
            "title": "한국은행 \"반도체 수출 호조세 지속...경제 성장 견인\"",
            "description": "한은 보고서는 HBM 수요 확대로 반도체 수출 증가세가 이어질 것으로 내다봤다.",
            "url": "https://example.test/semi-6",
        },
        {
            "title": "한은 \"HBM 중심 반도체 수출 호조...성장률 상방 요인\"",
            "description": "한국은행은 메모리 반도체 수출 회복이 설비투자로 확산 중이라고 진단했다.",
            "url": "https://example.test/semi-7",
        },
    ]

    search_calls = 0
    embed_batches = 0

    async def fake_embed_batch(
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int = 768,
        task_type: str | None = None,
        batch_size: int = 100,
    ) -> list[list[float]]:
        nonlocal embed_batches
        embed_batches += 1
        return _char_ngram_vectors(texts, dimensions)

    async def fake_search_page(keyword: str, *, start: int, size: int) -> NaverNewsPage:
        nonlocal search_calls
        search_calls += 1
        return NaverNewsPage(
            articles=articles,
            total=len(articles),
            start=1,
            end=len(articles),
        )

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "google_api_key", "unit-test-fake-key")
    monkeypatch.setattr(llm, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(news_fetcher, "fetch_search_page", fake_search_page)

    with TestClient(app) as client:
        first = client.get("/api/v1/keywords/map", params={"q": "반도체"})
        second = client.get("/api/v1/keywords/map", params={"q": " 반도체 "})

    assert first.status_code == 200
    assert second.status_code == 200
    assert search_calls == 1
    assert embed_batches == 2

    payload = first.json()
    entity_names = {n["keyword"] for n in payload["nodes"] if n["kind"] == "entity"}
    assert "삼성전자" in entity_names
    assert "SK하이닉스" in entity_names
    assert "닉스" not in entity_names
    assert "은행" not in entity_names
    assert "마켓뷰" not in entity_names


def test_keyword_map_handles_empty_whitespace_and_unavailable_embeddings(monkeypatch):
    async def empty_search_page(keyword: str, *, start: int, size: int) -> NaverNewsPage:
        return NaverNewsPage(articles=[], total=0, start=1, end=0)

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(news_fetcher, "fetch_search_page", empty_search_page)

    with TestClient(app) as client:
        assert client.get("/api/v1/keywords/map", params={"q": "   "}).status_code == 422
        empty_resp = client.get("/api/v1/keywords/map", params={"q": "없는키워드"})
        assert empty_resp.status_code == 200
        assert empty_resp.json() == {
            "query": "없는키워드",
            "total_articles": 0,
            "keywords": [],
            "nodes": [],
        }

    async def non_empty_page(keyword: str, *, start: int, size: int) -> NaverNewsPage:
        return NaverNewsPage(
            articles=[
                {
                    "title": "환율 상승에 수출주 주목",
                    "description": "원·달러 환율 상승",
                    "url": "https://a.test/1",
                }
            ],
            total=1,
            start=1,
            end=1,
        )

    # google_api_key is "" by default in conftest -> embed_batch returns [] -> 503
    monkeypatch.setattr(news_fetcher, "fetch_search_page", non_empty_page)
    with TestClient(app) as client:
        missing_resp = client.get("/api/v1/keywords/map", params={"q": "환율"})
        assert missing_resp.status_code == 503


def test_keyword_map_prevents_bare_leading_gyumo_and_preserves_quantified_modifiers():
    from app.agents.keyword_map_extractor import get_keyword_map_extractor

    extractor = get_keyword_map_extractor()

    shipping_phrases = [
        p
        for p, _lemmas, _anc, _ev in extractor._extract_title_phrases(
            "SM그룹 대한해운, 포스코플로우와 720억 규모 장기화물운송계약",
            "해운",
        )
    ]
    assert not any(p == "규모" or p.startswith("규모 ") for p in shipping_phrases)
    assert "720억 규모 장기화물운송계약" in shipping_phrases

    kofr_phrases = [
        p
        for p, _lemmas, _anc, _ev in extractor._extract_title_phrases(
            "산은·기은, 26일 1조원 규모 KOFR 대출 출시...다음 주 금리 동결 전망",
            "금리",
        )
    ]
    assert not any(p.startswith("규모 ") for p in kofr_phrases)
    assert not any(p.startswith("주 ") for p in kofr_phrases)
    assert "1조원 규모 KOFR 대출" in kofr_phrases

    compound_phrases = [
        p
        for p, _lemmas, _anc, _ev in extractor._extract_title_phrases(
            "정부, 대규모 반도체 투자 지원책 발표",
            "반도체",
        )
    ]
    assert any(p.startswith("대규모 ") for p in compound_phrases)


def test_anisotropic_sim_threshold_calibration_prevents_dense_cone_overmerge():
    import numpy as np

    from app.agents.keyword_map_extractor import get_keyword_map_extractor

    extractor = get_keyword_map_extractor()
    fx_articles = _fixture_topics()["환율"]["articles"]
    corpus = extractor.prepare_corpus("환율", fx_articles)
    titles = [a["_clean_title"] for a in corpus.deduped]
    assert len(titles) == 4

    # Construct an anisotropic high-cosine cone mimicking `gemini-embedding-001` on `환율`:
    # - Query vector on axis 0
    # - Shared topic cone along axis 1 (so projected pairwise cosine ~ 0.72 even for article #4)
    # - BOK report sub-event shared by articles 0, 1, 2 on axis 2; distinct MZ feature on axis 3
    dim = 16
    q_vec = np.zeros(dim, dtype=np.float32)
    q_vec[0] = 1.0

    art_vecs = np.zeros((4, dim), dtype=np.float32)
    for i in range(3):
        art_vecs[i, 0] = 0.65
        art_vecs[i, 1] = 0.62
        art_vecs[i, 2] = 0.44
    art_vecs[3, 0] = 0.65
    art_vecs[3, 1] = 0.66
    art_vecs[3, 3] = 0.38
    art_vecs /= np.linalg.norm(art_vecs, axis=1, keepdims=True)

    # Without anisotropy calibration (fixed 0.42), the 4th MZ article wrongly merges into 1 cluster of 4
    uncalibrated = extractor._cluster_articles(
        titles, art_vecs, q_vec, sim_threshold=0.42, calibrate_anisotropy=False
    )
    assert len(uncalibrated[0]) == 4

    # With default calibrated threshold, only the 3 BOK report articles merge (100% purity)
    calibrated = extractor._cluster_articles(titles, art_vecs, q_vec)
    assert sorted(calibrated[0]) == [0, 1, 2]


def test_keyword_map_quota_charges_only_uncached_supplier_calls(monkeypatch):
    topics = _fixture_topics()
    shipping_articles = topics["해운"]["articles"]

    async def fake_embed_batch(
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int = 768,
        task_type: str | None = None,
        batch_size: int = 100,
    ) -> list[list[float]]:
        return _char_ngram_vectors(texts, dimensions)

    async def fake_search_page(keyword: str, *, start: int, size: int) -> NaverNewsPage:
        return NaverNewsPage(
            articles=shipping_articles,
            total=len(shipping_articles),
            start=start,
            end=len(shipping_articles),
        )

    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "google_api_key", "unit-test-fake-key")
    monkeypatch.setattr(settings, "news_search_client_hourly_limit", 1)
    monkeypatch.setattr(llm, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(news_fetcher, "fetch_search_page", fake_search_page)

    with TestClient(app) as client:
        # Invalid whitespace query does not consume supplier quota
        assert client.get("/api/v1/keywords/map", params={"q": "   "}).status_code == 422
        # First real query succeeds and consumes the 1 allowed supplier call
        assert client.get("/api/v1/keywords/map", params={"q": "해운"}).status_code == 200
        # Cached repeat succeeds without consuming supplier quota
        assert client.get("/api/v1/keywords/map", params={"q": "해운"}).status_code == 200
        # Second distinct uncached query hits 429 with Retry-After
        over = client.get("/api/v1/keywords/map", params={"q": "환율"})
        assert over.status_code == 429
        assert int(over.headers["Retry-After"]) >= 1

