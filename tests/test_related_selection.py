from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import article_embeddings as embeddings
from app.agents import llm
from app.agents.related_selector import (
    cosine_similarity,
    keyword_tokens,
    select_related,
)
from app.api.v1 import news
from app.config import settings
from app.main import app


def article(nid, title, description="", **extra):
    return {"news_id": nid, "title": title, "description": description,
            "url": f"https://example.test/{nid}", "_search_keyword": "AI", **extra}


def vector(x, y, dimensions=768):
    return [x, y] + [0.0] * (dimensions - 2)


@pytest.fixture
def corpus(monkeypatch):
    articles = [
        article("center", "엔비디아 AI GPU 출시", "새 GPU 제품 출시"),
        article("a-low", "엔비디아 사옥 이전", "본사 직원 이사"),
        article("b-mid", "GPU 공급 계약", "새 AI 제품 공급 계약"),
        article("z-high", "AI GPU 판매 개시", "새 GPU 제품 출시"),
        article("unrelated", "주택 전세 계약", "부동산 임대차"),
    ]
    vectors = [vector(1, 0), vector(0.2, 0.98), vector(0.85, 0.53), vector(0.99, 0.1), vector(0, 1)]
    by_title = dict(zip([a["title"] for a in articles], vectors))
    calls = []

    async def embed(text, **kwargs):
        calls.append((text, kwargs))
        return by_title[text.splitlines()[0].removeprefix("Title: ")]

    async def fetch(*args, **kwargs):
        return deepcopy(articles)

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news, "fetch_news", fetch)
    store.news_cache.update({a["news_id"]: a for a in articles})
    return articles, calls


def ids(items):
    return [item.article["news_id"] for item in items]


async def test_rank_entire_pool_then_limit_and_do_not_fill(corpus):
    articles, calls = corpus
    assert ids(await select_related(articles[0], articles[1:], limit=1)) == ["z-high"]
    assert ids(await select_related(articles[0], articles[1:], limit=6)) == ["z-high", "b-mid"]
    assert len(calls) == 5  # Repeated selection reuses each article vector.


def test_free_filters_sorts_and_matches_graph(corpus):
    client = TestClient(app)
    response = client.get("/api/v1/news/center/related?limit=10&tier=FREE&include_score=false")
    assert response.status_code == 200
    items = response.json()["related_news"]
    assert [item["news_id"] for item in items] == ["z-high", "b-mid"]
    assert all(item["relevance_score"] is None for item in items)
    graph = client.get("/api/v1/news/center/graph?limit=10&depth=3").json()
    assert [item["news_id"] for item in graph["nodes"][1:]] == ["z-high", "b-mid"]
    assert all(item["distance"] == 1 for item in graph["nodes"][1:])
    assert client.get("/api/v1/news/center/related?limit=1").json()["related_news"][0]["news_id"] == "z-high"
    assert client.get("/api/v1/news/center/related?min_relevance=1").json()["related_news"] == []


async def test_changing_center_recalculates_ranking(corpus):
    articles, _ = corpus
    result = await select_related(articles[-1], articles, limit=6)
    assert ids(result) == ["a-low"]


async def test_self_exact_copies_and_url_duplicates_excluded_but_topic_kept(corpus):
    articles, _ = corpus
    duplicate = {**articles[3], "news_id": "dup", "url": "https://other.test/copy"}
    same_url = {**articles[2], "news_id": "alias", "url": articles[0]["url"] + "?utm_source=feed"}
    result = await select_related(articles[0], [*articles, duplicate, same_url], limit=20)
    assert len(result) == 2
    assert "b-mid" in ids(result)
    assert {"dup", "z-high"} & set(ids(result))
    assert "center" not in ids(result) and "alias" not in ids(result)


async def test_different_description_same_title_is_not_a_duplicate(monkeypatch):
    async def embed(text, **kwargs):
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    center = article("center", "GPU", "새 제품")
    candidates = [article("a", "AI 발표", "제품 출시"), article("b", "AI 발표", "후속 공급 계약")]
    assert len(await select_related(center, candidates, limit=6)) == 2


async def test_stable_ties_and_title_only_input(monkeypatch):
    calls = []
    async def embed(text, **kwargs):
        calls.append(text)
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    center = article("center", "HBM3E")
    candidates = [article("z", "HBM3E 공급"), article("a", "HBM3E 생산"), article("empty", "")]
    expected = ["a", "z"]
    assert ids(await select_related(center, candidates, limit=6)) == expected
    assert ids(await select_related(center, candidates[::-1], limit=6)) == expected
    assert calls[0] == "Title: HBM3E\nDescription: "
    assert all("summary" not in text for text in calls)


def test_keywords_preserve_short_numeric_terms_aliases_and_grounding():
    a = article("a", "NVIDIA AI HBM3E 금리", keywords=["엔비디아", "AI", "HBM3E", "무관한사건"])
    tokens = keyword_tokens(a)
    assert {"엔비디아", "ai", "hbm3e", "금리"} <= tokens
    assert "무관한사건" not in tokens
    assert "hbm" not in tokens  # HBM3E is a distinct product term.
    assert keyword_tokens(article("b", "chair")) == {"chair"}
    assert cosine_similarity(vector(1, 0), vector(-1, 0)) == -1
    assert keyword_tokens(article("c", "SK하이닉스의 HBM 생산")) >= {"sk하이닉스", "hbm"}
    assert "하이닉스의" not in keyword_tokens(article("c", "SK하이닉스의 HBM 생산"))


def test_legacy_summary_is_not_used_as_a_search_description():
    from app import database
    loaded = database.article_from_news_doc({"title": "HBM", "summary": "Generated content"})
    assert embeddings.map_text(loaded) == "Title: HBM\nDescription: "


def test_free_paid_limits_and_score_visibility(monkeypatch):
    articles = [article(str(i), f"AI GPU 기사 {i}", f"출시 정보 {i}") for i in range(7)]
    store.news_cache.update({a["news_id"]: a for a in articles})
    async def embed(text, **kwargs):
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    client = TestClient(app)
    for tier in ("FREE", "BASIC"):
        response = client.get(f"/api/v1/news/0/related?limit=6&tier={tier}&include_score=true")
        assert len(response.json()["related_news"]) == 3
        assert all(a["relevance_score"] is None for a in response.json()["related_news"])
    paid = client.get("/api/v1/news/0/related?limit=5&tier=PAID").json()["related_news"]
    graph = client.get("/api/v1/news/0/graph?limit=5").json()["nodes"][1:]
    assert len(paid) == 5 and all(a["relevance_score"] >= 0.65 for a in paid)
    assert [a["news_id"] for a in paid] == [a["news_id"] for a in graph]


async def test_entity_only_overlap_does_not_gain_keyword_bonus(monkeypatch):
    async def embed(text, **kwargs):
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    center = article("center", "엔비디아 GPU 출시", "AI 제품 생산")
    other = article("other", "Nvidia 사옥 이전", "직원 이사", keywords=["Nvidia"] * 10)
    result = await select_related(center, [other], limit=6)
    assert result[0].score == pytest.approx(0.8)
    other["keywords"] = []
    assert (await select_related(center, [other], limit=6))[0].score == result[0].score


@pytest.mark.parametrize("failed", [[], [float("nan")] * 768, [1.0], [0.0] * 768, "exception"])
def test_missing_invalid_or_failed_embeddings_are_503(monkeypatch, corpus, failed):
    async def embed(text, **kwargs):
        if failed == "exception":
            raise RuntimeError("private provider error")
        return failed
    monkeypatch.setattr(llm, "embed", embed)
    client = TestClient(app)
    for route in ("related", "graph"):
        response = client.get(f"/api/v1/news/center/{route}")
        assert response.status_code == 503
        assert "private" not in response.text
        assert "임베딩" in response.json()["detail"]


async def test_candidate_failure_does_not_become_zero_or_partial_success(monkeypatch, corpus):
    original = llm.embed
    async def embed(text, **kwargs):
        if "주택" in text:
            return []
        return await original(text, **kwargs)
    monkeypatch.setattr(llm, "embed", embed)
    with pytest.raises(embeddings.EmbeddingUnavailable):
        await select_related(corpus[0][0], corpus[0][1:], limit=6)


async def test_vector_reuse_and_content_configuration_refresh(monkeypatch):
    calls = []
    async def embed(text, **kwargs):
        calls.append((text, kwargs))
        return vector(1, 0, kwargs.get("dimensions", 768))
    monkeypatch.setattr(llm, "embed", embed)
    art = article("a", "AI GPU", "출시")
    await embeddings.article_vector(art)
    await embeddings.article_vector(deepcopy(art))
    assert len(calls) == 1
    art["description"] = "공급 계약"
    await embeddings.article_vector(art)
    monkeypatch.setattr(settings, "news_map_embedding_model", "test-model")
    await embeddings.article_vector(art)
    monkeypatch.setattr(settings, "news_map_embedding_dimensions", 128)
    await embeddings.article_vector(art)
    monkeypatch.setattr(settings, "news_map_embedding_task_type", "CLUSTERING")
    await embeddings.article_vector(art)
    monkeypatch.setattr(embeddings, "MAP_VERSION", "test-version")
    await embeddings.article_vector(art)
    assert len(calls) == 6
    assert art["news_map_embedding"]["metadata"]["input_hash"]
    assert settings.embedding_model == "gemini-embedding-001"  # RAG unchanged.
    assert len(await embeddings.article_vector(art, "rag", persist=False)) == 768
    assert calls[-1][1] == {}


async def test_legacy_vector_without_provenance_is_not_reused(monkeypatch):
    async def embed(text, **kwargs):
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    art = article("a", "AI", embedding=vector(0, 1))
    assert await embeddings.article_vector(art) == vector(1, 0)
    assert art["embedding"] == vector(0, 1)  # Map vector does not overwrite RAG.


async def test_concurrent_same_input_single_provider_call(monkeypatch):
    import asyncio
    calls = 0
    async def embed(text, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return vector(1, 0)
    monkeypatch.setattr(llm, "embed", embed)
    await asyncio.gather(*(embeddings.article_vector(article(str(i), "AI")) for i in range(5)))
    assert calls == 1


async def test_candidate_budget_is_independent_of_display_limit(monkeypatch, corpus):
    monkeypatch.setattr(settings, "news_map_candidate_limit", 3)
    candidates = await news._related_articles("center")
    assert len(candidates) == 3
    monkeypatch.setattr(settings, "news_map_candidate_limit", 40)
    assert len(await news._related_articles("center")) == 4
