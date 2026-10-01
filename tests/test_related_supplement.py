"""Bounded supplemental collection with mock HTTP/vectors, never live services."""
import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import database, store
from app.agents import article_embeddings, article_metadata, llm, related_candidates
from app.agents.naver_client import NaverNewsError
from app.api.v1 import news
from app.config import Settings, settings
from app.main import app
from app.utils import cache_articles
from tests.test_news_map_diversity import article, release, vec


@pytest.fixture
def supplement(monkeypatch):
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 2)
    center = release("center")
    copies = [release(str(i)) for i in range(5)]
    for a in copies:
        a["title"] += " 신제품 소식"
        a["description"] += " 공개 행사 소식이다."
    store.news_cache.update({a["news_id"]: a for a in [center, *copies]})
    calls = []

    async def embed(text, **kwargs):
        calls.append(text)
        return vec()

    monkeypatch.setattr(llm, "embed", embed)
    return center, calls


def new_angle(nid="reaction"):
    return article(nid, "삼성전자 갤럭시 S26 소비자 반응", "갤럭시 S26 신제품에 대한 소비자 반응을 조사했다.")


async def test_raw_pool_sufficient_but_repeats_trigger_grounded_supplement(monkeypatch, supplement):
    center, embedding_calls = supplement
    calls = []

    async def fetch(query, page_size):
        calls.append((query, page_size))
        return [new_angle()] if len(calls) == 1 else []

    monkeypatch.setattr(news, "fetch_news", fetch)
    result = await news._selected_related(center, 6, tier="FREE")
    assert [a.article["news_id"] for a in result] == ["reaction"]
    assert len(calls) == 2 and all(size == 12 for _, size in calls)
    assert len({q for q, _ in calls}) == 2
    assert all(q != center["_search_keyword"] for q, _ in calls)
    assert len(embedding_calls) == 3  # Center, one exact-copy representative, new angle.


async def test_enough_valid_original_candidates_skip_external_search(monkeypatch, supplement):
    center, _ = supplement
    original = [release("model-a", "S27"), release("model-b", "S28"), new_angle()]
    store.news_cache.update({a["news_id"]: a for a in original})

    async def forbidden(*args, **kwargs):
        pytest.fail("Enough original candidates must not trigger collection")

    monkeypatch.setattr(news, "fetch_news", forbidden)
    assert len(await news._selected_related(center, 6, tier="FREE")) == 3


async def test_collection_stops_as_soon_as_requested_shortage_is_resolved(monkeypatch, supplement):
    center, _ = supplement
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        return [new_angle()]

    monkeypatch.setattr(news, "fetch_news", fetch)
    assert len(await news._selected_related(center, 1)) == 1
    assert len(calls) == 1


async def test_supplement_adds_no_generation_or_rag_embedding_calls(monkeypatch, supplement, metadata_ai):
    center, _ = supplement
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "mongodb_required", True)
    saved, vectors, calls = [], [], []

    async def none(*args):
        return None

    async def candidates(*args):
        return []

    async def save(doc):
        saved.append(deepcopy(doc))

    async def save_vector(nid, purpose, record):
        vectors.append((nid, purpose))

    async def fetch(query, page_size):
        return [new_angle()]

    async def embed(text, **kwargs):
        assert kwargs.get("task_type") == "SEMANTIC_SIMILARITY", "Supplement cannot embed RAG"
        calls.append(text)
        return vec()

    async def forbidden(*args, **kwargs):
        pytest.fail("Supplement must not issue text generation")

    monkeypatch.setattr(database, "get_news", none)
    monkeypatch.setattr(database, "get_news_embedding", none)
    monkeypatch.setattr(database, "news_candidates", candidates)
    monkeypatch.setattr(database, "save_news", save)
    monkeypatch.setattr(database, "save_news_embedding", save_vector)
    monkeypatch.setattr(news, "fetch_news", fetch)
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(article_metadata, "generate", forbidden)
    assert len(await news._selected_related(center, 1)) == 1
    assert len(calls) == 3
    assert len(saved) == 1 and "embedding" not in saved[0]
    assert saved[0]["metadata_extraction"]["reason"] == "news_map_supplement"
    assert all(purpose == "news_map" for _, purpose in vectors)


@pytest.mark.parametrize("revised", [False, True])
async def test_map_only_save_reuses_valid_rag_and_removes_stale_rag(monkeypatch, revised):
    original = new_angle()
    _, metadata = article_embeddings._input(original, "rag")
    original.update(embedding=vec(), embedding_metadata=metadata)
    store.news_cache[original["news_id"]] = original
    fresh = {k: v for k, v in original.items() if not k.startswith("embedding")}
    if revised:
        fresh["description"] += " 원문 수정 정보."
    monkeypatch.setattr(settings, "use_mongodb", True)
    saved = []

    async def save(doc):
        saved.append(doc)

    async def forbidden(*args, **kwargs):
        pytest.fail("Map-only save cannot generate RAG")

    monkeypatch.setattr(database, "save_news", save)
    monkeypatch.setattr(llm, "embed", forbidden)
    await cache_articles([fresh], news_map_only=True)
    assert ("embedding" in saved[0]) == (not revised)


async def test_display_limit_shortage_and_candidate_budget_bound_requests(monkeypatch, supplement):
    center, _ = supplement
    monkeypatch.setattr(settings, "news_map_supplement_page_size", 3)
    monkeypatch.setattr(settings, "news_map_supplement_candidate_limit", 2)
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        return [release(str(i), f"S{27+i}") for i in range(10, 20)]

    monkeypatch.setattr(news, "fetch_news", fetch)
    result = await news._selected_related(center, 50, tier="PAID")
    assert len(result) == 2
    assert len(calls) == 1
    # Only accepted, capped additions are enriched/embedded/cached.
    assert len(store.news_cache) == 8  # 1 center + 5 original + 2 additional.


async def test_supplement_empty_or_irrelevant_stays_empty(monkeypatch, supplement):
    center, _ = supplement
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        return [article("unrelated", "주택 전세 시장", "임대차 계약")] if len(calls) == 1 else []

    async def embed(text, **kwargs):
        return vec(0) if "주택" in text else vec()

    monkeypatch.setattr(news, "fetch_news", fetch)
    monkeypatch.setattr(llm, "embed", embed)
    assert await news._selected_related(center, 3) == []
    assert len(calls) == 2


def test_related_graph_share_supplement_cache_and_order(monkeypatch, supplement):
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        return [new_angle(), release("model-a", "S27")]

    monkeypatch.setattr(news, "fetch_news", fetch)
    client = TestClient(app)
    related = client.get("/api/v1/news/center/related").json()["related_news"]
    graph = client.get("/api/v1/news/center/graph").json()["nodes"][1:]
    assert len(calls) == 2  # Repeated endpoints reuse both queries including exhausted results.
    assert [a["news_id"] for a in related] == [a["news_id"] for a in graph]
    assert len(related) == 2


@pytest.mark.parametrize("status", [502, 503, 504])
def test_supplement_failure_remains_error_with_cooldown(monkeypatch, supplement, status):
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        raise NaverNewsError("추가 검색 실패", status)

    monkeypatch.setattr(news, "fetch_news", fetch)
    client = TestClient(app)
    for endpoint in ("related", "graph"):
        response = client.get(f"/api/v1/news/center/{endpoint}")
        assert response.status_code == status
        assert response.json() == {"detail": "추가 검색 실패"}
    assert len(calls) == 1  # Failed search is coalesced/cached as failure, never normal empty data.


def test_extra_embedding_failure_never_returns_partial_success(monkeypatch, supplement):
    async def fetch(query, page_size):
        return [new_angle()]

    async def embed(text, **kwargs):
        return [] if "소비자" in text else vec()

    monkeypatch.setattr(news, "fetch_news", fetch)
    monkeypatch.setattr(llm, "embed", embed)
    response = TestClient(app).get("/api/v1/news/center/related")
    assert response.status_code == 503
    assert "임베딩" in response.json()["detail"]


def test_supplement_deadline_propagates_504(monkeypatch, supplement):
    monkeypatch.setattr(settings, "news_map_supplement_timeout_seconds", .01)

    async def fetch(query, page_size):
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr(news, "fetch_news", fetch)
    response = TestClient(app).get("/api/v1/news/center/related")
    assert response.status_code == 504
    assert "시간" in response.json()["detail"]


async def test_queries_single_flight_success_empty_expiry_and_no_mutation(monkeypatch):
    calls = []

    async def fetch(query, page_size):
        calls.append(query)
        await asyncio.sleep(0)
        return [new_angle()]

    a, b = await asyncio.gather(*(related_candidates.supplemental_search(q, fetch) for q in ("삼성전자 GPU", "삼성전자  gpu")))
    assert len(calls) == 1
    a[0]["title"] = "mutated"
    assert b[0]["title"] != a[0]["title"]
    c = await related_candidates.supplemental_search("삼성전자 GPU", fetch)
    assert c == b
    key = next(iter(related_candidates._cache))
    related_candidates._cache[key] = (0, [])
    assert await related_candidates.supplemental_search("삼성전자 GPU", fetch) == b
    assert len(calls) == 2


def test_grounded_queries_do_not_invent_angles_or_ungrounded_metadata():
    center = release("center", keywords=["소비자 반응", "삼성전자", "HBM3E"])
    queries = related_candidates.grounded_queries(center)
    assert queries and len(queries) == len(set(queries))
    assert all("소비자" not in q and "HBM3E" not in q and len(q) <= 100 for q in queries)
    assert related_candidates.grounded_queries(article("empty", "", keywords=["invented"])) == []


async def test_legacy_fallback_provider_is_not_used_for_supplement(monkeypatch, supplement):
    center, _ = supplement
    monkeypatch.setattr(settings, "news_provider", "gdelt")
    monkeypatch.setattr(settings, "use_mock_news", False)

    async def forbidden(*args, **kwargs):
        pytest.fail("Legacy silent sample fallback cannot fill a news map")

    monkeypatch.setattr(news, "fetch_news", forbidden)
    assert await news._selected_related(center, 3) == []


@pytest.mark.parametrize("key,value", [
    ("news_map_mmr_lambda", 1.1), ("news_map_repeat_cosine", -1),
    ("news_map_repeat_text_similarity", 2), ("news_map_repeat_short_text_similarity", -1),
    ("news_map_repeat_max_hours", 0), ("news_map_supplement_max_searches", 4),
    ("news_map_repeat_description_min_chars", 501), ("news_map_repeat_novelty_ratio", -1),
    ("news_map_supplement_page_size", 21), ("news_map_supplement_candidate_limit", 61),
    ("news_map_supplement_timeout_seconds", 61), ("news_map_supplement_cache_ttl_seconds", 0),
])
def test_new_configuration_rejects_out_of_bounds(key, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{key: value})
