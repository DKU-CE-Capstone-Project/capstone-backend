"""Bounded 20→50 candidate expansion with mock search/vectors, never live services.

Articles are fictional. Search pages are controlled fixtures that carry raw ranks
the way NAVER positions do, including duplicates that must not count twice.
"""
import asyncio
import logging
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import database, store
from app.agents import (
    article_embeddings,
    article_metadata,
    llm,
    news_fetcher,
    news_map,
    related_candidates,
)
from app.agents.naver_client import NaverNewsError, NaverNewsPage
from app.config import Settings, settings
from app.main import app
from tests.test_news_map_diversity import OIL_CENTER, oil, vec

SESSION = "2026-10-01T22:00:00+00:00"


def cached(article, rank, end=20, session=SESSION):
    return {**article, "_search_rank": rank, "_search_end": end, "_searched_at": session,
            "naver_url": f"https://n.news.naver.com/article/001/{article['news_id']}"}


def repeat(i):
    """Another outlet's report of the center's event, repeating its information."""
    return oil(f"repeat-{i:02d}", f"국제유가, 미국 원유 재고 감소에 상승…브렌트유 1.2%↑ ({i}언론)",
               "미국 원유 재고가 줄었다는 소식에 국제유가가 1일(현지시간) 올랐다. 브렌트유 선물은 "
               "전장 대비 1.2% 오른 배럴당 90.5달러에 거래를 마쳤다.")


ANGLES = [
    ("forecast", "국제유가 상승에 연말 물가 부담 커질 듯",
     "증권가는 유가 상승이 연말 소비자물가에 부담을 줄 것으로 전망했다. 항공·화학 업종의 원가 타격 우려도 나온다."),
    ("refiners", "정유사 정제마진 개선 기대",
     "국제유가 상승과 휘발유 재고 감소로 국내 정유사의 정제마진이 개선될 것이라는 분석이 나왔다."),
    ("airlines", "항공사 유류할증료 다음 달 인상",
     "국제유가 상승으로 국내 항공사들이 다음 달 국제선 유류할증료를 올린다. 편도 기준 최대 5만원이 부과된다."),
    ("opec", "산유국 감산 연장 논의 착수",
     "산유국 협의체가 원유 감산 연장 여부를 다음 주 논의한다. 공급 축소가 이어지면 유가 상승 압력이 커질 수 있다."),
]


def angle(name, rank=None):
    nid, title, description = next(a for a in ANGLES if a[0] == name)
    return oil(nid, title, description)


def page(articles, start, size):
    window = articles[start - 1:start - 1 + size]
    return NaverNewsPage(articles=[{**deepcopy(a), "_search_rank": start + i} for i, a in enumerate(window)],
                         total=len(articles), start=start, end=start + len(window) - 1)


@pytest.fixture
def oil_map(monkeypatch):
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 3)
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mock_news", False)
    center = cached(oil("center", *OIL_CENTER), 1)
    store.news_cache[center["news_id"]] = center
    calls = []

    async def embed(text, **kwargs):
        calls.append(text)
        return vec()

    async def passthrough(articles):
        return articles

    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(news_fetcher, "inspect_search_articles", passthrough)
    return center, calls


def put(articles):
    store.news_cache.update({a["news_id"]: a for a in articles})


def searches(monkeypatch, results):
    """results: query -> ordered raw list (or exception). Records (query, start, size)."""
    seen = []

    async def fetch(query, *, start, size):
        seen.append((query, start, size))
        value = results.get(query, results.get("*", []))
        if isinstance(value, Exception):
            raise value
        return page(value, start, size)

    monkeypatch.setattr(news_fetcher, "fetch_search_page", fetch)
    return seen


def neighbour_ids(result):
    return [g.article["news_id"] for g in result.items]


async def test_initial_candidates_sufficient_skip_external_search(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(angle(name), rank) for rank, name in enumerate(["forecast", "refiners", "airlines"], 2)])
    put([cached(repeat(i), 10 + i) for i in range(5)])
    seen = searches(monkeypatch, {})
    result = await news_map.build_news_map(center, target=3)
    assert result.status == "complete" and len(result.items) == 3
    assert seen == [] and result.stats["searches"] == 0
    assert result.stats["center_repeats"] == 5


async def test_initial_pool_full_of_repeats_expands_with_grounded_then_original_next_page(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(repeat(i), i + 2) for i in range(19)])
    raw = [center, *[repeat(i) for i in range(19)], repeat(30), angle("forecast"), angle("refiners"),
           angle("airlines"), angle("opec")]
    seen = searches(monkeypatch, {"국제유가": raw, "*": []})
    result = await news_map.build_news_map(center, target=3)
    assert result.status == "complete" and len(result.items) == 3
    # The center's grounded query runs first; the original relevance search then
    # continues after raw position 20, not page 1 again.
    size = settings.news_map_supplement_page_size
    assert seen[0][0] != "국제유가" and seen[0][1:] == (1, size)
    assert seen[1] == ("국제유가", 21, size)
    assert result.stats["initial"] == 19 and result.stats["search_steps"] == 2
    assert result.stats["center_repeats"] == 20  # Reprints never consume neighbour slots.
    assert result.stats["evaluated"] <= settings.news_map_max_candidates
    # Expansion candidates keep the original session/rank for later pools.
    assert store.news_cache["forecast"]["_searched_at"] == SESSION
    assert store.news_cache["forecast"]["_search_rank"] == 22


async def test_page_size_change_neither_repeats_nor_skips_raw_positions(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_map_supplement_page_size", 7)
    put([cached(repeat(i), i + 2) for i in range(19)])
    raw = [center, *[repeat(i) for i in range(19)], *[repeat(30 + i) for i in range(14)], angle("forecast")]
    seen = searches(monkeypatch, {"국제유가": raw, "*": []})
    first = await news_map.build_news_map(center, target=1)
    assert ("국제유가", 21, 7) in seen
    assert all(store.news_cache[f"repeat-{30 + i:02d}"]["_search_end"] == 27 for i in range(7))
    # A later map for the same session continues at 28, after the 7-item page.
    other = store.news_cache["repeat-00"]
    seen.clear()
    await news_map.build_news_map(other, target=1)
    assert [start for query, start, _ in seen if query == "국제유가"] == [28]
    assert first.status in {"insufficient", "complete"}


async def test_pool_order_follows_session_and_rank_not_article_id(monkeypatch, oil_map):
    center, calls = oil_map
    monkeypatch.setattr(settings, "news_map_initial_candidates", 3)
    late = cached(angle("opec"), 3, session="2026-09-30T00:00:00+00:00")
    put([cached(angle("refiners"), 4), cached(angle("forecast"), 3), cached(angle("airlines"), 2), late])
    searches(monkeypatch, {})
    result = await news_map.build_news_map(center, target=3)
    assert result.status == "complete"
    assert all("산유국" not in text for text in calls)  # Older session, beyond the initial 3.
    assert result.stats["initial"] == 3


async def test_fifty_unique_cap_counts_only_unique_candidates(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_map_supplement_page_size", 50)
    put([cached(repeat(i), i + 2) for i in range(30)])  # 20 initial + 10 cached remainder.
    raw = [center, *[repeat(i) for i in range(60)]]
    seen = searches(monkeypatch, {"국제유가": raw, "*": []})
    result = await news_map.build_news_map(center, target=3)
    assert result.stats["evaluated"] == 50
    assert result.status == "insufficient" and result.reason == "candidate_limit"
    assert result.items == [] and result.stats["center_repeats"] == 50
    # Raw positions 21-61: reports 19-29 were already pooled and are not unique candidates.
    assert result.stats["raw"] == 41 and result.stats["raw_duplicates"] == 11
    assert result.stats["added_cached_pool"] == 10 and result.stats["added_original_next"] == 20
    assert len(seen) == 2 and seen[1] == ("국제유가", 21, 50)


async def test_expansion_stops_at_search_budget_and_reports_shortage(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 2)
    monkeypatch.setattr(settings, "news_map_supplement_page_size", 3)
    put([cached(repeat(i), i + 2) for i in range(5)])
    seen = searches(monkeypatch, {"*": [repeat(40 + i) for i in range(40)]})
    result = await news_map.build_news_map(center, target=3)
    assert len(seen) == 2 and result.stats["search_steps"] == 2
    assert result.status == "insufficient" and result.reason == "search_limit"
    assert result.stats["evaluated"] < settings.news_map_max_candidates


async def test_fifty_evaluated_with_valid_shortage_returns_fewer_nodes(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_map_supplement_page_size", 50)
    put([cached(repeat(i), i + 2) for i in range(19)])
    raw = [*[repeat(100 + i) for i in range(40)], angle("forecast")]
    searches(monkeypatch, {"*": raw})
    result = await news_map.build_news_map(center, target=3)
    assert result.stats["evaluated"] == 50
    assert neighbour_ids(result) == [] or len(result.items) < 3
    assert result.status == "insufficient"


async def test_irrelevant_expansion_candidates_never_fill_slots(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(repeat(i), i + 2) for i in range(3)])

    async def embed(text, **kwargs):
        return vec(0) if "전세" in text else vec()

    monkeypatch.setattr(llm, "embed", embed)
    unrelated = [oil(f"house-{i}", f"주택 전세 계약 {i}", "임대차 계약 갱신 사례") for i in range(10)]
    searches(monkeypatch, {"*": unrelated})
    result = await news_map.build_news_map(center, target=3)
    assert result.items == [] and result.status == "insufficient"
    assert result.stats["below_threshold"] >= 10


async def test_search_failure_keeps_valid_initial_neighbours_as_partial(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(angle("forecast"), 2), cached(repeat(1), 3)])
    searches(monkeypatch, {"*": NaverNewsError("추가 검색 실패", 503)})
    result = await news_map.build_news_map(center, target=3)
    assert neighbour_ids(result) == ["forecast"]
    assert result.status == "partial" and result.reason == "search_failed"


async def test_expansion_timeout_is_partial(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_map_supplement_timeout_seconds", .05)
    put([cached(angle("forecast"), 2)])

    async def slow(query, *, start, size):
        await asyncio.sleep(1)

    monkeypatch.setattr(news_fetcher, "fetch_search_page", slow)
    result = await news_map.build_news_map(center, target=3)
    assert neighbour_ids(result) == ["forecast"]
    assert result.status == "partial" and result.reason == "timeout"


async def test_expansion_embedding_failure_is_partial_but_first_round_failure_raises(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(angle("forecast"), 2)])
    searches(monkeypatch, {"*": [angle("refiners")]})

    async def embed(text, **kwargs):
        return [] if "정제마진" in text else vec()

    monkeypatch.setattr(llm, "embed", embed)
    result = await news_map.build_news_map(center, target=3)
    assert neighbour_ids(result) == ["forecast"]
    assert result.status == "partial" and result.reason == "embedding_failed"

    async def broken(text, **kwargs):
        return []

    monkeypatch.setattr(llm, "embed", broken)
    article_embeddings._cache.clear()
    store.news_cache["forecast"].pop("news_map_embedding", None)
    with pytest.raises(article_embeddings.EmbeddingUnavailable):
        await news_map.build_news_map(center, target=3)


async def test_repeated_request_reuses_search_cache_and_vectors(monkeypatch, oil_map):
    center, calls = oil_map
    put([cached(repeat(i), i + 2) for i in range(4)])
    seen = searches(monkeypatch, {"국제유가": [center, *[repeat(i) for i in range(19)], angle("forecast"),
                                          angle("airlines"), angle("opec")]})
    first = await news_map.build_news_map(center, target=3)
    embedded = len(calls)
    second = await news_map.build_news_map(center, target=3)
    assert neighbour_ids(second) == neighbour_ids(first)
    # Grounded query (empty) + original continuation once; the continued page joined the
    # original search session, so the next map completes from that pool with no search.
    assert len(seen) == 2 and second.stats["search_steps"] == 0 and second.stats["embedding_calls"] == 0
    assert second.stats["initial"] == first.stats["evaluated"]
    assert len(calls) == embedded and second.stats["embedding_reused"] > 0


async def test_concurrent_requests_share_one_search(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(repeat(i), i + 2) for i in range(3)])
    seen = []

    async def fetch(query, *, start, size):
        seen.append(query)
        await asyncio.sleep(.01)
        return page([angle("forecast"), angle("airlines")], start, size)

    monkeypatch.setattr(news_fetcher, "fetch_search_page", fetch)
    a, b = await asyncio.gather(news_map.build_news_map(center, target=3),
                                news_map.build_news_map(center, target=3))
    assert neighbour_ids(a) == neighbour_ids(b)
    assert seen.count("국제유가") == 1


async def test_expand_false_reports_expandable_and_full_request_appends(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(angle("forecast"), 2), *[cached(repeat(i), i + 3) for i in range(5)]])
    seen = searches(monkeypatch, {"*": [angle("airlines"), angle("opec")]})
    initial = await news_map.build_news_map(center, target=3, expand=False)
    assert initial.status == "expandable" and seen == []
    full = await news_map.build_news_map(center, target=3)
    assert neighbour_ids(full)[:len(initial.items)] == neighbour_ids(initial)
    assert full.status == "complete"


async def test_grounded_query_precedes_original_and_never_repeats_it(monkeypatch, oil_map):
    center, _ = oil_map
    put([cached(repeat(i), i + 2) for i in range(3)])
    seen = searches(monkeypatch, {"국제유가": [repeat(i) for i in range(30)], "*": [angle("forecast")]})
    result = await news_map.build_news_map(center, target=1)
    # The grounded query found a meaningful neighbour, so no continuation was needed.
    assert len(seen) == 1 and seen[0][0] != "국제유가" and seen[0][1] == 1
    assert neighbour_ids(result) == ["forecast"]
    # Grounded-query results keep their own search membership.
    assert store.news_cache["forecast"]["_search_keyword"] == seen[0][0]


async def test_legacy_fallback_provider_is_not_used_for_expansion(monkeypatch, oil_map):
    center, _ = oil_map
    monkeypatch.setattr(settings, "news_provider", "gdelt")

    async def forbidden(*args, **kwargs):
        pytest.fail("Legacy silent sample fallback cannot fill a news map")

    monkeypatch.setattr(news_fetcher, "fetch_search_page", forbidden)
    result = await news_map.build_news_map(center, target=3)
    assert result.items == [] and result.status == "insufficient" and result.reason == "no_source"


async def test_expansion_adds_no_generation_body_or_rag_calls(monkeypatch, oil_map, metadata_ai):
    center, _ = oil_map
    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(settings, "mongodb_required", True)
    saved, vectors = [], []

    async def none(*args):
        return None

    async def no_candidates(*args):
        return []

    async def save(doc):
        saved.append(deepcopy(doc))

    async def save_vector(nid, purpose, record):
        vectors.append(purpose)

    async def embed(text, **kwargs):
        assert kwargs.get("task_type") == "SEMANTIC_SIMILARITY", "Expansion cannot embed RAG"
        return vec()

    async def forbidden(*args, **kwargs):
        pytest.fail("Expansion must not generate text or extract bodies")

    monkeypatch.setattr(database, "get_news", none)
    monkeypatch.setattr(database, "get_news_embedding", none)
    monkeypatch.setattr(database, "news_candidates", no_candidates)
    monkeypatch.setattr(database, "save_news", save)
    monkeypatch.setattr(database, "save_news_embedding", save_vector)
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(article_metadata, "generate", forbidden)
    from app.agents import diffbot_client
    monkeypatch.setattr(diffbot_client, "extract_article", forbidden, raising=False)
    searches(monkeypatch, {"*": [angle("forecast")]})
    result = await news_map.build_news_map(center, target=1)
    assert neighbour_ids(result) == ["forecast"]
    assert len(saved) == 1 and "embedding" not in saved[0]
    assert saved[0]["metadata_extraction"]["reason"] == "news_map_supplement"
    assert set(vectors) <= {"news_map"}


async def test_diagnostic_log_has_counts_without_titles_queries_or_vectors(monkeypatch, oil_map, caplog):
    center, _ = oil_map
    put([cached(repeat(1), 2)])
    searches(monkeypatch, {"*": [angle("forecast")]})
    with caplog.at_level(logging.INFO, logger="econmind.news_map"):
        await news_map.build_news_map(center, target=1)
    line = next(r.getMessage() for r in caplog.records if r.name == "econmind.news_map")
    for key in ("evaluated=", "center_repeats=", "searches=", "raw=", "embedding_calls=", "elapsed_ms=", "status="):
        assert key in line
    assert "국제유가" not in line and "브렌트" not in line and "0.0" not in line.split("elapsed_ms")[0]


# ── API contract ────────────────────────────────────────────────────────────

def test_related_graph_share_status_representatives_and_tier_policy(monkeypatch, oil_map):
    _ = oil_map
    put([cached(repeat(i), i + 2) for i in range(2)])
    searches(monkeypatch, {"*": NaverNewsError("추가 검색 실패", 503)})
    put([cached(angle("forecast"), 10)])
    client = TestClient(app)
    for tier in ("FREE", "PAID"):
        related = client.get("/api/v1/news/center/related", params={"tier": tier, "limit": 5}).json()
        graph = client.get("/api/v1/news/center/graph", params={"tier": tier, "limit": 5}).json()
        assert [a["news_id"] for a in related["related_news"]] == ["forecast"]
        assert [a["news_id"] for a in graph["nodes"][1:]] == ["forecast"]
        assert related["selection"] == graph["selection"]
        assert related["selection"]["status"] == "partial" and related["selection"]["reason"] == "search_failed"
        assert related["selection"]["requested"] == (5 if tier == "PAID" else 3)
        assert "center_same_story" not in related and "center_same_story_total" not in related
        assert all("same_story" not in a for a in graph["nodes"])
        assert (related["related_news"][0]["relevance_score"] is None) == (tier == "FREE")


def test_expand_query_and_openapi_contract(monkeypatch, oil_map):
    _ = oil_map
    put([cached(angle("forecast"), 2)])
    seen = searches(monkeypatch, {"*": [angle("airlines")]})
    client = TestClient(app)
    first = client.get("/api/v1/news/center/related", params={"expand": "false"}).json()
    assert first["selection"]["status"] == "expandable" and seen == []
    full = client.get("/api/v1/news/center/related").json()
    assert [a["news_id"] for a in full["related_news"]][:1] == ["forecast"]
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"same_story", "same_story_total"}.isdisjoint(schema["RelatedNewsItem"]["properties"])
    assert {"same_story", "same_story_total"}.isdisjoint(schema["GraphNode"]["properties"])
    assert "selection" in schema["RelatedResponse"]["properties"]
    assert {"center_same_story", "center_same_story_total"}.isdisjoint(schema["RelatedResponse"]["properties"])
    assert "SameStoryArticle" not in schema
    assert schema["NewsMapSelection"]["properties"]["status"]["enum"] == ["complete", "insufficient", "partial",
                                                                          "expandable"]


def test_first_round_embedding_failure_is_503(monkeypatch, oil_map):
    put([cached(angle("forecast"), 2)])

    async def broken(text, **kwargs):
        raise RuntimeError("private provider error")

    monkeypatch.setattr(llm, "embed", broken)
    for route in ("related", "graph"):
        response = TestClient(app).get(f"/api/v1/news/center/{route}")
        assert response.status_code == 503 and "private" not in response.text


# ── search cache and query grounding ────────────────────────────────────────

async def test_cached_search_single_flight_expiry_failure_and_no_mutation():
    calls = []

    async def fetch(query, *, start, size):
        calls.append((query, start, size))
        await asyncio.sleep(0)
        return page([angle("forecast")], start, size)

    a, b = await asyncio.gather(*(related_candidates.cached_search(q, start=1, size=5, fetch=fetch)
                                  for q in ("국제유가 재고", "국제유가  재고")))
    assert len(calls) == 1
    a.articles[0]["title"] = "mutated"
    assert b.articles[0]["title"] != "mutated"
    await related_candidates.cached_search("국제유가 재고", start=6, size=5, fetch=fetch)
    assert len(calls) == 2  # Another raw window is another key.
    key = next(iter(related_candidates._cache))
    related_candidates._cache[key] = (0, None)
    await related_candidates.cached_search("국제유가 재고", start=1, size=5, fetch=fetch)
    assert len(calls) == 3

    async def failing(query, *, start, size):
        calls.append(query)
        raise NaverNewsError("실패", 502)

    for _ in range(2):
        with pytest.raises(NaverNewsError):
            await related_candidates.cached_search("실패", start=1, size=5, fetch=failing)
    assert calls.count("실패") == 1  # Failure is cached briefly as a failure, not as empty data.


def test_grounded_queries_do_not_invent_angles_or_ungrounded_metadata():
    center = oil("center", *OIL_CENTER, )
    center["keywords"] = ["소비자 반응", "국제유가", "HBM3E"]
    queries = related_candidates.grounded_queries(center)
    assert queries and len(queries) == len(set(queries))
    assert all("소비자" not in q and "HBM3E" not in q and len(q) <= 100 for q in queries)
    assert all(q != center["_search_keyword"] for q in queries)
    assert related_candidates.grounded_queries(oil("empty", "", "")) == []


@pytest.mark.parametrize("key,value", [
    ("news_map_mmr_lambda", 1.1), ("news_map_repeat_cosine", -1),
    ("news_map_repeat_text_similarity", 2), ("news_map_repeat_title_similarity", -1),
    ("news_map_repeat_short_text_similarity", -1), ("news_map_repeat_max_hours", 0),
    ("news_map_repeat_description_min_chars", 501), ("news_map_repeat_novelty_ratio", -1),
    ("news_map_initial_candidates", 2), ("news_map_initial_candidates", 51), ("news_map_max_candidates", 51),
    ("news_map_supplement_max_searches", 6),
    ("news_map_supplement_page_size", 51), ("news_map_supplement_timeout_seconds", 61),
    ("news_map_supplement_cache_ttl_seconds", 0),
])
def test_configuration_rejects_out_of_bounds(key, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{key: value})


def test_defaults_define_the_20_to_50_budget():
    defaults = Settings(_env_file=None)
    assert (defaults.news_map_initial_candidates, defaults.news_map_max_candidates) == (20, 50)
    assert defaults.news_map_supplement_max_searches == 3 and defaults.news_map_supplement_page_size == 20
    assert defaults.news_map_supplement_timeout_seconds == 20
