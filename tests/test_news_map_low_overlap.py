"""Frozen tariff snippets reproduce an unconfirmed repeat, using only mock vectors.

The fixture keeps the two supplied articles' saved search inputs and provenance.
Additional articles below are fictional controls. Global offline fixtures block
external sockets and disable MongoDB; search and embedding functions are mocked.
"""
import json
import logging
import math
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import news_fetcher, news_map, related_selector
from app.agents.naver_client import NaverNewsPage
from app.agents.related_selector import NewsMapSelector, cosine_similarity
from app.agents.repeated_coverage import Judgment, Relation, compare, event_evidence
from app.config import settings
from app.main import app

FIXTURE = json.loads((Path(__file__).parent / "fixtures/news_map_tariff_low_overlap.json").read_text())
COSINE = FIXTURE["cosine"]


def vector(cosine=1.0):
    return [cosine, math.sqrt(1 - cosine ** 2)] + [0.0] * 766


def control(center, nid, **changes):
    return {**deepcopy(center), "news_id": nid, "url": f"https://fictional.test/{nid}",
            "naver_url": f"https://fictional.test/{nid}", "source": "fictional", **changes}


def analysis(center):
    return control(center, "analysis", title="관세 15% 유지와 대미투자의 기업 영향 분석",
                   description="연구진은 관세 15% 유지가 수출기업의 실적과 투자에 미칠 영향을 분석했다. "
                               "기업별 원가 부담과 향후 매출 전망을 비교한 결과도 제시했다.")


@pytest.fixture
def tariff_map(monkeypatch):
    center, candidate = deepcopy(FIXTURE["center"]), deepcopy(FIXTURE["candidate"])

    async def article_vector(article):
        # Cloned uncertain candidates have the same controlled vector as the fixture.
        return vector(COSINE if article["title"] == candidate["title"] else 1.0)

    monkeypatch.setattr(related_selector, "article_vector", article_vector)
    return center, candidate


def setup_expansion(monkeypatch, center, candidate, additions):
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_map_initial_candidates", 1)
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 1)
    store.news_cache.update({a["news_id"]: a for a in (center, candidate)})
    calls = []

    async def fetch(query, *, start, size):
        calls.append((query, start, size))
        articles = deepcopy(additions[start - 1:start - 1 + size])
        return NaverNewsPage(articles=articles, total=len(additions), start=start,
                             end=start + len(articles) - 1)

    async def inspect(articles):
        return articles

    monkeypatch.setattr(news_fetcher, "fetch_search_page", fetch)
    monkeypatch.setattr(news_fetcher, "inspect_search_articles", inspect)
    return calls


@pytest.mark.parametrize("reverse", [False, True])
def test_tariff_pair_is_low_overlap_insufficient(tariff_map, reverse):
    pair = tariff_map[::-1] if reverse else tariff_map
    assert cosine_similarity(vector(), vector(COSINE)) == pytest.approx(COSINE)
    judgment = compare(*map(event_evidence, pair), COSINE)
    assert judgment == Judgment(Relation.INSUFFICIENT, "low_overlap")
    assert judgment.withheld


@pytest.mark.parametrize("reason,withheld", [
    ("low_overlap", True), ("short_text_covered", True),
    ("short_text", False), ("time_unknown", False), ("future_reason", False),
])
def test_only_explicit_insufficient_reasons_are_withheld(reason, withheld):
    assert Judgment(Relation.INSUFFICIENT, reason).withheld is withheld
    assert not Judgment(Relation.NEW_INFO, reason).withheld


@pytest.mark.parametrize("reverse", [False, True])
async def test_low_overlap_is_not_a_neighbour(tariff_map, reverse):
    center, candidate = tariff_map[::-1] if reverse else tariff_map
    selector = NewsMapSelector(center)
    assert await selector.add([candidate]) == 1
    assert selector.select(1) == []
    assert selector.remaining == [] and selector.center_repeats == 0
    assert selector.withheld == 1
    assert selector.evaluated == 1  # Evaluation budget consumed; display target not filled.


async def test_low_overlap_against_a_neighbour_is_also_withheld(tariff_map):
    representative, candidate = tariff_map
    center = control(representative, "later-center", published_at="2026-10-04T09:00:00Z")
    assert compare(event_evidence(center), event_evidence(candidate), COSINE).relation is Relation.DIFFERENT
    selector = NewsMapSelector(center)
    await selector.add([representative])
    await selector.add([candidate])
    groups = selector.select(2)
    assert [g.article["news_id"] for g in groups] == [representative["news_id"]]
    assert selector.center_repeats == 0
    assert selector.withheld == 1


async def test_time_unknown_remains_an_eligible_insufficient_candidate(tariff_map):
    center, candidate = tariff_map
    candidate["published_at"] = ""
    assert compare(event_evidence(center), event_evidence(candidate), COSINE) == Judgment(
        Relation.INSUFFICIENT, "time_unknown")
    selector = NewsMapSelector(center)
    await selector.add([candidate])
    assert len(selector.select(1)) == 1
    assert selector.withheld == selector.center_repeats == 0


async def test_confirmed_repeat_and_added_analysis_are_preserved(tariff_map):
    center, candidate = tariff_map
    repeat, added = control(center, "repeat"), analysis(center)
    assert compare(event_evidence(center), event_evidence(repeat), 1.0).relation is Relation.REPEAT
    assert compare(event_evidence(center), event_evidence(added), 1.0) == Judgment(
        Relation.NEW_INFO, "title_addition")
    selector = NewsMapSelector(center)
    await selector.add([candidate, repeat, added])
    groups = selector.select(3)
    assert [g.article["news_id"] for g in groups] == ["analysis"]
    assert selector.neighbour_repeats == 0
    assert selector.center_repeats == 1
    assert selector.withheld == 1


async def test_withheld_candidate_triggers_cached_pool_expansion(monkeypatch, tariff_map):
    center, candidate = tariff_map
    calls = setup_expansion(monkeypatch, center, candidate, [])
    added = analysis(center)
    added["_search_rank"] = 21
    store.news_cache[added["news_id"]] = added
    result = await news_map.build_news_map(center, target=1)
    assert result.status == "complete" and result.reason is None
    assert [g.article["news_id"] for g in result.items] == ["analysis"]
    assert result.stats["initial"] == result.stats["withheld"] == 1
    assert result.stats["added_cached_pool"] == 1 and result.stats["evaluated"] == 2
    assert result.stats["center_repeats"] == 0 and calls == []


@pytest.mark.parametrize("valid_count", [0, 1, 2])
async def test_withheld_candidate_expands_and_returns_only_valid_results(
    monkeypatch, tariff_map, caplog, valid_count,
):
    center, candidate = tariff_map
    follow_up = control(center, "follow-up", published_at="2026-10-04T09:00:00Z")
    calls = setup_expansion(monkeypatch, center, candidate, [analysis(center), follow_up][:valid_count])
    with caplog.at_level(logging.INFO, logger="econmind.news_map"):
        result = await news_map.build_news_map(center, target=2)
    assert len(calls) == result.stats["search_steps"] == 1
    assert result.stats["initial"] == result.stats["withheld"] == 1
    assert result.stats["evaluated"] == 1 + valid_count
    assert result.stats["selected"] == len(result.items) == valid_count
    assert result.stats["center_repeats"] == result.stats["neighbour_repeats"] == 0
    assert candidate["news_id"] not in [g.article["news_id"] for g in result.items]
    assert result.stats["center_repeats"] == result.stats["neighbour_repeats"] == 0
    assert (result.status, result.reason) == (
        ("complete", None) if valid_count == 2 else ("insufficient", "search_limit"))
    line = next(r.getMessage() for r in caplog.records if r.name == "econmind.news_map")
    assert "withheld=1" in line and f"selected={valid_count}" in line


async def test_withheld_candidates_still_obey_total_evaluation_budget(monkeypatch, tariff_map):
    center, candidate = tariff_map
    another_uncertain = control(candidate, "uncertain-copy")
    calls = setup_expansion(monkeypatch, center, candidate, [another_uncertain, analysis(center)])
    monkeypatch.setattr(settings, "news_map_max_candidates", 2)
    result = await news_map.build_news_map(center, target=1)
    assert (result.status, result.reason) == ("insufficient", "candidate_limit")
    assert result.items == [] and result.stats["center_repeats"] == 0
    assert result.stats["evaluated"] == result.stats["withheld"] == 2 and len(calls) == 1


@pytest.mark.parametrize("tier,requested", [("FREE", 3), ("BASIC", 3), ("PAID", 4)])
def test_related_graph_withhold_expand_and_report_the_same_shortage(monkeypatch, tariff_map, tier, requested):
    center, candidate = tariff_map
    calls = setup_expansion(monkeypatch, center, candidate, [analysis(center)])
    monkeypatch.setattr(settings, "news_map_initial_candidates", 20)
    repeat = control(center, "repeat", _search_rank=2)
    store.news_cache[repeat["news_id"]] = repeat
    client = TestClient(app)

    def request(endpoint, expand):
        response = client.get(f"/api/v1/news/{center['news_id']}/{endpoint}",
                              params={"tier": tier, "limit": 4, "expand": expand})
        assert response.status_code == 200
        return response.json()

    initial_related, initial_graph = request("related", False), request("graph", False)
    assert calls == []
    assert initial_related["selection"] == initial_graph["selection"] == {
        "status": "expandable", "reason": None, "requested": requested, "returned": 0, "excluded": 0,
    }
    assert initial_related["related_news"] == [] and len(initial_graph["nodes"]) == 1
    related, graph = request("related", True), request("graph", True)
    assert len(calls) == 1  # The second endpoint reuses the mocked search page cache.
    assert related["selection"] == graph["selection"] == {
        "status": "insufficient", "reason": "search_limit", "requested": requested, "returned": 1, "excluded": 0,
    }
    assert [a["news_id"] for a in related["related_news"]] == ["analysis"]
    assert [a["news_id"] for a in graph["nodes"][1:]] == ["analysis"]
    assert "center_same_story" not in related and "center_same_story_total" not in related
    for card in [*related["related_news"], *graph["nodes"]]:
        assert "same_story" not in card and "same_story_total" not in card
    assert (related["related_news"][0]["relevance_score"] is None) == (tier != "PAID")
    assert (graph["nodes"][1]["relevance_score"] is None) == (tier != "PAID")
