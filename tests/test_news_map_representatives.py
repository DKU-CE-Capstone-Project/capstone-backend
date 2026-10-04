"""Visible representatives, not transitive groups; offline supplied-news regressions."""
import json
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import llm, news_map, related_selector
from app.agents.related_selector import NewsMapSelector, cosine_similarity
from app.agents.repeated_coverage import (
    Judgment,
    Relation,
    compare,
    event_evidence,
    has_issue_connection,
)
from app.config import settings
from app.main import app
from tests.test_news_map_diversity import article, release, vec

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures/news_map_representatives.json").read_text())


def ids(items):
    return [item.article["news_id"] for item in items]


@pytest.fixture
def no_provider(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Replay must reuse the saved Gemini vectors without a provider request")
    monkeypatch.setattr(llm, "embed", forbidden)


async def test_discarded_bridge_cannot_exclude_another_valid_candidate(monkeypatch):
    center = article("center", "center")
    rows = [article(nid, nid) for nid in ("a", "b", "c")]

    async def vector(*args):
        return vec()

    def judgment(a, b, cosine):
        # A repeats B; B repeats C; A and C add different information.
        if {a.title, b.title} in ({"a", "b"}, {"b", "c"}):
            return Judgment(Relation.REPEAT, "controlled_repeat")
        return Judgment(Relation.DIFFERENT, "controlled_different")

    monkeypatch.setattr(related_selector, "article_vector", vector)
    monkeypatch.setattr(related_selector, "compare", judgment)
    for order in permutations(rows):
        selector = NewsMapSelector(center)
        await selector.add(list(order))
        assert ids(selector.select(3)) == ["a", "c"]
        assert selector.neighbour_repeats == 1


async def test_unselected_high_relevance_copy_cannot_hide_mmr_preferred_representative(monkeypatch):
    center = release("center")
    a, b, c = release("a", "S27"), release("b", "S28"), release("c", "S28")
    c["title"] = "삼성전자 갤럭시 S28 선보여…신제품 공개"
    vectors = {"center": vec(), "a": vec(.995), "b": vec(.99), "c": vec(.985, -1)}

    async def vector(row):
        return vectors[row["news_id"]]

    monkeypatch.setattr(related_selector, "article_vector", vector)
    # Isolate selection order from the keyword support differences of the two headlines.
    monkeypatch.setattr(NewsMapSelector, "_relevance", lambda self, row, cosine: cosine)
    assert compare(event_evidence(b), event_evidence(c), cosine_similarity(vectors["b"], vectors["c"])).relation is Relation.REPEAT
    for order in permutations([a, b, c]):
        selector = NewsMapSelector(center)
        await selector.add(list(order))
        assert ids(selector.select(3)) == ["a", "c"]
        assert selector.neighbour_repeats == 1


async def test_representative_ties_use_description_completeness_then_fixed_key(monkeypatch):
    center = release("center")
    rows = [release("z", "S27"), release("a", "S27"), release("short", "S27", description="신제품 공개")]

    async def vector(row):
        return vec()

    monkeypatch.setattr(related_selector, "article_vector", vector)
    monkeypatch.setattr(NewsMapSelector, "_relevance", lambda *args: .9)
    for order in permutations(rows):
        selector = NewsMapSelector(center)
        await selector.add(list(order))
        assert ids(selector.select(3)) == ["a"]


async def test_later_better_copy_is_excluded_and_diagnostics_are_not_double_counted(monkeypatch):
    center, early, late = release("center"), release("early", "S27"), release("late", "S27")

    async def vector(row):
        return vec(.99 if row["news_id"] == "late" else 1)

    monkeypatch.setattr(related_selector, "article_vector", vector)
    selector = NewsMapSelector(center)
    await selector.add([early])
    assert ids(selector.select(3)) == ["early"]
    await selector.add([late])
    assert ids(selector.select(3)) == ["early"]
    assert ids(selector.select(3)) == ["early"]
    assert selector.neighbour_repeats == 1


@pytest.mark.parametrize("topic", list(SNAPSHOT["topics"]))
async def test_saved_representatives_are_deterministic_and_pairwise_nonrepeating(topic, no_provider):
    fixture = SNAPSHOT["topics"][topic]
    rows = deepcopy(fixture["articles"])
    center = next(a for a in rows if a["news_id"] == fixture["center_id"])
    candidates = [a for a in rows if a is not center]
    orders = [candidates, list(reversed(candidates)), candidates[1:] + candidates[:1]]
    results = []
    for order in orders:
        selector = NewsMapSelector(deepcopy(center))
        await selector.add(deepcopy(order))
        selected = selector.select(3)
        results.append(ids(selected))
        visible = [center, *(item.article for item in selected)]
        for index, a in enumerate(visible):
            for b in visible[index + 1:]:
                cosine = cosine_similarity(a["news_map_embedding"]["values"], b["news_map_embedding"]["values"])
                judgment = compare(event_evidence(a), event_evidence(b), cosine)
                assert judgment.relation is not Relation.REPEAT and not judgment.withheld
    assert results[0] == results[1] == results[2]
    if topic == "해운":
        assert results[0] == []  # Contract reprints and unrelated relocation reports.
    elif topic == "환율":
        assert results[0] == ["77176e920ff6"]  # Actual generational differences remain.
    elif topic == "사이버보안":
        assert results[0] == ["d844b5cf9ac8"]  # Corporate penalties provide a different policy context.
    else:
        assert results[0] == ["7fffbcb60b1e"]  # Supplied pregnancy/youth information, one safety report.


@pytest.mark.parametrize("topic", list(SNAPSHOT["topics"]))
def test_saved_related_graph_contract_and_storage_retention(topic, no_provider, monkeypatch):
    fixture = deepcopy(SNAPSHOT["topics"][topic])
    store.news_cache.update({a["news_id"]: a for a in fixture["articles"]})
    # Force cached-pool expansion after the first candidate. No external search.
    monkeypatch.setattr(settings, "news_map_initial_candidates", 1)
    client = TestClient(app)
    path = f"/api/v1/news/{fixture['center_id']}"
    first = client.get(f"{path}/related?expand=false").json()
    assert first["selection"]["status"] == "expandable"
    related = client.get(f"{path}/related").json()
    graph = client.get(f"{path}/graph").json()
    assert related["selection"] == graph["selection"]
    assert related["related_news"] == [{k: v for k, v in a.items() if k != "is_center"} for a in graph["nodes"][1:]]
    assert "center_same_story" not in related and "center_same_story_total" not in related
    for node in graph["nodes"]:
        assert "same_story" not in node and "same_story_total" not in node
    assert len(store.news_cache) == len(fixture["articles"])


def test_topic_only_news_fails_connection_but_explicit_impact_survives():
    center = article("center", "자동차용 반도체 공급 중단", "자동차용 반도체 공장의 화재로 공급이 중단됐다. 생산 복구까지 상당한 시간이 걸릴 것으로 예상된다.")
    unrelated = article("other", "반도체 기업 사옥 이전", "반도체 기업이 새로운 사옥으로 이전했다고 밝혔다. 직원들은 다음 달부터 새로운 사무실에서 근무한다.")
    impact = article("impact", "자동차 생산 차질", "자동차용 반도체 공급 중단으로 완성차 업체가 생산을 줄인다. 부품 재고가 부족해 조립 공장이 일시 중단된다.")
    assert not has_issue_connection(event_evidence(center), event_evidence(unrelated), "반도체")
    assert has_issue_connection(event_evidence(center), event_evidence(impact), "반도체")


async def test_expansion_preserves_distinct_background_and_excludes_copies(monkeypatch):
    from tests.test_news_map_expansion import angle, cached, searches
    center = cached(article("center", "국제유가, 미국 원유 재고 감소에 상승…브렌트유 1.2%↑",
                            "국제유가가 1일 미국 원유 재고 감소로 상승했다. 브렌트유는 1.2% 오른 90.5달러에 거래됐다.",
                            _search_keyword="국제유가"), 1)
    first = cached(angle("airlines"), 2)
    copy = {**first, "news_id": "copy", "url": "https://fictional.test/copy", "naver_url": ""}
    store.news_cache.update({a["news_id"]: a for a in [center, first]})
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "news_map_supplement_max_searches", 1)

    async def vector(row):
        return vec()

    async def passthrough(rows):
        return rows

    monkeypatch.setattr(related_selector, "article_vector", vector)
    from app.agents import news_fetcher
    monkeypatch.setattr(news_fetcher, "inspect_search_articles", passthrough)
    seen = searches(monkeypatch, {"*": [copy, angle("opec")]})
    result = await news_map.build_news_map(center, target=3)
    assert ids(result.items) == ["airlines", "opec"]
    assert len(seen) == result.stats["neighbour_repeats"] == 1
    assert result.status == "insufficient" and result.reason == "search_limit"
