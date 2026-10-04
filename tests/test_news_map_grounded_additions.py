"""Saved search snippets and mock vectors; no live news, LLM or database calls.

The tariff inputs are frozen real snippets, not a benchmark of all news quality.
Other cases are authored controls for names versus separately supplied facts.
"""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import news_map, related_selector
from app.agents.related_selector import NewsMapSelector, cosine_similarity
from app.agents.repeated_coverage import (
    Judgment,
    Relation,
    _new_items,
    compare,
    event_evidence,
)
from app.config import settings
from app.main import app
from tests.test_news_map_low_overlap import control, setup_expansion, vector

FIXTURE = json.loads((Path(__file__).parent / "fixtures/news_map_tariff_followup.json").read_text())
CENTER = FIXTURE["center"]
WOW, HERALD, KOOKJE, NEWSIS = FIXTURE["candidates"]


@pytest.fixture
def saved_vectors(monkeypatch):
    cosines = {row["article"]["news_id"]: row["cosine"] for row in FIXTURE["candidates"]}

    async def article_vector(article):
        return vector(cosines.get(article["news_id"], 1.0))

    monkeypatch.setattr(related_selector, "article_vector", article_vector)


@pytest.mark.parametrize("row,reason", [(WOW, "low_overlap"), (HERALD, "semantic_difference"), (KOOKJE, "low_overlap")])
@pytest.mark.parametrize("reverse", [False, True])
def test_saved_unconfirmed_pairs_are_withheld(row, reason, reverse):
    pair = (row["article"], CENTER) if reverse else (CENTER, row["article"])
    judgment = compare(*map(event_evidence, pair), row["cosine"])
    assert cosine_similarity(vector(), vector(row["cosine"])) == pytest.approx(row["cosine"])
    assert judgment == Judgment(Relation.INSUFFICIENT, reason)
    assert judgment.withheld


@pytest.mark.parametrize("phrase,name", [
    ("김정관 장관", "김정관"),
    ("김정관 산업부 장관", "김정관"),
    ("김정관 산업통상부 장관", "김정관"),
    ("정부는 김정관 산업통상자원부 장관은", "김정관"),
    ("이창용 한국은행 총재", "이창용"),
    ("박민수 보건복지부 차관", "박민수"),
    ("홍길동 기획재정부 장관", "홍길동"),
    ("김민수 삼성전자 대표", "김민수"),
    ("박지민 미래기술연구원 원장", "박지민"),
    ("남궁민 미래기술연구원 원장", "남궁민"),
])
def test_whole_person_name_is_distinct_from_institution_and_role(phrase, name):
    evidence = event_evidence({"title": phrase})
    assert evidence.people == {name}


@pytest.mark.parametrize("phrase", [
    "산업부 장관", "산업통상부 장관", "산업통상자원부 장관", "한국은행 총재",
    "대한상공회의소 회장", "연방준비제도 의장", "미래기술연구원 원장",
    "대한상의 회장", "새로운 대표", "정부 장관", "경제 장관", "김정관", "김정관 알수없는말 장관",
])
def test_institution_suffix_or_ambiguous_text_is_never_invented_as_a_person(phrase):
    assert event_evidence({"title": phrase}).people == set()


def test_saved_kookje_has_one_person_and_no_institution_people():
    evidence = event_evidence(KOOKJE["article"])
    assert evidence.people == {"김정관"}
    assert {"산업부", "산업통상부"} <= evidence.organizations
    assert "산업부" not in evidence.names and "업통상부" not in evidence.names
    assert _new_items(evidence, event_evidence(CENTER), title=False) == []


def test_even_two_new_names_are_not_description_addition_units():
    a = event_evidence(CENTER)
    b = replace(event_evidence(HERALD["article"]), names=frozenset({"홍길동", "김민수"}))
    assert _new_items(b, a, title=False) == []
    assert compare(a, b, HERALD["cosine"]) == Judgment(Relation.INSUFFICIENT, "semantic_difference")


def test_new_person_without_a_complete_supplied_proposition_is_withheld():
    candidate = control(CENTER, "fragment", description=f"{CENTER['description']} 박민수 보건복지부 차관은 최근")
    judgment = compare(event_evidence(CENTER), event_evidence(candidate), 0.90)
    assert judgment == Judgment(Relation.INSUFFICIENT, "semantic_difference")


def test_new_country_and_its_explicit_decision_are_not_removed_as_person_names():
    center = control(CENTER, "plain", title="관세 15% 유지 확인",
                     description="정부가 미국 측으로부터 관세율을 기존 합의선인 15% 이내로 유지하겠다는 입장을 확인했다.")
    candidate = control(center, "country", title="중국 관세 15% 유지…수출 지원금 신설",
                        description="중국은 관세 15%를 유지하면서 수출 지원금을 신설하기로 결정했다. 정부가 자금 지원을 발표했다.")
    assert compare(event_evidence(center), event_evidence(candidate), 0.90).relation is Relation.NEW_INFO


@pytest.mark.parametrize("cosine", [0.90, 0.97])
@pytest.mark.parametrize("short", [False, True])
def test_new_person_and_institution_mentions_only_do_not_add_information(cosine, short):
    description = ("관세 15% 유지 확인" if short else
                   "정부가 미국 측으로부터 관세율을 기존 합의선인 15% 이내로 유지하겠다는 입장을 확인했다.")
    center = control(CENTER, "plain", title="관세 15% 유지 확인", description=description)
    candidate = control(center, "named", title="김정관 장관·이창용 총재, 관세 15% 유지 확인",
                        description=f"김정관 산업부 장관과 이창용 한국은행 총재. {description}")
    judgment = compare(event_evidence(center), event_evidence(candidate), cosine)
    assert judgment.relation is not Relation.NEW_INFO
    assert judgment.relation in {Relation.REPEAT, Relation.INSUFFICIENT}
    if not short:
        assert judgment.relation is Relation.REPEAT or judgment.withheld


def test_unattributed_name_list_is_not_a_fact_or_confirmed_people():
    center = control(CENTER, "plain", title="관세 15% 유지 확인",
                     description="정부가 미국 측으로부터 관세율을 기존 합의선인 15% 이내로 유지하겠다는 입장을 확인했다.")
    candidate = control(center, "bare", title="김정관·이창용, 관세 15% 유지 확인",
                        description=f"김정관·이창용. {center['description']}")
    assert event_evidence(candidate).people == set()
    judgment = compare(event_evidence(center), event_evidence(candidate), 0.97)
    assert judgment.relation is Relation.REPEAT or judgment.withheld


def test_institution_abbreviations_and_titles_do_not_duplicate_one_person():
    candidate = control(CENTER, "aliases", title="김정관 산업부 장관, 관세 15% 유지 확인",
                        description="김정관 장관과 김정관 산업통상부 장관은 같은 인물이다. "
                                    "김정관 산업통상자원부 장관이라는 호칭도 사용했다.")
    assert event_evidence(candidate).people == {"김정관"}


@pytest.mark.parametrize("addition", [
    "박민수 보건복지부 차관은 관세 협상 실무단을 구성했다고 발표했다.",
    "김민수 삼성전자 대표는 관세 대응을 위해 해외 공장 신설을 결정했다.",
    "홍길동 기획재정부 장관은 관세 피해기업의 지원금 지급을 완료했다고 밝혔다.",
])
def test_separately_stated_actor_action_decision_and_result_are_kept(addition):
    center = control(CENTER, "plain", title="관세 15% 유지 확인",
                     description="정부가 미국 측으로부터 관세율을 기존 합의선인 15% 이내로 유지하겠다는 입장을 확인했다.")
    candidate = control(center, "added", description=f"{center['description']} {addition}")
    judgment = compare(event_evidence(center), event_evidence(candidate), 0.90)
    assert judgment == Judgment(Relation.NEW_INFO, "description_addition")
    assert not judgment.withheld


def test_saved_newsis_us_statement_is_preserved():
    judgment = compare(event_evidence(CENTER), event_evidence(NEWSIS["article"]), NEWSIS["cosine"])
    assert judgment == Judgment(Relation.NEW_INFO, "title_addition")
    assert "무역법 제301조" in NEWSIS["article"]["description"]
    assert "각국과의 무역 협정" in NEWSIS["article"]["description"]
    assert not judgment.withheld


async def test_saved_unconfirmed_reports_are_not_neighbours(saved_vectors):
    selector = NewsMapSelector(CENTER)
    await selector.add([deepcopy(row["article"]) for row in FIXTURE["candidates"]])
    groups = selector.select(3)
    assert [g.article["news_id"] for g in groups] == [NEWSIS["article"]["news_id"]]
    assert selector.center_repeats == selector.neighbour_repeats == 0
    assert selector.withheld == 3



async def test_same_input_with_confirmed_repetition_is_excluded(saved_vectors):
    center = control(CENTER, "plain", title="관세 15% 유지 확인",
                     description="정부가 미국 측으로부터 관세율을 기존 합의선인 15% 이내로 유지하겠다는 입장을 확인했다.")
    candidate = control(center, "named", title="김정관 산업부 장관, 관세 15% 유지 확인",
                        description=f"김정관 산업통상부 장관은 설명했다. {center['description']}")
    selector = NewsMapSelector(center)
    await selector.add([candidate])
    assert selector.select(3) == [] and selector.withheld == 0
    assert selector.center_repeats == 1


@pytest.mark.parametrize("row", [HERALD, KOOKJE])
async def test_new_withheld_reasons_apply_against_neighbour_representatives(saved_vectors, row):
    center = control(CENTER, "later", published_at="2026-10-04T09:00:00Z")
    selector = NewsMapSelector(center)
    await selector.add([CENTER])
    await selector.add([row["article"]])
    groups = selector.select(2)
    assert [g.article["news_id"] for g in groups] == [CENTER["news_id"]]
    assert selector.center_repeats == 0
    assert selector.withheld == 1


@pytest.mark.parametrize("row", [HERALD, KOOKJE])
@pytest.mark.parametrize("add_valid", [False, True])
async def test_withheld_reports_expand_and_budget_shortage_is_normal(monkeypatch, saved_vectors, row, add_valid):
    calls = setup_expansion(monkeypatch, deepcopy(CENTER), deepcopy(row["article"]),
                            [deepcopy(NEWSIS["article"])] if add_valid else [])
    result = await news_map.build_news_map(CENTER, target=2)
    assert calls and result.stats["withheld"] == 1
    assert result.stats["search_steps"] == 1
    assert (result.status, result.reason) == ("insufficient", "search_limit")
    assert len(result.items) == int(add_valid)
    assert result.stats["center_repeats"] == result.stats["neighbour_repeats"] == 0


@pytest.mark.parametrize("tier,requested", [("FREE", 3), ("BASIC", 3), ("PAID", 4)])
def test_related_and_graph_share_saved_pair_policy_expansion_and_shortage(monkeypatch, saved_vectors, tier, requested):
    calls = setup_expansion(monkeypatch, deepcopy(CENTER), deepcopy(HERALD["article"]),
                            [deepcopy(NEWSIS["article"])])
    store.news_cache.update({row["article"]["news_id"]: deepcopy(row["article"]) for row in (KOOKJE, WOW)})
    monkeypatch.setattr(settings, "news_map_initial_candidates", 20)
    client = TestClient(app)

    def request(endpoint, expand):
        response = client.get(f"/api/v1/news/{CENTER['news_id']}/{endpoint}",
                              params={"tier": tier, "limit": 4, "expand": expand})
        assert response.status_code == 200
        return response.json()

    initial_related, initial_graph = request("related", False), request("graph", False)
    assert initial_related["selection"] == initial_graph["selection"] == {
        "status": "expandable", "reason": None, "requested": requested, "returned": 0, "excluded": 0}
    assert initial_related["related_news"] == [] and len(initial_graph["nodes"]) == 1
    assert calls == []
    related, graph = request("related", True), request("graph", True)
    assert len(calls) == 1
    assert related["selection"] == graph["selection"] == {
        "status": "insufficient", "reason": "search_limit", "requested": requested, "returned": 1, "excluded": 0}
    assert [a["news_id"] for a in related["related_news"]] == [NEWSIS["article"]["news_id"]]
    assert [a["news_id"] for a in graph["nodes"][1:]] == [NEWSIS["article"]["news_id"]]
    assert "center_same_story" not in related and "same_story" not in graph["center_node"]
    assert "same_story" not in related["related_news"][0] and "same_story" not in graph["nodes"][1]
    assert (related["related_news"][0]["relevance_score"] is None) == (tier != "PAID")
    assert related["related_news"][0]["relevance_score"] == graph["nodes"][1]["relevance_score"]


def test_semantic_difference_only_withholds_insufficient_judgments():
    assert Judgment(Relation.INSUFFICIENT, "semantic_difference").withheld
    assert not Judgment(Relation.NEW_INFO, "semantic_difference").withheld
    assert not Judgment(Relation.DIFFERENT, "semantic_difference").withheld
    assert not Judgment(Relation.INSUFFICIENT, "time_unknown").withheld
