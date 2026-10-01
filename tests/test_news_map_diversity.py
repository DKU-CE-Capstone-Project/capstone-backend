"""Authored Korean examples, not scraped news or a semantic-quality benchmark.

All examples are fictional. Expected judgments follow the literal product,
date, action, amount and angle differences; vectors are controlled mocks.
"""
import math
from copy import deepcopy
from itertools import permutations

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import article_embeddings, llm
from app.agents.related_selector import select_related
from app.agents.repeated_coverage import event_evidence
from app.config import settings
from app.main import app


def article(nid, title, description="", **extra):
    return {"news_id": nid, "title": title, "description": description,
            "url": f"https://fictional.test/{nid}", "source": f"fictional-{nid}",
            "published_at": "2026-09-30T09:00:00Z", "_search_keyword": "삼성전자", **extra}


def release(nid, model="S26", date="9월 30일", **extra):
    return article(nid, f"삼성전자 갤럭시 {model} 신제품 공개",
                   extra.pop("description", f"삼성전자가 {date} 갤럭시 {model} 신제품을 공개했다. 새 스마트폰 발표 행사를 열어 제품을 선보였다."),
                   **extra)


def vec(cosine=1.0, side=1):
    return [cosine, side * math.sqrt(1 - cosine ** 2)] + [0.] * 766


def ids(items):
    return [item.article["news_id"] for item in items]


@pytest.fixture
def identical_vectors(monkeypatch):
    calls = []

    async def embed(text, **kwargs):
        calls.append(text)
        return vec()

    monkeypatch.setattr(llm, "embed", embed)
    return calls


async def test_center_reprints_do_not_occupy_slots_but_new_angles_survive(identical_vectors):
    center = release("center")
    repeats = [release("copy-a"), release("copy-b")]
    repeats[0]["title"] = "갤럭시 S26 선보인 삼성전자, 신제품 발표"
    repeats[1]["description"] = "9월 30일 삼성전자는 새 스마트폰 갤럭시 S26을 선보였다. 신제품 공개 행사를 열고 제품을 발표했다."
    novel = [
        article("comparison", "갤럭시 S26과 S25 성능 비교", "신제품 공개 이후 갤럭시 S26과 S25의 사양과 성능 차이를 비교한다."),
        article("reaction", "갤럭시 S26 소비자 반응", "9월 30일 신제품 공개 이후 갤럭시 S26에 대한 소비자 반응을 조사했다."),
        article("impact", "갤럭시 S26 실적 영향 전망", "삼성전자 신제품 갤럭시 S26 발표가 매출과 주가에 미칠 영향을 분석했다."),
    ]
    result = await select_related(center, [*repeats, *novel], limit=3)
    assert set(ids(result)) == {"comparison", "reaction", "impact"}
    assert not set(ids(result)) & {"copy-a", "copy-b"}


async def test_candidate_reprints_have_one_deterministic_representative(identical_vectors):
    center = release("center")
    a = release("a", "S27")
    b = release("b", "S27")
    b["title"] = "삼성전자 갤럭시 S27 선보여…신제품 공개"
    b["description"] = a["description"].replace("공개했다", "선보였다")
    c = article("c", "삼성전자 HBM4 양산", "삼성전자가 HBM4 생산을 확대했다. 반도체 공장에서 양산에 나섰다.")
    results = [ids(await select_related(center, list(order), limit=6)) for order in permutations([a, b, c])]
    assert all(result == results[0] for result in results)
    assert len(set(results[0]) & {"a", "b"}) == 1
    assert "c" in results[0]


@pytest.mark.parametrize("candidate", [
    release("model", "S27"),
    release("date", date="10월 1일"),
    release("year", date="2025년 9월 30일"),
    release("availability", description=""),
])
async def test_model_time_or_missing_description_cannot_be_inferred(identical_vectors, candidate):
    center = release("center", date="2026년 9월 30일")
    # An absent description cannot assert the center's date or information.
    if candidate["news_id"] == "availability":
        candidate["title"] = "삼성전자 갤럭시 S26 판매 개시"
    assert ids(await select_related(center, [candidate], limit=3)) == [candidate["news_id"]]


async def test_changed_measurement_and_new_information_keep_article(identical_vectors):
    center = release("center")
    center["description"] += " 출고가는 100만원이다."
    revised = release("price-change")
    revised["description"] += " 출고가는 120만원이다."
    extra = release("new-fact")
    extra["description"] += " 배터리 성능과 가격 정보를 처음 설명했다."
    assert set(ids(await select_related(center, [revised, extra], limit=3))) == {"price-change", "new-fact"}


@pytest.mark.parametrize("a,b", [("2.5%", "25%"), ("0.25%", "0.025%"), ("128 GB", "256 GB")])
async def test_decimal_measurements_are_not_dates_or_equal_numbers(identical_vectors, a, b):
    center = release("center")
    center["description"] += f" 변화 수치는 {a}다."
    candidate = release("changed")
    candidate["description"] += f" 변화 수치는 {b}다."
    assert ids(await select_related(center, [candidate], limit=3)) == ["changed"]
    assert event_evidence(center).dates == {(None, 9, 30)}
    assert event_evidence(center).amounts != event_evidence(candidate).amounts


@pytest.mark.parametrize("a,b", [("S26", "S26 Ultra"), ("HBM3E", "HBM4"), ("GPT-5.1", "GPT-51"), ("S26", "S26+")])
async def test_model_variants_generations_and_decimal_versions_remain_distinct(identical_vectors, a, b):
    assert ids(await select_related(release("center", a), [release("changed", b)], limit=3)) == ["changed"]


async def test_person_and_policy_event_reprints_use_literal_subject_evidence(identical_vectors):
    examples = [
        ("이재용 회장 UAE 원전 수주 계약 발표", "UAE 원전 수주 계약 체결, 이재용 회장",
         "이재용 회장은 9월 30일 UAE 원전 수주 계약 체결을 발표했다. 원전 계약을 체결했다고 밝혔다."),
        ("한국은행 기준금리 인하 결정", "기준금리 인하 결정한 한국은행",
         "한국은행은 9월 30일 기준금리 인하를 결정했다. 통화정책 회의에서 기준금리를 내리기로 결정했다."),
    ]
    for title, paraphrase, description in examples:
        center = article("center", title, description)
        assert await select_related(center, [article("copy", paraphrase, description)], limit=3) == []


async def test_short_description_literal_new_content_is_not_erased(identical_vectors):
    center = release("center", description="")
    candidate = release("new", description="서비스 센터 수리 절차와 보상 정책을 설명했다.")
    assert ids(await select_related(center, [candidate], limit=3)) == ["new"]


async def test_partial_date_does_not_invent_a_shared_year(identical_vectors):
    center = release("center", date="2026년 9월 30일")
    candidate = release("old", date="9월 30일", published_at="2025-09-30T09:00:00Z")
    assert ids(await select_related(center, [candidate], limit=3)) == ["old"]
    candidate["published_at"] = ""
    assert ids(await select_related(center, [candidate], limit=3)) == ["old"]


async def test_high_cosine_or_same_company_alone_never_means_same_event(identical_vectors):
    center = release("center")
    candidates = [article("factory", "삼성전자 공장 증설 투자", "삼성전자 공장 증설 계획과 투자 규모를 발표했다."),
                  article("earnings", "삼성전자 분기 실적 발표", "삼성전자 영업이익과 매출을 공개했다."),
                  article("unknown", "삼성전자 새로운 소식", "")]
    assert set(ids(await select_related(center, candidates, limit=6))) == {"factory", "earnings", "unknown"}


async def test_short_text_requires_known_time_and_strong_title_evidence(identical_vectors):
    center = release("center", description="")
    copy = release("copy", description="신제품 공개")
    assert await select_related(center, [copy], limit=3) == []
    copy["published_at"] = ""
    assert ids(await select_related(center, [copy], limit=3)) == ["copy"]
    copy["published_at"] = "2026-10-05T09:00:00Z"
    assert ids(await select_related(center, [copy], limit=3)) == ["copy"]
    copy["published_at"] = center["published_at"]
    copy["title"] = "갤럭시 S26 새 이야기"
    assert ids(await select_related(center, [copy], limit=3)) == ["copy"]


async def test_complete_link_prevents_chain_merging(monkeypatch):
    center = release("center")
    candidates = [release(nid, "S27") for nid in ("a", "b", "c")]
    for a in candidates:
        a["title"] += f" {a['news_id']}언론"
    vectors = {center["title"]: vec(), candidates[0]["title"]: vec(0.94),
               candidates[1]["title"]: vec(), candidates[2]["title"]: vec(0.94, -1)}

    async def embed(text, **kwargs):
        return vectors[text.splitlines()[0].removeprefix("Title: ")]

    monkeypatch.setattr(llm, "embed", embed)
    result = await select_related(center, candidates, limit=6)
    assert len(result) == 2  # A~B, B~C, A!~C must not become one connected component.
    assert ids(result)[0] == "b"


async def test_mmr_order_can_differ_from_relevance_and_cannot_include_unrelated(monkeypatch):
    articles = [article("center", "중심 기사"), article("a", "가장 관련된 기사"),
                article("b", "첫 기사와 겹치는 기사"), article("c", "다른 정보를 담은 기사"),
                article("unrelated", "관련 없는 기사")]
    vectors = dict(zip([a["title"] for a in articles], [vec(), vec(.97), vec(.96), vec(.88, -1), vec(0)]))
    calls = []

    async def embed(text, **kwargs):
        calls.append(text)
        return vectors[text.splitlines()[0].removeprefix("Title: ")]

    monkeypatch.setattr(llm, "embed", embed)
    result = await select_related(articles[0], articles[1:], limit=3)
    assert ids(result) == ["a", "c", "b"]
    assert result[1].score < result[2].score  # Neither MMR nor duplicate score replaces relevance.
    assert len(calls) == 5
    assert ids(await select_related(articles[0], list(reversed(articles[1:])), limit=3)) == ids(result)
    assert len(calls) == 5  # Pair comparisons reuse article vectors, zero pair API calls.
    monkeypatch.setattr(settings, "news_map_mmr_lambda", 0)
    assert "unrelated" not in ids(await select_related(articles[0], articles[1:], limit=20))


async def test_stored_vectors_reused_for_center_pairs_and_mmr(monkeypatch):
    center = release("center")
    candidates = [release("a", "S27"), release("b", "S27"), article("c", "삼성전자 HBM4 양산")]
    for a in [center, *candidates]:
        _, metadata = article_embeddings._input(a, "news_map")
        a["news_map_embedding"] = {"metadata": metadata, "values": vec()}

    async def forbidden(*args, **kwargs):
        pytest.fail("Compatible stored vectors must avoid provider calls")

    monkeypatch.setattr(llm, "embed", forbidden)
    result = await select_related(deepcopy(center), deepcopy(candidates), limit=6)
    assert len(result) == 2


@pytest.mark.parametrize("tier,limit", [("FREE", 6), ("BASIC", 2), ("PAID", 5)])
def test_apis_match_repeated_cleanup_order_policy_and_metadata(monkeypatch, identical_vectors, tier, limit):
    center = release("center")
    candidates = [release("copy"), release("a", "S27"), release("b", "S27"),
                  article("reaction", "삼성전자 갤럭시 S26 소비자 반응", "신제품을 평가한 소비자 반응이다.",
                          keywords=["갤럭시 S26"], categories=["반도체"])]
    store.news_cache.update({a["news_id"]: a for a in [center, *candidates]})
    client = TestClient(app)
    params = {"tier": tier, "limit": limit}
    related_response = client.get("/api/v1/news/center/related", params=params)
    graph_response = client.get("/api/v1/news/center/graph", params=params)
    assert related_response.status_code == graph_response.status_code == 200
    related = related_response.json()["related_news"]
    graph = graph_response.json()
    assert len(related) == 2
    assert [a["news_id"] for a in related] == [a["news_id"] for a in graph["nodes"][1:]]
    assert all(a["news_id"] != "copy" and (a["relevance_score"] is not None) == (tier == "PAID") for a in related)
    assert graph["center_node"]["distance"] == 0
    for card, node, edge in zip(related, graph["nodes"][1:], graph["edges"]):
        assert card == {k: v for k, v in node.items() if k != "is_center"}
        assert card["distance"] == edge["distance"] == 1
    assert len(identical_vectors) == 3  # Exact input copies share/exclude vectors before pair comparisons.
