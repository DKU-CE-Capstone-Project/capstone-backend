"""Authored Korean examples, not scraped news or a semantic-quality benchmark.

All examples are fictional. Expected judgments follow the literal product, target,
direction, date, figure and angle differences; vectors are controlled mocks.
"""
import math
from copy import deepcopy
from itertools import permutations

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import article_embeddings, llm
from app.agents.related_selector import NewsMapSelector, select_related
from app.agents.repeated_coverage import Judgment, Relation, compare, event_evidence
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


def oil(nid, title, description, published="2026-10-01T21:00:00Z"):
    return article(nid, title, description, published_at=published, _search_keyword="국제유가")


OIL_CENTER = ("국제유가, 미국 원유 재고 감소에 상승…브렌트유 1.2%↑",
              ("국제유가가 1일(현지시간) 미국 원유 재고가 줄었다는 소식에 상승했다. 브렌트유 선물은 "
               "전장 대비 1.2% 오른 배럴당 90.5달러에 거래를 마쳤다."))


def vec(cosine=1.0, side=1):
    return [cosine, side * math.sqrt(1 - cosine ** 2)] + [0.] * 766


def ids(items):
    return [item.article["news_id"] for item in items]


def judge(a, b, cosine=0.97):
    return compare(event_evidence(a), event_evidence(b), cosine)


@pytest.fixture
def identical_vectors(monkeypatch):
    calls = []

    async def embed(text, **kwargs):
        calls.append(text)
        return vec()

    monkeypatch.setattr(llm, "embed", embed)
    return calls


# ── same event / repeated information judgment ──────────────────────────────

def test_same_title_with_paraphrased_description_is_a_repeat():
    center = oil("center", *OIL_CENTER)
    same_title = oil("copy", OIL_CENTER[0], "미국 원유 재고가 감소했다는 소식에 국제유가가 1일(현지시간) 상승했다. "
                                           "이날 브렌트유는 전장보다 1.2% 오른 배럴당 90.5달러였다.")
    assert judge(center, same_title).relation is Relation.REPEAT


def test_same_title_is_not_enough_when_the_description_adds_analysis():
    center = oil("center", *OIL_CENTER)
    analysis = oil("analysis", OIL_CENTER[0], "증권가는 산유국 감산 연장 가능성까지 겹쳐 연말 물가 부담이 커질 것으로 "
                                              "전망했다. 항공·화학 업종의 원가 타격 우려도 나온다.")
    assert judge(center, analysis).relation is Relation.NEW_INFO


def test_reworded_report_of_the_same_event_is_a_repeat():
    center = oil("center", *OIL_CENTER)
    reworded = oil("reworded", "미국 원유 재고 줄자 국제유가 올라…브렌트유 1.2% 상승",
                   "1일(현지시간) 국제유가는 미국의 원유 재고 감소 영향으로 올랐다. 브렌트유 선물 가격은 "
                   "전장보다 1.2% 상승한 배럴당 90.5달러로 마감했다.")
    assert judge(center, reworded).relation is Relation.REPEAT


def test_reordered_headline_fully_covered_by_the_center_is_a_repeat():
    center = oil("center", "산유국 원유 공급 회복에도 국제유가 1% 상승…미국·이란 협상 교착",
                 "국제유가가 미국과 이란의 협상 교착과 미국의 연료 재고 감소에 1%가량 올랐다. 산유국의 원유 수출이 "
                 "회복되고 있지만 석유제품 공급이 빠듯한 점이 유가를 지지했다.")
    subset = oil("subset", "미국·이란 협상 교착에 석유재고 감소까지…국제유가 상승",
                 "미국과 이란의 협상이 진전되지 못한 상황에서 미국 석유제품 재고까지 줄자 국제유가가 1일(현지시간) "
                 "소폭 올랐다. 브렌트유는 전장보다 0.92% 오른 배럴당 90달러였다.")
    assert judge(center, subset, 0.95).relation is Relation.REPEAT
    # In reverse, the wider headline adds the supply-recovery subject to the narrower report.
    assert judge(subset, center, 0.95).relation is Relation.NEW_INFO


def test_rounded_figure_and_one_sided_background_date_do_not_create_new_information():
    center = article("center", "미국 10년물 국채금리 5.1% 돌파…19년 만에 최고",
                     "미국 10년 만기 국채 금리가 30일(현지시간) 5.1%를 돌파하며 19년 만에 최고 수준으로 올라섰다. "
                     "강한 성장세에 고금리 장기화 우려가 커졌다.", _search_keyword="금리")
    background = article("background", "미국 10년물 국채금리 5.1% 돌파…19년 만에 최고치 경신",
                         "미국 10년 만기 국채 금리가 30일(현지시간) 5.1%를 넘어섰다. 장중 한때 5.104%까지 올라 "
                         "2007년 기록한 장중 고점 5.103%를 웃돌았다.", _search_keyword="금리")
    judgment = judge(center, background)
    assert judgment.relation is Relation.REPEAT, judgment


RESEARCH = ("한국은행 \"환율 급등 경험 많을수록 소비 영향 작아\"",
            ("한국은행 연구에 따르면 환율 급등을 여러 번 겪은 가계일수록 환율이 올라도 소비를 덜 줄였다. "
             "외환위기 이후 반복된 충격에 가계가 적응했다는 분석이다."))


def test_editorial_headline_wording_is_not_new_information_but_reported_facts_are():
    center = article("center", *RESEARCH, _search_keyword="환율")
    rhetoric = article("rhetoric", "환율 올라도 지갑 안 닫았다…가계에 생긴 내성",
                       "환율 급등을 반복해 겪은 가계는 환율이 올라도 소비를 크게 줄이지 않았다는 한은 연구 결과가 "
                       "나왔다. 외환위기 이후 충격에 적응한 결과다.", _search_keyword="환율")
    assert judge(center, rhetoric, 0.96).relation is Relation.REPEAT
    # Unsupported wording with merely high similarity confirms neither a repeat nor an addition.
    judgment = judge(center, rhetoric, 0.93)
    assert judgment == Judgment(Relation.INSUFFICIENT, "low_overlap")
    assert judgment.withheld
    generations = article("generations", "환율 급등기 소비…MZ세대는 줄이고 X세대는 그대로",
                          "한국은행 연구에서 외환위기를 겪은 X세대는 환율이 올라도 소비를 유지했지만 MZ세대는 "
                          "소비를 줄였다. 자산이 적을수록 차이가 컸다.", _search_keyword="환율")
    assert judge(center, generations, 0.96).relation is Relation.NEW_INFO


def test_newsroom_abbreviation_and_rounded_large_amounts_are_the_same_facts():
    assert "한국은행" in event_evidence(article("a", "한은 \"금리 동결\"", "한은이 금리를 동결했다.")).compact
    center = article("center", "9월 수출 1209억달러 역대 최대…반도체 600억달러 돌파",
                     "9월 수출이 1209억달러로 역대 최대를 기록했다. 반도체 수출은 603억달러였다.",
                     _search_keyword="수출")
    rounded = article("rounded", "9월 수출 1200억달러 돌파…반도체 역대 최대",
                      "지난달 수출이 처음으로 1200억달러를 넘었다. 반도체가 월 수출 600억달러를 넘어섰다.",
                      _search_keyword="수출")
    judgment = judge(center, rounded)
    assert judgment.relation is Relation.REPEAT, judgment
    other_month = article("other", "8월 수출 1150억달러…반도체 560억달러",
                          "8월 수출이 1150억달러를 기록했다.", _search_keyword="수출",
                          published_at="2026-09-01T09:00:00Z")
    assert judge(center, other_month).relation is Relation.DIFFERENT


def test_missing_figure_is_not_a_conflict_but_a_changed_labelled_figure_is():
    center = release("center")
    center["description"] += " 출고가는 100만원이다."
    missing = release("missing")
    changed = release("changed")
    changed["description"] += " 출고가는 120만원이다."
    assert judge(center, missing).relation is Relation.REPEAT
    assert judge(center, changed).relation is Relation.DIFFERENT
    assert judge(center, changed).reason == "figure"


@pytest.mark.parametrize("a,b", [("2.5%", "25%"), ("0.25%", "0.025%"), ("128 GB", "256 GB")])
def test_decimal_measurements_are_not_dates_or_equal_numbers(a, b):
    center = release("center")
    center["description"] += f" 변화 수치는 {a}다."
    candidate = release("changed")
    candidate["description"] += f" 변화 수치는 {b}다."
    assert judge(center, candidate).relation is Relation.DIFFERENT
    assert event_evidence(center).lead_date == (None, 9, 30)
    assert {f.value for f in event_evidence(center).figures} != {f.value for f in event_evidence(candidate).figures}


def test_same_company_different_events_and_high_cosine_alone_never_merge():
    center = release("center")
    others = [article("factory", "삼성전자 공장 증설 투자", "삼성전자 공장 증설 계획과 투자 규모를 발표했다."),
              article("earnings", "삼성전자 분기 실적 발표", "삼성전자 영업이익과 매출을 공개했다."),
              article("price", "삼성전자, 갤럭시 S26 가격 인상", "삼성전자가 갤럭시 S26 국내 판매 가격을 올렸다."),
              article("unknown", "삼성전자 새로운 소식", "")]
    for other in others:
        assert judge(center, other, 1.0).relation is not Relation.REPEAT, other["news_id"]


def test_follow_up_direction_change_and_decision_are_kept():
    center = oil("center", *OIL_CENTER)
    next_day = oil("next-day", "미국 원유 재고 증가에 국제유가 하락",
                   "국제유가가 2일(현지시간) 미국 원유 재고가 늘었다는 소식에 떨어졌다. 브렌트유는 1.5% 내렸다.",
                   published="2026-10-02T21:00:00Z")
    assert judge(center, next_day).relation is Relation.DIFFERENT
    hold = article("hold", "한국은행 기준금리 동결", "한국은행은 1일 통화정책회의에서 기준금리를 연 3.0%로 동결했다.")
    hike = article("hike", "한국은행 기준금리 인상", "한국은행은 1일 통화정책회의에서 기준금리를 연 3.25%로 올렸다.")
    assert judge(hold, hike).relation is Relation.DIFFERENT
    deadlock = article("deadlock", "가나노조 임금 협상 결렬", "가나노조와 회사의 임금 협상이 1일 결렬됐다. 노조는 파업을 예고했다.",
                       published_at="2026-10-01T09:00:00Z")
    deal = article("deal", "가나노조 임금 협상 타결", "가나노조와 회사가 3일 임금 협상을 타결했다. 파업은 철회됐다.",
                   published_at="2026-10-03T09:00:00Z")
    assert judge(deadlock, deal).relation is Relation.DIFFERENT
    deal["published_at"] = deadlock["published_at"]
    deal["description"] = deal["description"].replace("3일", "1일")
    assert judge(deadlock, deal) == Judgment(Relation.DIFFERENT, "stage")  # 결렬 → 타결 is a new decision.


def test_bond_tenors_and_oil_inventory_versus_supply_are_distinct_targets():
    ten = article("ten", "미국 10년물 국채금리 5.1% 돌파", "미국 10년 만기 국채 금리가 30일 5.1%를 넘었다.")
    thirty = article("thirty", "미국 30년물 국채금리 5.4% 돌파", "미국 30년 만기 국채 금리가 30일 5.4%를 넘었다.")
    assert judge(ten, thirty).relation is Relation.DIFFERENT
    assert "bond:us:10y" in event_evidence(ten).title_targets
    center = oil("center", *OIL_CENTER)
    supply = oil("supply", "산유국 원유 공급 회복에도 국제유가 1% 상승",
                 "산유국의 원유 수출이 회복되고 있지만 미국 원유 재고 감소로 국제유가가 1일(현지시간) 1%가량 올랐다.")
    assert judge(center, supply).relation is Relation.NEW_INFO


def test_word_first_alone_is_not_follow_up_evidence():
    center = release("center")
    first = release("first")
    first["description"] += " 이번 행사에서 신제품을 처음 선보였다."
    assert judge(center, first).relation is Relation.REPEAT


def test_lexicon_miss_falls_back_to_overlap_instead_of_never_repeating():
    # No registered target, product, stage or company: wording and figures decide.
    center = article("center", "가나시 시내버스 요금 2천원으로 조정", "가나시는 시내버스 기본요금을 내년부터 2천원으로 "
                                                         "조정하기로 했다. 시의회 심의를 거친 요금안이다.")
    copy = article("copy", "가나시, 시내버스 요금 2천원으로 조정", "가나시가 시내버스 기본요금을 내년부터 2천원으로 "
                                                         "조정한다. 시의회가 요금안을 심의해 의결했다.")
    other = article("other", "가나시 시내버스 노선 개편", "가나시는 시내버스 노선 12개를 내년부터 개편하기로 했다.")
    assert judge(center, copy).relation is Relation.REPEAT
    assert judge(center, other).relation is not Relation.REPEAT


@pytest.mark.parametrize("candidate", [
    release("model", "S27"),
    release("date", date="10월 1일"),
    release("year", date="2025년 9월 30일"),
])
def test_model_or_explicit_time_conflicts_are_different_events(candidate):
    center = release("center", date="2026년 9월 30일")
    assert judge(center, candidate, 1.0).relation is Relation.DIFFERENT


@pytest.mark.parametrize("a,b", [("S26", "S26 Ultra"), ("HBM3E", "HBM4"), ("GPT-5.1", "GPT-51"), ("S26", "S26+")])
def test_model_variants_generations_and_decimal_versions_remain_distinct(a, b):
    assert judge(release("center", a), release("changed", b), 1.0).relation is Relation.DIFFERENT


def test_model_name_for_a_product_line_already_named_is_a_detail_not_a_new_subject():
    center = article("center", "삼성 갤럭시탭·이어폰 신제품 공개",
                     "삼성전자가 1일 갤럭시탭과 클립형 이어폰 신제품을 공개했다. 태블릿에는 포토샵을 기본 탑재했다.")
    detail = article("detail", "삼성, 갤럭시 탭 S12 공개…클립형 이어폰도 선보여",
                     "삼성전자가 1일 갤럭시 탭 S12와 클립형 이어폰 신제품을 공개했다. 태블릿에 포토샵을 기본 탑재했다.")
    assert judge(center, detail).relation is Relation.REPEAT
    named = article("named", "삼성 갤럭시 탭 S11 신제품 공개", center["description"].replace("갤럭시탭", "갤럭시 탭 S11"))
    assert judge(named, detail).relation is Relation.DIFFERENT  # Two different named models still conflict.


def test_short_or_missing_descriptions_are_insufficient_not_repeats():
    center = release("center", description="")
    same = release("same", description="신제품 공개")
    assert judge(center, same, 1.0) .relation is Relation.INSUFFICIENT
    assert judge(center, same, 1.0).withheld  # Visibly nothing new, but unconfirmed.
    new = release("new", description="서비스 센터 수리 절차와 보상 정책을 설명했다.")
    assert judge(center, new, 1.0).relation is Relation.NEW_INFO
    unknown_time = release("unknown", description="신제품 공개", published_at="")
    assert not judge(center, unknown_time, 1.0).withheld
    reworded = release("reworded", description="신제품 공개")
    reworded["title"] = "갤럭시 S26 새 이야기"
    assert not judge(center, reworded, 1.0).withheld


def test_verbatim_short_reprint_is_a_repeat_but_unknown_time_is_not():
    center = release("center", description="신제품 공개")
    verbatim = release("verbatim", description="신제품 공개", url="https://other.test/verbatim")
    assert judge(center, verbatim, 1.0) == Judgment(Relation.REPEAT, "identical_text")
    verbatim["published_at"] = ""
    assert judge(center, verbatim, 1.0).relation is not Relation.REPEAT


def test_partial_date_does_not_invent_a_shared_year():
    center = release("center", date="2026년 9월 30일")
    old = release("old", date="9월 30일", published_at="2025-09-30T09:00:00Z")
    assert judge(center, old, 1.0).relation is Relation.DIFFERENT
    old["published_at"] = ""
    assert judge(center, old, 1.0).relation is not Relation.REPEAT


def test_dairy_raw_milk_is_not_read_as_crude_oil():
    milk = article("milk", "흰우유용 원유 2.6만톤 감축 합의", "낙농가와 유업계가 원유 생산 물량 가운데 음용유용 2만6000톤을 줄이기로 했다.")
    crude = article("crude", "중동 원유 수출 80% 회복", "중동 산유국의 원유 수출량이 전쟁 전의 80% 수준까지 회복됐다.")
    assert not {t for t in event_evidence(milk).targets if t.startswith("oil")}
    assert "oil_supply" in event_evidence(crude).targets
    assert judge(milk, crude).relation is Relation.DIFFERENT


# ── grouping, withholding and MMR ───────────────────────────────────────────

async def test_center_reprints_are_excluded_without_occupying_slots_and_new_angles_survive(identical_vectors):
    center = release("center")
    repeats = [release("copy-a"), release("copy-b")]
    repeats[0]["title"] = "갤럭시 S26 선보인 삼성전자, 신제품 발표"
    repeats[1]["description"] = "9월 30일 삼성전자는 새 스마트폰 갤럭시 S26을 선보였다. 신제품 공개 행사를 열고 제품을 발표했다."
    novel = [
        article("comparison", "갤럭시 S26과 S25 성능 비교", "신제품 공개 이후 갤럭시 S26과 S25의 사양과 성능 차이를 비교한다."),
        article("reaction", "갤럭시 S26 소비자 반응", "9월 30일 신제품 공개 이후 갤럭시 S26에 대한 소비자 반응을 조사했다."),
        article("impact", "갤럭시 S26 실적 영향 전망", "삼성전자 신제품 갤럭시 S26 발표가 매출과 주가에 미칠 영향을 분석했다."),
    ]
    selector = NewsMapSelector(center)
    await selector.add([*repeats, *novel])
    assert {g.article["news_id"] for g in selector.select(3)} == {"comparison", "reaction", "impact"}
    assert selector.center_repeats == 2


async def test_exact_copy_from_another_url_is_evaluated_and_excluded(identical_vectors):
    center = release("center")
    copy = {**center, "news_id": "copy", "url": "https://other-outlet.test/copy", "source": "other"}
    selector = NewsMapSelector(center)
    assert await selector.add([copy]) == 1  # Not an identity duplicate: different URL.
    assert selector.center_repeats == 1
    assert selector.select(3) == []


async def test_candidate_reprints_have_one_deterministic_representative(identical_vectors):
    center = release("center")
    a = release("a", "S27")
    b = release("b", "S27")
    b["title"] = "삼성전자 갤럭시 S27 선보여…신제품 공개"
    b["description"] = a["description"].replace("공개했다", "선보였다")
    c = article("c", "갤럭시 부품 공급 확대", "갤럭시 부품 공급 업체가 배터리 생산을 확대했다. 공장에서 양산에 나섰다.")
    results = []
    for order in permutations([a, b, c]):
        selector = NewsMapSelector(center)
        await selector.add(list(order))
        groups = selector.select(6)
        results.append(ids(groups))
        assert selector.neighbour_repeats == 1
    assert all(result == results[0] for result in results)
    assert len(results[0]) == 2
    assert len(set(results[0]) & {"a", "b"}) == 1 and "c" in results[0]



async def test_withheld_short_copy_is_not_a_neighbour(identical_vectors):
    center = release("center", description="")
    copy = release("copy", description="신제품 공개")
    selector = NewsMapSelector(center)
    await selector.add([copy])
    assert selector.select(3) == [] and selector.center_repeats == 0
    assert selector.withheld == 1
    copy["published_at"] = ""  # Unknown time cannot even suggest the same event.
    assert ids(await select_related(center, [copy], limit=3)) == ["copy"]


async def test_selected_bridge_directly_excludes_both_copies(monkeypatch):
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
    assert len(result) == 1  # B remains visible and directly covers both A and C.
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
    assert result[1].score < result[2].score  # MMR never replaces the center relevance score.
    assert len(calls) == 5
    assert ids(await select_related(articles[0], list(reversed(articles[1:])), limit=3)) == ids(result)
    assert len(calls) == 5  # Pair comparisons reuse article vectors, zero pair API calls.
    monkeypatch.setattr(settings, "news_map_mmr_lambda", 0)
    assert "unrelated" not in ids(await select_related(articles[0], articles[1:], limit=20))


async def test_later_rounds_append_without_replacing_selected_neighbours(monkeypatch):
    center = release("center")
    early = article("early", "갤럭시 부품 공급 확대", "갤럭시 부품 공급 업체가 배터리 생산을 확대했다. 공장에서 양산에 나섰다.")
    better = article("better", "갤럭시 S26 소비자 반응", "9월 30일 신제품 공개 이후 갤럭시 S26 소비자 반응을 조사했다.")
    vectors = {center["title"]: vec(), early["title"]: vec(.9), better["title"]: vec(.99)}

    async def embed(text, **kwargs):
        return vectors[text.splitlines()[0].removeprefix("Title: ")]

    monkeypatch.setattr(llm, "embed", embed)
    selector = NewsMapSelector(center)
    await selector.add([early])
    assert ids(selector.select(2)) == ["early"]
    await selector.add([better])
    assert ids(selector.select(2)) == ["early", "better"]


async def test_stored_vectors_reused_for_center_pairs_and_mmr(monkeypatch):
    center = release("center")
    candidates = [release("a", "S27"), release("b", "S27"), article("c", "갤럭시 부품 공급 확대")]
    for a in [center, *candidates]:
        _, metadata = article_embeddings._input(a, "news_map")
        a["news_map_embedding"] = {"metadata": metadata, "values": vec()}

    async def forbidden(*args, **kwargs):
        pytest.fail("Compatible stored vectors must avoid provider calls")

    monkeypatch.setattr(llm, "embed", forbidden)
    result = await select_related(deepcopy(center), deepcopy(candidates), limit=6)
    assert len(result) == 2


@pytest.mark.parametrize("tier,limit", [("FREE", 6), ("BASIC", 2), ("PAID", 5)])
def test_apis_match_representatives_order_policy_and_remove_group_fields(monkeypatch, identical_vectors, tier, limit):
    center = release("center")
    candidates = [{**release("copy"), "url": "https://other.test/copy"}, release("a", "S27"), release("b", "S27"),
                  article("reaction", "삼성전자 갤럭시 S26 소비자 반응", "신제품을 평가한 소비자 반응이다.",
                          keywords=["갤럭시 S26"], categories=["반도체"])]
    store.news_cache.update({a["news_id"]: a for a in [center, *candidates]})
    client = TestClient(app)
    params = {"tier": tier, "limit": limit}
    related_response = client.get("/api/v1/news/center/related", params=params)
    graph_response = client.get("/api/v1/news/center/graph", params=params)
    assert related_response.status_code == graph_response.status_code == 200
    body, graph = related_response.json(), graph_response.json()
    related = body["related_news"]
    assert len(related) == 2
    assert [a["news_id"] for a in related] == [a["news_id"] for a in graph["nodes"][1:]]
    assert all(a["news_id"] != "copy" and (a["relevance_score"] is not None) == (tier == "PAID") for a in related)
    assert "center_same_story" not in body and "center_same_story_total" not in body
    for card in [*related, *graph["nodes"], graph["center_node"]]:
        assert "same_story" not in card and "same_story_total" not in card
        assert card["source_url"] and card["title"] and card["source_name"]
    assert "copy" in store.news_cache  # Exclusion never deletes stored articles.
    assert body["selection"] == graph["selection"]
    assert body["selection"]["requested"] == (limit if tier == "PAID" else min(limit, 3))
    assert body["selection"]["returned"] == 2
    assert graph["center_node"]["distance"] == 0
    for card, node, edge in zip(related, graph["nodes"][1:], graph["edges"]):
        assert card == {k: v for k, v in node.items() if k != "is_center"}
        assert card["distance"] == edge["distance"] == 1
    assert len(identical_vectors) == 3  # Exact input copies share one vector before pair comparisons.
