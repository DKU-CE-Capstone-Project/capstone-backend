"""Offline contract, selection and API failure tests; real sockets stay blocked."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import store
from app.agents import article_embeddings, decision_client, news_map
from app.agents.decision_client import (
    MODEL,
    DecisionClient,
    DecisionUnavailable,
    normalize,
    payload_for,
)
from app.agents.decision_policy import CENTER_QUESTIONS, PAIR_QUESTIONS
from app.agents.decision_selector import DecisionNewsMapSelector
from app.config import Settings, settings
from app.main import app


def article(name, rank=1, **extra):
    return {"news_id": name, "title": name, "description": "검색 기사 설명", "url": f"https://example.test/{name}",
            "published_at": "2026-10-08T07:00:00Z", "_search_keyword": "반도체", "_search_rank": rank, **extra}


def response(questions, **top):
    rows = []
    defaults = {"occurrence": "different", "contribution": "background", "helpfulness": "3", "redundancy": "distinct"}
    for name, question in questions.items():
        keys = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
        best = top.get(name, defaults[name])
        row = {"name": name, "type": question["type"], "confidence": 1,
               "probabilities": [{"value": k if question["type"] == "choice" else int(k),
                                  "probability": float(k == best)} for k in keys]}
        row["choice" if question["type"] == "choice" else "score"] = best if question["type"] == "choice" else int(best)
        rows.append(row)
    return {"model": MODEL, "answers": rows, "usage": {"input_tokens": 100, "output_tokens": 0}}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("unit-test-private-key"))
    monkeypatch.setattr(settings, "news_map_selector", "decision")


def test_evidence_allowlist_and_key_alias():
    a = article("center", secret="never send", cleaned_content="never send", news_map_embedding=[1, 0])
    payload = payload_for(a, a, CENTER_QUESTIONS)
    assert set(json.loads(payload["input"])["article_a"]) == {"title", "description", "published_at"}
    assert "never send" not in json.dumps(payload)
    assert [q["name"] for q in payload["questions"]] == list(CENTER_QUESTIONS)
    local = Settings(_env_file=None, OAI_KEY="alias-secret")
    assert local.openai_api_key.get_secret_value() == "alias-secret"
    assert "alias-secret" not in repr(local)


@pytest.mark.parametrize("mutation", ["model", "refusal", "names", "duplicate", "nan", "sum", "score", "confidence", "usage", "row"])
def test_invalid_provider_evidence_is_rejected(mutation):
    body = response(CENTER_QUESTIONS)
    if mutation == "model":
        body["model"] = "other"
    elif mutation == "refusal":
        body["answers"][0]["type"] = "refusal"
    elif mutation == "names":
        body["answers"][1]["name"] = body["answers"][0]["name"]
    elif mutation == "duplicate":
        body["answers"][0]["probabilities"].append(body["answers"][0]["probabilities"][0])
    elif mutation == "nan":
        body["answers"][0]["probabilities"][0]["probability"] = float("nan")
    elif mutation == "sum":
        body["answers"][0]["probabilities"][0]["probability"] = .8
    elif mutation == "score":
        body["answers"][-1]["score"] = True
    elif mutation == "confidence":
        body["answers"][0]["confidence"] = 2
    elif mutation == "usage":
        body["usage"]["input_tokens"] = True
    else:
        body["answers"][0]["probabilities"] = [None]
    with pytest.raises(ValueError):
        normalize(body, CENTER_QUESTIONS)


async def test_content_cache_singleflight_and_revision_invalidation(enabled):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        await asyncio.sleep(0)
        return httpx.Response(200, json=response(PAIR_QUESTIONS))

    async with DecisionClient(transport=httpx.MockTransport(handler)) as client:
        a, b = article("a"), article("b")
        answers = await asyncio.gather(*(client.ask(a, b, PAIR_QUESTIONS) for _ in range(4)))
        assert all(r == answers[0] for r in answers)
        assert len(requests) == 1 and client.stats["decision_cache_hits"] == 3
        answers[0]["redundancy"]["probabilities"]["distinct"] = 0
        assert (await client.ask(a, b, PAIR_QUESTIONS))["redundancy"]["probabilities"]["distinct"] == 1
        await client.ask(a, {**b, "description": "새 설명"}, PAIR_QUESTIONS)
        assert len(requests) == 2
        assert client.stats["decision_input_tokens"] == 200
    assert not decision_client._locks[asyncio.get_running_loop()]


@pytest.mark.parametrize("status", [400, 401, 403, 200])
async def test_invalid_or_access_error_is_not_cached_or_retried(enabled, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text='{"error": "private provider detail"}')

    async with DecisionClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DecisionUnavailable) as exc:
            await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
        assert "private provider detail" not in str(exc.value)
        assert len(calls) == 1 and not decision_client._cache


async def test_retry_budget_counts_attempts_and_total_deadline(enabled, monkeypatch):
    async def no_wait(_):
        pass

    monkeypatch.setattr(decision_client.asyncio, "sleep", no_wait)
    monkeypatch.setattr(settings, "news_map_decision_max_calls", 2)
    async with DecisionClient(transport=httpx.MockTransport(lambda r: httpx.Response(429))) as client:
        with pytest.raises(DecisionUnavailable, match="budget"):
            await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
        assert client.stats["decision_attempts"] == 2
    monkeypatch.setattr(settings, "news_map_decision_total_timeout_seconds", .01)

    async def blocked(_):
        await asyncio.Event().wait()

    async with DecisionClient(transport=httpx.MockTransport(blocked)) as client:
        with pytest.raises(DecisionUnavailable, match="deadline"):
            await client.ask(article("a"), article("b"), PAIR_QUESTIONS)
    assert not decision_client._locks[asyncio.get_running_loop()]


class FakeDecisions:
    def __init__(self, spec=None, redundant_pairs=(), fail=()):
        self.stats = {}
        self.spec, self.pairs, self.fail = spec or {}, set(redundant_pairs), set(fail)
        self.calls = []

    async def ask(self, a, b, questions):
        self.calls.append((a["news_id"], b["news_id"], tuple(questions)))
        if b["news_id"] in self.fail:
            raise DecisionUnavailable("unit-test failure")
        if questions is CENTER_QUESTIONS:
            body = response(questions, **self.spec.get(b["news_id"], {}))
        else:
            pair = frozenset((a["news_id"], b["news_id"]))
            body = response(questions, redundancy="redundant" if pair in self.pairs else "distinct")
        return normalize(body, questions)


async def test_repeat_first_roles_identity_and_uncertainty():
    fake = FakeDecisions({
        "repeat": {"occurrence": "same", "contribution": "repeat", "helpfulness": "1"},
        "uncertain": {"helpfulness": "1"}, "off": {"contribution": "tangential", "helpfulness": "1"},
        "perspective": {"contribution": "perspective"}, "detail": {"occurrence": "same", "contribution": "detail"}})
    selector = DecisionNewsMapSelector(article("center"), fake)
    candidates = [article(n, i) for i, n in enumerate(("repeat", "uncertain", "off", "bg1", "bg2", "perspective", "detail"), 1)]
    alias = article("alias", url="https://example.test/center?utm_source=feed")
    assert await selector.add([alias, *candidates]) == 7
    items = await selector.select(3)
    assert [i.article["news_id"] for i in items] == ["bg1", "perspective", "detail"]
    assert (selector.center_repeats, selector.withheld, selector.unconnected) == (1, 1, 1)
    assert not any(b == "alias" for _, b, _ in fake.calls)


async def test_discarded_bridge_does_not_exclude_new_representative():
    fake = FakeDecisions(redundant_pairs=[frozenset(("n1", "n2")), frozenset(("n2", "n3"))])
    selector = DecisionNewsMapSelector(article("center"), fake)
    await selector.add([article(f"n{i}", i) for i in (1, 2, 3)])
    assert [i.article["news_id"] for i in await selector.select(3)] == ["n1", "n3"]
    assert selector.neighbour_repeats == 1
    assert not any(a == "n2" and b == "n3" for a, b, _ in fake.calls)


async def test_failed_round_preserves_evaluations_and_visible_articles():
    fake = FakeDecisions()
    selector = DecisionNewsMapSelector(article("center"), fake)
    await selector.add([article("first")])
    await selector.select(3)
    fake.fail.add("bad")
    with pytest.raises(DecisionUnavailable):
        await selector.add([article("good"), article("bad")])
    assert selector.evaluated == 1 and selector.is_new(article("good"))
    fake.fail.clear()
    await selector.add([article("second")])
    fake.fail.add("second")
    with pytest.raises(DecisionUnavailable):
        await selector.select(3)
    assert [i.article["news_id"] for i in selector.selected] == ["first"]
    assert [i.article["news_id"] for i in selector.remaining] == ["second"]


def mock_http(monkeypatch, fail_title=None):
    original = DecisionClient.__init__
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        evidence = json.loads(payload["input"])
        b = evidence["article_b"]["title"]
        calls.append(evidence)
        if b == fail_title:
            return httpx.Response(403, json={"error": {"message": "unit-test-private-key"}})
        questions = CENTER_QUESTIONS if len(payload["questions"]) == 3 else PAIR_QUESTIONS
        spec = {}
        if questions is CENTER_QUESTIONS and b.startswith("repeat"):
            spec = {"occurrence": "same", "contribution": "repeat", "helpfulness": "1"}
        return httpx.Response(200, json=response(questions, **spec))

    def init(self):
        original(self, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(DecisionClient, "__init__", init)
    return calls


def seed(names):
    articles = [article(n, i) for i, n in enumerate(["center", *names], 1)]
    store.news_cache.update({a["news_id"]: a for a in articles})
    return articles[0]


def test_related_and_graph_share_decisions_without_embeddings(enabled, monkeypatch):
    calls = mock_http(monkeypatch)
    seed(["repeat", "background", "perspective", "detail"])

    async def forbidden(*args, **kwargs):
        raise AssertionError("Decision selection must not request Gemini embeddings")

    monkeypatch.setattr(article_embeddings, "article_vector", forbidden)
    with TestClient(app) as client:
        related = client.get("/api/v1/news/center/related").json()
        graph = client.get("/api/v1/news/center/graph").json()
        paid = client.get("/api/v1/news/center/related?tier=PAID&include_score=true").json()
    expected_ids = ["background", "perspective", "detail"]
    assert [a["news_id"] for a in related["related_news"]] == expected_ids
    assert [a["news_id"] for a in graph["nodes"][1:]] == expected_ids
    assert all(a["relevance_score"] is None for a in related["related_news"])
    assert all(a["relevance_score"] == 1 for a in paid["related_news"])
    assert len(calls) == 7  # Four center questions, three representative comparisons; later endpoints reuse them.
    assert "same_story" not in json.dumps(related)


async def test_shortage_only_expands_until_candidate_budget(enabled, monkeypatch):
    calls = mock_http(monkeypatch)
    center = seed([f"repeat{i}" for i in range(60)])
    result = await news_map.build_news_map(center, target=3)
    assert result.stats["evaluated"] == 50 and len(calls) == 50
    assert result.stats["center_repeats"] == 50 and not result.items
    assert result.status == "insufficient"


def test_decision_exclusions_skip_ids_and_url_aliases_before_provider_work(enabled, monkeypatch):
    calls = mock_http(monkeypatch)
    seed(["shown", "one", "two", "three"])
    store.news_cache["alias"] = article("alias", url=store.news_cache["shown"]["url"] + "?utm_source=other")
    with TestClient(app) as client:
        response = client.get("/api/v1/news/center/related?exclude_ids=shown,shown")
        assert response.status_code == 200
        data = response.json()
        assert [row["news_id"] for row in data["related_news"]] == ["one", "two", "three"]
        assert data["selection"]["excluded"] == 2
        assert all(call["article_b"]["title"] not in {"shown", "alias"} for call in calls)
        empty = client.get("/api/v1/news/center/related?exclude_ids=shown,one,two,three").json()
        assert empty["related_news"] == [] and empty["selection"]["returned"] == 0


async def test_expansion_failure_returns_completed_initial_selection(enabled, monkeypatch):
    mock_http(monkeypatch, "failure")
    monkeypatch.setattr(settings, "news_map_initial_candidates", 3)
    center = seed(["first", "repeat1", "repeat2", "failure"])
    result = await news_map.build_news_map(center, target=3)
    assert (result.status, result.reason) == ("partial", "decision_failed")
    assert [a.article["news_id"] for a in result.items] == ["first"]
    assert result.stats["evaluated"] == 3


def test_first_round_failure_returns_sanitized_503(enabled, monkeypatch):
    mock_http(monkeypatch, "failure")
    seed(["failure"])
    with TestClient(app) as client:
        for path in ("related", "graph"):
            response = client.get(f"/api/v1/news/center/{path}")
            assert response.status_code == 503
            assert "unit-test-private-key" not in response.text
    assert not decision_client._cache
