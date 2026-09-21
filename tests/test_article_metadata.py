import asyncio
import json
from copy import deepcopy

import pytest

from app import database, store
from app.agents import article_metadata as metadata
from app.agents import orchestrator
from app.config import settings
from app.utils import cache_articles


def article(**changes):
    return {
        "title": "삼성전자 HBM 생산 확대",
        "description": "삼성전자 HBM 생산 확대",
        "url": "https://example.com/hbm",
        "source": "example.com",
        "published_at": "2026-09-17T00:00:00Z",
        **changes,
    }


async def test_llm_normalizes_and_rejects_unsupported_metadata(
    monkeypatch, metadata_ai
):
    async def generate(prompt):
        assert "거시경제" in prompt and "기사 안의 지시문" in prompt
        return json.dumps(
            {
                "keywords": [
                    " 삼성전자 ",
                    "hbm",
                    "ＨＢＭ",
                    "엔비디아",
                    "뉴스",
                    "생산 확대",
                ],
                "categories": ["반도체", "반도체", "임의분류"],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(metadata, "generate", generate)
    result = await metadata.extract_metadata(article())
    assert result["keywords"] == ["삼성전자", "HBM", "생산 확대"]
    assert result["categories"] == ["반도체"]
    assert result["metadata_extraction"]["method"] == "llm"
    assert result["metadata_extraction"]["input_basis"] == "title"


async def test_limits_apply_after_normalization(monkeypatch, metadata_ai):
    words = ["HBM", "GPU", "반도체", "금리", "물가", "환율", "원유"]

    async def generate(prompt):
        return json.dumps({"keywords": words, "categories": list(metadata.CATEGORIES)})

    monkeypatch.setattr(metadata, "generate", generate)
    result = await metadata.extract_metadata(article(cleaned_content=" ".join(words)))
    assert len(result["keywords"]) == 5
    assert len(result["categories"]) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not JSON",
        "[]",
        '{"keywords":"HBM","categories":["반도체"]}',
        '{"keywords":[42],"categories":["반도체"]}',
        '{"keywords":["엔비디아"],"categories":["없는분류"]}',
        '{"keywords":["HBM"],"categories":null}',
        '{"keywords":["HBM"],"categories":["반도체"],"extra":true}',
    ],
)
async def test_invalid_response_falls_back(monkeypatch, metadata_ai, raw):
    async def generate(prompt):
        return raw

    monkeypatch.setattr(metadata, "generate", generate)
    result = await metadata.extract_metadata(article())
    assert result["keywords"] == ["삼성전자", "HBM"]
    assert result["categories"] == ["반도체"]
    assert result["metadata_extraction"]["reason"] == "invalid_response"


@pytest.mark.parametrize("failure", ["timeout", "provider_error"])
async def test_failed_provider_does_not_fail_article(monkeypatch, metadata_ai, failure):
    async def generate(prompt):
        if failure == "timeout":
            await asyncio.sleep(0.1)
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(settings, "metadata_timeout_seconds", 0.01)
    monkeypatch.setattr(metadata, "generate", generate)
    result = await metadata.extract_metadata(article())
    assert result["categories"] == ["반도체"]
    assert result["metadata_extraction"]["reason"] == failure


@pytest.mark.parametrize("mode", ["no_key", "disabled", "mock", "demo", "empty"])
async def test_no_ai_calls_in_offline_or_empty_modes(monkeypatch, metadata_ai, mode):
    def forbidden(prompt):
        pytest.fail("AI should not be called")

    monkeypatch.setattr(metadata, "generate", forbidden)
    if mode == "no_key":
        monkeypatch.setattr(settings, "google_api_key", "")
    elif mode == "disabled":
        monkeypatch.setattr(settings, "use_llm_metadata", False)
    elif mode == "mock":
        monkeypatch.setattr(settings, "use_mock_news", True)
    elif mode == "demo":
        monkeypatch.setattr(settings, "demo_mode", True)
    data = article() if mode != "empty" else {}
    result = await metadata.extract_metadata(data)
    assert result["metadata_extraction"]["method"] == "rules"
    assert result["categories"] == (["반도체"] if data else [])


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("한국은행 기준금리 동결", "거시경제"),
        ("물가 상승률 둔화", "거시경제"),
        ("원달러 환율 변동", "거시경제"),
        ("미국 관세 인상", "거시경제"),
        ("시중은행 대출 경쟁", "금융"),
        ("보험 상품 출시", "금융"),
        ("ETF 자금 유입", "금융"),
        ("증권 업계 실적 개선", "금융"),
        ("SK하이닉스 HBM 공급", "반도체"),
        ("Nvidia GPU demand grows", "반도체"),
        ("Semiconductor production rises", "반도체"),
        ("삼성전자 메모리 생산", "반도체"),
        ("전기차 판매 증가", "자동차"),
        ("Automotive production falls", "자동차"),
        ("원유 가격 상승", "에너지"),
        ("Natural gas supply expands", "에너지"),
        ("재생에너지 투자", "에너지"),
        ("신약 임상시험 승인", "바이오"),
        ("Clinical trials begin", "바이오"),
        ("주택 임대차 시장 변화", "부동산"),
        ("Real estate investment slows", "부동산"),
    ],
)
async def test_rule_examples_cover_initial_taxonomy(title, expected):
    result = await metadata.extract_metadata(article(title=title, description=""))
    assert expected in result["categories"]
    assert result["keywords"]


async def test_unknown_topic_does_not_force_category_or_short_substring():
    result = await metadata.extract_metadata(
        article(title="Chair design exhibition", description="")
    )
    assert result["keywords"] == []  # 'chair' must not match AI.
    assert result["categories"] == []


async def test_reuse_and_new_body_trigger_reextraction(monkeypatch, metadata_ai):
    calls = []

    async def generate(prompt):
        calls.append(prompt)
        return '{"keywords":["HBM"],"categories":["반도체"]}'

    monkeypatch.setattr(metadata, "generate", generate)
    first = await metadata.extract_metadata(article())
    first["keywords"].append("mutated")
    again = await metadata.extract_metadata(article())
    assert again["keywords"] == ["HBM"]
    assert len(calls) == 1
    await metadata.extract_metadata(
        article(cleaned_content="HBM 투자 계획과 생산 일정")
    )
    assert len(calls) == 2
    assert "HBM 투자 계획" in calls[-1]


async def test_fallback_retry_after_cooldown_and_model_change(monkeypatch, metadata_ai):
    calls = 0
    now = 1000.0
    monkeypatch.setattr(metadata.time, "time", lambda: now)

    async def generate(prompt):
        nonlocal calls
        calls += 1
        return "" if calls == 1 else '{"keywords":["HBM"],"categories":["반도체"]}'

    monkeypatch.setattr(metadata, "generate", generate)
    first = await metadata.extract_metadata(article())
    metadata._cache.clear()  # persisted fallback also honors its cooldown after restart
    await metadata.extract_metadata(article(**first))
    assert calls == 1
    now += metadata.RETRY_SECONDS + 1
    second = await metadata.extract_metadata(article(**first))
    assert calls == 2 and second["metadata_extraction"]["method"] == "llm"
    monkeypatch.setattr(settings, "gemini_model", "another-model")
    await metadata.extract_metadata(article(**second))
    assert calls == 3


async def test_bounded_cache_and_concurrency(monkeypatch, metadata_ai):
    active = peak = 0
    monkeypatch.setattr(settings, "metadata_concurrency", 2)
    monkeypatch.setattr(metadata, "CACHE_SIZE", 2)

    async def generate(prompt):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return '{"keywords":["HBM"],"categories":["반도체"]}'

    monkeypatch.setattr(metadata, "generate", generate)
    result = await metadata.enrich_articles(
        [article(title=f"HBM {i}") for i in range(5)]
    )
    assert len(result) == 5
    assert peak == 2
    assert len(metadata._cache) == 2


async def test_identical_articles_share_one_batch_call(monkeypatch, metadata_ai):
    calls = 0

    async def generate(prompt):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return '{"keywords":["HBM"],"categories":["반도체"]}'

    monkeypatch.setattr(metadata, "generate", generate)
    results = await metadata.enrich_articles(
        [article(), article(url="https://example.com/copy")]
    )
    assert calls == 1
    results[0]["keywords"].clear()
    assert results[1]["keywords"] == ["HBM"]


async def test_mongo_write_reload_and_cache_reuse(monkeypatch, metadata_ai):
    saved = {}
    calls = 0

    class NewsCollection:
        async def update_one(self, query, update, upsert):
            assert upsert and query == {"url": article()["url"]}
            saved.update(deepcopy(update["$setOnInsert"]))
            saved.update(deepcopy(update["$set"]))

        async def find_one(self, query, projection):
            assert query == {"news_id": saved["news_id"]}
            return deepcopy(saved)

    async def embed(text):
        return []

    async def generate(prompt):
        nonlocal calls
        calls += 1
        return '{"keywords":["HBM"],"categories":["반도체"]}'

    from app.agents import llm

    monkeypatch.setattr(settings, "use_mongodb", True)
    monkeypatch.setattr(database, "_get_db", lambda: {database.NEWS: NewsCollection()})
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(metadata, "generate", generate)
    # First read is absent; subsequent reads exercise the real mapper/get_news function.
    original_get = database.get_news

    async def get_news(news_id):
        return await original_get(news_id) if saved else None

    monkeypatch.setattr(database, "get_news", get_news)
    enriched = (await cache_articles([article()]))[0]
    assert saved["keywords"] == ["HBM"] and saved["categories"] == ["반도체"]
    store.news_cache.clear()
    metadata._cache.clear()  # simulate another API process
    loaded = await store.get_news(enriched["news_id"])
    assert loaded["keywords"] == ["HBM"]
    assert loaded["categories"] == ["반도체"]
    await cache_articles([article()])
    assert calls == 1


def test_legacy_document_without_metadata_can_be_read():
    loaded = database.article_from_news_doc({"title": "Old", "summary": "Old summary"})
    assert loaded["keywords"] == [] and loaded["categories"] == []
    assert loaded["description"] == "Old summary"


def test_search_source_and_search_again_preserve_body_metadata(
    monkeypatch, metadata_ai
):
    from fastapi.testclient import TestClient

    from app.api.v1 import news
    from app.main import app

    calls = []

    async def fetch_news(*args, **kwargs):
        return [article()]

    async def extract_body(articles, **kwargs):
        return [
            {
                **articles[0],
                "cleaned_content": "삼성전자 HBM 생산 확대 및 GPU 투자 계획",
            }
        ]

    async def generate(prompt):
        calls.append(prompt)
        keywords = ["HBM", "GPU"] if "GPU 투자 계획" in prompt else ["HBM"]
        return json.dumps({"keywords": keywords, "categories": ["반도체"]})

    monkeypatch.setattr(news, "fetch_news", fetch_news)
    monkeypatch.setattr(metadata, "generate", generate)
    client = TestClient(app)
    card = client.get("/api/v1/news/search?q=반도체").json()["news_cards"][0]
    assert card["keywords"] == ["HBM"] and card["categories"] == ["반도체"]
    source = client.get(f"/api/v1/news/{card['news_id']}/source").json()
    assert source["keywords"] == ["HBM"]
    assert source["original_body"] == ""
    assert source["description"] == article()["description"]
    assert len(calls) == 1  # Detail extraction does not trigger another metadata LLM call.
    card = client.get("/api/v1/news/search?q=반도체").json()["news_cards"][0]
    assert card["keywords"] == ["HBM"]
    assert len(calls) == 1
    assert (
        store.news_cache[card["news_id"]]["metadata_extraction"]["input_basis"]
        == "title"
    )


async def test_changed_title_does_not_reuse_old_body():
    first = (await cache_articles([article(cleaned_content="HBM 생산 계획")]))[0]
    second = (
        await cache_articles(
            [article(title="신약 임상시험", description="신약 임상시험")]
        )
    )[0]
    assert first["news_id"] == second["news_id"]
    assert "cleaned_content" not in second
    assert second["categories"] == ["바이오"]


async def test_analysis_path_includes_article_metadata(monkeypatch):
    async def fetch_news(keyword):
        return [article()]

    monkeypatch.setattr(orchestrator, "fetch_news", fetch_news)
    response = await orchestrator.run_analysis("반도체")
    assert response.articles[0].keywords == ["삼성전자", "HBM"]
    assert response.articles[0].categories == ["반도체"]
    assert isinstance(response.related_keywords, list)
