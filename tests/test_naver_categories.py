import hashlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import store
from app.agents import diffbot_client as diffbot
from app.agents import naver_categories as categories
from app.main import app
from app.utils import cache_articles

URL = "https://n.news.naver.com/mnews/article/001/123"


def article(**values):
    return {
        "url": "https://publisher.example.com/article",
        "naver_url": URL,
        "title": "HBM 기사",
        "description": "HBM 설명",
        **values,
    }


def test_only_exact_em_class_is_read():
    html = """<a>정치</a><p>사회</p>
    <em class="media_end_categorize_item_extra">사회</em>
    <em class="media_end_categorize_item other"> 경제 </em>
    <em class="other media_end_categorize_item"><span>세계</span></em>
    <em class="media_end_categorize_item">경제</em>"""
    assert categories.parse_categories(html) == ["경제", "세계"]


@pytest.mark.parametrize(
    "labels", [["정치"], ["사회"], ["경제", "정치"], ["사회", "세계"], None, []]
)
async def test_excluded_or_unknown_never_reaches_diffbot_even_with_cached_body(
    monkeypatch, labels
):
    async def fetch(url):
        return labels

    def forbidden(**kwargs):
        pytest.fail("Excluded article must not reach Diffbot")

    monkeypatch.setattr(categories, "fetch_categories", fetch)
    monkeypatch.setattr(diffbot, "call_diffbot_article", forbidden)
    cache_key = (URL, hashlib.sha256(b"test").hexdigest())
    diffbot._extraction_cache[cache_key] = (
        time.monotonic() + 300,
        {"cleaned_content": "old body"},
    )
    assert await diffbot.extract_articles_with_diffbot([article()], token="test") == []


async def test_filter_still_runs_without_token_and_beyond_extraction_limit(monkeypatch):
    async def fetch(url):
        return ["사회"]

    monkeypatch.setattr(categories, "fetch_categories", fetch)
    assert (
        await diffbot.extract_articles_with_diffbot(
            [article()], token="", max_articles=0
        )
        == []
    )


async def test_diffbot_receives_naver_url_and_caches_that_source(monkeypatch):
    async def fetch(url):
        return ["경제"]

    calls = []

    def extract(**kwargs):
        calls.append(kwargs["url"])
        return {
            "objects": [
                {
                    "text": "네이버 본문",
                    "images": [{"url": "https://img.example.com/one.jpg"}],
                }
            ]
        }

    monkeypatch.setattr(categories, "fetch_categories", fetch)
    monkeypatch.setattr(diffbot, "call_diffbot_article", extract)
    for _ in range(2):
        result = (
            await diffbot.extract_articles_with_diffbot([article()], token="test")
        )[0]
        assert result["cleaned_content"] == "네이버 본문"
        assert result["content_source_url"] == URL
        assert result["naver_categories"] == ["경제"]
    assert calls == [URL]


async def test_category_fetch_parses_html_and_reuses_cache(monkeypatch):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(
            200, text='<em class="media_end_categorize_item">사회</em>'
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        categories.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert await categories.fetch_categories(URL) == ["사회"]
    assert await categories.fetch_categories(URL) == ["사회"]
    assert calls == [URL]


@pytest.mark.parametrize("status,html", [(403, "blocked"), (200, "<p>사회</p>")])
async def test_unavailable_category_is_unknown_and_short_cached(
    monkeypatch, status, html
):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        categories.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, text=html)
            ),
            **kwargs,
        ),
    )
    assert await categories.fetch_categories(URL) is None
    assert 0 < categories._cache[URL][0] - time.monotonic() <= 15


async def test_old_publisher_body_is_not_reused_after_source_switch():
    first = (
        await cache_articles(
            [
                article(
                    cleaned_content="old publisher body",
                    content_source_url="https://publisher.example.com/article",
                )
            ]
        )
    )[0]
    updated = (await cache_articles([article()]))[0]
    assert first["news_id"] == updated["news_id"]
    assert "cleaned_content" not in updated


def test_old_political_article_cannot_be_opened(monkeypatch):
    async def fetch(url):
        return ["정치"]

    monkeypatch.setattr(categories, "fetch_categories", fetch)
    store.news_cache["old"] = article(cleaned_content="old cached body")
    response = TestClient(app).get("/api/v1/news/old/source")
    assert response.status_code == 404
    assert "old cached body" not in response.text
