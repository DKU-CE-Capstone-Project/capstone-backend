import httpx
import pytest
from fastapi.testclient import TestClient

from app import database, store
from app.agents import diffbot_client as diffbot
from app.agents import naver_categories, news_fetcher
from app.agents import naver_client as naver
from app.config import Settings, settings
from app.main import app


@pytest.fixture
def naver_mode(monkeypatch):
    monkeypatch.setattr(settings, "news_provider", "naver")
    monkeypatch.setattr(settings, "use_mock_news", False)
    monkeypatch.setattr(settings, "naver_client_id", "unit-test-id")
    monkeypatch.setattr(settings, "naver_client_secret", "unit-test-secret")
    monkeypatch.setattr(settings, "diffbot_token", "unit-test-diffbot")

    async def known_category(url):
        return ["경제"]
    monkeypatch.setattr(naver_categories, "fetch_categories", known_category)
    monkeypatch.setattr(naver_categories, "page_image", lambda url: "https://cdn.example.com/hbm.jpg")


def item(**changes):
    return {
        "title": "삼성<b>전자</b>, HBM &amp; AI 투자",
        "description": "<b>HBM</b> 공급을 &quot;확대&quot;한다.",
        "originallink": "https://www.example.com/article/1",
        "link": "https://n.news.naver.com/article/001/123",
        "pubDate": "Fri, 18 Sep 2026 09:30:00 +0900",
        **changes,
    }


def mock_naver_http(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        naver.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_naver_is_default_provider(monkeypatch):
    monkeypatch.delenv("NEWS_PROVIDER", raising=False)
    assert Settings(_env_file=None).news_provider == "naver"


def test_naver_normalization_uses_original_link_and_utc():
    result = naver.normalize_naver_article(item())
    assert result["title"] == "삼성전자, HBM & AI 투자"
    assert result["description"] == 'HBM 공급을 "확대"한다.'
    assert result["summary"] == result["description"]
    assert result["source"] == "example.com"
    assert result["url"] == "https://www.example.com/article/1"
    assert result["published_at"] == "2026-09-18T00:30:00Z"
    assert result["thumbnail_url"] == ""


def test_missing_original_link_and_invalid_date():
    result = naver.normalize_naver_article(
        item(originallink="javascript:bad", pubDate="bad-date")
    )
    assert result["url"] == item()["link"]
    assert result["published_at"] == ""
    assert naver.normalize_naver_article(item(originallink="", link="")) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://news.naver.com/main/read.naver?oid=001&aid=123",
        "https://n.news.naver.com/article/001/123",
        "https://m.news.naver.com/read.naver?oid=001&aid=123",
    ],
)
def test_naver_hosted_news_is_accepted(url):
    article = naver.normalize_naver_article(item(link=url))
    assert article["naver_url"] == url
    assert article["url"] == item()["originallink"]


@pytest.mark.parametrize(
    "url",
    [
        "",
        None,
        "https://example.com/article",
        "https://www.naver.com/",
        "https://news.naver.com.evil.test/article",
        "https://evilnews.naver.com/article",
        "https://news.naver.com@evil.test/article",
        "javascript:alert(1)",
    ],
)
def test_non_naver_news_links_are_excluded(url):
    assert naver.normalize_naver_article(item(link=url)) is None


async def test_default_batch_is_20_and_only_hosted_news_reaches_diffbot(
    monkeypatch, naver_mode
):
    searches = []
    extracted_urls = []
    items = [
        item(
            originallink=f"https://example.com/{i}",
            link=f"https://n.news.naver.com/article/001/{i}"
            if i % 2 == 0
            else f"https://example.com/{i}",
        )
        for i in range(20)
    ]

    def handler(request):
        searches.append(dict(request.url.params))
        assert request.url.params["display"] == "20"
        return httpx.Response(200, json={"items": items, "total": 500})

    def extract(**kwargs):
        extracted_urls.append(kwargs["url"])
        return {"objects": [{"images": [{"url": "https://cdn.example.com/image.jpg"}]}]}

    mock_naver_http(monkeypatch, handler)
    monkeypatch.setattr(diffbot, "call_diffbot_article", extract)
    articles = await news_fetcher.fetch_news("반도체")
    assert len(searches) == 1  # Do not fetch more pages to refill the filtered batch.
    assert len(articles) == 10
    assert extracted_urls == []  # Diffbot is deferred until a detail click.
    assert all(a["naver_url"].startswith("https://n.news.naver.com/") for a in articles)


def test_all_external_links_return_empty_api_result(monkeypatch, naver_mode):
    mock_naver_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "items": [item(link="https://example.com/article")],
                "total": 1,
            },
        ),
    )

    def forbidden(**kwargs):
        pytest.fail("Excluded articles must not call Diffbot")

    monkeypatch.setattr(diffbot, "call_diffbot_article", forbidden)
    response = TestClient(app).get("/api/v1/news/search?q=반도체")
    assert response.status_code == 200
    assert response.json()["news_cards"] == []


async def test_naver_headers_korean_query_pagination_and_dedupe(
    monkeypatch, naver_mode
):
    def handler(request):
        assert request.url.host == "naverapihub.apigw.ntruss.com"
        assert request.url.path == "/search/v1/news"
        assert request.headers["X-NCP-APIGW-API-KEY-ID"] == "unit-test-id"
        assert request.headers["X-NCP-APIGW-API-KEY"] == "unit-test-secret"
        assert "X-Naver-Client-Id" not in request.headers
        assert "X-Naver-Client-Secret" not in request.headers
        assert dict(request.url.params) == {
            "query": "반도체",
            "display": "3",
            "start": "4",
            "sort": "date",
            "format": "json",
        }
        return httpx.Response(
            200, json={"total": 321, "items": [item(), item(), {"title": "URL 없음"}]}
        )

    mock_naver_http(monkeypatch, handler)
    result = await naver.fetch_naver_page(" 반도체 ", page=2, size=3, sort="latest")
    assert result.total == 321 and len(result.articles) == 1


@pytest.mark.parametrize(("page", "size"), [(0, 10), (1, 101), (1, 0), (101, 10)])
async def test_invalid_pagination_never_calls_api(page, size):
    with pytest.raises(naver.NaverNewsError) as exc:
        await naver.fetch_naver_page("반도체", page=page, size=size)
    assert exc.value.status_code == 422


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_naver_failures_are_explicit_and_do_not_leak_credentials(
    monkeypatch, naver_mode, status
):
    mock_naver_http(
        monkeypatch, lambda request: httpx.Response(status, text="unit-test-secret")
    )
    response = TestClient(app).get("/api/v1/news/search?q=반도체")
    assert response.status_code in (502, 503)
    assert "unit-test-secret" not in response.text
    assert "news_cards" not in response.json()


@pytest.mark.parametrize(
    "payload", [{}, [], {"items": [], "total": "many"}, {"items": None, "total": 0}]
)
async def test_invalid_naver_response(monkeypatch, naver_mode, payload):
    mock_naver_http(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(naver.NaverNewsError):
        await naver.fetch_naver_page("반도체")


async def test_naver_missing_keys_is_not_empty_news():
    with pytest.raises(naver.NaverNewsError) as exc:
        await naver.fetch_naver_page("반도체")
    assert exc.value.status_code == 503


async def test_naver_empty_results_skip_diffbot(monkeypatch, naver_mode):
    mock_naver_http(
        monkeypatch, lambda request: httpx.Response(200, json={"items": [], "total": 0})
    )

    def forbidden(**kwargs):
        pytest.fail("Empty results must not call Diffbot")

    monkeypatch.setattr(diffbot, "call_diffbot_article", forbidden)
    assert await news_fetcher.fetch_news("결과없음") == []


def test_diffbot_primary_image_works_without_article_body(monkeypatch):
    monkeypatch.setattr(
        diffbot,
        "call_diffbot_article",
        lambda **kwargs: {
            "objects": [
                {
                    "images": [
                        {"url": "data:image/png;base64,xxx", "primary": True},
                        {"url": "https://cdn.example.com/secondary.jpg"},
                        {"url": "//cdn.example.com/primary.jpg", "primary": True},
                    ]
                }
            ]
        },
    )
    result = diffbot._extract_one_sync(
        {"url": "https://example.com/article"}, "test", 1000, 0
    )
    assert result["thumbnail_url"] == "https://cdn.example.com/primary.jpg"
    assert "cleaned_content" not in result


def test_diffbot_token_loads_settings_and_environment_precedence(monkeypatch):
    monkeypatch.setattr(settings, "diffbot_token", "configured-in-dotenv")
    assert diffbot.load_diffbot_token() == "configured-in-dotenv"
    monkeypatch.setenv("DIFFBOT_TOKEN", "environment-token")
    assert diffbot.load_diffbot_token() == "environment-token"
    monkeypatch.setenv("DIFFBOT_API_KEY", "environment-api-key")
    assert diffbot.load_diffbot_token() == "environment-api-key"


def test_naver_api_images_body_and_reload_preserved(monkeypatch, naver_mode):
    requests_seen = []
    mock_naver_http(
        monkeypatch,
        lambda request: httpx.Response(200, json={"items": [item()], "total": 321}),
    )

    def extract(**kwargs):
        requests_seen.append(kwargs)
        assert kwargs["url"] == item()["link"]
        assert kwargs["max_retries"] == 0
        return {
            "objects": [
                {
                    "text": "HBM 본문 정보",
                    "images": [
                        {"url": "https://cdn.example.com/hbm.jpg", "primary": True}
                    ],
                }
            ]
        }

    monkeypatch.setattr(diffbot, "call_diffbot_article", extract)
    client = TestClient(app)
    response = client.get("/api/v1/news/search?q=반도체&page=2&size=3&sort=latest")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 321
    card = data["news_cards"][0]
    assert card["thumbnail_url"] == "https://cdn.example.com/hbm.jpg"
    assert card["summary"] == 'HBM 공급을 "확대"한다.'
    assert (
        client.get(f"/api/v1/news/{card['news_id']}/thumbnail").json()["fallback_used"]
        is False
    )
    assert requests_seen == []  # Search gets only NAVER description and page image.
    source = client.get(f"/api/v1/news/{card['news_id']}/source").json()
    assert source["original_body"] == ""
    assert source["description"] == card["summary"]
    assert source["source_url"] == item()["link"]
    assert len(requests_seen) == 0
    # Search again reuses Diffbot extraction and keeps current NAVER fields.
    client.get("/api/v1/news/search?q=반도체&size=1")
    assert len(requests_seen) == 0
    doc = database.news_doc_from_article(store.news_cache[card["news_id"]])
    restored = database.article_from_news_doc(doc)
    assert restored["thumbnail_url"] == card["thumbnail_url"]
    assert not restored.get("cleaned_content")
    assert restored["_news_provider"] == "naver"


async def test_diffbot_failure_keeps_naver_article_and_is_cached(
    monkeypatch, naver_mode
):
    mock_naver_http(
        monkeypatch,
        lambda request: httpx.Response(200, json={"items": [item()], "total": 1}),
    )
    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("unavailable")

    monkeypatch.setattr(diffbot, "call_diffbot_article", fail)
    first = await news_fetcher.fetch_news("반도체", page_size=1)
    second = await news_fetcher.fetch_news("반도체", page_size=1)
    assert calls == 0
    assert first == second and first[0]["thumbnail_url"] == "https://cdn.example.com/hbm.jpg"
    assert first[0]["title"] == "삼성전자, HBM & AI 투자"


async def test_naver_timeout_is_reported(monkeypatch, naver_mode):
    def timeout(request):
        raise httpx.ReadTimeout("test", request=request)

    mock_naver_http(monkeypatch, timeout)
    with pytest.raises(naver.NaverNewsError) as exc:
        await naver.fetch_naver_page("반도체")
    assert exc.value.status_code == 504
