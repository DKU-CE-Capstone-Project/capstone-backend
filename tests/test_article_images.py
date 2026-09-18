import pytest

from app.agents import diffbot_client as diffbot
from app.api.v1.news import _thumb

NAVER = "https://n.news.naver.com/mnews/article/003/0014198961"
QR = "https://imgnews.pstatic.net/image/003/2026/09/18/QRCODE_0014198961_001.jpg"
PHOTO = "https://imgnews.pstatic.net/image/003/2026/09/18/NISI20260918_photo.jpg"


def test_naver_requests_page_meta():
    assert diffbot.build_article_params(NAVER, "test", 1000)["fields"] == "meta"


def test_naver_qr_primary_uses_diffbot_opengraph_photo():
    obj = {
        "images": [{"url": QR, "primary": True}],
        "meta": {"og": {"og:image": PHOTO}},
    }
    assert diffbot.extract_image_url(obj, NAVER) == PHOTO


def test_top_level_meta_and_twitter_fallback():
    obj = {"images": [{"url": QR, "primary": True}]}
    assert (
        diffbot.extract_image_url(obj, NAVER, {"twitter": {"twitter:image": PHOTO}})
        == PHOTO
    )


def test_qr_primary_never_wins_over_real_secondary():
    assert (
        diffbot.extract_image_url(
            {"images": [{"url": QR, "primary": True}, {"url": PHOTO}]}, NAVER
        )
        == PHOTO
    )


@pytest.mark.parametrize(
    "url",
    [
        QR,
        QR.lower(),
        QR.replace("QRCODE", "%51%52CODE"),
        "https://cdn.example.com/qr-code.png",
    ],
)
def test_old_qr_thumbnails_are_not_served(url):
    thumbnail, fallback = _thumb({"thumbnail_url": url})
    assert fallback and thumbnail != url


def test_only_qr_means_no_photo():
    assert diffbot.extract_image_url({"images": [{"url": QR}]}, NAVER) == ""


def test_full_extraction_returns_photo_and_body(monkeypatch):
    monkeypatch.setattr(
        diffbot,
        "call_diffbot_article",
        lambda **kw: {
            "meta": {"og": {"og:image": PHOTO}},
            "objects": [
                {"text": "기사 본문", "images": [{"url": QR, "primary": True}]}
            ],
        },
    )
    result = diffbot._extract_one_sync(
        {"url": NAVER, "thumbnail_url": QR}, "test", 1000, 0
    )
    assert result["thumbnail_url"] == PHOTO
    assert result["cleaned_content"] == "기사 본문"


def test_qr_only_response_clears_old_thumbnail(monkeypatch):
    monkeypatch.setattr(
        diffbot,
        "call_diffbot_article",
        lambda **kw: {"objects": [{"text": "본문", "images": [{"url": QR}]}]},
    )
    result = diffbot._extract_one_sync(
        {"url": NAVER, "thumbnail_url": QR}, "test", 1000, 0
    )
    assert result["thumbnail_url"] == ""
