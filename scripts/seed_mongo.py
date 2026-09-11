"""
데모용 시드 데이터 삽입.

기존 MongoDB Atlas 클러스터는 삭제되어 이관할 데이터가 없다. 대신 저장소에 있는
mock 기사(fixtures/news_mock.json)를 설계 `news` 스키마로 변환해 넣어, 빈 DB에서도
검색/그래프/리포트 흐름을 바로 확인할 수 있게 한다.

변환은 app.utils.to_news_document을 그대로 쓴다 — 런타임 저장 경로와 같은 함수라
시드 데이터와 실제 수집 데이터의 형태가 어긋나지 않는다.

사용:
    # 로컬
    USE_MONGODB=true MONGODB_URI=... python scripts/seed_mongo.py

    # 컨테이너 (compose가 환경변수를 이미 주입한다)
    docker compose exec econmind-api python scripts/seed_mongo.py

    # 임베딩까지 생성 (GOOGLE_API_KEY 필요, 벡터검색 확인용)
    docker compose exec econmind-api python scripts/seed_mongo.py --embed

옵션:
    --keyword <검색어>  시드 기사에 붙일 _search_keyword (기본: trump)
    --embed            Gemini 임베딩을 생성해 함께 저장 (쿼터 소모)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 저장소 루트를 import 경로에 넣는다 (스크립트 직접 실행 대응)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import database  # noqa: E402
from app.config import settings  # noqa: E402
from app.utils import make_news_id, to_news_document  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "news_mock.json"


def _load_fixture() -> list[dict]:
    """NewsAPI 형태의 mock 응답을 앱 내부 기사 dict로 정규화한다."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    articles = []
    for item in raw.get("articles", []):
        url = item.get("url") or ""
        if not url:
            continue
        articles.append({
            "news_id": make_news_id(url),
            "title": item.get("title") or "",
            "url": url,
            "source": (item.get("source") or {}).get("name", "") or "",
            "published_at": item.get("publishedAt") or "",
            "description": item.get("description") or "",
            "thumbnail_url": item.get("urlToImage") or "",
        })
    return articles


async def main() -> int:
    parser = argparse.ArgumentParser(description="MongoDB 시드 데이터 삽입")
    parser.add_argument("--keyword", default="trump", help="_search_keyword 값")
    parser.add_argument("--embed", action="store_true", help="임베딩까지 생성")
    args = parser.parse_args()

    if not settings.use_mongodb or not settings.mongodb_uri:
        print("[seed] USE_MONGODB=false 이거나 MONGODB_URI가 비어 있습니다. 중단합니다.")
        return 1

    if not database.enabled():
        print("[seed] MongoDB 연결을 만들 수 없습니다. MONGODB_URI를 확인하세요.")
        return 1

    # 인덱스/벡터인덱스 보장 (mongo-init이 이미 만들었다면 no-op)
    await database.ensure_indexes()

    articles = _load_fixture()
    if not articles:
        print(f"[seed] {FIXTURE} 에서 기사를 찾지 못했습니다.")
        return 1

    embed_fn = None
    if args.embed:
        from app.agents.llm import embed as embed_fn  # noqa: F811

    inserted = 0
    for art in articles:
        art["_search_keyword"] = args.keyword
        doc = to_news_document(art)
        if embed_fn:
            text = f"{doc['title']} {doc['summary'] or doc['content']}".strip()
            vec = await embed_fn(text)
            if vec:
                doc["embedding"] = vec
        await database.save_news(doc)
        inserted += 1
        print(f"[seed] upsert {doc['news_id']}  {doc['title'][:50]}")

    # 검증: 실제로 몇 건이 들어갔는지 되읽어 확인한다
    stored = await database.find_news_by_keyword(args.keyword, limit=100)
    print(f"[seed] 완료 — upsert {inserted}건, DB 조회 {len(stored)}건 (keyword={args.keyword})")

    if len(stored) != inserted:
        print("[seed] ⚠️ upsert 건수와 조회 건수가 다릅니다. validator 거부 로그를 확인하세요.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
