# EconMind Backend

실시간 경제 뉴스 검색·뉴스맵·AI 리포트 API. FastAPI / Python 3.11 이상.
프로젝트 정본은 [econmind-docs](https://github.com/DKU-CE-Capstone-Project/econmind-docs)다. 이 README는 **2026-10-01 로컬 `article-api` 브랜치**의 뉴스 경로와 뉴스맵 선정 설정을 설명한다. 작업 시작 커밋은 `c65dfbd866620f993d37d3c670abf73b4c64a06b`이며 미커밋 변경 없이 시작했다. 결과 코드 커밋은 정본 `docs/99-verification.md`의 2026-10-01 기록에 남긴다. 기존 뉴스 세션 경로는 정본의 2026-09-21 병합 기록을 따르며, 이번 뉴스맵 변경은 로컬 구현·검증 범위다. [API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)의 과거 계약과 날짜별 추가 내용을 구분해 읽는다. 원격 반영·운영 배포는 수행하지 않았다.

## 뉴스 공급원 변경 이유

**GDELT에서 반복되는 HTTP 429 오류로 검색과 테스트가 어려워져 기본 공급원을 NCP NAVER API HUB로 변경했다. 해외 뉴스는 검토 예정이다.**
기존 GDELT·NewsAPI 코드는 선택 가능한 이전 경로로 남아 있다. GDELT 429 대응용 별도 백그라운드 수집 기능은 적용하지 않는다. 기존 `/jobs` 데모 큐와 이번 뉴스 검색은 별개다.

## 뉴스 처리 순서

1. 프론트가 `/api/v1/news/search?q=반도체&size=20`을 호출한다. NCP 검색 결과 20건 중 `news.naver.com` 또는 그 하위 도메인의 `link`가 있는 기사만 남긴다. 부족한 수를 추가 검색으로 채우지 않는다.
2. NAVER 기사 HTML의 `em.media_end_categorize_item`을 확인한다. **정치 또는 사회가 포함된 기사와 분류를 확인하지 못한 기사는 제외**한다. 같은 HTML의 `og:image`를 썸네일로 사용한다. 검색 단계에서는 Diffbot을 호출하지 않는다.
3. 제목·description을 바탕으로 Gemini가 키워드 최대 5개, 카테고리 최대 2개를 JSON으로 추출한다. 허용 분류·본문 근거·별칭·중복을 검증하고, 실패하면 보수적인 사전 규칙을 사용한다. 입력과 모델 설정에 따른 캐시·동시성·시간 제한이 있다.
4. **뉴스 상세는 NAVER description만 표시**한다. `/source` 역시 `original_body=""`을 반환하고 Diffbot을 호출하지 않는다.
5. **리포트 보기에서 선택 기사 1건의 NAVER URL을 Diffbot으로 추출**한다. 본문과 출처를 저장하고 리포트·검증 에이전트에 전달한다. 연관 뉴스는 요약 근거다. 본문 추출 실패 시 502를 반환한다.
6. 본문 URL이 같으면 저장된 본문을 재사용한다. 현재 프로세스에서 같은 기사·연관 기사 ID 조합의 리포트를 재사용한다. 프로세스 재시작 후에도 기존 ID의 뉴스·리포트·전략은 MongoDB에서 읽을 수 있지만, 리포트 중복 생성 방지 인덱스는 메모리다. `POST /strategies`는 호출할 때마다 새 전략을 만든다.

분류 체계: `거시경제`, `금융`, `반도체`, `자동차`, `에너지`, `바이오`, `부동산`. 이는 NAVER의 정치·사회 제외용 분류와 별개다. `related_tickers` 자동 매핑은 아직 구현하지 않았다.

### 이미지와 본문

- NAVER 검색 API에는 이미지 필드가 없다. 검색 썸네일은 NAVER 페이지의 OG 이미지에서 얻는다.
- 리포트용 Diffbot 추출은 NAVER URL을 사용하며 `meta.og`의 기사 이미지를 우선한다. QR 코드 URL·캡션 등은 제외하고 이미지가 없으면 기본 이미지를 쓴다.
- `description`, `naver_url`, `naver_categories`, `keywords`, `categories`, `metadata_extraction`, `content_source_url`을 MongoDB 왕복 변환에서 보존한다.
- Diffbot 성공 캐시는 1시간, 실패 캐시는 30초이며 이미지 선택 규칙 버전을 키에 포함한다. NAVER 분류 캐시는 5분, 실패 캐시는 15초다.

## 뉴스맵 반복 보도 정리와 다양성 선정 (2026-10-01)

이미 정해진 중심 기사의 제목·NAVER description을 기준으로 주변 기사를 선정한다. 최초 중심 기사 선정과 실제 다단계 그래프 확장은 이번 범위에서 변경하지 않았다.

기존 방식은 ID·URL·정규화 제목/설명이 다른 재서술 기사를 남긴 뒤 중심과의 연관도만으로 정렬했다. 따라서 동일 발표의 반복 보도가 주변 자리를 차지할 수 있었다. 명확한 중복 제거는 기사 식별 중복을, 반복 보도 판별은 같은 사건의 같은 전달 정보를, MMR은 통과 후보 사이의 과도한 유사성을 각각 다룬다.

1. 같은 검색어의 메모리·MongoDB 기사를 먼저 모은다. 기본 40건이며 최종 표시 limit과 별개다. 원시 후보 3건을 기준으로 재검색하던 분기를 제거했다.
2. 중심 자신·같은 ID/정규화 URL/정규화 제목과 설명의 중복·입력 없는 후보를 제외한다. 같은 주제의 별개 기사는 일괄 중복 처리하지 않는다.
3. 기사별 `Title: {title}\nDescription: {description}` 형식으로 Gemini 임베딩을 생성하거나 저장 벡터를 재사용한다. HTML·공백을 정리하고 제목 500자·설명 6,000자까지 사용한다. 설명이 없으면 제목만 사용하며 본문·생성 summary를 보충하지 않는다. 뉴스맵에서는 Diffbot을 새로 호출하지 않는다.
4. 서버 코사인·근거 있는 키워드 보조 점수·등록 조직명만 겹칠 때 감점을 유지한다. 설정과 요청 중 큰 최소 연관도를 적용한다.
5. 중심과 각 후보를 직접 비교해 사실상 같은 전달 정보인 후보를 제외한다. 후보끼리도 비교하고, 중심과 다른 반복 보도 묶음은 대표 1건만 남긴다. 묶음의 **모든 구성원과** 반복 판정이 성립해야 합류하므로 A~B·B~C만으로 A와 C를 합치지 않는다. 대표는 연관도 → description 길이 → 기사 ID·URL·텍스트 등의 고정 순서로 결정한다.
6. 첫 기사는 최고 연관도, 이후는 `λ × 중심 연관도 − (1−λ) × 이미 선택한 주변 기사와의 최대 코사인`으로 하나씩 선택한다. MMR 동점은 연관도·고정 기사 키로 결정한다. 기존 벡터와 기사 쌍별 코사인을 재사용하며 기사 쌍마다 생성 API를 호출하지 않는다.
7. 요금제·요청의 최종 개수를 적용한다. 유효 후보가 부족할 때만 아래 제한 수집을 거쳐 같은 기준으로 다시 평가하며, 끝까지 부족하면 **0·1·2건도 정상 결과**다.

[반복 보도 규칙](app/agents/repeated_coverage.py)은 제공된 제목·description의 대상/모델·사건 종류·명시 날짜/기간·수치·관점과 문자 bigram Dice 유사도, 발행 시각을 함께 사용한다. 코사인이나 회사명만으로 합치지 않는다. 모델·숫자·시점 또는 비교·소비자 반응·실적 영향 등의 단서가 다르면 보존한다. 명시 날짜가 충돌하면 보존한다. 양쪽에 일치하는 연도 포함 사건 날짜가 없는 경우에는 알려진 timezone 포함 발행 시각의 근접성도 필요하다. 발행 시각을 사건 날짜로 만들거나 빠진 연도를 추측하지 않는다. 설명이 40자 미만이면 공통 모델·긴 제목의 높은 유사도와 짧은 설명의 새 문구 비율을 추가 확인한다. 근거가 없으면 보존하는 보수적 규칙이며 제한된 어휘 밖의 사건과 표현 변화는 놓칠 수 있다.

[공통 API 수집 경로](app/api/v1/news.py)의 보충은 기존 후보의 선정 결과가 유효 목표 `min(limit,3)`(FREE/BASIC) 또는 `limit`(PAID)에 못 미칠 때만 시작한다. 실제 제목·description·근거 있는 키워드로 검색어를 만들고 원래 검색어/중복 검색은 제외한다. 기본 **추가 검색 2회, 페이지당 12건, 추가 고유 후보 20건, 추가 수집·저장·선정 전체 20초**다. 총 평가 후보는 기본 40+20건 이하다. 목표 달성 또는 예산 소진 시 멈춘다. 성공·빈 검색은 프로세스 로컬 60초/128키 캐시, 실패는 5초 오류 캐시, 동시 동일 검색은 단일 요청으로 공유한다. 재시작·여러 프로세스 간 캐시는 공유하지 않는다.

추가 수집은 실패를 샘플로 대체하지 않는 NAVER 경로와 명시적 mock 모드에 한정한다. 샘플 fallback이 남은 실 GDELT/NewsAPI 경로에서는 보충하지 않는다. NAVER URL·분류 제외 정책은 그대로 적용한다. 새 보충 기사 메타데이터는 기존 검증된 결과를 재사용하거나 로컬 규칙으로 채우며 **Flex 생성·RAG 임베딩을 새로 호출하지 않는다**. 규칙 캐시는 정상 검색의 AI 추출을 막지 않는 별도 모드다. 정상 검색의 메타데이터/RAG 처리는 유지하며 호환되지 않는 기존 RAG 벡터는 보충 저장에서 제거한다. 보충 후보의 뉴스맵 벡터는 기존 구조로 생성·저장한다. 기본 상한에서 새 뉴스맵 임베딩은 보충 기사 최대 20건이며 NAVER 페이지 분류/OG 이미지 HTTP 비용도 추가된다. 실제 비용·지연은 측정하지 않았다.

`/related`와 `/graph`는 [공통 선정 함수](app/agents/related_selector.py)와 같은 요금제 정책을 사용한다. 두 API 모두 `tier=FREE`, `limit=10`(1~50), `min_relevance=0`을 기본값으로 받는다. FREE/BASIC도 내부 평가·필터·정렬을 수행하며 마지막에 `min(limit, 3)`건만 반환하고 `relevance_score=null`로 숨긴다. PAID는 요청 limit에 따라 반환하고 점수를 노출한다. `include_score`는 호환용이며 FREE/BASIC의 점수 숨김을 해제하지 않는다. `tier`는 쿼리 값이며 실제 구독 확인 기능은 아직 없다. **기존 `/graph` 호출도 tier를 생략하면 이제 주변 최대 3건**이다. 모든 주변 노드는 직접 연결이므로 `distance=1`이다. `/relations`의 기존 단어 중첩 점수는 별도 의미다.

검색·연관 기사·그래프 노드는 [공통 카드 변환](app/news_cards.py)을 사용해 `description`, `source_name`, `source_url`, `published_at`, `thumbnail_url`, `keywords`, `categories`를 함께 반환한다. `/related`는 주변 기사 목록, `/graph`는 중심 기사와 전체 노드·연결선을 반환하며 중심의 `distance=0`, `relevance_score=null`이다. `limit`은 중심을 제외한 주변 개수다. 설명은 자르지 않고, `summary`는 description 우선·기존 summary 대체 규칙을 유지한다. `/related`의 응답도 `RelatedResponse`로 OpenAPI에 명시한다.

최종 점수는 `clip((1-w) * max(cosine, 0) + w * J - p, 0, 1)`이다. `J`는 조직명을 제외한 키워드 집합의 Jaccard 값이며, `p`는 등록된 조직명이 겹치면서 다른 핵심어는 겹치지 않을 때만 적용한다. `AI`, `HBM`, `HBM3E`, `금리`와 확인된 별칭을 보존하고, 같은 제목·설명에서 추출된 메타데이터는 집합으로 합쳐 반복 가산하지 않는다. 현재 가중치·임계값은 **실제 뉴스 품질 평가 전 초기값**이다.

| 환경변수 | 기본값 | 역할 |
|---|---|---|
| `NEWS_MAP_EMBEDDING_MODEL` | `gemini-embedding-001` | 뉴스맵 전용 모델 |
| `NEWS_MAP_EMBEDDING_DIMENSIONS` | `768` | 요청·검증 차원 |
| `NEWS_MAP_EMBEDDING_TASK_TYPE` | `SEMANTIC_SIMILARITY` | 중심·후보에 같은 용도 적용 |
| `NEWS_MAP_CANDIDATE_LIMIT` | `40` | 최종 표시 수와 별개인 후보 상한 |
| `NEWS_MAP_EMBEDDING_CONCURRENCY` | `3` | 벡터 생성 동시성 |
| `NEWS_MAP_MIN_RELEVANCE` | `0.65` | 최소 최종 점수; 요청값과 큰 쪽 적용 |
| `NEWS_MAP_KEYWORD_WEIGHT` | `0.10` | 보조 키워드 비중 |
| `NEWS_MAP_ENTITY_ONLY_PENALTY` | `0.10` | 등록된 조직명만 겹칠 때 감점 |
| `NEWS_MAP_MMR_LAMBDA` | `0.70` | 관련성 비중; 나머지 0.30은 주변 간 최대 유사도 감점 |
| `NEWS_MAP_REPEAT_COSINE` | `0.92` | 반복 판별의 필요조건, 단독 판정 금지 |
| `NEWS_MAP_REPEAT_TEXT_SIMILARITY` | `0.55` | 설명 및 제목/설명 평균 Dice 유사도 하한 |
| `NEWS_MAP_REPEAT_SHORT_TEXT_SIMILARITY` | `0.85` | 짧은 설명에서 제목 유사도 하한 |
| `NEWS_MAP_REPEAT_MAX_HOURS` | `48` | 완전한 공통 사건 날짜가 없을 때 발행 시각 간격 상한 |
| `NEWS_MAP_REPEAT_DESCRIPTION_MIN_CHARS` | `40` | 짧은 설명 처리 경계 |
| `NEWS_MAP_REPEAT_NOVELTY_RATIO` | `0.25` | 짧은 설명의 새 bigram 비중이 넘으면 보존 |
| `NEWS_MAP_SUPPLEMENT_MAX_SEARCHES` | `2` | 추가 검색 상한; 0이면 비활성화 |
| `NEWS_MAP_SUPPLEMENT_PAGE_SIZE` | `12` | 검색당 요청 기사 수 |
| `NEWS_MAP_SUPPLEMENT_CANDIDATE_LIMIT` | `20` | 추가 고유 후보 상한; 0이면 비활성화 |
| `NEWS_MAP_SUPPLEMENT_TIMEOUT_SECONDS` | `20` | 추가 수집·저장·재선정 전체 제한 시간 |
| `NEWS_MAP_SUPPLEMENT_CACHE_TTL_SECONDS` | `60` | 추가 검색 성공/빈 결과 캐시 TTL |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | 기존 임베딩 호출 제한 시간 |

λ=0.70은 관련성을 우선하면서 중복 감점을 주기 위한 출발점이다. 높은 반복 코사인·문자 유사도·40자/48시간 기준은 오탐을 줄이고, 2회/20건/20초는 추가 비용을 제한하기 위한 초기 예산이다. **모든 새 임계값과 가중치는 실뉴스 평가로 튜닝해야 하며 최적값으로 검증하지 않았다.** 설정 범위는 [Settings](app/config.py)에서 검증한다.

공개 `relevance_score`는 위 중심 연관도 공식 그대로다. MMR 점수나 반복 점수로 덮어쓰지 않으며 **최종 응답 순서는 relevance_score 내림차순과 다를 수 있다**. 추가 검색의 인증/공급원/시간 오류는 기존 502·503·504를 유지하고 추가 전체 시간 초과는 504다. 임베딩 실패는 503이며 이미 고른 일부 결과를 성공으로 반환하지 않는다. 정상 빈 결과와 실패를 구분한다.

[벡터 저장·재사용](app/agents/article_embeddings.py)은 입력 SHA-256·모델·차원·용도·전처리 버전을 확인한다. 뉴스맵은 `news_map_embedding`, 기존 리포트 RAG는 `embedding`·`embedding_metadata`와 `EMBEDDING_MODEL`·768차원 인덱스를 사용한다. 내용·설정이 바뀐 기사만 필요할 때 다시 생성하며 전체 DB 삭제·재생성은 하지 않는다. 평가에 필요한 벡터가 없거나 API 실패·차원 불일치 등이 발생하면 503을 반환한다. 실패를 0점이나 일부 후보의 성공 목록으로 처리하지 않는다.

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env  # 처음 설정할 때만. 기존 파일은 보존한다.
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`.env` 실제 값은 Git에 올리지 않는다. 프론트에는 API 비밀키를 전달하지 않는다.

| 설정 | 현재 권장값 / 역할 |
|---|---|
| `NEWS_PROVIDER` | `naver` (기본값) |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | **NCP NAVER API HUB** 애플리케이션의 인증키 |
| `DIFFBOT_TOKEN` | Diffbot Article API 토큰. `.env`에서 읽는다 |
| `GOOGLE_API_KEY` | Gemini API 키 |
| `LLM_PROVIDER` | `gemini` |
| `GEMINI_MODEL`, `GEMINI_SERVICE_TIER` | `gemini-3.5-flash-lite`, `flex` |
| `USE_MOCK_NEWS`, `DEMO_MODE` | 실제 연동은 모두 `false` |
| `USE_LLM_METADATA`, `USE_LLM_SUMMARIES` | 각각 `true`, `false` 권장. 카드 설명은 NAVER 원문 description |
| `USE_MONGODB`, `MONGODB_URI`, `MONGODB_DB_NAME` | 기본 `true`; 연결할 환경의 앱 계정 URI가 필요하며 DB 이름 기본값은 `capstone_news` |
| `MONGODB_REQUIRED` | 로컬 기본 `false`, 서버 `true`. 필수 저장 실패는 503 |
| `USE_RAG`, `USE_CRITIC` | 유사 기사 근거 검색, 생성 리포트 검증 |
| `CORS_ORIGINS` | 로컬은 `http://localhost:5173,http://127.0.0.1:5173` |

NCP 요청은 `https://naverapihub.apigw.ntruss.com/search/v1/news`에 `X-NCP-APIGW-API-KEY-ID` / `X-NCP-APIGW-API-KEY` 헤더를 보낸다. NAVER Developers 키와 구별한다. NCP에서 해당 애플리케이션의 뉴스 검색 API 사용 권한을 활성화해야 한다.
응답의 `title`/`description` HTML을 제거하고, `pubDate`를 UTC로 정규화하며 `originallink`(언론사 출처)와 `link`(NAVER 본문 URL)를 모두 보존한다. `total_count`는 필터 전 공급원 전체 검색 수이며 현재 반환 카드 수가 아니다.

### Flex 대기 시간

텍스트 생성 전체 제한은 600초(재시도 포함), 기사 메타데이터와 검색 배치 전체 제한은 660초, 임베딩은 30초다. 메타데이터 동시성은 3, 텍스트 출력 상한은 2048토큰이다. Flex는 즉시 응답을 보장하지 않으며 제한 시간 이후 메타데이터는 규칙 결과를 사용한다. 서버 프록시는 리포트 생성과 검증 대기를 고려해 1500초를 허용한다.

## API

| 메서드 / 경로 | 역할 |
|---|---|
| `GET /health` | 프로세스 생존 확인. MongoDB 실제 ping을 대신하지 않는다 |
| `GET /ready` | MongoDB 실제 ping. 연결 실패 또는 필수 DB 비활성화 시 503 |
| `GET /api/v1/keywords/recommended` | 추천 검색어 |
| `GET /api/v1/news/search`, `/cards` | 뉴스 카드·description·키워드·카테고리 |
| `GET /api/v1/news/{id}/source` | 출처·description. 전체 본문은 반환하지 않음 |
| `GET /api/v1/news/{id}/thumbnail`, `/graph`, `/related` | 썸네일·뉴스맵·연관 뉴스 |
| `POST /api/v1/news/selections`, `GET /api/v1/news/{id}/relations` | `tier=PAID` 쿼리 조건의 선택 묶음·관계 점수. 사용자 인증은 아직 없음 |
| `POST /api/v1/reports`, `GET /api/v1/reports/{id}` | 선택 기사 본문 추출·리포트 생성 및 조회 |
| `POST /api/v1/strategies`, `GET /api/v1/strategies/{id}` | 전략 생성 및 조회 |
| `POST /jobs`, `GET /jobs/{id}` | 기존 NATS·Redis 작업 큐 |
| `POST /analyze` | 기존 분석 호환 경로 |
| `GET`·`DELETE /api/v1/session`, `POST /api/v1/session/mindmap/expand`·`collapse`, `DELETE /api/v1/session/mindmap` | 쿠키 기반 세션 식별·마인드맵 상태. 아래 「세션」 참고 |

전체 요청·응답 형식은 [뉴스 세션 API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)와 해당 브랜치 실행 중인 `/docs`를 참고한다. `/health`는 실제 DB ping이 아니며 `/ready`가 이를 검사한다. 현재 프론트 `article-api`는 `/related?tier=FREE` 결과를 서버 순서대로 표시하고 `/graph`로 보충하지 않는다. 주변 기사 선정만 제한적으로 재개했으며, 키워드맵 알고리즘·다단계 확장·세션 상태 연동은 계속 보류한다. NCP 인증·응답 오류는 안전한 메시지로 전달하며, 검색 오류를 mock 뉴스로 대체하지 않는다.

## 세션 (쿠키 기반 사용자 식별)

로그인을 넣지 않기로 해서 계정으로 사용자를 구분할 수 없다. 대신 `econmind_sid` 쿠키를
발급하고 세션 상태를 **Redis에 TTL과 함께** 저장한다. TTL 만료 = 세션 소멸 = 데이터 삭제다.
Redis가 없거나 죽어 있으면 프로세스 로컬 dict로 폴백하지만, api를 여러 개로 띄우면
세션이 인스턴스마다 갈리므로 운영에서는 Redis를 전제로 한다.

```bash
SESSION_TTL_SECONDS=86400      # 세션 수명(초). 접근할 때마다 갱신
SESSION_COOKIE_SECURE=false    # HTTPS로 서비스할 때 true
SESSION_COOKIE_SAMESITE=lax    # 프론트와 API가 다른 출처면 none(+secure=true)
```

| 메서드 / 경로 | 역할 |
|---|---|
| `GET /api/v1/session` | 세션 id + 마인드맵 상태 조회 (쿠키 없으면 발급) |
| `DELETE /api/v1/session` | 세션 전체 초기화 |
| `POST /api/v1/session/mindmap/expand` | 펼친 노드를 세션에 기록 |
| `POST /api/v1/session/mindmap/collapse` | 펼친 노드 기록 해제 |
| `DELETE /api/v1/session/mindmap` | 마인드맵 상태만 초기화 |

쿠키는 `HttpOnly`로 발급하며 CORS는 `allow_credentials=True`다. 프론트엔드는
`fetch(..., { credentials: 'include' })`로 호출해야 쿠키가 오간다.

> **현재 범위**: 세션은 어떤 노드를 펼쳤는지 *기록*만 한다. `GET /news/{id}/graph`는
> 아직 세션의 확장 상태를 읽지 않는다 — 그 연동은 `app/api/v1/news.py`의 마인드맵 확장
> 포팅에 딸려 있고, 해당 포팅은 [2026-09-19 보류 결정](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/04-roadmap.md#마인드맵-알고리즘-보류-결정-2026-09-19)
> 대상이다. 화면 연동도 아직 없다.

## 검증과 배포

2026-10-01: Python 3.13.5 프로젝트 가상환경에서 **238 passed**(별도 실제 로컬 MongoDB 4.4.18 왕복 3개 포함), 변경 Python 12개 파일 Ruff 통과, `compileall` 성공, `uv build --offline --wheel` 성공. Gemini·NAVER·Diffbot은 fixture/mock이며 기본 테스트는 외부 소켓을 차단한다. MongoDB 검사는 opt-in URI의 고유 테스트 DB만 사용·정리한다. 기본 실행은 MongoDB 검사 3개를 건너뛴다. 자세한 사례는 [다양성 테스트](tests/test_news_map_diversity.py)·[제한 수집 테스트](tests/test_related_supplement.py)에 있다. 한국어 제목·설명은 직접 작성한 가상 사례이며 실제 기사 출처나 실뉴스 의미 품질 검증을 뜻하지 않는다.

```bash
NEWS_MAP_TEST_MONGODB_URI=mongodb://127.0.0.1:27028 .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app tests
uv build --offline --wheel --out-dir /tmp/econmind-news-map-wheels-20261001
```

Ruff 대상: `app/agents/{article_metadata,related_selector,repeated_coverage,related_candidates}.py`, `app/{config,utils}.py`, `app/api/v1/news.py`, `tests/{conftest,test_article_metadata,test_news_map_mongo,test_news_map_diversity,test_related_supplement}.py`다. 아래 2026-09-18·2026-09-30 결과는 과거 기록으로 보존한다. 이번 앱 Docker 이미지 빌드·Python 3.11 실행·실제 외부 API·운영 배포는 수행하지 않았다.

```bash
.venv/bin/python -m pytest -q
# 서버와 같은 아키텍처로 빌드
docker build --platform linux/amd64 -t econmind-backend:release-20260918 .
```

2026-09-18: 로컬 Python 3.13과 Docker Python 3.11에서 **129개 테스트 통과**. NCP 형식·URL 제한·분류 제외·QR 이미지 제외·상세 무추출·리포트 단건 추출·메타데이터·MongoDB 필수 저장·readiness를 검증했다. 단위 테스트는 외부 소켓을 차단하며 실제 API 호출 결과와 구분한다. 2026-09-20 로컬 브랜치의 `/ready`는 MongoDB 연결 상태 `on`을 반환했다.

2026-09-30 선정 구현 단계: 외부 API mock과 별도 로컬 MongoDB로 **167개 테스트 통과**(MongoDB 왕복 2개 포함). 신규 임베딩·선정 모듈 및 해당 테스트 4개 파일의 Ruff 검사도 통과했다. 같은 날 공통 카드·요금제 통일 후 기본 전체 검사는 **180 passed, 2 skipped**, 변경 Python 파일 6개의 Ruff 검사도 통과했다. 추가 계약 테스트는 검색 캐시와 독립적인 메타데이터, 두 API의 요금제·개수·점수·임계값 일치와 OpenAPI 스키마를 확인한다. 이번 후속 검증에서는 실제 MongoDB 왕복을 재실행하지 않았다. 기본 `pytest -q`에서는 MongoDB opt-in 검사 2개를 건너뛴다. 별도 테스트 MongoDB를 준비했을 때만 아래 URI를 지정한다. 테스트는 고유 DB를 만들고 정리하며 Gemini·뉴스 HTTP는 계속 mock한다.

```bash
NEWS_MAP_TEST_MONGODB_URI=mongodb://127.0.0.1:27028 .venv/bin/python -m pytest -q
```

뒤쪽 후보 우선 선정, FREE 필터·정렬, 중심 변경, 적은 결과, 중복 제외, API 선정 일치, 빈 설명·벡터 실패·재사용·설정 변경을 검사했다. 실제 Gemini 성공 응답·한국어 뉴스 선정 품질·가중치 최적화·운영 배포는 검증하지 않았다. 날짜별 상세 근거는 정본 `docs/99-verification.md`에 기록한다.
서버 배포·백업 절차는 [배포 런북](https://github.com/DKU-CE-Capstone-Project/capstone-deploy)을, 실제 검증 범위와 운영 미검증 상태는 [검증 기록](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/99-verification.md#2026-09-20-뉴스-세션-브랜치-api-검증)을 따른다.

남은 범위: JWT·사용자별 세션 통합, reports/strategies 설계 정합화와 strict validator 승격, 종목 매핑, 해외 뉴스 공급원 검토. 생성 실패 안내용 백엔드 fallback 리포트는 남아 있으며, 프론트는 이를 성공 리포트로 표시하지 않는다.
