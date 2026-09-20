# EconMind Backend

실시간 경제 뉴스 검색·뉴스맵·AI 리포트 API. FastAPI / Python 3.11 이상.
프로젝트 정본은 [econmind-docs](https://github.com/DKU-CE-Capstone-Project/econmind-docs)이며, 이 문서는 **`codex/news-session-20260918` 작업 브랜치**의 뉴스 경로를 설명한다. 이 변경은 [백엔드 PR #6](https://github.com/DKU-CE-Capstone-Project/capstone-backend/pull/6)에서 검토 중이고 코드 `main`에는 아직 병합되지 않았다. 운영 서버의 현재 상태는 여기서 검증하지 않았다. [브랜치 API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)에는 전체 18개 operation과 검증 범위가 있다.

## 뉴스 공급원 변경 이유

**GDELT에서 반복되는 HTTP 429 오류로 검색과 테스트가 어려워져 기본 공급원을 NCP NAVER API HUB로 변경했다. 해외 뉴스는 검토 예정이다.**
기존 GDELT·NewsAPI 코드는 선택 가능한 이전 경로로 남아 있다. GDELT 429 대응용 별도 백그라운드 수집 기능은 적용하지 않는다. 기존 `/jobs` 데모 큐와 이번 뉴스 검색은 별개다.

## 작업 브랜치의 처리 순서

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

전체 요청·응답 형식은 [뉴스 세션 API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)와 해당 브랜치 실행 중인 `/docs`를 참고한다. `/health`는 실제 DB ping이 아니며 `/ready`가 이를 검사한다. `/graph`·`/related`는 구현돼 있지만 현재 프론트 뉴스맵 화면은 호출하지 않는다. 마인드맵 알고리즘 적용은 [보류 결정](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/04-roadmap.md#마인드맵-알고리즘-보류-결정-2026-09-19)에 따른다. NCP 인증·응답 오류는 안전한 메시지로 전달하며, 검색 오류를 mock 뉴스로 대체하지 않는다.

## 검증과 배포

```bash
.venv/bin/python -m pytest -q
# 서버와 같은 아키텍처로 빌드
docker build --platform linux/amd64 -t econmind-backend:release-20260918 .
```

2026-09-18: 로컬 Python 3.13과 Docker Python 3.11에서 **129개 테스트 통과**. NCP 형식·URL 제한·분류 제외·QR 이미지 제외·상세 무추출·리포트 단건 추출·메타데이터·MongoDB 필수 저장·readiness를 검증했다. 단위 테스트는 외부 소켓을 차단하며 실제 API 호출 결과와 구분한다. 2026-09-20 로컬 브랜치의 `/ready`는 MongoDB 연결 상태 `on`을 반환했다.
서버 배포·백업 절차는 [배포 런북](https://github.com/DKU-CE-Capstone-Project/capstone-deploy)을, 실제 검증 범위와 운영 미검증 상태는 [검증 기록](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/99-verification.md#2026-09-20-뉴스-세션-브랜치-api-검증)을 따른다.

남은 범위: JWT·사용자별 세션 통합, reports/strategies 설계 정합화와 strict validator 승격, 종목 매핑, 해외 뉴스 공급원 검토. 생성 실패 안내용 백엔드 fallback 리포트는 남아 있으며, 프론트는 이를 성공 리포트로 표시하지 않는다.
