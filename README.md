# EconMind Backend

실시간 경제 뉴스 검색·뉴스맵·AI 리포트 API. FastAPI / Python 3.11 이상.
프로젝트 정본은 [econmind-docs](https://github.com/DKU-CE-Capstone-Project/econmind-docs)다. 이 README는 **2026-10-01 로컬 `article-api` 브랜치**의 뉴스 경로와 뉴스맵 선정 설정을 설명한다. 같은 날 2차 작업 시작 커밋은 `0a462f8ba080f957e6855e3cb224b17007fd876d`이며 미커밋 변경 없이 시작했다. 결과 코드 커밋은 정본 `docs/99-verification.md`의 2026-10-01 기록에 남긴다. 기존 뉴스 세션 경로는 정본의 2026-09-21 병합 기록을 따르며, 이번 뉴스맵 변경은 로컬 구현·검증 범위다. [API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)의 과거 계약과 날짜별 추가 내용을 구분해 읽는다. 원격 반영·운영 배포는 수행하지 않았다.

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

## 뉴스맵 대표 기사와 반복 보도 제외 (2026-10-02)

현재 `article-api`는 중심 기사와 추가 정보를 주는 주변 대표만 제공한다. 반복 보도는 표시에서 제외하며 저장 기사/DB를 삭제하지 않는다. 기존 미커밋 low_overlap·semantic_difference 보류, 추가 사실 근거 검사와 인물 추출 개선을 보존해 이번 변경에 포함했다.

제품 기준은 **“중심 이슈를 이해하는 데 필요한 배경·이전 변화·후속 전개·영향·다른 관점을 제공하는가?”**, 최종 기준은 **“중심과 이미 선정한 주변에서 얻지 못한 유용한 정보를 제공하는가?”**다.

- 같은 ID/URL 제외 → 기존 Gemini 연관도/최소 기준 → 중심 반복·명시적 보류 제외 → 입력에 근거한 연결 검사 → 기존 MMR로 하나씩 선정하면서 **실제로 남는 주변 대표와 직접 비교** → 요금제 상한. 탈락/미선정 기사는 다른 후보를 제외하는 기준이 되지 않는다.
- MMR 값 동점은 중심 연관도, 정제한 description 길이, 고정 기사 키 순으로 결정한다. 같은 라운드의 입력 순서를 바꿔도 결정적이다. 확장 라운드는 이미 표시한 대표를 유지하고, 뒤늦게 도착한 더 높은 점수의 재보도도 제외한다.
- 같은 사건 전체를 제거하지 않는다. 근거 있는 세대별 소비 차이, 임신·청소년 안전 정보, 별도 정책 배경/후속/원가 영향은 남길 수 있다. 이름·기관·숫자·인용의 단순 추가는 근거가 아니며 `short_text_covered`·`low_overlap`·`semantic_difference` 보류를 유지한다.
- 사건 단서는 제목과 서로의 설명을 함께 확인한다. 제목의 새 관점도 자신의 description에 뒷받침되어야 하며, 발표/전달 관용어는 별도 사실로 세지 않는다. 등록 회사명만 공유하는 후보와 충분한 설명에서 주제/기관만 공유하는 후보는 제외한다. 연결 검사는 어휘 기반 보조 장치이므로 짧은 입력·동의어·검색 의도·실제 유용성을 완전히 판정하지 못한다.
- **응답 계약 변경:** 카드의 `same_story`·`same_story_total`, `/related`의 `center_same_story`·`center_same_story_total`, `SameStoryArticle` 스키마와 묶음 직렬화를 제거했다. `/related`·`/graph`는 같은 선정 결과·공통 카드·tier·distance·selection 계약을 쓴다. 내부에는 `center_repeats`·`neighbour_repeats`·`withheld`·`entity_only`·`unconnected` 진단 수만 남긴다.
- `NEWS_MAP_SAME_STORY_LIMIT`을 Settings/환경 예시에서 삭제했다. Gemini 모델·입력·연관도 공식/임계값·MMR 가중치, 최초 20/총 고유 50개, 기존 검색 단계/시간/캐시를 유지한다. 부족하면 기존 확장을 쓰고 0·1·2개와 정상 부족 상태를 반환한다.

로컬 검증: `.venv/bin/python -m pytest -q` **348 passed, 3 skipped**(선택 MongoDB 검사), 변경 Python 13개 Ruff, compileall와 offline wheel 성공. [대표 회귀](tests/test_news_map_representatives.py) 14개는 연쇄 제외 방지, 미선정 대표의 오제외 방지, 동점/설명 기준, 확장 중복, 저장 실뉴스 19건/기존 Gemini 벡터, API·저장 유지 등을 검증한다. 기존 [low_overlap](tests/test_news_map_low_overlap.py)·[추가 사실/인물](tests/test_news_map_grounded_additions.py) 회귀도 통과했다. NAVER/Gemini/Diffbot/DB 실호출 없이 외부 소켓을 차단했다. 전체 실뉴스 의미 품질·운영 환경 검증은 아니다.

아래 2026-10-01 절은 당시 구현 기록이다. 묶음 제공과 당시 공개 필드/검증 수치는 현재 동작을 뜻하지 않으며 위 2026-10-02 계약이 우선한다.

## 뉴스맵 추가 사실 근거와 인물 추출 (2026-10-01 후속 2)

기존 `article-api/553d204`의 미커밋 low_overlap 수정·회귀를 보존하고 이어서 구현했다. 앞선 추가 정보 조건을 통과하지 못한 낮은 코사인 분기는 `INSUFFICIENT/semantic_difference`다. `Judgment.withheld`는 `short_text_covered`·`low_overlap`·`semantic_difference`만 제외하여, 노드/묶음/표시 개수에 넣지 않고 기존 `withheld` 진단에 집계한다. 부족하면 기존 확장 예산을 쓰고 적은 결과와 정상 `insufficient`를 반환한다. 다른 INSUFFICIENT와 사건 충돌은 일괄 제외하지 않는다.

인물은 전체 한국어 이름 토큰·성씨 형태·명시적 직함과 기관 형태를 구분해 추출하며 기관명 끝 2~4자를 잘라 인물로 만들지 않는다. `김정관 장관`·`김정관 산업부 장관`·`김정관 산업통상부 장관`은 같은 인물이다. 이름/기관/직함만 추가된 것은 제목·설명 추가 사실로 세지 않는다. 새 발언/행동/결정/결과는 후보 설명의 같은 절에 행위자·완결 표현과 별도 내용 단어가 뒷받침될 때 보존하며 기사에 없는 사실을 생성하지 않는다.

저장 입력+mock 벡터에서 헤럴드 semantic_difference와 국제신문 잘못된 인물 name 2개 기반 추가 정보가 보류로 바뀌었고, 한국경제TV 보류 및 뉴시스 미국 측 발표는 유지된다. 신규 회귀 55개, 관련 회귀 189개, 전체 **335 passed·3 skipped**, 변경 Python Ruff, 프론트 **20 passed**·타입 검사·빌드와 실제 로컬 관세 맵(뉴시스 1개/부족 안내)을 확인했다. 외부 API·DB는 미사용이며 프론트/배포·임베딩/코사인/MMR/검색 상한·요금제·공개 API 계약은 그대로다. 코드/문서는 미커밋이며 push·병합·운영 배포는 없다. 전체 뉴스 의미 품질을 검증한 것은 아니다. 상세 명령·입력·화면·방향별 한계는 정본 `docs/99-verification.md`의 후속 2 기록을 따른다.

아래 low_overlap 절은 이전 후속 기록이며 현재 동작에는 이 후속 2를 우선 적용한다.

## 뉴스맵 low_overlap 판단 보류 (2026-10-01 후속)

`article-api`의 `553d204` 기반 후속 수정이다. 반복 확인 근거가 부족한 `low_overlap`을 새 정보 발견과 구분해 `insufficient_evidence`로 판정한다. `Judgment.withheld`는 `short_text_covered`와 `low_overlap`만 명시적으로 보류한다. 중심 또는 주변 대표와 비교해 보류된 후보는 독립 주변 노드·중심/주변의 같은 소식 묶음에 모두 넣지 않고, 기존 진단 로그의 `withheld`에 집계한다. 다른 근거 부족 사유와 실제 추가 정보 판정은 기존대로다.

보류는 평가 후보 예산에는 포함되지만 표시 목표 개수에는 포함되지 않는다. 부족하면 아래 기존 20→50 후보·검색 단계·시간 예산으로 확장하고, 끝까지 부족하면 `/related`·`/graph` 모두 정상적인 적은 결과와 `selection.status="insufficient"`를 반환한다. API 응답 형식·요금제·임베딩·코사인·MMR·본문 수집은 유지한다. 고정 관세 기사 쌍과 mock 벡터 회귀 20개, 전체 **280 passed·3 skipped**(MongoDB opt-in), 변경 Python Ruff 검사를 로컬에서 통과했다. 외부 API·DB·운영 환경은 호출하지 않았다. 상세 근거는 정본 `docs/99-verification.md`의 low_overlap 후속 검증에 있다.

## 뉴스맵 같은 소식 묶음과 20→50 후보 확장 (2026-10-01 2차)

이미 정해진 중심 기사의 제목·NAVER description을 기준으로 주변 기사를 고른다. 최초 중심 기사 선정과 다단계 그래프 확장은 바꾸지 않았다. 1차(`0a462f8`)의 반복 판별은 제한된 동작 어휘(공개·출시 등)를 못 찾으면 무조건 "반복 아님"을 반환해 금리·국제유가처럼 제품이 아닌 경제 뉴스의 재보도가 주변을 차지했다. 이번에는 판별 규칙을 다시 쓰고, 반복 보도를 버리지 않고 **같은 소식 묶음**으로 응답한다.

### 처리 순서

후보 확보 → 같은 ID·같은 기사 URL 제외 → Gemini 임베딩·코사인 중심 연관도 → 최소 연관도 → 같은 소식 판별·묶음 → MMR → 표시 개수 제한 → 부족하면 예산 안에서 후보 확장 후 같은 기준으로 이어서 선정.

1. **최초 후보**: 같은 검색어로 저장된 메모리·MongoDB 기사를 **최근 검색 세션 → NAVER 원시 순위** 순서로 정렬해 고유 후보 `NEWS_MAP_INITIAL_CANDIDATES`(20)개를 평가한다. 기사 ID 순으로 자르지 않는다. 검색 시 `_search_rank`(원시 위치)·`_search_end`(요청한 마지막 원시 위치)·`_searched_at`를 저장하고 MongoDB 왕복에서도 보존한다.
2. **식별 중복**: 중심 자신과 같은 ID·정규화 URL(`url`, `naver_url`)만 제외한다. 제목·설명이 같아도 다른 매체 URL이면 별도 후보이며, 아래 판별에서 같은 소식으로 묶인다.
3. **연관도**: 기존 공식 `clip(0.9·max(cos,0) + 0.1·J − p)`과 최소 연관도(설정·요청 중 큰 값)를 그대로 쓴다. 벡터는 입력 해시·모델·차원·용도·전처리 버전이 같으면 재사용한다.
4. **같은 소식 판별**([repeated_coverage.py](app/agents/repeated_coverage.py)): 후보마다 중심과 직접 비교해 `repeat`(같은 사건·정보 대부분 반복) / `same_event_new_info`(같은 사건+의미 있는 추가 사실·분석) / `different_event`(관련 있지만 다른 사건·시점·결과) / `insufficient_evidence`(텍스트·시간·반복 확인 근거 부족) 중 하나로 판정한다. 중심과 `repeat`이면 중심 묶음, 명시적 보류 사유(`short_text_covered`·`low_overlap`)면 선정에서 제외한다. 나머지는 앞서 남은 대표와 같은 방식으로 비교해 보류하거나 대표 묶음에 넣거나 새 대표가 된다(완전 연결: 묶음의 모든 구성원과 같은 소식이어야 합류).
5. **MMR**: 묶음 대표끼리 `λ·중심 연관도 − (1−λ)·이미 고른 대표와의 최대 코사인`으로 고른다. 공개 `relevance_score`는 중심 연관도이며 MMR 점수가 아니다. 후보 확장 라운드는 이미 고른 주변 기사를 바꾸지 않고 뒤에 이어 붙인다.
6. **확장**: 고른 대표가 표시 목표(FREE/BASIC `min(limit,3)`, PAID `limit`)보다 적을 때만 ①남은 저장 후보 → ②중심 제목·설명·근거 키워드의 첫 보충 검색어 → ③원래 검색어의 다음 원시 위치(`_search_end + 1`부터) → ④나머지 보충 검색어 순으로 추가한다. 목표 도달·고유 후보 50개·외부 검색 단계 수·전체 시간 중 먼저 닿는 곳에서 멈춘다. 관련성 기준은 낮추지 않으며 끝까지 부족하면 적은 결과를 정상으로 반환한다.

### 같은 소식 판별 기준

- **같은 사건의 단서**: 제목의 경제 대상(국채 만기·국가별 금리, 기준금리, 주담대·신용대출·예금 금리, 유가·브렌트·WTI, 원유 재고·공급, 환율, 증시, 물가, 성장률 등), 제품 종류·모델명, 공유 제목 수치, 제목 핵심어 2개 이상. 회사명만 겹치는 것은 단서가 아니다. 어휘에 없는 사건도 제목·설명 겹침과 아래 충돌 검사로 판단하며, 어휘를 못 찾았다는 이유만으로 "반복 아님"을 반환하지 않는다.
- **다른 사건(보존)**: 서로 다른 모델명, 다른 진행 단계(공개 vs 가격 인상·판매 개시·결렬 vs 타결 등), 같은 대상의 반대 방향(상승 vs 하락, 인상 vs 동결), 같은 종류 제목 수치의 불일치, 제목 날짜·설명 첫 날짜의 명시적 불일치, 발행 시각 48시간 초과.
- **누락과 충돌 구분**: 한쪽에만 있는 수치·날짜는 충돌이 아니다. 수치는 반올림(5.3% ↔ 5.304%), 만·억·조 단위의 1% 이내 반올림(1209억 ↔ 1200억)을 같은 값으로 본다. 설명 스니펫 수치는 같은 라벨·단위의 값이 양쪽에 하나씩만 있을 때만 비교하며 2.5% 이내 보도 차이는 허용한다. "30일(현지시간)"과 "1일"처럼 12시간 안의 인접 일자는 시차로 본다.
- **추가 정보(보존)**: 제목에 새 경제 대상·제품·모델·분석 관점(전망·영향·분석·비교·반응·위험 등)이 있거나, 제목의 새 단어가 그 기사 설명에도 실제로 나오거나 영문·숫자 고유명(예: MZ, HBM4)인 경우(2개 이상·비율 기준), 설명에 새 대상·관점이 2개 이상이면서 중심 본문과 겹침이 낮은 경우. "처음" 같은 단어 하나, 한쪽에만 있는 배경 날짜·수치, 설명에 근거 없는 편집 제목 표현은 추가 정보로 보지 않는다.
- **반복 확인**: 위 조건을 모두 통과하고 코사인 0.92 이상·발행 시각 48시간 이내이며, 제목 유사도·설명 포함률·공유 구조 2개 이상·후보 제목 핵심어 전부 포함·매우 높은 의미 유사도(≈0.947) 중 하나가 있어야 `repeat`이다. 설명에 근거가 없는 새 제목 단어가 2개 이상이면 매우 높은 의미 유사도일 때만 반복으로 인정한다.
- **짧은 설명**: 40자 미만이면 판단 근거 부족이다. 제목·설명이 정규화 후 완전히 같은 전재 기사는 반복으로 묶고, 보이는 내용이 중심과 같지만 확인할 수 없는 기사는 묶지도 주변에 두지도 않는다(진단 로그 `withheld`). 새 내용이 있으면 주변 후보로 남긴다.
- 별칭은 `한은 → 한국은행`, 한자 표기(`美`, `弗`) 정규화만 한다. 낙농 문맥의 "원유"는 원유(석유) 대상으로 읽지 않지만 검색어 "원유"를 석유로 치환하지는 않는다.

규칙 사전과 임계값은 2026-10-01 실제 뉴스 표본으로 보정한 **초기값**이다(정본 `docs/99-verification.md`). 새 생성형 AI 호출은 추가하지 않았다. 기사 쌍 판별을 Flex 생성 모델로 하면 요청마다 수십 쌍의 지연·비용·비결정성이 생기기 때문이며, 필요하면 별도 상한·캐시 설계 후 후속 과제로 다룬다.

### 후보·호출·시간 상한

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `NEWS_MAP_INITIAL_CANDIDATES` | `20` | 최초 라운드 고유 후보 수(검색 순서) |
| `NEWS_MAP_MAX_CANDIDATES` | `50` | 요청당 평가하는 **고유 주변 후보 총상한**. 중심 제외, 같은 ID·URL 중복 제외. 저장 후보·원래 검색 다음 위치·보충 검색 후보를 합산 |
| `NEWS_MAP_SUPPLEMENT_MAX_SEARCHES` | `3` | 외부 검색 **단계** 상한(원래 검색 다음 위치 + 보충 검색어). 0이면 외부 검색 없음 |
| `NEWS_MAP_SUPPLEMENT_PAGE_SIZE` | `20` | 외부 검색 한 번의 NAVER 원시 요청 건수 |
| `NEWS_MAP_SUPPLEMENT_TIMEOUT_SECONDS` | `20` | 확장 전체(검색·분류 확인·저장·임베딩·재선정) 시간 |
| `NEWS_MAP_SUPPLEMENT_CACHE_TTL_SECONDS` | `60` | 검색 페이지 성공/빈 결과 프로세스 캐시. 실패는 5초 실패 캐시 |
| `NEWS_MAP_REPEAT_TITLE_SIMILARITY` | `0.50` | 반복 확인의 제목 문자 유사도 기준 |
| `NEWS_MAP_REPEAT_COSINE` | `0.92` | 반복의 필요조건, 단독 판정 금지 |
| `NEWS_MAP_REPEAT_TEXT_SIMILARITY` | `0.55` | 설명 포함률 기준 |
| `NEWS_MAP_REPEAT_SHORT_TEXT_SIMILARITY` | `0.85` | 짧은 설명에서 제목 유사도 기준 |
| `NEWS_MAP_REPEAT_MAX_HOURS` | `48` | 반복 인정 발행 시각 간격 |
| `NEWS_MAP_REPEAT_DESCRIPTION_MIN_CHARS` | `40` | 짧은 설명 경계 |
| `NEWS_MAP_REPEAT_NOVELTY_RATIO` | `0.25` | 제목 새 단어 비율 기준(공유 구조가 강하면 1.6배) |
| `NEWS_MAP_EMBEDDING_MODEL` / `DIMENSIONS` / `TASK_TYPE` | `gemini-embedding-001` / `768` / `SEMANTIC_SIMILARITY` | 뉴스맵 전용 벡터 |
| `NEWS_MAP_EMBEDDING_CONCURRENCY` | `3` | 벡터 생성 동시성 |
| `NEWS_MAP_MIN_RELEVANCE`, `KEYWORD_WEIGHT`, `ENTITY_ONLY_PENALTY`, `MMR_LAMBDA` | `0.65`, `0.10`, `0.10`, `0.70` | 기존 연관도·MMR 설정 |

`NEWS_MAP_CANDIDATE_LIMIT`(40)과 `NEWS_MAP_SUPPLEMENT_CANDIDATE_LIMIT`(20)은 하나의 총상한으로 대체해 삭제했다. 최초 수가 총상한보다 크면 설정 검증이 실패한다. 원시 검색 결과 수와 필터 후 고유 후보 수는 따로 집계하며, 고유 후보를 채우려고 원시 검색을 반복하지 않는다(단계 수·시간 상한). 원래 검색 이어 받기는 저장된 `_search_end` 다음 위치에서 시작하므로 페이지 크기가 바뀌어도 원시 위치가 겹치거나 빠지지 않는다. 다만 실시간 검색 순위가 그사이 바뀌는 것까지는 막을 수 없다.

확장 후보는 NAVER 분류 확인(정치·사회 제외)을 필요한 수만큼만 하고, 규칙 메타데이터와 뉴스맵 벡터만 만든다. **Diffbot 본문 추출·리포트 생성·Flex 메타데이터·RAG 임베딩은 실행하지 않는다.** 원래 검색 이어 받기 결과는 같은 검색 세션 후보로 저장돼 다음 요청은 외부 검색 없이 재사용한다. 레거시 GDELT/NewsAPI(실패 시 샘플 대체)에서는 외부 확장을 하지 않는다.

### API 계약(`/related`, `/graph` 공통)

- 새 쿼리 `expand`(기본 `true`). `false`면 최초 후보만 평가하고, 더 찾을 수 있으면 `selection.status="expandable"`을 준다. 프론트는 이 결과를 먼저 그린 뒤 `expand=true`로 다시 요청한다.
- 주변 카드에 `same_story`(같은 소식 다른 보도 카드 목록, 발행 순)·`same_story_total`. `/related`는 `center_same_story`·`center_same_story_total`, `/graph`는 `center_node.same_story`를 쓴다. 묶인 카드는 기존 카드 필드(제목·설명·출처·발행 시각·원문 링크·썸네일·키워드·카테고리)만 있고 **점수·거리 필드가 없다.** 묶음은 주변 노드 수를 소비하지 않는다.
- `selection`: `status`(`complete`·`insufficient`·`partial`·`expandable`), `reason`(`exhausted`·`candidate_limit`·`search_limit`·`no_source`·`timeout`·`search_failed`·`embedding_failed`·`storage_failed`), `requested`(요금제 적용 후 목표), `returned`.
- 최초 라운드의 임베딩 실패는 503이다. 확장 중 검색 실패·시간 초과·확장 후보 임베딩 실패·저장 실패는 이미 확정된 최초 결과를 200 `partial`로 반환한다(1차의 502/503/504 오류 응답에서 변경).
- FREE/BASIC 최대 3건·점수 `null`, PAID 요청 limit·중심 연관도 점수는 그대로다. `tier`는 쿼리이며 실제 구독 인증은 아니다.

### 진단 로그

요청마다 `econmind.news_map` 로거가 한 줄을 남긴다: 저장 후보 수, 최초 평가 수, 외부 검색 단계·실제 호출·캐시 적중, 원시 결과·원시 중복·분류 제외, 라운드, 단계별 추가 수, 상태·이유, 평가 수, 연관도 미달, 중심 반복·묶음·보류 수, 그룹·선택 수, 임베딩 실제 호출·재사용 수, 처리 시간. 기사 ID(해시) 외에 제목·검색어·키·벡터는 남기지 않는다.

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
| `GET /api/v1/keywords/related?q=` | 검색어별 연관 키워드 (검색 기사 키워드 빈도 집계 + 추천 목록 보충) |
| `GET /api/v1/news/search`, `/cards` | 뉴스 카드·description·키워드·카테고리 |
| `GET /api/v1/news/{id}/source` | 출처·description. 전체 본문은 반환하지 않음 |
| `GET /api/v1/news/{id}/thumbnail`, `/graph`, `/related` | 썸네일·뉴스맵·연관 뉴스 |
| `POST /api/v1/news/selections`, `GET /api/v1/news/{id}/relations` | `tier=PAID` 쿼리 조건의 선택 묶음·관계 점수. 사용자 인증은 아직 없음 |
| `POST /api/v1/reports`, `GET /api/v1/reports/{id}` | 선택 기사 본문 추출·리포트 생성 및 조회. `news_ids`(1~5)면 비동기(202 → `status`·`stage` 폴링), `news_id`면 기존 동기(201) |
| `POST /api/v1/strategies`, `GET /api/v1/strategies/{id}` | 전략 생성 및 조회 |
| `POST /jobs`, `GET /jobs/{id}` | 기존 NATS·Redis 작업 큐 |
| `POST /analyze` | 기존 분석 호환 경로 |
| `GET`·`DELETE /api/v1/session`, `POST /api/v1/session/mindmap/expand`·`collapse`, `DELETE /api/v1/session/mindmap` | 쿠키 기반 세션 식별·마인드맵 상태. 아래 「세션」 참고 |

전체 요청·응답 형식은 [뉴스 세션 API 명세](https://github.com/DKU-CE-Capstone-Project/econmind-docs/blob/main/docs/07-api-spec.md)와 해당 브랜치 실행 중인 `/docs`를 참고한다. `/health`는 실제 DB ping이 아니며 `/ready`가 이를 검사한다. 현재 프론트 `article-api`는 `/related?tier=FREE&expand=false` 결과를 먼저 서버 순서대로 표시하고, `expandable`이면 같은 중심으로 `expand=true`를 다시 요청한다. `/graph`로 보충하지 않는다. 주변 기사 선정만 제한적으로 재개했으며, 키워드맵 알고리즘·다단계 확장·세션 상태 연동은 계속 보류한다. NCP 인증·응답 오류는 안전한 메시지로 전달하며, 검색 오류를 mock 뉴스로 대체하지 않는다.

## 뉴스맵 프로토타입 연결 계약 (2026-10-04)

정본 econmind-docs `docs/10-newsmap-api-contract-draft.md`의 확정 결정(D1~D6)을 구현했다. 기존 v1 응답은 바꾸지 않고 필드·파라미터만 추가했다.

- `/related?exclude_ids=`(최대 100): 맵에 이미 있는 기사를 선정 전에 후보에서 뺀다. 하위 펼치기·재검색은 같은 엔드포인트를 쓰고, 결과 0건도 정상(200)이다. `selection.excluded`는 후보 중 제외된 수.
- `/keywords/related`: 검색 결과 20건의 `keywords`를 기사 수로 집계(2건 이상, 별칭 정규화, 검색어 제외). 부족분은 추천 목록으로 채우고 `source`로 구분한다. 추가 LLM 호출 없음. 검색 실패는 502.
- `POST /reports {news_ids}`: `app/report_jobs.py`가 응답 후 같은 API 프로세스에서 처리한다. 상태는 Redis `report_job:{id}`(TTL 1시간, Redis 없으면 메모리). 모든 근거 본문을 Diffbot으로 추출(동시 2, 재시도 0)하고 1건 이상 추출되면 생성, 나머지는 `body_status=description_only`. LLM 1회로 리포트·`stock_impacts`·`strategy`를 받고, 근거에 이름이 없는 종목·허용되지 않은 값은 버린다. LLM 실패는 `is_fallback=true`이며 재사용하지 않는다. 완료본은 `reports`에 `schema_version=2`(`sections`·`evidence`·`reuse_key`)로 저장되고 같은 기사 집합(순서 무관)이면 200으로 재사용한다. 세션당 진행 중 1건(409). 최종 요청 한도·응답 코드는 M5(SEC-05) 담당과 확정한다.

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

2026-10-01 2차: Python 3.13.5 프로젝트 가상환경에서 기본 실행 **260 passed, 3 skipped**, 별도 로컬 MongoDB 4.4.18(127.0.0.1:27028, 테스트마다 고유 DB 생성·삭제) 지정 시 **263 passed**. 변경 Python 16개 파일 Ruff 통과(`app/database.py`의 기존 8건은 이번에 수정하지 않은 줄), `compileall`·`uv build --offline --wheel` 성공. Gemini·NAVER·Diffbot은 fixture/mock이며 기본 테스트는 외부 소켓을 차단한다. [판별·묶음·MMR 테스트](tests/test_news_map_diversity.py)·[확장·상태·캐시·API 계약 테스트](tests/test_news_map_expansion.py)의 한국어 기사는 직접 작성한 가상 사례이며 실뉴스 품질 평가가 아니다. 실뉴스 표본 비교와 남은 한계는 정본 `docs/99-verification.md`에 별도로 기록한다.

```bash
NEWS_MAP_TEST_MONGODB_URI=mongodb://127.0.0.1:27028 .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app tests
uv build --offline --wheel --out-dir /tmp/econmind-news-map-wheels-20261001
```

Ruff 대상: `app/agents/{article_embeddings,graph_builder,naver_client,news_fetcher,news_map,related_candidates,related_selector,repeated_coverage}.py`, `app/{config,main,schemas}.py`, `app/api/v1/news.py`, `tests/{test_news_map_diversity,test_news_map_expansion,test_news_map_mongo,test_related_selection}.py`. 아래 이전 결과는 과거 기록으로 보존한다. 앱 Docker 이미지 빌드·Python 3.11 실행·운영 배포는 이번에 수행하지 않았다.

2026-10-01 1차(`0a462f8`): 238 passed(로컬 MongoDB 3개 포함). 1차의 `tests/test_related_supplement.py`는 2차 확장 설계로 대체되어 `tests/test_news_map_expansion.py`가 되었다.

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
