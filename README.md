# 캡스톤 백엔드 — 실시간 뉴스 기반 멀티 에이전트 투자 판단 지원 시스템

> **FastAPI + Google Gemini** 기반 뉴스 수집·분석·투자 전략 생성 백엔드 서버

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [디렉터리 구조](#3-디렉터리-구조)
4. [기술 스택](#4-기술-스택)
5. [에이전트 파이프라인 상세](#5-에이전트-파이프라인-상세)
6. [데이터 흐름도](#6-데이터-흐름도)
7. [API 명세](#7-api-명세)
8. [데이터 모델](#8-데이터-모델)
9. [인메모리 캐시 설계](#9-인메모리-캐시-설계)
10. [티어 권한 시스템](#10-티어-권한-시스템)
11. [환경 변수](#11-환경-변수)
12. [로컬 실행 방법](#12-로컬-실행-방법)
13. [테스트](#13-테스트)
14. [향후 개발 계획](#14-향후-개발-계획)

---

## 1. 프로젝트 개요

사용자가 투자 관련 **키워드**를 입력하면 실시간으로 영문 뉴스를 수집하고, AI가 한국어로 요약·분석하여 **마인드맵형 뉴스 탐색 → AI 리포트 → 투자 전략**까지 자동으로 생성해주는 시스템의 백엔드.

| 항목 | 내용 |
|------|------|
| 개발 기간 | 2025년 캡스톤 디자인 |
| 아키텍처 | FastAPI 모놀리식 (MVP), 추후 마이크로서비스 분리 예정 |
| 데이터 소스 | NewsAPI.org (영문 뉴스, 무료 100req/일) |
| AI 모델 | Google Gemini (`gemini-flash-latest`) |
| 언어 | Python 3.11+ |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│                   (http://localhost:5173)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  Legacy API  │    │           /api/v1 Router             │   │
│  │ POST /analyze│    │                                      │   │
│  └──────┬───────┘    │  /news/*   /keywords/*               │   │
│         │            │  /reports/* /strategies/*            │   │
│         │            └──────────────────┬───────────────────┘   │
│         │                               │                       │
│         └───────────────┬───────────────┘                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Agent Pipeline                          │   │
│  │                                                          │   │
│  │  ① news_fetcher  →  ② filter_agent  →  ③ summarizer    │   │
│  │       ↓                                      ↓           │   │
│  │  NewsAPI.org                         ④ keyword_expander  │   │
│  │  (+ KO→EN 번역)                               ↓           │   │
│  │                                       ⑤ report_generator │   │
│  │                                               ↓           │   │
│  │                                      ⑥ strategy_generator│   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               In-Memory Cache (store.py)                 │   │
│  │  news_cache │ report_cache │ strategy_cache │ selection  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               Google Gemini API                          │   │
│  │         (요약 / 리포트 / 전략 생성 — LLM 호출)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 디렉터리 구조

```
backend/
├── app/
│   ├── main.py                  # FastAPI 앱 진입점, CORS, 라우터 등록
│   ├── config.py                # 환경변수 로드 (pydantic-settings)
│   ├── schemas.py               # Pydantic 요청/응답 모델 (22개)
│   ├── store.py                 # 인메모리 캐시 (news/report/strategy/selection)
│   ├── utils.py                 # news_id 생성, 캐싱 헬퍼, 티어 체크
│   │
│   ├── api/
│   │   ├── routes.py            # POST /analyze (legacy, 프론트 호환용)
│   │   └── v1/
│   │       ├── news.py          # /api/v1/news/* — 뉴스 관련 8개 엔드포인트
│   │       ├── keywords.py      # /api/v1/keywords/recommended
│   │       ├── reports.py       # /api/v1/reports — AI 리포트 생성·조회
│   │       └── strategies.py    # /api/v1/strategies — 투자 전략 생성·조회
│   │
│   └── agents/
│       ├── orchestrator.py      # 분석 파이프라인 총괄 (fetch→filter→summarize→expand)
│       ├── news_fetcher.py      # NewsAPI.org 호출 + KO→EN 키워드 번역
│       ├── filter_agent.py      # URL/제목 중복 제거 (overlap coefficient)
│       ├── summarizer.py        # Gemini 기사 요약 (한국어 1문장)
│       ├── keyword_expander.py  # Gemini 관련 키워드 5~7개 생성
│       ├── graph_builder.py     # 마인드맵 노드·엣지 데이터 생성
│       ├── report_generator.py  # Gemini AI 투자 분석 리포트 생성
│       ├── strategy_generator.py# Gemini 투자 전략 생성 (risk/period 기반)
│       └── llm.py               # Gemini 클라이언트 래퍼 (에러 시 "" 반환)
│
├── fixtures/
│   └── news_mock.json           # 오프라인 테스트용 목 뉴스 데이터 (8건)
│
├── tests/
│   └── test_smoke.py            # 통합 스모크 테스트 (pytest)
│
└── pyproject.toml               # 의존성 + 프로젝트 메타데이터
```

---

## 4. 기술 스택

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| 웹 프레임워크 | FastAPI | ≥0.115 | REST API 서버 |
| ASGI 서버 | uvicorn | ≥0.30 | 비동기 HTTP 서버 |
| HTTP 클라이언트 | httpx | ≥0.27 | NewsAPI.org 비동기 호출 |
| 데이터 검증 | pydantic / pydantic-settings | ≥2.7 | 스키마 정의 및 환경변수 로드 |
| AI 모델 | google-genai | latest | Gemini API (요약·리포트·전략) |
| 뉴스 데이터 | NewsAPI.org | v2 | 영문 뉴스 수집 |
| 테스트 | pytest + pytest-asyncio | ≥8.0 | 비동기 통합 테스트 |
| 린터 | ruff | ≥0.6 | 코드 스타일 검사 |

---

## 5. 에이전트 파이프라인 상세

### 5.1 전체 파이프라인 흐름

```
키워드 입력 (예: "트럼프 관세")
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ① news_fetcher.py                                   │
│                                                     │
│  KO→EN 번역: "트럼프 관세" → "trump tariff"          │
│  NewsAPI 호출: GET /v2/everything                    │
│   - searchIn=title,description (관련 기사 필터링)    │
│   - sortBy=publishedAt, language=en                  │
│   - 결과 없으면 language 제한 제거 후 재시도          │
│  후처리: 검색어가 title/description에 포함된 것만 유지│
│                                                     │
│  Output: List[{title, url, source, published_at,    │
│                description, thumbnail_url}]          │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ ② filter_agent.py  (중복 제거, LLM 호출 없음)        │
│                                                     │
│  Step 1 — URL 완전 일치 중복 제거                    │
│  Step 2 — 제목 토큰 overlap coefficient 계산         │
│            |A∩B| / min(|A|,|B|) ≥ 0.4 → 중복으로 판단│
│            (Reuters/Bloomberg 동일 사건 기사 제거)   │
│                                                     │
│  Output: 중복 제거된 기사 리스트                     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ ③ summarizer.py  (Gemini 호출)                      │
│                                                     │
│  최대 6건 asyncio.gather() 병렬 요약                 │
│  프롬프트: "한국어 1문장, 50자 이내,                 │
│            투자 판단에 유의미한 사실 중심"            │
│  Fallback: Gemini 실패 시 description 앞 80자       │
│                                                     │
│  Output: 각 기사에 "summary" 필드 추가               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ ④ keyword_expander.py  (Gemini 호출)                │
│                                                     │
│  프롬프트: "요약 목록을 보고 관련 키워드 5~7개 한국어│
│            콤마로 출력, 입력 키워드 제외"             │
│  Fallback: 빈도 기반 추출 + 영→한 레이블 변환        │
│            (nvidia→엔비디아, tariff→관세 등)         │
│                                                     │
│  Output: ["관세", "무역전쟁", "달러", ...]           │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
               AnalyzeResponse 반환 + 캐시 저장
```

### 5.2 리포트/전략 생성 파이프라인

```
POST /api/v1/reports
    │  {news_id, related_news_ids, ticker_symbols}
    ▼
┌─────────────────────────────────────────────────────┐
│ ⑤ report_generator.py  (Gemini 호출)                │
│                                                     │
│  입력: 기준 뉴스 title+summary + 연관 뉴스 요약 5건  │
│  프롬프트: JSON 형식으로 한국어 리포트 작성 요청      │
│  출력 JSON:                                         │
│   - title: 리포트 제목                              │
│   - summary: 핵심 요약 (200자)                      │
│   - event_analysis: 사건 분석 (300자)               │
│   - market_impact: 시장 영향 분석 (300자)            │
│   - risk_factors: 리스크 요인 리스트                 │
│  Fallback: "(AI 분석 준비 중)" 더미 리포트           │
│                                                     │
└──────────────────────────┬──────────────────────────┘
                           │ report_id 발급 후 캐시 저장
                           ▼
POST /api/v1/strategies
    │  {report_id, risk_level, period}
    ▼
┌─────────────────────────────────────────────────────┐
│ ⑥ strategy_generator.py  (Gemini 호출)              │
│                                                     │
│  입력: 리포트 요약 + risk_level + period             │
│  risk_level: low(보수적) / medium(중립) / high(공격적)│
│  period: short(1-3개월) / mid(3-12개월) / long(1년+) │
│  출력 JSON:                                         │
│   - expected_return: 기대 수익률 (%)                 │
│   - strategy_items: 종목별 {ticker, action, reason} │
│     action: buy / hold / sell / watch               │
│  Fallback: 삼성전자·SK하이닉스·엔비디아 더미 전략    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.3 마인드맵 그래프 생성

```
GET /api/v1/news/{news_id}/graph
    │
    ▼
① store.news_cache에서 center article 조회
    │
    ▼
② 동일 검색 키워드로 캐싱된 기사 먼저 확인
   (3건 이상이면 API 재호출 없이 반환)
    │ (3건 미만)
    ▼
③ 원래 검색 키워드로 NewsAPI 재호출 (최대 12건)
    │
    ▼
④ graph_builder.build_graph()
   ┌─────────────────────────────────────┐
   │  distance 계산:                     │
   │  center ↔ other 제목 토큰 overlap   │
   │  ≥ 0.4 → distance = 1              │
   │  < 0.4 → distance = 2              │
   └─────────────────────────────────────┘
    │
    ▼
GraphResponse {
  center_node: {news_id, title, summary, distance:0, is_center:true}
  nodes: [{news_id, title, summary, distance, is_center}, ...]
  edges: [{source, target, relation_type, distance}, ...]
}
```

---

## 6. 데이터 흐름도

### 6.1 검색 → 카드 표시

```
[Client]                [FastAPI]              [NewsAPI]    [Gemini]
   │                       │                      │             │
   │ GET /news/search?q=X  │                      │             │
   │──────────────────────▶│                      │             │
   │                       │ GET /v2/everything   │             │
   │                       │─────────────────────▶│             │
   │                       │ [{title,url,...}]    │             │
   │                       │◀─────────────────────│             │
   │                       │                      │             │
   │                       │ generate(요약 프롬프트)│             │
   │                       │──────────────────────────────────▶│
   │                       │ "엔비디아는..."       │             │
   │                       │◀──────────────────────────────────│
   │                       │                      │             │
   │                       │ cache_articles()     │             │
   │                       │ (news_id 부여, store 저장)         │
   │                       │                      │             │
   │ {news_cards[], total} │                      │             │
   │◀──────────────────────│                      │             │
```

### 6.2 리포트 → 전략 생성

```
[Client]               [FastAPI / Cache]            [Gemini]
   │                         │                          │
   │ POST /reports           │                          │
   │ {news_id}               │                          │
   │────────────────────────▶│                          │
   │                         │ store.news_cache[news_id]│
   │                         │ → center article 조회    │
   │                         │                          │
   │                         │ generate(리포트 프롬프트) │
   │                         │─────────────────────────▶│
   │                         │ {title, summary, ...}    │
   │                         │◀─────────────────────────│
   │                         │                          │
   │                         │ store.report_cache[uuid] │
   │ {report_id, status}     │                          │
   │◀────────────────────────│                          │
   │                         │                          │
   │ POST /strategies        │                          │
   │ {report_id, risk, period}                          │
   │────────────────────────▶│                          │
   │                         │ store.report_cache 조회  │
   │                         │ generate(전략 프롬프트)  │
   │                         │─────────────────────────▶│
   │                         │ {expected_return, items} │
   │                         │◀─────────────────────────│
   │                         │                          │
   │ {strategy_id, status}   │ store.strategy_cache 저장│
   │◀────────────────────────│                          │
```

---

## 7. API 명세

### Base URL

```
http://localhost:8000/api/v1
```

### 공통 응답 코드

| 코드 | 의미 |
|------|------|
| 200 | 성공 |
| 201 | 생성 완료 |
| 400 | 잘못된 요청 (파라미터 오류) |
| 403 | 권한 없음 (티어 부족) |
| 404 | 리소스 없음 (캐시 미존재) |
| 500 | 서버 내부 오류 |

---

### 7.1 뉴스 검색

```
GET /api/v1/news/search
```

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `q` | string | ✅ | — | 검색어 (한국어/영어 모두 지원) |
| `page` | int | | 1 | 페이지 번호 |
| `size` | int | | 10 | 페이지당 기사 수 (최대 50) |
| `sort` | string | | relevance | `relevance` \| `latest` |

**응답 예시**
```json
{
  "news_cards": [
    {
      "news_id": "82cb9c6ab34c",
      "title": "Nvidia appoints former Goldman Sachs Vice Chairman to board",
      "summary": "엔비디아가 재무 감독 강화를 위해 골드만삭스 전 부회장을 이사회에 선임했다.",
      "thumbnail_url": "https://...",
      "source_name": "Crypto Briefing",
      "published_at": "2026-05-09T05:41:08Z",
      "related_stock_names": []
    }
  ],
  "total_count": 6
}
```

---

### 7.2 추천 키워드

```
GET /api/v1/keywords/recommended?limit=8
```

**응답 예시**
```json
{
  "keywords": [
    { "keyword": "엔비디아",    "category": "기업",      "rank": 1 },
    { "keyword": "삼성전자",    "category": "기업",      "rank": 2 },
    { "keyword": "트럼프 관세", "category": "경제·정치", "rank": 3 },
    { "keyword": "반도체",      "category": "산업",      "rank": 4 }
  ]
}
```

---

### 7.3 마인드맵 그래프

```
GET /api/v1/news/{news_id}/graph?depth=2&limit=10
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `depth` | int | 2 | 탐색 깊이 (1~3) |
| `limit` | int | 10 | 최대 노드 수 |
| `include_distance` | bool | true | 거리값 포함 여부 |

**응답 예시**
```json
{
  "center_node": {
    "news_id": "82cb9c6ab34c",
    "title": "Nvidia appoints former Goldman Sachs...",
    "summary": "엔비디아가 골드만삭스 전 부회장을 선임했다.",
    "distance": 0,
    "is_center": true
  },
  "nodes": [
    { "news_id": "1b842bd96d78", "title": "NVIDIA Stock Climbs...", "distance": 2, "is_center": false }
  ],
  "edges": [
    { "source": "82cb9c6ab34c", "target": "1b842bd96d78", "relation_type": "related", "distance": 2 }
  ]
}
```

---

### 7.4 연관 뉴스

```
GET /api/v1/news/{news_id}/related?tier=FREE&limit=3
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `limit` | int | 10 | 조회 수 (FREE는 max 3 강제 적용) |
| `min_relevance` | float | 0.0 | 최소 연관도 (0~1) |
| `include_score` | bool | false | 연관도 점수 포함 여부 |
| `tier` | string | FREE | `FREE` \| `BASIC` \| `PAID` |

**FREE vs PAID 차이**

| 항목 | FREE | PAID |
|------|------|------|
| 최대 기사 수 | 3건 | 제한 없음 |
| relevance_score | null | 0.0~1.0 반환 |

---

### 7.5 뉴스 원문 출처

```
GET /api/v1/news/{news_id}/source
```

```json
{
  "news_id": "82cb9c6ab34c",
  "source_name": "Crypto Briefing",
  "source_url": "https://cryptobriefing.com/...",
  "published_at": "2026-05-09T05:41:08Z",
  "original_title": "Nvidia appoints former Goldman Sachs Vice Chairman to board"
}
```

---

### 7.6 썸네일

```
GET /api/v1/news/{news_id}/thumbnail
```

```json
{
  "news_id": "82cb9c6ab34c",
  "thumbnail_url": "https://images.unsplash.com/...",
  "fallback_used": true
}
```

> `fallback_used: true` — NewsAPI가 이미지를 제공하지 않아 Unsplash 대체 이미지 사용

---

### 7.7 AI 리포트 생성

```
POST /api/v1/reports
Content-Type: application/json
```

**요청**
```json
{
  "news_id": "82cb9c6ab34c",
  "related_news_ids": ["1b842bd96d78", "c94f1d04f18d"],
  "ticker_symbols": ["NVDA", "005930"],
  "language": "ko",
  "report_type": "investment"
}
```

**응답 (생성)**
```json
{
  "report_id": "a53f44e8-7624-4920-bc3b-c8e4242361b6",
  "status": "completed",
  "created_at": "2026-05-10T06:21:04.575738+00:00"
}
```

---

### 7.8 AI 리포트 조회

```
GET /api/v1/reports/{report_id}
```

```json
{
  "report_id": "a53f44e8-...",
  "title": "엔비디아 이사회 변화와 AI 반도체 투자 전망",
  "summary": "엔비디아가 재무 전문가를 이사회에 선임하며 거버넌스를 강화했다...",
  "event_analysis": "골드만삭스 전 부회장 수잔 노라 존슨 선임은...",
  "market_impact": "엔비디아 주가는 AI 칩 수요 증가와 이사회 안정화로...",
  "related_stocks": ["NVDA", "005930"],
  "evidence_news": [
    { "news_id": "82cb9c6ab34c", "title": "Nvidia appoints..." }
  ],
  "risk_factors": ["AI 버블 붕괴 가능성", "반도체 수출 규제 강화", "경쟁사 기술 추격"],
  "created_at": "2026-05-10T06:21:04.575738+00:00"
}
```

---

### 7.9 투자 전략 생성

```
POST /api/v1/strategies
Content-Type: application/json
```

**요청**
```json
{
  "report_id": "a53f44e8-...",
  "risk_level": "medium",
  "period": "short",
  "strategy_type": "simulation"
}
```

**응답 (생성)**
```json
{
  "strategy_id": "b576c0f8-...",
  "status": "completed",
  "created_at": "2026-05-10T06:22:11.123456+00:00"
}
```

---

### 7.10 투자 전략 조회

```
GET /api/v1/strategies/{strategy_id}
```

```json
{
  "strategy_id": "b576c0f8-...",
  "expected_return": 12.5,
  "risk": "medium",
  "period": "short",
  "strategy_summary": "AI 반도체 수요 강세를 고려한 단기 성장주 중심 포트폴리오",
  "strategy_items": [
    {
      "ticker": "NVDA",
      "stock_name": "엔비디아",
      "action": "buy",
      "reason": "AI 데이터센터 GPU 수요 지속 증가, 이사회 안정화"
    },
    {
      "ticker": "005930",
      "stock_name": "삼성전자",
      "action": "hold",
      "reason": "HBM 공급 증가로 메모리 가격 압박, 관망 권장"
    }
  ],
  "created_at": "2026-05-10T06:22:11.123456+00:00"
}
```

---

### 7.11 유료 — 뉴스 카드 선택 저장

```
POST /api/v1/news/selections?tier=PAID
```

**요청**
```json
{
  "news_ids": ["82cb9c6ab34c", "1b842bd96d78"],
  "selection_type": "report_source"
}
```

---

### 7.12 유료 — 연관도 점수

```
GET /api/v1/news/{news_id}/relations?target_news_ids=id1,id2&tier=PAID
```

```json
{
  "relations": [
    {
      "source_news_id": "82cb9c6ab34c",
      "target_news_id": "1b842bd96d78",
      "relevance_score": 0.571,
      "relation_reason": "공통 키워드 4개 공유",
      "shared_keywords": ["nvidia", "stock", "chip", "market"]
    }
  ]
}
```

---

### 7.13 레거시 (호환성 유지)

```
POST /analyze
Content-Type: application/json

{ "keyword": "트럼프 관세" }
```

---

## 8. 데이터 모델

### 8.1 내부 기사 dict (store에 저장되는 형태)

```python
{
    "news_id": "82cb9c6ab34c",        # URL MD5[:12] — 결정론적 ID
    "title": "Nvidia appoints...",
    "url": "https://...",
    "source": "Crypto Briefing",
    "published_at": "2026-05-09T05:41:08Z",
    "description": "원문 설명",
    "summary": "엔비디아가 골드만삭스 전 부회장을...",  # Gemini 생성
    "thumbnail_url": "https://...",   # NewsAPI urlToImage (없으면 "")
    "_search_keyword": "엔비디아",    # 재검색용 원래 키워드
}
```

### 8.2 news_id 생성 규칙

```python
import hashlib

def make_news_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

# 예) "https://cryptobriefing.com/nvidia-appoints-..."
#   → "82cb9c6ab34c"
```

- **결정론적**: 같은 URL은 항상 같은 ID
- **URL-safe**: 영소문자+숫자 12자
- **짧음**: 경로 파라미터로 사용하기 적합

---

## 9. 인메모리 캐시 설계

```python
# app/store.py
news_cache:      dict[str, dict]  # news_id → 기사 dict
report_cache:    dict[str, dict]  # report_id (UUID4) → 리포트
strategy_cache:  dict[str, dict]  # strategy_id (UUID4) → 전략
selection_cache: dict[str, dict]  # selection_id (UUID4) → 선택 묶음
```

**특성**

| 항목 | 내용 |
|------|------|
| 생존 범위 | 서버 프로세스 단위 (재시작 시 초기화) |
| 스레드 안전성 | uvicorn 단일 워커 기준 안전 |
| 메모리 관리 | MVP — LRU/TTL 미적용. 장기 운영 시 추가 필요 |
| 향후 대체 | Redis 또는 PostgreSQL로 교체 예정 |

**캐시 활용 흐름**
```
/news/search 호출
    → run_analysis() 실행
    → cache_articles() 호출
    → news_cache[news_id] = article  ← 저장

/news/{id}/graph 호출
    → news_cache[news_id] 조회       ← 재활용
    → 동일 키워드 캐싱 기사 반환
    → (부족 시) NewsAPI 재호출
```

---

## 10. 티어 권한 시스템

MVP에서는 인증 없이 쿼리 파라미터로 티어를 전달합니다.

```
?tier=FREE   (기본값)
?tier=BASIC
?tier=PAID
```

**티어 계층**

```
PAID  ▶  BASIC  ▶  FREE
  │          │        │
선택 저장   리포트   검색/카드
연관도 점수  전략    마인드맵
무제한 연관              연관 3개
```

**구현**
```python
# app/utils.py
_TIER_ORDER = {"FREE": 0, "BASIC": 1, "PAID": 2}

def tier_ok(user_tier: str, required: str) -> bool:
    return _TIER_ORDER.get(user_tier.upper(), 0) \
        >= _TIER_ORDER.get(required.upper(), 0)
```

> 향후 JWT 기반 인증으로 교체 시 `tier_ok()` 함수 내부만 변경하면 됩니다.

---

## 11. 환경 변수

`.env` 파일을 프로젝트 루트(`Capstone/`)에 생성합니다.

```env
NEWSAPI_KEY=your_newsapi_org_key       # NewsAPI.org 발급 키
GOOGLE_API_KEY=your_google_ai_key      # Google AI Studio 발급 키
USE_MOCK_NEWS=false                    # true로 설정 시 fixture 사용
```

| 변수 | 필수 | 없을 때 동작 |
|------|------|-------------|
| `NEWSAPI_KEY` | 권장 | `USE_MOCK_NEWS=true`로 강제 전환, fixture 뉴스 반환 |
| `GOOGLE_API_KEY` | 권장 | Gemini 호출 건너뜀, 설명 기반 fallback 사용 |
| `USE_MOCK_NEWS` | — | 기본 `false` |

**Gemini 없을 때 fallback 동작**

| 기능 | Fallback |
|------|----------|
| 기사 요약 | `description` 앞 80자 잘라서 반환 |
| 관련 키워드 | 제목·설명에서 빈도 기반 추출 (영→한 레이블 변환) |
| AI 리포트 | `"(AI 분석 준비 중)"` 텍스트 더미 리포트 |
| 투자 전략 | 삼성전자·SK하이닉스·엔비디아 고정 더미 전략 |

---

## 12. 로컬 실행 방법

### 요구사항

- Python 3.11+
- NewsAPI.org 무료 API 키 ([발급](https://newsapi.org/register))
- Google AI Studio API 키 ([발급](https://aistudio.google.com))

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/DKU-CE-Capstone-Project/capstone-backend.git
cd capstone-backend

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -e .

# 4. 환경 변수 설정 (프로젝트 루트에 .env 파일 생성)
cat > ../.env << 'EOF'
NEWSAPI_KEY=your_newsapi_key_here
GOOGLE_API_KEY=your_google_api_key_here
USE_MOCK_NEWS=false
EOF

# 5. 서버 실행
uvicorn app.main:app --reload --port 8000
```

### 동작 확인

```bash
# 헬스체크
curl http://localhost:8000/health
# → {"status": "ok"}

# 추천 키워드
curl "http://localhost:8000/api/v1/keywords/recommended?limit=5"

# 뉴스 검색 (한국어 지원)
curl "http://localhost:8000/api/v1/news/search?q=엔비디아&size=3"

# Swagger UI
open http://localhost:8000/docs
```

### 엔드-투-엔드 테스트 (검색→리포트→전략)

```bash
# 1. 검색하여 news_id 획득
NEWS_ID=$(curl -s "http://localhost:8000/api/v1/news/search?q=엔비디아&size=3" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['news_cards'][0]['news_id'])")
echo "NEWS_ID=$NEWS_ID"

# 2. 그래프 조회
curl -s "http://localhost:8000/api/v1/news/$NEWS_ID/graph?limit=5" | python3 -m json.tool

# 3. 리포트 생성
REPORT_ID=$(curl -s -X POST http://localhost:8000/api/v1/reports \
  -H "Content-Type: application/json" \
  -d "{\"news_id\":\"$NEWS_ID\",\"language\":\"ko\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['report_id'])")

# 4. 리포트 조회
curl -s "http://localhost:8000/api/v1/reports/$REPORT_ID" | python3 -m json.tool

# 5. 전략 생성 및 조회
STRAT_ID=$(curl -s -X POST http://localhost:8000/api/v1/strategies \
  -H "Content-Type: application/json" \
  -d "{\"report_id\":\"$REPORT_ID\",\"risk_level\":\"medium\",\"period\":\"short\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['strategy_id'])")
curl -s "http://localhost:8000/api/v1/strategies/$STRAT_ID" | python3 -m json.tool
```

---

## 13. 테스트

```bash
# 전체 테스트 실행
cd backend
source .venv/bin/activate
pytest tests/ -v
```

**테스트 커버리지**

| 테스트 | 내용 |
|--------|------|
| `test_health` | `GET /health` → `{"status":"ok"}` |
| `test_analyze_with_mocked_news` | URL 중복 제거, 제목 유사도 필터, 응답 구조 검증 |

> 통합 테스트는 `USE_MOCK_NEWS=true` 강제 설정 + Gemini API 없이 동작 (CI 환경 대응)

---

## 14. 향후 개발 계획

| 단계 | 기능 | 상태 |
|------|------|------|
| v1.0 | 기본 파이프라인 + `/api/v1` 전체 엔드포인트 | ✅ 완료 |
| v1.1 | 한국투자증권 MCP 연동 (실시간 시세) | 🔜 예정 |
| v1.2 | RAG 도입 (ChromaDB 벡터 검색) | 🔜 예정 |
| v1.3 | Google ADK SequentialAgent 전환 | 🔜 예정 |
| v2.0 | 사용자 인증 (JWT) + PostgreSQL 영속화 | 📋 계획 |
| v2.1 | Docker 컨테이너화 + 클라우드 배포 | 📋 계획 |

### 한국투자증권 MCP 연동 구조 (예정)

```
/api/v1/strategies/{id}
    ↓
strategy_generator
    ↓
mcp__kis__get_current_price(ticker)   ← 실시간 시세
mcp__kis__get_orderbook(ticker)       ← 호가창
    ↓
실제 현재가 기반 전략 보정
```

---

## 라이선스

본 프로젝트는 단국대학교 캡스톤 디자인 과목 학습 목적으로 작성되었습니다.
