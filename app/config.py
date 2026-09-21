from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env 위치: 저장소 루트(.env.example 옆). 뒤의 것이 우선하므로 저장소 루트가
    # 이긴다. 앞의 경로는 옛 모노레포(Capstone/backend/) 시절 위치 — 기존 로컬 설정 호환용.
    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parents[2] / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    newsapi_key: str = ""
    news_provider: Literal["naver", "gdelt", "newsapi"] = "naver"
    naver_client_id: str = ""
    naver_client_secret: str = ""
    naver_timeout_seconds: float = Field(default=10, gt=0, le=30)
    diffbot_token: str = ""
    diffbot_api_key: str = ""
    diffbot_search_timeout_ms: int = Field(default=10000, ge=1000, le=30000)
    diffbot_search_concurrency: int = Field(default=3, ge=1, le=5)
    google_api_key: str = ""
    use_mock_news: bool = False
    # Gemini 텍스트 생성 기본값. Flex는 지연을 허용하는 저비용 처리 티어.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_service_tier: Literal["flex", "standard"] = "flex"
    gemini_timeout_seconds: float = Field(default=600.0, gt=0, le=1800)
    llm_max_output_tokens: int = Field(default=2048, ge=512, le=8192)
    embedding_timeout_seconds: float = Field(default=30, gt=0, le=120)

    # ── LLM 제공자 (텍스트 생성) ─────────────────────────────────────────
    # auto를 명시하면 Claude 우선, 실패 시 Gemini. 기본값은 Gemini 고정.
    # 임베딩은 Claude에 API가 없으므로 항상 Gemini 사용(google_api_key 유지).
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"
    llm_provider: str = "gemini"  # auto | anthropic | gemini

    # 구 GDELT 경로 내부 호환용. 공급원 선택은 NEWS_PROVIDER를 사용한다.
    use_gdelt: bool = True

    # LLM 카드 요약 토글: Gemini 무료 티어(5 req/min)에서 검색당 요약 6건이
    # 쿼터를 소진해 리포트 생성이 fallback 되는 문제 방지. False면 요약은
    # 기사 description을 그대로 사용하고, Gemini는 리포트/전략에만 사용.
    use_llm_summaries: bool = True

    # 기사별 메타데이터: 실패/키 없음/mock 모드에서는 사전 기반 추출.
    use_llm_metadata: bool = True
    # Flex의 서버 대기 및 재시도를 감안한 기사별 전체 제한 시간.
    metadata_timeout_seconds: float = Field(default=660.0, gt=0, le=1800)
    metadata_batch_timeout_seconds: float = Field(default=660.0, gt=0, le=1800)
    metadata_concurrency: int = Field(default=3, ge=1, le=10)

    # ── 클라우드/이벤트 인프라 (CNCF 재구성) ─────────────────────────────
    nats_url: str = "nats://localhost:4222"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # demo_mode: burst 부하 데모의 재현성 확보용.
    # 외부 API(GDELT 타임아웃/Gemini 쿼터) 변동을 제거하기 위해 mock 뉴스 + 고정 처리지연 사용.
    demo_mode: bool = False
    demo_delay_seconds: float = 2.0

    # ── MongoDB Atlas (영속화 + 벡터검색) ────────────────────────────────
    # 기본적으로 MongoDB를 사용한다. 로컬에서 DB 없이 실행할 때만 USE_MONGODB=false로 지정한다.
    mongodb_uri: str = ""
    mongodb_db_name: str = "capstone_news"
    use_mongodb: bool = True
    mongodb_required: bool = False  # 서버에서는 저장 실패를 성공으로 처리하지 않는다.

    # ── 세션 (쿠키 기반 사용자 식별) ──────────────────────────────────────
    # 로그인을 빼기로 해서 계정으로 사용자를 구분할 수 없다. 대신 쿠키에 세션 id를
    # 심어 사용자별 마인드맵 상태를 분리한다(12주차 회의 결정).
    # 상태는 Redis에 TTL과 함께 저장되므로, 만료 = 세션 소멸 = 데이터 삭제다.
    session_ttl_seconds: int = 86400  # 24시간
    # HTTPS로 서비스할 때 true. http로 접속하면 true일 때 쿠키가 아예 안 실린다.
    session_cookie_secure: bool = False
    # 프론트와 API가 다른 출처면 "none"(+secure=true) 필요. 같은 출처면 "lax"로 충분.
    session_cookie_samesite: str = "lax"

    # ── AI 에이전트 (RAG 그라운딩 + 검증) ────────────────────────────────
    embedding_model: str = "gemini-embedding-001"  # Gemini 임베딩 (768차원 요청, generate와 별도 쿼터)
    use_rag: bool = True       # 리포트 생성 시 벡터검색으로 유사 과거 뉴스 근거 주입
    use_critic: bool = True    # 생성된 리포트를 검증(critic) 에이전트로 점검

    @property
    def mock_news_active(self) -> bool:
        return self.use_mock_news or self.demo_mode

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
