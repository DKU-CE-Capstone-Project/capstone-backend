from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    newsapi_key: str = ""
    google_api_key: str = ""
    use_mock_news: bool = True
    # Gemini 모델 — 무료 쿼터는 모델별 독립이므로 한 모델 소진 시 교체 가능
    gemini_model: str = "gemini-flash-latest"

    # 뉴스 소스 토글: GDELT가 막힌(429/타임아웃) 환경에서는 False로 두어
    # NewsAPI를 바로 사용 (응답 지연·행 방지). 기본 True(한국어 GDELT 우선).
    use_gdelt: bool = True

    # LLM 카드 요약 토글: Gemini 무료 티어(5 req/min)에서 검색당 요약 6건이
    # 쿼터를 소진해 리포트 생성이 fallback 되는 문제 방지. False면 요약은
    # 기사 description을 그대로 사용하고, Gemini는 리포트/전략에만 사용.
    use_llm_summaries: bool = True

    # ── 클라우드/이벤트 인프라 (CNCF 재구성) ─────────────────────────────
    nats_url: str = "nats://localhost:4222"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # demo_mode: burst 부하 데모의 재현성 확보용.
    # 외부 API(GDELT 타임아웃/Gemini 쿼터) 변동을 제거하기 위해 mock 뉴스 + 고정 처리지연 사용.
    demo_mode: bool = False
    demo_delay_seconds: float = 2.0

    # ── MongoDB Atlas (영속화 + 벡터검색) ────────────────────────────────
    # 연결 불가/미설정 시 use_mongodb=False로 두면 전 기능 graceful no-op (기존 흐름 유지).
    mongodb_uri: str = ""
    mongodb_db_name: str = "capstone_news"
    use_mongodb: bool = False

    # ── AI 에이전트 (RAG 그라운딩 + 검증) ────────────────────────────────
    embedding_model: str = "gemini-embedding-001"  # Gemini 임베딩 (768차원 요청, generate와 별도 쿼터)
    use_rag: bool = True       # 리포트 생성 시 벡터검색으로 유사 과거 뉴스 근거 주입
    use_critic: bool = True    # 생성된 리포트를 검증(critic) 에이전트로 점검

    @property
    def mock_news_active(self) -> bool:
        return self.use_mock_news or self.demo_mode or not self.newsapi_key

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
