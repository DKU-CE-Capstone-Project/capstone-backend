from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── 기존 (POST /analyze 호환 유지) ────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=100)


class Article(BaseModel):
    title: str
    url: str
    source: str
    published_at: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    keyword: str
    articles: list[Article]
    related_keywords: list[str]


# ── /api/v1/news ──────────────────────────────────────────────────────────────

class NewsCard(BaseModel):
    description: str = ""
    news_id: str
    title: str
    summary: str
    thumbnail_url: str
    source_name: str
    published_at: str
    related_stock_names: list[str] = []
    source_url: str = ""
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    news_cards: list[NewsCard]
    total_count: int


class ThumbnailResponse(BaseModel):
    news_id: str
    thumbnail_url: str
    fallback_used: bool


class SourceResponse(BaseModel):
    news_id: str
    source_name: str
    source_url: str
    published_at: str
    original_title: str
    thumbnail_url: str = ""
    original_body: str = ""
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


# ── /api/v1/news/{id}/graph ───────────────────────────────────────────────────

class NewsMapCard(NewsCard):
    distance: int
    relevance_score: float | None = None  # PAID only; the center has no pair score.


class NewsMapSelection(BaseModel):
    """complete: target reached. insufficient: normal shortage after the bounded search.
    partial: expansion failed or timed out; the returned neighbours are still valid.
    expandable: only with expand=false; call again with expand=true to search further."""
    status: Literal["complete", "insufficient", "partial", "expandable"]
    reason: str | None = None
    requested: int
    returned: int
    excluded: int = 0  # Caller-supplied exclude_ids that appeared among the candidates.


class GraphNode(NewsMapCard):
    is_center: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str = "related"
    distance: int


class GraphResponse(BaseModel):
    center_node: GraphNode
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    selection: NewsMapSelection | None = None


# ── /api/v1/news/{id}/related ─────────────────────────────────────────────────

class RelatedNewsItem(NewsMapCard):
    pass


class RelatedResponse(BaseModel):
    related_news: list[RelatedNewsItem]
    selection: NewsMapSelection | None = None


# ── /api/v1/news/selections ───────────────────────────────────────────────────

class NewsSelectionRequest(BaseModel):
    news_ids: list[str] = Field(min_length=1)
    selection_type: str = "report_source"


class NewsSelectionResponse(BaseModel):
    selection_id: str
    selected_news: list[dict[str, Any]]
    created_at: str


# ── /api/v1/news/{id}/relations ───────────────────────────────────────────────

class RelationScore(BaseModel):
    source_news_id: str
    target_news_id: str
    relevance_score: float
    relation_reason: str
    shared_keywords: list[str]


class RelationsResponse(BaseModel):
    relations: list[RelationScore]


# ── /api/v1/keywords ──────────────────────────────────────────────────────────

class RecommendedKeyword(BaseModel):
    keyword: str
    category: str
    rank: int


class RecommendedKeywordsResponse(BaseModel):
    keywords: list[RecommendedKeyword]


class RelatedKeyword(BaseModel):
    keyword: str
    article_count: int  # Search articles carrying the keyword; 0 for recommended fill-ins.
    source: Literal["articles", "recommended"]


class RelatedKeywordsResponse(BaseModel):
    query: str
    keywords: list[RelatedKeyword]
    article_total: int


# ── /api/v1/reports ───────────────────────────────────────────────────────────

class ReportCreateRequest(BaseModel):
    """news_ids(1~5)가 있으면 선택 기사 비동기 경로(202), 없으면 기존 동기 경로(news_id, 201)."""

    news_id: str | None = None
    related_news_ids: list[str] = []
    news_ids: list[str] | None = Field(default=None, min_length=1, max_length=5)
    ticker_symbols: list[str] = []
    language: str = "ko"
    report_type: str = "investment"

    @model_validator(mode="after")
    def _one_path(self) -> ReportCreateRequest:
        if self.news_ids is None and not self.news_id:
            raise ValueError("news_id 또는 news_ids 중 하나가 필요합니다.")
        if self.news_ids is not None:
            self.news_ids = list(dict.fromkeys(i for i in self.news_ids if i))
            if not self.news_ids:
                raise ValueError("news_ids에 유효한 ID가 없습니다.")
        return self


class ReportCreateResponse(BaseModel):
    report_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: str


class ReportProgress(BaseModel):
    done: int
    total: int


class StockImpact(BaseModel):
    name: str
    ticker: str = ""
    direction: Literal["up", "down", "mixed"]  # 기사 영향 해석. 가격 예측이 아니다.
    action: Literal["buy", "hold", "sell", "watch"]
    comment: str = ""


class ReportStrategy(BaseModel):
    stance: str
    rationale: str = ""
    watchlist: list[str] = []
    risk_warning: str = ""


class ReportError(BaseModel):
    code: str
    message: str


class ReportResponse(BaseModel):
    report_id: str
    title: str
    summary: str
    event_analysis: str
    market_impact: str
    related_stocks: list[str]
    evidence_news: list[dict[str, Any]]
    risk_factors: list[str]
    created_at: str
    # AI 에이전트 강화: RAG 근거(유사 과거 뉴스 제목) + 검증(critic) 결과
    rag_sources: list[str] = []
    verification: dict[str, Any] | None = None
    # 선택 기사 비동기 경로 (docs/10 § 3.6). 기존 동기 리포트는 completed/done·빈 값.
    status: Literal["pending", "processing", "completed", "failed"] = "completed"
    stage: Literal["queued", "extracting", "analyzing", "strategy", "done"] = "done"
    progress: ReportProgress | None = None
    requested_news_ids: list[str] = []
    stock_impacts: list[StockImpact] = []
    strategy: ReportStrategy | None = None
    is_fallback: bool = False
    error: ReportError | None = None
    updated_at: str = ""


# ── /api/v1/strategies ────────────────────────────────────────────────────────

class StrategyCreateRequest(BaseModel):
    report_id: str
    risk_level: Literal["low", "medium", "high"] = "medium"
    period: Literal["short", "mid", "long"] = "short"
    strategy_type: str = "simulation"


class StrategyCreateResponse(BaseModel):
    strategy_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: str


class StrategyItem(BaseModel):
    ticker: str
    stock_name: str
    action: Literal["buy", "hold", "sell", "watch"]
    reason: str


class StrategyResponse(BaseModel):
    strategy_id: str
    expected_return: float
    risk: str
    period: str
    strategy_summary: str
    strategy_items: list[StrategyItem]
    created_at: str


# ── 세션 (쿠키 기반 사용자 식별) ──────────────────────────────────────────────


class MindmapState(BaseModel):
    """세션에 저장되는 마인드맵 상태. DB에는 저장하지 않는다(설계 결정)."""

    center_news_id: str = ""
    expanded_news_ids: list[str] = Field(default_factory=list)
    query: str = ""


class SessionResponse(BaseModel):
    session_id: str
    mindmap: MindmapState
    viewed_news_ids: list[str] = Field(default_factory=list)
