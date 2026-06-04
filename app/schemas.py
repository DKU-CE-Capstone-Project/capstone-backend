from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── 기존 (POST /analyze 호환 유지) ────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=100)


class Article(BaseModel):
    title: str
    url: str
    source: str
    published_at: str
    summary: str


class AnalyzeResponse(BaseModel):
    keyword: str
    articles: list[Article]
    related_keywords: list[str]


# ── /api/v1/news ──────────────────────────────────────────────────────────────

class NewsCard(BaseModel):
    news_id: str
    title: str
    summary: str
    thumbnail_url: str
    source_name: str
    published_at: str
    related_stock_names: list[str] = []


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


# ── /api/v1/news/{id}/graph ───────────────────────────────────────────────────

class GraphNode(BaseModel):
    news_id: str
    title: str
    summary: str
    distance: int
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


# ── /api/v1/news/{id}/related ─────────────────────────────────────────────────

class RelatedNewsItem(BaseModel):
    news_id: str
    title: str
    summary: str
    thumbnail_url: str
    relevance_score: float | None = None  # PAID 전용
    distance: int


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


# ── /api/v1/reports ───────────────────────────────────────────────────────────

class ReportCreateRequest(BaseModel):
    news_id: str
    related_news_ids: list[str] = []
    ticker_symbols: list[str] = []
    language: str = "ko"
    report_type: str = "investment"


class ReportCreateResponse(BaseModel):
    report_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: str


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
