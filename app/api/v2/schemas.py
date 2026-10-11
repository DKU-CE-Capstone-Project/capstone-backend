"""Public v2 DTOs. Persistence and worker-only fields stay private."""
from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool
    details: dict


class NewsCard(BaseModel):
    news_id: str
    title: str
    description: str
    source_name: str
    published_at: str | None
    source_url: str | None
    thumbnail_url: str | None
    keywords: list[str]
    categories: list[str]


class Selection(BaseModel):
    status: Literal['complete', 'insufficient', 'partial']
    reason: str | None
    requested: int = Field(ge=1)
    returned: int = Field(ge=0)
    shortage: bool
    budget_exhausted: bool


class Node(BaseModel):
    node_id: str
    news_id: str
    parent_node_id: str | None
    level: int = Field(ge=0)
    expanded: bool
    batch: int = Field(ge=0)
    news: NewsCard


class Branch(BaseModel):
    parent_node_id: str
    child_node_ids: list[str]
    display_selection: Selection
    last_attempt: Selection


class ResultLink(BaseModel):
    report_id: str
    strategy_id: str | None


class Map(BaseModel):
    map_id: str
    revision: int = Field(ge=1)
    query: str
    state: Literal['active', 'empty']
    root_node_id: str | None
    nodes: list[Node]
    branches: list[Branch]
    initial_selection: Selection
    selected_news_ids: list[str] = Field(max_length=10)
    detail_news_id: str | None
    report_panel_open: bool
    active_report_job_id: str | None
    last_result: ResultLink | None
    expires_at: str


class Capabilities(BaseModel):
    max_report_articles: int
    initial_related_target: Literal[4]
    branch_related_target: Literal[3]
    max_map_nodes: int
    max_map_depth: int


class Session(BaseModel):
    session_epoch: str
    resumed: bool
    expires_at: str
    prototype_tier: Literal['FREE', 'PAID']
    capabilities: Capabilities


class Progress(BaseModel):
    requested: int
    processed: int
    succeeded: int
    failed: int


class JobResult(BaseModel):
    report_id: str | None
    strategy_id: str | None
    report_reused: bool
    strategy_reused: bool


class ReportJob(BaseModel):
    job_id: str
    client_request_id: str
    map_id: str
    accepted_map_revision: int
    status: Literal['queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled']
    stage: Literal['queued', 'extracting', 'analyzing', 'verifying', 'strategy', 'saving', 'done']
    request_news_ids: list[str]
    progress: Progress
    result: JobResult | None
    warnings: list[str]
    error: ErrorDetail | None
    created_at: str
    updated_at: str
    expires_at: str


class UsedEvidence(BaseModel):
    news_id: str
    title: str
    source_name: str
    source_url: str
    published_at: str | None
    body_id: str
    content_hash: str
    input_char_count: int
    truncated: bool


class FailedEvidence(BaseModel):
    news_id: str
    code: str
    message: str
    retryable: bool


class RagEvidence(BaseModel):
    news_id: str
    title: str
    source_url: str | None
    description: str
    description_hash: str


class StockImpact(BaseModel):
    ticker: str | None
    name: str
    impact: str
    direction: Literal['up', 'down', 'mixed', 'unknown']
    evidence_news_ids: list[str]


class Coverage(BaseModel):
    requested_count: int
    used_count: int
    failed_count: int


class Verification(BaseModel):
    status: Literal['passed', 'warning', 'unavailable']
    notes: list[str]


class Report(BaseModel):
    schema_version: Literal[2]
    report_id: str
    title: str
    event_summary: str
    event_analysis: str
    market_impact: str
    stock_impacts: list[StockImpact]
    risk_factors: list[str]
    requested_news_ids: list[str]
    used_evidence: list[UsedEvidence]
    failed_evidence: list[FailedEvidence]
    coverage: Coverage
    rag_evidence: list[RagEvidence]
    verification: Verification
    created_at: str


class StrategyItem(BaseModel):
    ticker: str | None
    stock_name: str
    action: Literal['buy', 'hold', 'sell', 'watch']
    reason: str


class Strategy(BaseModel):
    schema_version: Literal[2]
    strategy_id: str
    report_id: str
    risk_level: Literal['low', 'medium', 'high']
    period: Literal['short', 'mid', 'long']
    strategy_summary: str
    risk_warning: str
    strategy_items: list[StrategyItem]
    created_at: str
