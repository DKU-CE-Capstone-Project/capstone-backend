"""
Module-level in-memory caches.
Survive re-renders / re-imports within a single server process.
Reset on server restart (stateless by design for MVP).
"""
from __future__ import annotations

from typing import Any

# news_id (MD5[:12] of URL) → normalized article dict
news_cache: dict[str, dict[str, Any]] = {}

# report_id (UUID4 str) → report dict
report_cache: dict[str, dict[str, Any]] = {}

# strategy_id (UUID4 str) → strategy dict
strategy_cache: dict[str, dict[str, Any]] = {}

# selection_id (UUID4 str) → {news_ids, selected_news, created_at}
selection_cache: dict[str, dict[str, Any]] = {}

# ── 결과 캐싱 (LLM 재호출 방지 → 비용·지연 절감) ─────────────────────────────
# content_hash(제목+설명) → {"title_ko", "summary_ko"}  : 같은 기사 재번역/재요약 방지
summary_cache: dict[str, dict[str, str]] = {}

# content_key(news_id + 정렬된 related_ids) → report_id : 같은 뉴스 리포트 재생성 방지
report_index: dict[str, str] = {}
