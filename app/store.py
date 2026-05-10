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
