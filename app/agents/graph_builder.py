"""Build mind-map (graph) data from a center article and related articles."""
from __future__ import annotations

from typing import Any

from app.agents.filter_agent import _content_tokens, _overlap_coefficient
from app.utils import make_news_id

_FALLBACK_UNSPLASH = [
    "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1494412519320-aa613dfb7738?auto=format&fit=crop&w=520&q=80",
    "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=520&q=80",
]


def _relevance_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Overlap coefficient of title tokens as a 0-1 relevance score."""
    ta = _content_tokens(a.get("title", "") + " " + a.get("description", ""))
    tb = _content_tokens(b.get("title", "") + " " + b.get("description", ""))
    return round(_overlap_coefficient(ta, tb), 3)


def build_graph(
    center: dict[str, Any],
    related: list[dict[str, Any]],
    include_distance: bool = True,
) -> dict[str, Any]:
    """
    Build GraphResponse-compatible dict from a center article and related articles.

    Args:
        center: Normalized article dict (must have news_id).
        related: List of normalized article dicts (may or may not have news_id).
        include_distance: Whether to populate distance fields.

    Returns:
        Dict matching GraphResponse schema.
    """
    center_id = center.get("news_id") or make_news_id(center.get("url", ""))
    # The frontend uses graph summaries as article descriptions in the detail view.
    center_summary = center.get("description") or center.get("summary") or ""

    center_node = {
        "news_id": center_id,
        "title": center.get("title", ""),
        "summary": center_summary,
        "distance": 0,
        "is_center": True,
    }

    nodes: list[dict[str, Any]] = [center_node]
    edges: list[dict[str, Any]] = []

    for i, art in enumerate(related):
        nid = art.get("news_id") or make_news_id(art.get("url", f"unknown-{i}"))
        # Every selected article is directly connected to the center. Similarity
        # is not a hop count; depth remains reserved for future multi-hop expansion.
        dist = 1
        summary = art.get("description") or art.get("summary") or ""

        nodes.append({
            "news_id": nid,
            "title": art.get("title", ""),
            "summary": summary,
            "distance": dist,
            "is_center": False,
        })
        edges.append({
            "source": center_id,
            "target": nid,
            "relation_type": "related",
            "distance": dist,
        })

    return {
        "center_node": center_node,
        "nodes": nodes,
        "edges": edges,
    }
