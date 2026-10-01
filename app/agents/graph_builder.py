"""Build mind-map (graph) data from a center article and related articles."""
from __future__ import annotations

from typing import Any

from app.agents.filter_agent import _content_tokens, _overlap_coefficient
from app.news_cards import to_news_card


def _relevance_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Overlap coefficient of title tokens as a 0-1 relevance score."""
    ta = _content_tokens(a.get("title", "") + " " + a.get("description", ""))
    tb = _content_tokens(b.get("title", "") + " " + b.get("description", ""))
    return round(_overlap_coefficient(ta, tb), 3)


def build_graph(
    center: dict[str, Any],
    related: list[dict[str, Any]],
    include_distance: bool = True,
    *,
    scores: dict[str, float] | None = None,
    same_story: dict[str, tuple[list[Any], int]] | None = None,
) -> dict[str, Any]:
    """
    Build GraphResponse-compatible dict from a center article and related articles.

    Args:
        center: Normalized article dict (must have news_id).
        related: List of normalized article dicts (may or may not have news_id).
        include_distance: Legacy flag; direct edges always have distance 1.
        scores: Only caller-authorized scores; omitted for FREE/BASIC.
        same_story: Node ID -> (grouped repeat cards, total). Never scored.

    Returns:
        Dict matching GraphResponse schema.
    """
    def stories(nid: str) -> dict[str, Any]:
        cards, total = (same_story or {}).get(nid, ([], 0))
        return {"same_story": [c.model_dump() if hasattr(c, "model_dump") else c for c in cards],
                "same_story_total": total}

    center_card = to_news_card(center).model_dump()
    center_id = center_card["news_id"]
    center_node = {
        **center_card,
        "distance": 0,
        "is_center": True,
        "relevance_score": None,
        **stories(center_id),
    }

    nodes: list[dict[str, Any]] = [center_node]
    edges: list[dict[str, Any]] = []

    for i, art in enumerate(related):
        card = to_news_card(art, i).model_dump()
        nid = card["news_id"]
        # Every selected article is directly connected to the center. Similarity
        # is not a hop count; depth remains reserved for future multi-hop expansion.
        dist = 1

        nodes.append({
            **card,
            "relevance_score": (scores or {}).get(nid),
            "distance": dist,
            "is_center": False,
            **stories(nid),
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
