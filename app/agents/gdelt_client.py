"""Compatibility imports from the capstone-news-logic repository.

News collection and deduplication are maintained in that package so deployment
of news-logic changes also updates the API and worker.
"""
from capstone_news_logic.api import build_gdelt_params, fetch_gdelt_articles

__all__ = ["build_gdelt_params", "fetch_gdelt_articles"]
