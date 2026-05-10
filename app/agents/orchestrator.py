from app.agents.filter_agent import dedupe_articles
from app.agents.keyword_expander import expand_keywords
from app.agents.news_fetcher import fetch_news
from app.agents.summarizer import summarize_articles
from app.schemas import AnalyzeResponse, Article


# Deterministic pipeline orchestrator. Will graduate to a Google ADK SequentialAgent
# (or a higher-level reasoning Agent) once we add branching — e.g., choosing between
# OSS LLM vs Claude per step, or deciding when to expand the keyword tree further.
async def run_analysis(keyword: str) -> AnalyzeResponse:
    raw = await fetch_news(keyword)
    deduped = dedupe_articles(raw)
    summarized = await summarize_articles(deduped[:6])
    related = await expand_keywords(keyword, summarized)

    articles = [
        Article(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source=a.get("source", ""),
            published_at=a.get("published_at", ""),
            summary=a.get("summary", ""),
        )
        for a in summarized
    ]

    return AnalyzeResponse(keyword=keyword, articles=articles, related_keywords=related)
