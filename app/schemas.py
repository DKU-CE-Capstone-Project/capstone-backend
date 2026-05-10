from pydantic import BaseModel, Field


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
