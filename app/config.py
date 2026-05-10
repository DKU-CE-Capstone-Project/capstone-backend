from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    newsapi_key: str = ""
    google_api_key: str = ""
    use_mock_news: bool = True

    @property
    def mock_news_active(self) -> bool:
        return self.use_mock_news or not self.newsapi_key


settings = Settings()
