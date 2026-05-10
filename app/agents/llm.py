from __future__ import annotations

from app.config import settings

_MODEL_NAME = "gemini-1.5-flash"


def _client():
    if not settings.google_api_key:
        return None
    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(_MODEL_NAME)


async def generate(prompt: str) -> str:
    """Generate text via Gemini. Returns empty string if no API key configured."""
    model = _client()
    if model is None:
        return ""
    resp = await model.generate_content_async(prompt)
    return (getattr(resp, "text", "") or "").strip()
