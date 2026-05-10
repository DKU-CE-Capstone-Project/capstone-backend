from __future__ import annotations

from app.config import settings

_MODEL_NAME = "gemini-flash-latest"


def _client():
    if not settings.google_api_key:
        return None
    from google import genai

    return genai.Client(api_key=settings.google_api_key)


async def generate(prompt: str) -> str:
    """Generate text via Gemini. Returns empty string on error or missing key."""
    client = _client()
    if client is None:
        return ""
    try:
        from google.genai import types as gtypes

        resp = await client.aio.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )
        return (resp.text or "").strip()
    except Exception as e:
        # 할당량 초과·네트워크 오류 등은 조용히 fallback 처리
        print(f"[llm] Gemini error (fallback): {type(e).__name__}: {e}")
        return ""
