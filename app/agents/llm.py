from __future__ import annotations

from app.config import settings


def _model_name() -> str:
    return settings.gemini_model


def _client():
    if not settings.google_api_key:
        return None
    from google import genai

    return genai.Client(api_key=settings.google_api_key)


async def generate(prompt: str) -> str:
    """Generate text via Gemini. Returns empty string on error or missing key."""
    if not settings.google_api_key:
        return ""
    try:
        client = _client()  # SDK import은 try 안에서 — 미설치 시에도 graceful fallback
        if client is None:
            return ""
        from google.genai import types as gtypes

        resp = await client.aio.models.generate_content(
            model=_model_name(),
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
