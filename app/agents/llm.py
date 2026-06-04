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


async def embed(text: str) -> list[float]:
    """Gemini 임베딩 벡터(768차원) 반환. 키 없음/오류 시 빈 리스트(graceful).

    임베딩 쿼터는 generate_content와 별도 버킷이라 비교적 여유롭다.
    """
    if not settings.google_api_key or not (text or "").strip():
        return []
    try:
        client = _client()
        if client is None:
            return []
        from google.genai import types as gtypes

        resp = await client.aio.models.embed_content(
            model=settings.embedding_model,
            contents=text[:8000],
            config=gtypes.EmbedContentConfig(output_dimensionality=768),
        )
        # google-genai: resp.embeddings[0].values
        embs = getattr(resp, "embeddings", None)
        if embs:
            return list(embs[0].values)
        return []
    except Exception as e:  # noqa: BLE001
        print(f"[llm] embed error (skip): {type(e).__name__}: {e}")
        return []
