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
    """텍스트 생성. anthropic_api_key가 있으면 Claude 우선, 실패 시 Gemini로 fallback.

    임베딩(embed)은 항상 Gemini — Claude에는 임베딩 API가 없음.
    """
    use_claude = settings.anthropic_api_key and settings.llm_provider in ("auto", "anthropic")
    if use_claude:
        text = await _generate_claude(prompt)
        if text:
            return text
        # Claude 실패(쿼터/오류) → Gemini로 fallback (provider=anthropic 강제 시엔 빈 문자열)
        if settings.llm_provider == "anthropic":
            return ""
    return await _generate_gemini(prompt)


async def _generate_claude(prompt: str) -> str:
    """Claude(Anthropic) 호출. 오류/미설치 시 빈 문자열(graceful)."""
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        # opus-4-8/4.7는 temperature/top_p 미지원(400) — 전달하지 않음
        resp = await client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:  # noqa: BLE001
        print(f"[llm] Claude error (fallback): {type(e).__name__}: {e}")
        return ""


async def _generate_gemini(prompt: str) -> str:
    """Gemini 호출. 키 없음/오류 시 빈 문자열(graceful)."""
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
