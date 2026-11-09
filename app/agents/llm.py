from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any
from weakref import WeakKeyDictionary

from app.config import settings
from app.embedding_usage import record_embedding_usage
from app.telemetry import observe, token_usage, usage_context

_loop_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[str, Any]] = WeakKeyDictionary()
# Filled by the provider call when generate_with_usage() is waiting for its result.
_usage_sink: ContextVar[dict | None] = ContextVar("llm_usage_sink", default=None)


def _model_name() -> str:
    return settings.gemini_model


def _client():
    if not settings.google_api_key:
        return None
    from google import genai

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return genai.Client(api_key=settings.google_api_key)
    cached = _loop_clients.get(loop)
    if cached is not None and cached[0] == settings.google_api_key:
        return cached[1]
    client = genai.Client(api_key=settings.google_api_key)
    _loop_clients[loop] = (settings.google_api_key, client)
    return client


def _output_limit(max_output_tokens: int | None) -> int:
    """Per-call output cap within the same bounds as LLM_MAX_OUTPUT_TOKENS."""
    if max_output_tokens is None:
        return settings.llm_max_output_tokens
    return min(8192, max(512, int(max_output_tokens)))


def _capture_usage(service: str, *, status: str, model: str | None, usage: Any = None) -> None:
    sink = _usage_sink.get()
    if sink is None:
        return
    tokens = token_usage(usage, service)
    sink.update(service=service, model=model, status=status,
                tokens_in=tokens.get("input_tokens"), tokens_out=tokens.get("output_tokens"))


async def generate(prompt: str, *, purpose: str | None = None, max_output_tokens: int | None = None) -> str:
    """Text generation. ``purpose`` becomes the telemetry stage (bull, judge, ...).

    Without the keyword arguments the behaviour is unchanged, and a stage set by
    the caller with ``@stage`` stays in effect.
    """
    if purpose is None:
        return await _generate_with_timeout(prompt, max_output_tokens)
    with usage_context(stage=purpose):
        return await _generate_with_timeout(prompt, max_output_tokens)


async def _generate_with_timeout(prompt: str, max_output_tokens: int | None) -> str:
    try:
        return await asyncio.wait_for(
            _generate_with_provider(prompt, max_output_tokens=max_output_tokens),
            settings.gemini_timeout_seconds,
        )
    except TimeoutError:
        print("[llm] generation timeout")
        usage = _usage_sink.get()
        if usage is not None:
            usage["status"] = "timeout"
        return ""


async def generate_with_usage(
    prompt: str, *, purpose: str | None = None, max_output_tokens: int | None = None,
) -> tuple[str, dict]:
    """Like generate(), also returning {service, model, status, tokens_in, tokens_out}.

    If generation is unavailable before a provider can be selected, the status
    is error and the provider, model, and counts are None.
    """
    usage = {
        "service": None,
        "model": None,
        "status": "error",
        "tokens_in": None,
        "tokens_out": None,
    }
    token = _usage_sink.set(usage)
    try:
        text = await generate(prompt, purpose=purpose, max_output_tokens=max_output_tokens)
    finally:
        _usage_sink.reset(token)
    return text, usage


async def _generate_with_provider(prompt: str, *, max_output_tokens: int | None = None) -> str:
    """텍스트 생성. 기본 Gemini; auto 설정일 때만 Claude → Gemini fallback.

    임베딩(embed)은 항상 Gemini — Claude에는 임베딩 API가 없음.
    """
    use_claude = settings.anthropic_api_key and settings.llm_provider in ("auto", "anthropic")
    if use_claude:
        text = await _generate_claude(prompt, max_output_tokens=max_output_tokens)
        if text:
            return text
        # Claude 실패(쿼터/오류) → Gemini로 fallback (provider=anthropic 강제 시엔 빈 문자열)
        if settings.llm_provider == "anthropic":
            return ""
    return await _generate_gemini(prompt, max_output_tokens=max_output_tokens)


async def _generate_claude(prompt: str, *, max_output_tokens: int | None = None) -> str:
    """Claude(Anthropic) 호출. 오류/미설치 시 빈 문자열(graceful)."""
    model = settings.claude_model
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        # opus/sonnet 4.x는 temperature/top_p 미지원(400) — 전달하지 않음.
        # cache_control: 동일 프롬프트가 5분 내 반복되면 프리픽스 캐싱(프롬프트가 최소 길이 이상일 때).
        # 실질 비용 절감은 app-level 결과 캐싱(summary_cache/report_index)이 담당.
        with observe("anthropic", "generate", model=model) as usage_event:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024 if max_output_tokens is None else _output_limit(max_output_tokens),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            )
            usage_event.response(resp, usage=getattr(resp, "usage", None))
        _capture_usage("anthropic", status="ok", model=model, usage=getattr(resp, "usage", None))
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except asyncio.CancelledError:
        _capture_usage("anthropic", status="cancelled", model=model)
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[llm] Claude error (fallback): {type(e).__name__}")
        _capture_usage("anthropic", status="error", model=model)
        return ""


async def _generate_gemini(prompt: str, *, max_output_tokens: int | None = None) -> str:
    """Gemini 호출. 키 없음/오류 시 빈 문자열(graceful)."""
    if not settings.google_api_key:
        return ""
    model = _model_name()
    try:
        client = _client()  # SDK import은 try 안에서 — 미설치 시에도 graceful fallback
        if client is None:
            return ""
        from google.genai import types as gtypes

        with observe("gemini", "generate", model=model, service_tier=settings.gemini_service_tier) as usage_event:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=_output_limit(max_output_tokens),
                    service_tier=settings.gemini_service_tier,
                    http_options=gtypes.HttpOptions(
                        timeout=int(settings.gemini_timeout_seconds * 1000),
                        retry_options=gtypes.HttpRetryOptions(
                            attempts=3, initial_delay=1, max_delay=8, exp_base=2,
                            http_status_codes=[429, 503],
                        ),
                    ),
                ),
            )
            usage_event.response(resp, usage=getattr(resp, "usage_metadata", None))
        _capture_usage("gemini", status="ok", model=model, usage=getattr(resp, "usage_metadata", None))
        return (resp.text or "").strip()
    except asyncio.CancelledError:
        _capture_usage("gemini", status="cancelled", model=model)
        raise
    except Exception as e:  # noqa: BLE001 — optional provider errors use a safe fallback
        # 할당량 초과·네트워크 오류 등은 조용히 fallback 처리
        print(f"[llm] Gemini error (fallback): {type(e).__name__}")
        _capture_usage("gemini", status="error", model=model)
        return ""


async def embed(
    text: str, *, model: str | None = None, dimensions: int = 768,
    task_type: str | None = None,
) -> list[float]:
    try:
        return await _embed(text, model=model, dimensions=dimensions, task_type=task_type)
    except TimeoutError:
        return []


async def _embed(
    text: str, *, model: str | None = None, dimensions: int = 768,
    task_type: str | None = None,
) -> list[float]:
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

        with observe("gemini", "embed", model=model or settings.embedding_model) as usage_event:
            resp = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model=model or settings.embedding_model,
                    contents=text[:8000],
                    config=gtypes.EmbedContentConfig(output_dimensionality=dimensions, task_type=task_type),
                ),
                settings.embedding_timeout_seconds,
            )
            await record_embedding_usage(
                usage_event, resp, client=client,
                model=model or settings.embedding_model, texts=[text[:8000]],
            )
        # google-genai: resp.embeddings[0].values
        embs = getattr(resp, "embeddings", None)
        if embs:
            return list(embs[0].values)
        return []
    except Exception as e:  # noqa: BLE001
        print(f"[llm] embed error (skip): {type(e).__name__}")
        return []


async def embed_batch(
    texts: list[str],
    *,
    model: str | None = None,
    dimensions: int = 768,
    task_type: str | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """배치 Gemini 임베딩 반환 (`embed_content(contents=[...])`로 RPM 절약).

    실패·키 누락·개수 불일치 시 빈 리스트를 반환한다.
    """
    if not texts:
        return []
    if not settings.google_api_key:
        return []
    cleaned = [(t or "").strip()[:8000] for t in texts]
    if any(not c for c in cleaned):
        return []
    try:
        client = _client()
        if client is None:
            return []
        from google.genai import types as gtypes

        out: list[list[float]] = []
        step = max(1, min(batch_size, 100))
        for start in range(0, len(cleaned), step):
            chunk = cleaned[start : start + step]
            with observe("gemini", "embed_batch", model=model or settings.embedding_model) as usage_event:
                resp = await asyncio.wait_for(
                    client.aio.models.embed_content(
                        model=model or settings.embedding_model,
                        contents=chunk,
                        config=gtypes.EmbedContentConfig(
                            output_dimensionality=dimensions,
                            task_type=task_type,
                        ),
                    ),
                    settings.embedding_timeout_seconds,
                )
                await record_embedding_usage(
                    usage_event, resp, client=client,
                    model=model or settings.embedding_model, texts=chunk,
                )
            embs = getattr(resp, "embeddings", None)
            if not embs or len(embs) != len(chunk):
                return []
            for item in embs:
                vals = list(getattr(item, "values", None) or [])
                if len(vals) != dimensions:
                    return []
                out.append(vals)
        return out
    except TimeoutError:
        print("[llm] embed_batch timeout")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"[llm] embed_batch error (skip): {type(e).__name__}")
        return []
