"""
NEXUS Ω — Gemini Provider.

Hotfix v3.4: 3 reintentos con backoff exponencial + jitter
para manejar errores 503 intermitentes.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

from google import genai
from google.genai import types

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    logger,
)
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)


def _is_transient(error: Exception) -> bool:
    """Errores que vale la pena reintentar (503, 429, timeout)."""
    text = str(error).upper()
    return any(m in text for m in (
        "503", "502", "500",
        "429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "TOO MANY REQUESTS", "QUOTA",
        "TIMEOUT", "DEADLINE",
        "SERVICE_UNAVAILABLE", "UNAVAILABLE",
    ))


class GeminiProvider(BaseModelProvider):
    """
    Provider Gemini con reintentos automáticos.

    Estrategia de backoff: delay = base * 2^attempt + jitter(0..1s)
    Intento 1 → ~1s, Intento 2 → ~2s, Intento 3 → ~4s
    """

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None
        if self.is_configured:
            try:
                self._client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info(
                    "GeminiProvider inicializado | modelo=%s | max_retries=%d",
                    GEMINI_MODEL, GEMINI_MAX_RETRIES,
                )
            except Exception:
                logger.exception("No se pudo inicializar GeminiProvider.")
                self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return GEMINI_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(GEMINI_API_KEY)

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        if self._client is None:
            raise RuntimeError("Gemini no está disponible.")

        last_error: Exception = RuntimeError("No se intentó ninguna llamada.")

        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                response = await self._call(request)
                if attempt > 0:
                    logger.info(
                        "Gemini OK en intento %d/%d", attempt + 1, GEMINI_MAX_RETRIES
                    )
                return response

            except Exception as error:
                last_error = error
                is_last = attempt == GEMINI_MAX_RETRIES - 1

                if is_last or not _is_transient(error):
                    raise

                delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                logger.warning(
                    "Gemini intento %d/%d falló (%s). Reintentando en %.1fs...",
                    attempt + 1, GEMINI_MAX_RETRIES,
                    str(error)[:80],
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_error

    async def _call(self, request: GenerateRequest) -> ProviderResponse:
        started = time.perf_counter()

        contents = []
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg.content)])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=request.prompt)])
        )

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=request.system,
                        max_output_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ),
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - started
            raise RuntimeError(f"Gemini timeout después de {elapsed:.0f}s") from e

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise RuntimeError("Gemini devolvió una respuesta vacía.")

        elapsed = time.perf_counter() - started
        logger.info("Gemini OK | %.2fs", elapsed)

        return ProviderResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            duration_ms=int(elapsed * 1000),
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.aio.aclose()
                logger.info("GeminiProvider cerrado.")
            except Exception:
                logger.exception("Error cerrando GeminiProvider.")
