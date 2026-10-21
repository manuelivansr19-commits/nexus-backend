"""
NEXUS Ω — Gemini Provider.

Hotfix v3.4: 3 reintentos con backoff exponencial + jitter
para manejar errores 503 intermitentes.
"""

from __future__ import annotations

import asyncio
import random
import re
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


class GeminiTimeout(RuntimeError):
    """Timeout local de la llamada a Gemini (mismo mensaje de siempre; clase propia para clasificar)."""


_QUOTA = "quota"
_TRANSIENT = "transient"
_PERMANENT = "permanent"

_QUOTA_CODES = frozenset({429})
_TRANSIENT_CODES = frozenset({408, 500, 502, 503, 504})
_QUOTA_STATUSES = frozenset({"RESOURCE_EXHAUSTED"})
_TRANSIENT_STATUSES = frozenset({"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"})
# Solo se usan cuando el error NO trae code/status estructurados. Con limites de palabra:
# "15003 tokens" no es un 503.
_QUOTA_TEXT = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|\bQUOTA\b|RATE[ _]?LIMIT|TOO MANY REQUESTS")
_TRANSIENT_TEXT = re.compile(
    r"\b50[0234]\b|\bUNAVAILABLE\b|SERVICE_UNAVAILABLE|DEADLINE|\bTIMEOUT\b|TIMED OUT")


def _classify_error(error: Exception) -> str:
    """
    'quota'     cuota/limite agotado (429, RESOURCE_EXHAUSTED, QUOTA): reintentar solo gasta llamadas.
    'transient' fallo del servidor o timeout (500/502/503/504, UNAVAILABLE, INTERNAL, DEADLINE): reintentable.
    'permanent' todo lo demas (400, 401, 403, 404, respuesta vacia, errores de programa...): no reintentar.
    """
    if isinstance(error, (GeminiTimeout, asyncio.TimeoutError, TimeoutError)):
        return _TRANSIENT
    code = getattr(error, "code", None)
    code = code if isinstance(code, int) and not isinstance(code, bool) else None
    status = str(getattr(error, "status", "") or "").upper()
    if code in _QUOTA_CODES or status in _QUOTA_STATUSES:
        return _QUOTA
    if code in _TRANSIENT_CODES or status in _TRANSIENT_STATUSES:
        return _TRANSIENT
    if code is not None:
        return _PERMANENT          # otro codigo HTTP estructurado: no se adivina por el texto
    text = str(error).upper()
    if _QUOTA_TEXT.search(text):
        return _QUOTA
    if _TRANSIENT_TEXT.search(text):
        return _TRANSIENT
    return _PERMANENT


def _is_transient(error: Exception) -> bool:
    """Errores que vale la pena reintentar (5xx y timeouts). La cuota agotada NO lo es."""
    return _classify_error(error) == _TRANSIENT


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
                kind = _classify_error(error)

                if kind == _QUOTA:
                    # Cuota agotada: reintentar solo gasta llamadas. Se relanza el error
                    # original (su texto lo reconocen el router y el plan de autonomia).
                    logger.warning(
                        "Gemini cuota/limite alcanzado (%s); no se reintenta.",
                        str(error)[:80],
                    )
                    raise
                if is_last or kind != _TRANSIENT:
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
            raise GeminiTimeout(f"Gemini timeout después de {elapsed:.0f}s") from e

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
