"""
NEXUS Ω — OpenRouter Provider.

API compatible con OpenAI. Soporta lista de modelos fallback.
"""

from __future__ import annotations

import time

import httpx

from backend.config import (
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    logger,
)
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)


class OpenRouterProvider(BaseModelProvider):

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def model(self) -> str:
        return OPENROUTER_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(OPENROUTER_API_KEY)

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        if not self.is_configured:
            raise RuntimeError("OPENROUTER_API_KEY no configurada.")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexus-backend-wduu.onrender.com",
            "X-Title": "NEXUS AI",
        }

        messages = [{"role": "system", "content": request.system}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        # Modelos fallback de OpenRouter
        models = [self.model]
        if OPENROUTER_FALLBACK_MODELS:
            models.extend(
                m.strip()
                for m in OPENROUTER_FALLBACK_MODELS.split(",")
                if m.strip()
            )
        if len(models) > 1:
            payload["models"] = models

        started = time.perf_counter()

        response = await self._http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter no devolvió choices.")

        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("OpenRouter devolvió contenido vacío.")

        used_model = data.get("model", self.model)
        elapsed = time.perf_counter() - started

        logger.info("OpenRouter OK | modelo=%s | %.2fs", used_model, elapsed)

        return ProviderResponse(
            text=text.strip(),
            provider=self.name,
            model=used_model,
            duration_ms=int(elapsed * 1000),
        )
