"""
NEXUS Ω — Groq Provider.

API compatible con OpenAI. Inferencia de baja latencia.
"""

from __future__ import annotations

import time

import httpx

from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    logger,
)
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)


class GroqProvider(BaseModelProvider):

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return GROQ_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(GROQ_API_KEY)

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY no configurada.")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "system", "content": request.system}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens,
        }

        started = time.perf_counter()

        response = await self._http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq no devolvió choices.")

        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Groq devolvió contenido vacío.")

        elapsed = time.perf_counter() - started
        logger.info("Groq OK | modelo=%s | %.2fs", self.model, elapsed)

        return ProviderResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            duration_ms=int(elapsed * 1000),
        )
