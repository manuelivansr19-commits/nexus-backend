"""
NEXUS Ω — Ollama Provider.

Usa /api/chat (no el legacy /api/generate) para soporte
nativo de roles system/user/assistant.
"""

from __future__ import annotations

import time

import httpx

from backend.config import (
    MAX_OUTPUT_TOKENS,
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_URL,
    REQUEST_TIMEOUT_SECONDS,
    logger,
)
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)


class OllamaProvider(BaseModelProvider):

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return OLLAMA_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(OLLAMA_URL)

    @property
    def _base_url(self) -> str:
        """Extrae la base URL si OLLAMA_URL apunta a un endpoint específico."""
        url = OLLAMA_URL.rstrip("/")
        # Si la URL ya termina en /api/generate o /api/chat, usar la base
        for suffix in ("/api/generate", "/api/chat"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        if not self.is_configured:
            raise RuntimeError("OLLAMA_URL no configurada.")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        messages = [{"role": "system", "content": request.system}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        started = time.perf_counter()

        chat_url = f"{self._base_url}/api/chat"

        response = await self._http.post(
            chat_url,
            headers=headers,
            json=payload,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {response.status_code}")

        data = response.json()

        # /api/chat devuelve { message: { role, content } }
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama devolvió contenido vacío.")

        elapsed = time.perf_counter() - started
        logger.info("Ollama OK | modelo=%s | %.2fs", self.model, elapsed)

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            duration_ms=int(elapsed * 1000),
        )
