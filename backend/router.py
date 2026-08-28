"""
NEXUS Ω — Model Router.

Selecciona el provider correcto según disponibilidad y modo:

  NEXUS_LOCAL_ONLY=true  → solo providers locales (is_local=True)
  USE_OLLAMA_ONLY=true   → solo Ollama (legado)
  Normal                 → todos los providers en orden de prioridad

Orden de prioridad (modo normal):
  1. Local  (llama.cpp / Ollama local)
  2. Gemini
  3. OpenRouter
  4. Groq
  5. Ollama (externo)

Distingue:
  CONFIGURED  → tiene credenciales/config
  AVAILABLE   → responde activamente (verificado vía is_available())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.config import NEXUS_LOCAL_ONLY, USE_OLLAMA_ONLY, logger
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)


def is_rate_limit_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    if code == 429:
        return True
    text = str(error).upper()
    return any(m in text for m in (
        "429", "RESOURCE_EXHAUSTED", "RATE LIMIT",
        "TOO MANY REQUESTS", "QUOTA",
    ))


@dataclass
class RouterResult:
    response: ProviderResponse
    fallback: bool
    attempted_providers: list[str]
    primary_provider: str
    local_mode: bool = False


class ModelRouter:

    def __init__(self, providers: list[BaseModelProvider]) -> None:
        self._all = providers

    # ── Provider selection ───────────────────────────────────

    @property
    def configured_providers(self) -> list[BaseModelProvider]:
        """
        Providers con credenciales/config válidas,
        filtrados según modo de routing.
        """
        result = []
        for p in self._all:
            if not p.is_configured:
                continue

            if NEXUS_LOCAL_ONLY:
                # Solo proveedores locales
                if getattr(p, "is_local", False):
                    result.append(p)
                continue

            if USE_OLLAMA_ONLY:
                # Legado: solo Ollama
                if p.name == "ollama":
                    result.append(p)
                continue

            result.append(p)

        return result

    # ── Status ───────────────────────────────────────────────

    def provider_status(self) -> dict[str, bool]:
        return {p.name: p.is_configured for p in self._all}

    def model_names(self) -> dict[str, str]:
        return {p.name: p.model for p in self._all}

    def local_mode_active(self) -> bool:
        return NEXUS_LOCAL_ONLY

    # ── Generate ─────────────────────────────────────────────

    async def generate(self, request: GenerateRequest) -> RouterResult:
        providers = self.configured_providers

        if not providers:
            if NEXUS_LOCAL_ONLY:
                raise RuntimeError(
                    "NEXUS_LOCAL_ONLY=true pero ningún provider local está configurado. "
                    "Verifica LOCAL_MODEL_PATH o LOCAL_OLLAMA_URL."
                )
            raise RuntimeError("Ningún proveedor de IA está configurado.")

        primary  = providers[0].name
        errors:   list[str] = []
        attempted: list[str] = []

        for provider in providers:
            attempted.append(provider.name)
            logger.info(
                "Router intentando proveedor=%s [local=%s]",
                provider.name,
                getattr(provider, "is_local", False),
            )
            started = time.perf_counter()

            try:
                response = await provider.generate(request)
                elapsed  = time.perf_counter() - started
                logger.info(
                    "Router éxito | proveedor=%s | %.2fs", provider.name, elapsed
                )
                return RouterResult(
                    response=response,
                    fallback=(provider.name != primary),
                    attempted_providers=attempted,
                    primary_provider=primary,
                    local_mode=NEXUS_LOCAL_ONLY,
                )

            except Exception as error:
                msg = str(error)[:300]
                errors.append(f"{provider.name}: {msg}")

                if is_rate_limit_error(error):
                    logger.warning(
                        "Proveedor %s limitado (429). Saltando.", provider.name
                    )
                else:
                    logger.warning(
                        "Proveedor %s falló: %s", provider.name, msg
                    )

        raise RuntimeError(
            "Todos los proveedores fallaron | " + " | ".join(errors)
        )

    # ── Shutdown ─────────────────────────────────────────────

    async def shutdown(self) -> None:
        for p in self._all:
            try:
                await p.shutdown()
            except Exception:
                logger.exception("Error cerrando provider %s", p.name)
