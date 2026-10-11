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

import os
import re
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


# --- Modo degradado: clasificacion de errores y enfriamiento por proveedor ---

COOLDOWN_MAX_S = 300.0   # tope: ningun proveedor queda bloqueado mas de 5 min


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# Valores iniciales; sobrescribibles por variable de entorno.
COOLDOWN_RATE_LIMIT_S = _env_float("NEXUS_COOLDOWN_429_S", 60.0)
COOLDOWN_OVERLOAD_S   = _env_float("NEXUS_COOLDOWN_503_S", 30.0)

_OVERLOAD_CODE_RE = re.compile(r"\b(503|529)\b")
_RETRY_TEXT_RES = (
    re.compile(r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.I),
    re.compile(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)\s*s", re.I),
)


def is_overload_error(error: Exception) -> bool:
    """503/529 o texto tipo 'high demand' / 'overloaded' / 'UNAVAILABLE'."""
    if getattr(error, "code", None) in (503, 529):
        return True
    text = str(error)
    if _OVERLOAD_CODE_RE.search(text):
        return True
    upper = text.upper()
    return any(m in upper for m in ("UNAVAILABLE", "HIGH DEMAND", "OVERLOADED"))


def retry_after_seconds(error: Exception) -> float | None:
    """Mejor esfuerzo: atributo retry_after, header Retry-After o 'retry in 12.5s'."""
    candidates = [getattr(error, "retry_after", None)]
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is not None:
        try:
            candidates.append(headers.get("Retry-After") or headers.get("retry-after"))
        except Exception:
            pass
    for c in candidates:
        try:
            value = float(c)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    text = str(error)
    for rx in _RETRY_TEXT_RES:
        m = rx.search(text)
        if m:
            value = float(m.group(1))
            if value > 0:
                return value
    return None


def degraded_text(reasons: dict[str, str] | None = None) -> str:
    """Mensaje honesto de modo degradado. Nunca incluye el error crudo del proveedor."""
    causes = set((reasons or {}).values())
    if "rate_limit" in causes:
        causa = "la cuota del proveedor de IA está agotada por ahora"
    elif "overloaded" in causes:
        causa = "el proveedor de IA está saturado en este momento"
    else:
        causa = "hubo un fallo interno al consultar el modelo de IA"
    return (
        f"NEXUS está en modo degradado: {causa}. "
        "Esta respuesta no fue generada por un modelo de IA. "
        "Las herramientas locales siguen disponibles; intenta de nuevo en un minuto."
    )


class AllProvidersUnavailable(RuntimeError):
    """Hereda de RuntimeError: cualquier `except RuntimeError` existente sigue funcionando.

    reasons: {proveedor: "rate_limit" | "overloaded" | "error"}
    """

    def __init__(self, errors: list[str], reasons: dict[str, str]) -> None:
        super().__init__("Todos los proveedores fallaron | " + " | ".join(errors))
        self.errors = errors
        self.reasons = reasons


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
        # nombre -> (expira en time.monotonic(), causa). Se vence solo, sin flags
        # permanentes. Sin locks: asyncio es cooperativo y no hay await entre
        # leer y escribir este dict.
        self._cooldown: dict[str, tuple[float, str]] = {}

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

    def _start_cooldown(self, name: str, reason: str, default_s: float,
                        error: Exception) -> None:
        secs = retry_after_seconds(error) or default_s
        secs = min(max(secs, 1.0), COOLDOWN_MAX_S)
        self._cooldown[name] = (time.monotonic() + secs, reason)

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

        reasons: dict[str, str] = {}
        for provider in providers:
            entry = self._cooldown.get(provider.name)
            if entry is not None:
                until, cause = entry
                remaining = until - time.monotonic()
                if remaining > 0:
                    errors.append(
                        f"{provider.name}: en enfriamiento ({remaining:.0f}s, {cause})"
                    )
                    reasons[provider.name] = cause
                    logger.info(
                        "Router salta %s (enfriamiento %.0fs, %s)",
                        provider.name, remaining, cause,
                    )
                    continue
                self._cooldown.pop(provider.name, None)
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
                    reasons[provider.name] = "rate_limit"
                    self._start_cooldown(
                        provider.name, "rate_limit", COOLDOWN_RATE_LIMIT_S, error
                    )
                    logger.warning(
                        "Proveedor %s limitado (429). Enfriando.", provider.name
                    )
                elif is_overload_error(error):
                    reasons[provider.name] = "overloaded"
                    self._start_cooldown(
                        provider.name, "overloaded", COOLDOWN_OVERLOAD_S, error
                    )
                    logger.warning(
                        "Proveedor %s saturado (503). Enfriando.", provider.name
                    )
                else:
                    reasons[provider.name] = "error"
                    logger.warning(
                        "Proveedor %s falló: %s", provider.name, msg
                    )

        raise AllProvidersUnavailable(errors, reasons)

    # ── Shutdown ─────────────────────────────────────────────

    async def shutdown(self) -> None:
        for p in self._all:
            try:
                await p.shutdown()
            except Exception:
                logger.exception("Error cerrando provider %s", p.name)
