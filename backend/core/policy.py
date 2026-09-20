"""
NEXUS Ω — Inference Policy v3.8.0

Decide qué runtimes usar y en qué orden.
Nunca está dentro de NexusCore.

Responsabilidades:
  - runtime primario según modo
  - si cloud está permitido
  - qué hacer ante timeout / unavailable / invalid
  - orden de fallback
  - logging de decisiones
"""

from __future__ import annotations

from backend.config import ALLOW_CLOUD_INFERENCE, INFERENCE_MODE, logger
from backend.inference.errors import (
    AllRuntimesFailed, CloudInferenceBlocked,
    InferenceRateLimited, RuntimeUnavailable, classify_error,
)
from backend.inference.models import (
    InferenceMode, InferenceRequest, InferenceResult,
    RuntimeType, DEPRECATED_MODES,
)


class InferencePolicy:
    """
    Motor de decisión de inferencia.

    Aplica las reglas:
      - INFERENCE_MODE controla el orden de runtimes
      - ALLOW_CLOUD_INFERENCE bloquea cloud si es False
      - Fallback solo si la política lo permite
    """

    def __init__(
        self,
        mode:               InferenceMode  = None,
        allow_cloud:        bool           = None,
    ) -> None:
        # Leer de config si no se especifica (permite override en tests)
        self._mode        = mode or InferenceMode(INFERENCE_MODE)
        self._allow_cloud = allow_cloud if allow_cloud is not None else ALLOW_CLOUD_INFERENCE

        if self._mode in DEPRECATED_MODES:
            logger.warning(
                "InferencePolicy: modo '%s' está DEPRECATED. "
                "Usa LOCAL_FIRST o LOCAL_ONLY.",
                self._mode.value,
            )

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    @property
    def cloud_allowed(self) -> bool:
        return self._allow_cloud

    def select_runtimes(self, all_runtimes: list) -> list:
        """
        Ordenar y filtrar runtimes según la política activa.
        Retorna la lista en orden de intento.
        """
        local_runtimes = [r for r in all_runtimes if r.is_local]
        cloud_runtimes = [r for r in all_runtimes if not r.is_local]

        if self._mode == InferenceMode.LOCAL_ONLY:
            if not local_runtimes:
                raise CloudInferenceBlocked(
                    "LOCAL_ONLY: no hay runtimes locales configurados.",
                    runtime="none",
                )
            return local_runtimes

        if self._mode == InferenceMode.CLOUD_ONLY:
            if not self._allow_cloud:
                raise CloudInferenceBlocked(
                    "CLOUD_ONLY pero ALLOW_CLOUD_INFERENCE=false.",
                    runtime="none",
                )
            return cloud_runtimes

        if self._mode == InferenceMode.LOCAL_FIRST:
            if self._allow_cloud:
                return local_runtimes + cloud_runtimes
            return local_runtimes   # solo local, cloud no permitido

        if self._mode == InferenceMode.CLOUD_FIRST:
            # DEPRECATED — pero funciona
            if self._allow_cloud:
                return cloud_runtimes + local_runtimes
            return local_runtimes

        return local_runtimes + cloud_runtimes

    def should_retry(self, error: Exception) -> bool:
        """¿Vale la pena reintentar con el siguiente runtime?"""
        if isinstance(error, InferenceRateLimited):
            return True   # rate limit → intentar siguiente
        if isinstance(error, RuntimeUnavailable):
            return True   # runtime caído → intentar siguiente
        return True       # por defecto, intentar siguiente

    def is_cloud_blocked(self, runtime) -> bool:
        """¿Está este runtime cloud bloqueado por política?"""
        if not runtime.is_local and not self._allow_cloud:
            return True
        return False

    def log_decision(self, selected: list, blocked: list) -> None:
        logger.info(
            "InferencePolicy: mode=%s cloud=%s | runtimes=[%s] blocked=[%s]",
            self._mode.value,
            "allowed" if self._allow_cloud else "BLOCKED",
            ",".join(r.name for r in selected),
            ",".join(r.name for r in blocked),
        )
