"""
NEXUS Ω — Inference Registry v3.8.0

Registro dinámico de runtimes.
Permite agregar nuevos runtimes sin modificar NexusCore.
"""

from __future__ import annotations

from typing import Optional
from backend.config import logger
from backend.inference.interface import InferenceRuntime
from backend.inference.models import RuntimeHealth, RuntimeStatus, RuntimeType


class InferenceRegistry:
    """
    Registro central de runtimes de inferencia.

    Permite:
      register(runtime)      → registrar
      get(name)              → obtener por nombre
      list(type)             → listar por tipo
      health()               → estado de todos
      available_local()      → runtimes locales disponibles
      available_cloud()      → runtimes cloud disponibles
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, InferenceRuntime] = {}

    def register(self, runtime: InferenceRuntime) -> None:
        self._runtimes[runtime.name] = runtime
        logger.info(
            "InferenceRegistry: registrado '%s' [%s]",
            runtime.name, runtime.runtime_type.value,
        )

    def unregister(self, name: str) -> bool:
        existed = name in self._runtimes
        self._runtimes.pop(name, None)
        return existed

    def get(self, name: str) -> Optional[InferenceRuntime]:
        return self._runtimes.get(name)

    def list(self, runtime_type: Optional[RuntimeType] = None) -> list[InferenceRuntime]:
        runtimes = list(self._runtimes.values())
        if runtime_type:
            runtimes = [r for r in runtimes if r.runtime_type == runtime_type]
        return runtimes

    def all_runtimes(self) -> list[InferenceRuntime]:
        return list(self._runtimes.values())

    async def health(self) -> dict[str, RuntimeHealth]:
        """Health check de todos los runtimes registrados."""
        results: dict[str, RuntimeHealth] = {}
        for name, runtime in self._runtimes.items():
            try:
                results[name] = await runtime.health()
            except Exception as e:
                results[name] = RuntimeHealth(
                    status=RuntimeStatus.UNAVAILABLE,
                    runtime_name=name,
                    runtime_type=runtime.runtime_type,
                    message=str(e)[:100],
                )
        return results

    async def available_local(self) -> list[InferenceRuntime]:
        """Runtimes locales disponibles ahora mismo."""
        result = []
        for r in self.list(RuntimeType.LOCAL):
            try:
                h = await r.health()
                if h.is_available():
                    result.append(r)
            except Exception:
                pass
        return result

    def stats(self) -> dict:
        total  = len(self._runtimes)
        local  = sum(1 for r in self._runtimes.values() if r.is_local)
        cloud  = total - local
        return {
            "total": total, "local": local, "cloud": cloud,
            "names": list(self._runtimes.keys()),
        }

    async def shutdown(self) -> None:
        for r in self._runtimes.values():
            try:
                await r.shutdown()
            except Exception:
                pass
