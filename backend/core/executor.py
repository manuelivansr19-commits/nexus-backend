"""
NEXUS Ω — Executor.

Ejecuta herramientas de forma segura y controlada.

Flujo:
  Intent → Tool selection → Validation → Execution → Result → Memory

Restricciones absolutas:
  - No ejecución de código arbitrario
  - No comandos shell
  - No acceso irrestricto al sistema operativo
  - Tools con risk CRITICAL requieren autorización explícita
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import TOOL_TIMEOUT_SECONDS, logger
from backend.tools.base import RiskLevel, ToolInput, ToolResult


# Risk levels permitidos en ejecución automática
_AUTO_ALLOWED_RISKS = {RiskLevel.LOW, RiskLevel.MEDIUM}


@dataclass
class ExecutionResult:
    """Resultado completo de una ejecución."""
    success:       bool
    tool_name:     str
    tool_result:   Optional[ToolResult] = None
    error:         Optional[str]        = None
    duration_ms:   int                  = 0
    authorized:    bool                 = True
    skipped:       bool                 = False


class Executor:
    """
    Ejecuta herramientas con validación de seguridad.

    - Valida risk level antes de ejecutar
    - Aplica timeout configurable
    - Registra resultado en memoria si está disponible
    - No ejecuta tools CRITICAL automáticamente
    """

    def __init__(
        self,
        registry=None,
        memory=None,
        timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = registry
        self._memory   = memory
        self._timeout  = timeout

    async def execute_by_name(
        self,
        tool_name:  str,
        params:     dict,
        context:    str = "",
        request_id: str = "",
        authorized: bool = False,
    ) -> ExecutionResult:
        """Ejecutar una herramienta por nombre."""
        if self._registry is None:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error="ToolRegistry no disponible.",
            )

        tool = self._registry.get(tool_name)
        if tool is None:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' no encontrada.",
            )

        if not tool.enabled:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' está deshabilitada.",
                skipped=True,
            )

        # Validar risk level
        if tool.risk_level == RiskLevel.CRITICAL and not authorized:
            logger.warning(
                "Executor: tool '%s' requiere autorización explícita (CRITICAL).",
                tool_name,
            )
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error="Esta acción requiere autorización explícita.",
                authorized=False,
            )

        if tool.risk_level not in _AUTO_ALLOWED_RISKS and not authorized:
            logger.warning(
                "Executor: tool '%s' risk=%s requiere autorización.",
                tool_name, tool.risk_level.value,
            )
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool de riesgo '{tool.risk_level.value}' requiere autorización.",
                authorized=False,
            )

        tool_input = ToolInput(
            params=params,
            context=context,
            request_id=request_id,
        )

        # Validar input
        validation_error = await tool.validate(tool_input)
        if validation_error:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Input inválido: {validation_error}",
            )

        # Ejecutar con timeout
        started = time.perf_counter()
        try:
            tool_result = await asyncio.wait_for(
                tool.execute(tool_input),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.error(
                "Executor: timeout en tool '%s' (%.0fs)", tool_name, self._timeout
            )
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Timeout ({self._timeout:.0f}s).",
                duration_ms=elapsed,
            )
        except Exception as e:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.exception("Executor: error en tool '%s'", tool_name)
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                error=str(e)[:300],
                duration_ms=elapsed,
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        tool_result.tool_name   = tool_name
        tool_result.duration_ms = elapsed

        logger.info(
            "Executor: tool '%s' %s | %dms",
            tool_name,
            "OK" if tool_result.success else "FAIL",
            elapsed,
        )

        # Guardar resultado en memoria si disponible
        if tool_result.success and self._memory:
            try:
                from backend.core.memory import MemoryType
                self._memory.remember(
                    content=f"[TOOL:{tool_name}] {str(tool_result.output)[:300]}",
                    memory_type=MemoryType.WORKING,
                    importance=0.4,
                )
            except Exception:
                pass

        return ExecutionResult(
            success=tool_result.success,
            tool_name=tool_name,
            tool_result=tool_result,
            error=tool_result.error,
            duration_ms=elapsed,
        )

    async def execute_candidates(
        self,
        candidate_names: list[str],
        params:          dict,
        context:         str = "",
        request_id:      str = "",
    ) -> list[ExecutionResult]:
        """Ejecutar múltiples tools candidatas en paralelo."""
        tasks = [
            self.execute_by_name(
                tool_name=name,
                params=params,
                context=context,
                request_id=request_id,
            )
            for name in candidate_names
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    def collect_context_strings(
        self,
        results: list[ExecutionResult],
    ) -> list[str]:
        """Extraer strings de contexto de los resultados exitosos."""
        out = []
        for r in results:
            if r.success and r.tool_result:
                out.append(r.tool_result.to_context_string())
        return out
