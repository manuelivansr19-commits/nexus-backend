"""
NEXUS Ω — Built-in Tools.

Herramientas básicas incluidas por defecto.
Todas son de riesgo LOW o MEDIUM — sin acceso al sistema.
"""

from __future__ import annotations

import time
from datetime import datetime

from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult


class ClockTool(BaseTool):
    """Retorna fecha y hora actual."""
    enabled    = True
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str:
        return "clock"

    @property
    def description(self) -> str:
        return "Retorna la fecha y hora actual del sistema."

    @property
    def intent_keywords(self) -> list[str]:
        return ["hora", "tiempo", "fecha", "día", "time", "date", "clock", "cuando"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        now = datetime.now()
        return ToolResult(
            success=True,
            output={
                "time": now.strftime("%H:%M:%S"),
                "date": now.strftime("%A, %d de %B de %Y"),
                "iso":  now.isoformat(),
            },
            tool_name=self.name,
        )


class StatusTool(BaseTool):
    """Retorna el estado del sistema NEXUS."""
    enabled    = True
    risk_level = RiskLevel.LOW

    @property
    def name(self) -> str:
        return "system_status"

    @property
    def description(self) -> str:
        return "Retorna el estado operacional del sistema NEXUS."

    @property
    def intent_keywords(self) -> list[str]:
        return ["estado", "status", "sistema", "operativo", "funcionando", "activo"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(
            success=True,
            output="NEXUS Ω v3.5.0 operativo. Todos los subsistemas activos.",
            tool_name=self.name,
        )


class MemorySearchTool(BaseTool):
    """Busca en la memoria del sistema."""
    enabled    = True
    risk_level = RiskLevel.LOW

    def __init__(self, memory=None) -> None:
        self._memory = memory

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Busca en la memoria del sistema información almacenada previamente."

    @property
    def intent_keywords(self) -> list[str]:
        return ["recuerda", "recuerdo", "antes dijiste", "anteriormente",
                "memoria", "guardaste", "mencionaste"]

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        if self._memory is None:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                error="Memory no disponible.",
            )
        query   = tool_input.params.get("query", tool_input.context)
        results = self._memory.recall(query, limit=5)
        if not results:
            return ToolResult(
                success=True,
                output="No encontré información relevante en memoria.",
                tool_name=self.name,
            )
        texts = [e.content for e in results]
        return ToolResult(
            success=True,
            output="\n".join(f"• {t}" for t in texts),
            tool_name=self.name,
            metadata={"matches": len(results)},
        )


def create_default_registry(memory=None):
    """Crear un ToolRegistry con las tools built-in registradas."""
    from backend.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.register(ClockTool())
    registry.register(StatusTool())
    registry.register(MemorySearchTool(memory))
    return registry
