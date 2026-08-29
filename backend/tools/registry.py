"""
NEXUS Ω — Tool Registry.

Registro central de todas las herramientas disponibles.
El Executor consulta el Registry para encontrar y ejecutar tools.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.config import logger
from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult


class ToolRegistry:
    """
    Registro central de herramientas.

    Métodos:
      register(tool)            → registrar herramienta
      unregister(name)          → eliminar herramienta
      get(name)                 → obtener por nombre
      list(enabled_only)        → listar todas
      find_by_intent(message)   → encontrar candidatas por mensaje
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── CRUD ─────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Registrar una herramienta. Sobreescribe si ya existe."""
        if tool.name in self._tools:
            logger.warning("ToolRegistry: sobreescribiendo tool '%s'", tool.name)
        self._tools[tool.name] = tool
        logger.info(
            "ToolRegistry: registrada '%s' | risk=%s | enabled=%s",
            tool.name, tool.risk_level.value, tool.enabled,
        )

    def unregister(self, name: str) -> bool:
        """Eliminar herramienta. Retorna True si existía."""
        existed = name in self._tools
        self._tools.pop(name, None)
        if existed:
            logger.info("ToolRegistry: eliminada '%s'", name)
        return existed

    def get(self, name: str) -> Optional[BaseTool]:
        """Obtener herramienta por nombre."""
        return self._tools.get(name)

    def enable(self, name: str) -> bool:
        t = self._tools.get(name)
        if t:
            t.enabled = True
        return bool(t)

    def disable(self, name: str) -> bool:
        t = self._tools.get(name)
        if t:
            t.enabled = False
        return bool(t)

    # ── Discovery ─────────────────────────────────────────────

    def list(
        self,
        enabled_only: bool = True,
        max_risk: Optional[RiskLevel] = None,
    ) -> list[BaseTool]:
        """Listar herramientas, opcionalmente filtradas."""
        risk_order = [
            RiskLevel.LOW, RiskLevel.MEDIUM,
            RiskLevel.HIGH, RiskLevel.CRITICAL,
        ]
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        if max_risk is not None:
            max_idx = risk_order.index(max_risk)
            tools = [t for t in tools if risk_order.index(t.risk_level) <= max_idx]
        return tools

    def find_by_intent(
        self,
        message: str,
        max_results: int = 3,
        enabled_only: bool = True,
    ) -> list[BaseTool]:
        """
        Encontrar herramientas relevantes para un mensaje.
        Match por keywords declaradas en cada tool.
        """
        lower = message.lower()
        scored: list[tuple[int, BaseTool]] = []

        for tool in self.list(enabled_only=enabled_only):
            score = sum(
                1 for kw in tool.intent_keywords
                if kw.lower() in lower
            )
            if score > 0:
                scored.append((score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:max_results]]

    def stats(self) -> dict:
        """Estadísticas del registro."""
        all_tools  = list(self._tools.values())
        return {
            "total":    len(all_tools),
            "enabled":  sum(1 for t in all_tools if t.enabled),
            "disabled": sum(1 for t in all_tools if not t.enabled),
            "by_risk":  {
                r.value: sum(1 for t in all_tools if t.risk_level == r)
                for r in RiskLevel
            },
            "names": [t.name for t in all_tools],
        }

    def describe(self) -> list[dict]:
        """Descripción de todas las tools (para el LLM)."""
        return [
            {
                "name":        t.name,
                "description": t.description,
                "risk":        t.risk_level.value,
                "enabled":     t.enabled,
                "keywords":    t.intent_keywords,
            }
            for t in self._tools.values()
        ]
