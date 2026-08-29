"""
NEXUS Ω — Base Tool.

Interfaz abstracta para todas las herramientas del sistema.
Las Tools son capacidades discretas que el Executor puede invocar.

Ninguna Tool se ejecuta directamente — siempre pasan por el Executor
que valida el risk level y registra el resultado en Memory.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RiskLevel(str, Enum):
    """
    Nivel de riesgo de una herramienta.

    LOW      → solo lectura, sin efectos secundarios
    MEDIUM   → escribe en memoria o envía datos
    HIGH     → llama a APIs externas, modifica estado
    CRITICAL → acceso al sistema, ejecución de código
               (requiere autorización explícita, nunca automático)
    """
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class ToolInput:
    """Input tipado para una herramienta."""
    params: dict[str, Any] = field(default_factory=dict)
    context: str = ""           # contexto adicional del mensaje
    request_id: str = ""


@dataclass
class ToolResult:
    """Resultado de la ejecución de una herramienta."""
    success:     bool
    output:      Any            # texto, dict, lista, etc.
    tool_name:   str = ""
    duration_ms: int = 0
    error:       Optional[str] = None
    metadata:    dict = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Representación para incluir en el contexto del LLM."""
        if not self.success:
            return f"[TOOL:{self.tool_name}] ERROR: {self.error}"
        if isinstance(self.output, str):
            return f"[TOOL:{self.tool_name}] {self.output}"
        return f"[TOOL:{self.tool_name}] {str(self.output)[:500]}"


class BaseTool(abc.ABC):
    """
    Interfaz base para todas las herramientas de NEXUS.

    Implementar:
      - name (property)
      - description (property)
      - input_schema (property)
      - output_schema (property)
      - execute(input) async
    """

    enabled:    bool      = True
    risk_level: RiskLevel = RiskLevel.LOW

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identificador único. Snake_case. e.g. 'web_search'"""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Descripción clara de qué hace y cuándo usar."""

    @property
    def input_schema(self) -> dict:
        """JSON Schema del input esperado."""
        return {"type": "object", "properties": {}}

    @property
    def output_schema(self) -> dict:
        """JSON Schema del output producido."""
        return {"type": "string"}

    @property
    def intent_keywords(self) -> list[str]:
        """Keywords que sugieren el uso de esta tool."""
        return []

    @abc.abstractmethod
    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """Ejecutar la herramienta. Debe ser idempotente cuando sea posible."""

    async def validate(self, tool_input: ToolInput) -> Optional[str]:
        """
        Validar el input antes de ejecutar.
        Retorna string con error si inválido, None si OK.
        """
        return None

    def __repr__(self) -> str:
        return (
            f"<Tool name={self.name!r} "
            f"risk={self.risk_level.value} "
            f"enabled={self.enabled}>"
        )
