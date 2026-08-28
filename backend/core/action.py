"""
NEXUS Ω — Action.

Interfaz de ejecución de acciones.
Las acciones son siempre seguras y verificadas.
NO ejecuta comandos del sistema arbitrariamente.

Fase actual: acciones de texto + notificaciones.
Fase futura: acciones sobre hardware (servos, motores) vía
             interfaces seguras con sandbox + permisos.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    TEXT_RESPONSE  = "text_response"   # responder al usuario
    MEMORY_WRITE   = "memory_write"    # guardar en memoria
    TOOL_CALL      = "tool_call"       # invocar herramienta
    NOTIFICATION   = "notification"    # alerta interna
    MOTOR          = "motor"           # (FUTURO) actuador físico
    SERVO          = "servo"           # (FUTURO) servo


@dataclass
class Action:
    action_type: ActionType
    payload:     dict = field(default_factory=dict)
    action_id:   str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    safe:        bool = True           # marcar acciones de riesgo como False


@dataclass
class ActionResult:
    action_id:  str
    success:    bool
    output:     Any   = None
    error:      Optional[str] = None
    duration_ms: int  = 0


class ActionExecutor:
    """
    Ejecutor de acciones con registro de handlers.

    Para agregar un tipo nuevo:
      executor.register(ActionType.TOOL_CALL, my_handler)
    """

    def __init__(self) -> None:
        self._handlers: dict[ActionType, list] = {}

    def register(self, action_type: ActionType, handler) -> None:
        self._handlers.setdefault(action_type, []).append(handler)

    async def execute(self, action: Action) -> ActionResult:
        if not action.safe:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                error="Acción marcada como insegura. Requiere autorización explícita.",
            )

        handlers = self._handlers.get(action.action_type, [])
        if not handlers:
            return ActionResult(
                action_id=action.action_id,
                success=True,
                output=f"Acción {action.action_type.value} registrada (sin handler activo).",
            )

        started = time.perf_counter()
        try:
            result = None
            for h in handlers:
                result = await h(action) if hasattr(h, "__await__") else h(action)
            return ActionResult(
                action_id=action.action_id,
                success=True,
                output=result,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as e:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                error=str(e)[:300],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
