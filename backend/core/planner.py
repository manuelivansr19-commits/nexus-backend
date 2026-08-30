"""
NEXUS Ω — Planner v3.6.0

DOS NIVELES — nunca dependiente de un solo proveedor:

  NIVEL 1: LOCAL/DETERMINISTIC
    - Reglas, plantillas, heurísticas
    - Funciona sin LLM
    - Siempre disponible

  NIVEL 2: LLM-GUIDED
    - Usa ModelRouter (no llama directamente a Gemini)
    - Solo si el problema requiere razonamiento complejo
    - Si falla → fallback a Nivel 1

Flujo:
  LLM PLANNER
       ↓ falla
  LOCAL PLANNER
       ↓
  PLAN BÁSICO
       ↓
  EXECUTOR

SEPARACIÓN ESTRICTA:
  Planner  → decide QUÉ hacer
  Executor → hace lo autorizado
  Evaluator → comprueba el resultado

El Planner NUNCA ejecuta herramientas directamente.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import MAX_PLAN_STEPS, PLANNER_SYSTEM, logger


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    description:     str
    step_id:         str           = field(default_factory=lambda: str(uuid.uuid4())[:6])
    tool:            Optional[str] = None      # nombre de tool del registry, o None
    requires_llm:    bool          = True
    expected_output: str           = ""
    depends_on:      list[str]     = field(default_factory=list)
    status:          StepStatus    = StepStatus.PENDING
    result:          Optional[str] = None
    error:           Optional[str] = None
    retries:         int           = 0
    duration_ms:     int           = 0

    def to_dict(self) -> dict:
        return {
            "step_id":      self.step_id,
            "description":  self.description,
            "tool":         self.tool,
            "requires_llm": self.requires_llm,
            "status":       self.status.value,
            "result":       self.result[:100] if self.result else None,
            "error":        self.error[:100] if self.error else None,
            "retries":      self.retries,
        }


@dataclass
class Plan:
    goal:       str
    plan_id:    str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps:      list[PlanStep]  = field(default_factory=list)
    done:       bool            = False
    created_at: float           = field(default_factory=time.time)
    source:     str             = "local"      # "local" | "llm"
    metadata:   dict            = field(default_factory=dict)

    def next_step(self) -> Optional[PlanStep]:
        completed = {s.step_id for s in self.steps if s.status == StepStatus.DONE}
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed for dep in step.depends_on):
                return step
        return None

    def mark_done(self, step_id: str, result: str = "") -> None:
        for s in self.steps:
            if s.step_id == step_id:
                s.status = StepStatus.DONE
                s.result = result
        if all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps):
            self.done = True

    def mark_failed(self, step_id: str, reason: str = "") -> None:
        for s in self.steps:
            if s.step_id == step_id:
                s.status = StepStatus.FAILED
                s.error  = reason

    def mark_running(self, step_id: str) -> None:
        for s in self.steps:
            if s.step_id == step_id:
                s.status = StepStatus.RUNNING

    def progress(self) -> str:
        total = len(self.steps)
        done  = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        return f"{done}/{total}"

    def collect_results(self) -> str:
        parts = []
        for s in self.steps:
            if s.status == StepStatus.DONE and s.result:
                parts.append(f"[{s.description}]\n{s.result}")
        return "\n\n".join(parts) if parts else ""

    def summary(self) -> dict:
        return {
            "plan_id":      self.plan_id,
            "goal":         self.goal,
            "source":       self.source,
            "steps":        len(self.steps),
            "done":         self.done,
            "progress":     self.progress(),
            "steps_detail": [s.to_dict() for s in self.steps],
        }


# ============================================================
# NIVEL 1 — LOCAL DETERMINISTIC PLANNER
# ============================================================

def _steps_from_list(items: list[dict]) -> list[PlanStep]:
    return [
        PlanStep(
            description  = s["description"],
            tool         = s.get("tool"),
            requires_llm = s.get("requires_llm", True),
        )
        for s in items
    ]


_ANALYSIS_TEMPLATE = [
    {"description": "Identificar el problema central y sus variables clave", "requires_llm": True},
    {"description": "Analizar causas y factores relevantes del contexto", "requires_llm": True},
    {"description": "Identificar riesgos y oportunidades principales", "requires_llm": True},
    {"description": "Formular conclusiones y recomendaciones accionables", "requires_llm": True},
]

_DESIGN_TEMPLATE = [
    {"description": "Definir requisitos y restricciones del sistema", "requires_llm": True},
    {"description": "Proponer arquitectura o estructura general", "requires_llm": True},
    {"description": "Especificar componentes y sus interfaces", "requires_llm": True},
    {"description": "Identificar dependencias, riesgos y mitigaciones", "requires_llm": True},
    {"description": "Producir diseño preliminar consolidado", "requires_llm": True},
]

_TASK_TEMPLATE = [
    {"description": "Clarificar objetivo y criterios de éxito", "requires_llm": True},
    {"description": "Identificar recursos y herramientas disponibles", "requires_llm": True},
    {"description": "Definir pasos de implementación concretos", "requires_llm": True},
    {"description": "Identificar riesgos y plan de contingencia", "requires_llm": True},
    {"description": "Producir plan de acción detallado", "requires_llm": True},
]

_RESEARCH_TEMPLATE = [
    {"description": "Definir el alcance y preguntas de investigación", "requires_llm": True},
    {"description": "Recopilar y analizar información disponible", "requires_llm": True},
    {"description": "Identificar hallazgos principales y patrones", "requires_llm": True},
    {"description": "Sintetizar conclusiones y recomendaciones", "requires_llm": True},
]

_TEMPLATES: dict[str, list[dict]] = {
    "analysis":  _ANALYSIS_TEMPLATE,
    "design":    _DESIGN_TEMPLATE,
    "task":      _TASK_TEMPLATE,
    "research":  _RESEARCH_TEMPLATE,
}


class LocalPlanner:
    """
    NIVEL 1 — Planner determinístico.

    Siempre disponible. No depende de ningún modelo externo.
    Usa plantillas por tipo de intent.
    """

    def __init__(self, max_steps: int = MAX_PLAN_STEPS) -> None:
        self._max_steps = max_steps

    def plan(self, goal: str, intent_type: str = "task") -> Plan:
        """Crear plan con plantilla determinística."""
        template = _TEMPLATES.get(intent_type, _TASK_TEMPLATE)
        steps    = _steps_from_list(template[:self._max_steps])
        logger.info(
            "LocalPlanner: plan creado | intent=%s | pasos=%d",
            intent_type, len(steps),
        )
        return Plan(goal=goal, steps=steps, source="local")

    def single_step(self, description: str, tool: Optional[str] = None) -> Plan:
        """Plan de un solo paso."""
        return Plan(
            goal=description,
            steps=[PlanStep(description=description, tool=tool, requires_llm=(tool is None))],
            source="local",
        )


# ============================================================
# NIVEL 2 — LLM-GUIDED PLANNER
# ============================================================

class LLMPlanner:
    """
    NIVEL 2 — Planner guiado por LLM.

    Usa ModelRouter — no conoce Gemini directamente.
    Si falla, el Planner principal hace fallback a LocalPlanner.
    """

    def __init__(self, model_router, max_steps: int = MAX_PLAN_STEPS) -> None:
        self._router    = model_router
        self._max_steps = max_steps

    async def plan(self, goal: str, intent_type: str = "task", context: str = "") -> Plan:
        """Generar plan con LLM. Lanza excepción si falla."""
        from backend.providers.base import GenerateRequest

        prompt = (
            f"Crea un plan estructurado para este objetivo.\n"
            f"Objetivo: {goal}\n"
            f"Tipo: {intent_type}\n\n"
            f"Responde SOLO con JSON válido (sin markdown):\n"
            f'{{"goal":"...","steps":['
            f'{{"id":1,"description":"paso concreto","tool":null,"requires_llm":true}}'
            f']}}\n\n'
            f"Máximo {min(self._max_steps, 8)} pasos. "
            f"Solo análisis, planificación, razonamiento. "
            f"Sin acciones destructivas ni irreversibles."
        )

        result = await self._router.generate(
            GenerateRequest(
                prompt=prompt,
                system=PLANNER_SYSTEM,
                max_tokens=1500,
            )
        )

        raw = result.response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        data  = json.loads(raw)
        steps = [
            PlanStep(
                description  = s.get("description", "paso"),
                tool         = s.get("tool"),
                requires_llm = s.get("requires_llm", True),
                expected_output = s.get("expected_output", ""),
            )
            for s in data.get("steps", [])[:self._max_steps]
        ]

        if not steps:
            raise ValueError("LLM devolvió plan vacío.")

        logger.info(
            "LLMPlanner: plan generado | pasos=%d | provider=%s",
            len(steps), result.response.provider,
        )
        return Plan(goal=data.get("goal", goal), steps=steps, source="llm")


# ============================================================
# PLANNER PRINCIPAL — con fallback
# ============================================================

class Planner:
    """
    Planner principal con dos niveles y fallback automático.

    Si use_llm=True y el LLM falla → LocalPlanner automáticamente.
    NEXUS nunca queda inutilizado por un proveedor caído.
    """

    def __init__(
        self,
        model_router=None,
        max_steps: int = MAX_PLAN_STEPS,
    ) -> None:
        self._local     = LocalPlanner(max_steps=max_steps)
        self._llm       = LLMPlanner(model_router, max_steps) if model_router else None
        self._max_steps = max_steps

    async def plan(
        self,
        goal:        str,
        intent_type: str  = "task",
        use_llm:     bool = True,
        context:     str  = "",
    ) -> Plan:
        """
        Crear plan con fallback automático.

        Orden:
          1. LLM (si use_llm=True y router disponible)
          2. LOCAL (siempre disponible)
        """
        if use_llm and self._llm is not None:
            try:
                return await self._llm.plan(goal, intent_type, context)
            except Exception as e:
                logger.warning(
                    "Planner: LLM falló (%s), usando LocalPlanner.",
                    str(e)[:100],
                )

        return self._local.plan(goal, intent_type)

    def plan_sync(self, goal: str, intent_type: str = "task") -> Plan:
        """Plan síncrono (solo LocalPlanner). Para testing y emergencias."""
        return self._local.plan(goal, intent_type)

    def single_step(self, description: str, tool: Optional[str] = None) -> Plan:
        return self._local.single_step(description, tool)
