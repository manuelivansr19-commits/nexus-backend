"""
NEXUS Ω — Planner v3.8.0

El Planner usa InferenceLayer para LLM-guided planning.
NO conoce Gemini, Groq ni ningún proveedor concreto.
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
    tool:            Optional[str] = None
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
    source:     str             = "local"
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


# ── Templates locales ─────────────────────────────────────────

_TEMPLATES: dict[str, list[dict]] = {
    "analysis": [
        {"description": "Identificar el problema central y sus variables clave"},
        {"description": "Analizar causas y factores relevantes del contexto"},
        {"description": "Identificar riesgos y oportunidades principales"},
        {"description": "Formular conclusiones y recomendaciones accionables"},
    ],
    "design": [
        {"description": "Definir requisitos y restricciones del sistema"},
        {"description": "Proponer arquitectura o estructura general"},
        {"description": "Especificar componentes y sus interfaces"},
        {"description": "Identificar dependencias, riesgos y mitigaciones"},
        {"description": "Producir diseño preliminar consolidado"},
    ],
    "task": [
        {"description": "Clarificar objetivo y criterios de éxito"},
        {"description": "Identificar recursos y herramientas disponibles"},
        {"description": "Definir pasos de implementación concretos"},
        {"description": "Identificar riesgos y plan de contingencia"},
        {"description": "Producir plan de acción detallado"},
    ],
    "research": [
        {"description": "Definir el alcance y preguntas de investigación"},
        {"description": "Recopilar y analizar información disponible"},
        {"description": "Identificar hallazgos principales y patrones"},
        {"description": "Sintetizar conclusiones y recomendaciones"},
    ],
}


class LocalPlanner:
    """Nivel 1 — siempre disponible, sin inferencia."""

    def __init__(self, max_steps: int = MAX_PLAN_STEPS) -> None:
        self._max_steps = max_steps

    def plan(self, goal: str, intent_type: str = "task") -> Plan:
        template = _TEMPLATES.get(intent_type, _TEMPLATES["task"])
        steps = [
            PlanStep(description=s["description"], tool=s.get("tool"),
                     requires_llm=s.get("requires_llm", True))
            for s in template[:self._max_steps]
        ]
        return Plan(goal=goal, steps=steps, source="local")

    def single_step(self, description: str, tool: Optional[str] = None) -> Plan:
        return Plan(
            goal=description,
            steps=[PlanStep(description=description, tool=tool, requires_llm=(tool is None))],
            source="local",
        )


class LLMPlanner:
    """
    Nivel 2 — usa InferenceLayer (NO un proveedor concreto).
    Falla gracefully si la inferencia no está disponible.
    """

    def __init__(self, inference_layer, max_steps: int = MAX_PLAN_STEPS) -> None:
        self._inference  = inference_layer
        self._max_steps  = max_steps

    async def plan(self, goal: str, intent_type: str = "task", context: str = "") -> Plan:
        from backend.inference.models import InferenceRequest

        prompt = (
            f"Crea un plan estructurado para este objetivo.\n"
            f"Objetivo: {goal}\nTipo: {intent_type}\n\n"
            f"Responde SOLO con JSON (sin markdown):\n"
            f'{{"goal":"...","steps":[{{"id":1,"description":"paso","tool":null,"requires_llm":true}}]}}\n'
            f"Máximo {min(self._max_steps, 8)} pasos. Solo análisis y planificación."
        )

        result = await self._inference.generate(
            InferenceRequest(prompt=prompt, system=PLANNER_SYSTEM, max_tokens=1500)
        )

        raw = result.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        steps = [
            PlanStep(
                description  = s.get("description", "paso"),
                tool         = s.get("tool"),
                requires_llm = s.get("requires_llm", True),
            )
            for s in data.get("steps", [])[:self._max_steps]
        ]
        if not steps:
            raise ValueError("Plan LLM vacío.")

        logger.info("LLMPlanner: plan generado via InferenceLayer | pasos=%d", len(steps))
        return Plan(goal=data.get("goal", goal), steps=steps, source="llm")


class Planner:
    """
    Planner principal con fallback automático.
    Usa InferenceLayer — NO conoce Gemini ni ningún proveedor.
    """

    def __init__(self, inference_layer=None, model_router=None,
                 max_steps: int = MAX_PLAN_STEPS) -> None:
        self._local     = LocalPlanner(max_steps=max_steps)
        self._max_steps = max_steps
        # Priorizar InferenceLayer; legacy model_router como fallback
        if inference_layer is not None:
            self._llm = LLMPlanner(inference_layer, max_steps)
        elif model_router is not None:
            # Wrapper de compatibilidad para model_router legacy
            self._llm = _LegacyRouterPlanner(model_router, max_steps)
        else:
            self._llm = None

    async def plan(self, goal: str, intent_type: str = "task",
                   use_llm: bool = True, context: str = "") -> Plan:
        if use_llm and self._llm is not None:
            try:
                return await self._llm.plan(goal, intent_type, context)
            except Exception as e:
                logger.warning("Planner: LLM falló (%s) → LocalPlanner.", str(e)[:80])
        return self._local.plan(goal, intent_type)

    def plan_sync(self, goal: str, intent_type: str = "task") -> Plan:
        return self._local.plan(goal, intent_type)

    def single_step(self, description: str, tool: Optional[str] = None) -> Plan:
        return self._local.single_step(description, tool)


class _LegacyRouterPlanner:
    """Wrapper de compatibilidad — usa model_router legacy."""

    def __init__(self, router, max_steps: int) -> None:
        self._router    = router
        self._max_steps = max_steps

    async def plan(self, goal: str, intent_type: str = "task", context: str = "") -> Plan:
        from backend.providers.base import GenerateRequest
        import json

        prompt = (
            f"Crea un plan para: {goal}\n"
            f"Responde SOLO JSON: {{\"goal\":\"\",\"steps\":[{{\"description\":\"\",\"tool\":null}}]}}"
        )
        result = await self._router.generate(
            GenerateRequest(prompt=prompt, system=PLANNER_SYSTEM, max_tokens=1500)
        )
        raw  = result.response.text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        steps = [
            PlanStep(description=s.get("description","paso"), tool=s.get("tool"),
                     requires_llm=True)
            for s in data.get("steps",[])[:self._max_steps]
        ]
        return Plan(goal=data.get("goal", goal), steps=steps, source="llm_legacy")
