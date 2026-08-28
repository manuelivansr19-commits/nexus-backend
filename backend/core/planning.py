"""
NEXUS Ω — Planning.

El planificador descompone objetivos complejos en pasos
ejecutables. Fase actual: estructura básica.
Fase futura: LLM-guided planning con ReAct/CoT.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class PlanStep:
    description: str
    tool:        Optional[str] = None    # herramienta necesaria
    depends_on:  list[str] = field(default_factory=list)  # IDs de steps previos
    step_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:6])
    status:      StepStatus = StepStatus.PENDING
    result:      Optional[str] = None


@dataclass
class Plan:
    goal:    str
    steps:   list[PlanStep] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    done:    bool = False

    def next_step(self) -> Optional[PlanStep]:
        """Próximo step ejecutable (sin dependencias pendientes)."""
        completed = {s.step_id for s in self.steps if s.status == StepStatus.DONE}
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed for dep in step.depends_on):
                return step
        return None

    def progress(self) -> str:
        total = len(self.steps)
        done  = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        return f"{done}/{total} pasos completados"


class Planner:
    """
    Crea y gestiona planes.

    Fase actual: planes manuales (definidos por código).
    Fase futura: LLM genera el plan a partir de un objetivo.
    """

    def create(self, goal: str, steps: list[dict]) -> Plan:
        """
        steps = [{"description": "...", "tool": "web_search"}, ...]
        """
        plan_steps = [
            PlanStep(
                description=s["description"],
                tool=s.get("tool"),
                depends_on=s.get("depends_on", []),
            )
            for s in steps
        ]
        return Plan(goal=goal, steps=plan_steps)

    def mark_done(self, plan: Plan, step_id: str, result: str = "") -> None:
        for s in plan.steps:
            if s.step_id == step_id:
                s.status = StepStatus.DONE
                s.result = result
                break
        if all(s.status == StepStatus.DONE for s in plan.steps):
            plan.done = True

    def mark_failed(self, plan: Plan, step_id: str, reason: str = "") -> None:
        for s in plan.steps:
            if s.step_id == step_id:
                s.status = StepStatus.FAILED
                s.result = reason
                break
