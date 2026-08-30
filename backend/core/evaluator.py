"""
NEXUS Ω — Evaluator v3.6.0

Evalúa el resultado de cada paso del plan.

Estados posibles:
  SUCCESS          → paso completado correctamente
  PARTIAL          → resultado obtenido pero incompleto
  FAILED           → falló, no reintentar
  RETRY            → falló pero vale la pena reintentar
  NEEDS_USER_INPUT → se requiere información del usuario

El Evaluator NO modifica el plan directamente.
Retorna una evaluación — el Autonomy Loop decide qué hacer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import logger


class EvalStatus(str, Enum):
    SUCCESS          = "success"
    PARTIAL          = "partial"
    FAILED           = "failed"
    RETRY            = "retry"
    NEEDS_USER_INPUT = "needs_user_input"


@dataclass
class StepEvaluation:
    status:       EvalStatus
    score:        float          = 1.0    # 0.0 – 1.0
    reason:       str            = ""
    question:     Optional[str]  = None   # para NEEDS_USER_INPUT
    can_continue: bool           = True   # si el plan puede continuar
    metadata:     dict           = field(default_factory=dict)


@dataclass
class PlanEvaluation:
    plan_id:      str
    goal:         str
    steps_total:  int
    steps_done:   int
    steps_failed: int
    overall:      EvalStatus
    score:        float
    summary:      str
    duration_ms:  int = 0


class StepEvaluator:
    """
    Evalúa un paso individual del plan.

    Criterios:
    1. ¿Se ejecutó sin error?
    2. ¿Hay output?
    3. ¿El output es relevante (no vacío, no genérico)?
    4. ¿Es un error recuperable?
    """

    def evaluate(
        self,
        step_description: str,
        execution_success: bool,
        output: Optional[str],
        error: Optional[str] = None,
        retries_used: int = 0,
        max_retries: int = 3,
    ) -> StepEvaluation:

        # Falló sin output
        if not execution_success:
            if error and self._is_transient(error) and retries_used < max_retries:
                return StepEvaluation(
                    status=EvalStatus.RETRY,
                    score=0.0,
                    reason=f"Error transitorio: {error[:100]}",
                    can_continue=True,
                )
            if error and self._needs_input(error):
                return StepEvaluation(
                    status=EvalStatus.NEEDS_USER_INPUT,
                    score=0.0,
                    reason=error[:200],
                    question=f"Para continuar con '{step_description}' necesito más información: {error[:100]}",
                    can_continue=False,
                )
            return StepEvaluation(
                status=EvalStatus.FAILED,
                score=0.0,
                reason=error or "Sin detalles del error.",
                can_continue=True,   # plan puede continuar con siguiente paso
            )

        # Ejecutó pero sin output
        if not output or not output.strip():
            return StepEvaluation(
                status=EvalStatus.PARTIAL,
                score=0.3,
                reason="Ejecución sin output.",
                can_continue=True,
            )

        # Output muy corto (< 20 chars) — posiblemente vacío
        if len(output.strip()) < 20:
            return StepEvaluation(
                status=EvalStatus.PARTIAL,
                score=0.5,
                reason="Output muy corto.",
                can_continue=True,
            )

        # Score por longitud del output
        score = min(1.0, 0.6 + len(output) / 2000 * 0.4)

        return StepEvaluation(
            status=EvalStatus.SUCCESS,
            score=round(score, 2),
            reason="OK",
            can_continue=True,
        )

    def _is_transient(self, error: str) -> bool:
        upper = error.upper()
        return any(m in upper for m in [
            "TIMEOUT", "503", "502", "500", "RATE LIMIT",
            "429", "RESOURCE_EXHAUSTED", "CONNECTION",
        ])

    def _needs_input(self, error: str) -> bool:
        lower = error.lower()
        return any(m in lower for m in [
            "información insuficiente", "necesito saber",
            "falta el parámetro", "no especificado",
        ])


class PlanEvaluator:
    """
    Evalúa el plan completo al finalizar.
    """

    def __init__(self) -> None:
        self._history: list[PlanEvaluation] = []

    def evaluate_plan(
        self,
        plan,
        duration_ms: int = 0,
    ) -> PlanEvaluation:
        from backend.core.planner import StepStatus

        steps_total  = len(plan.steps)
        steps_done   = sum(1 for s in plan.steps if s.status == StepStatus.DONE)
        steps_failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)

        if steps_total == 0:
            score   = 0.0
            overall = EvalStatus.FAILED
            summary = "Plan vacío."
        elif steps_done == steps_total:
            score   = 1.0
            overall = EvalStatus.SUCCESS
            summary = f"Plan completado: {steps_done}/{steps_total} pasos."
        elif steps_done > 0:
            score   = round(steps_done / steps_total, 2)
            overall = EvalStatus.PARTIAL
            summary = f"Plan parcial: {steps_done}/{steps_total} pasos completados."
        else:
            score   = 0.0
            overall = EvalStatus.FAILED
            summary = f"Plan fallido: {steps_failed} errores."

        result = PlanEvaluation(
            plan_id=plan.plan_id,
            goal=plan.goal,
            steps_total=steps_total,
            steps_done=steps_done,
            steps_failed=steps_failed,
            overall=overall,
            score=score,
            summary=summary,
            duration_ms=duration_ms,
        )
        self._history.append(result)
        return result

    def stats(self) -> dict:
        if not self._history:
            return {"total_plans": 0}
        scores = [p.score for p in self._history]
        return {
            "total_plans": len(self._history),
            "avg_score":   round(sum(scores) / len(scores), 3),
            "success_rate": round(
                sum(1 for p in self._history if p.overall == EvalStatus.SUCCESS)
                / len(self._history), 3
            ),
        }
