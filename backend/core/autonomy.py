"""
NEXUS Ω — Autonomy Loop v3.6.0

Ciclo controlado de ejecución autónoma:

  PLAN → EXECUTE → EVALUATE → ¿completo?
                               ├── YES     → FINALIZE
                               ├── RETRY   → EXECUTE (mismo paso)
                               ├── REPLAN  → PLANNER
                               └── INPUT   → preguntar al usuario

Límites absolutos (nunca loops infinitos):
  MAX_PLAN_STEPS       — pasos máximos por plan
  MAX_EXECUTION_LOOPS  — iteraciones máximas del ciclo
  MAX_RETRIES_PER_STEP — reintentos por paso

La autonomía inicial permite:
  READ / ANALYZE / PLAN / CALCULATE / SEARCH / REMEMBER

NO permite:
  exec() / shell / acciones irreversibles / modificar servidor
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import (
    MAX_EXECUTION_LOOPS,
    MAX_PLAN_STEPS,
    MAX_RETRIES_PER_STEP,
    logger,
)
from backend.core.evaluator import EvalStatus, PlanEvaluator, StepEvaluator
from backend.core.planner import Plan, PlanStep, Planner, StepStatus


class LoopStatus(str, Enum):
    COMPLETED    = "completed"
    PARTIAL      = "partial"
    FAILED       = "failed"
    LIMIT_HIT    = "limit_hit"
    NEEDS_INPUT  = "needs_input"


@dataclass
class ExecutionTrace:
    """Registro completo de una ejecución autónoma."""
    run_id:      str                      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    plan_id:     str                      = ""
    goal:        str                      = ""
    loops:       int                      = 0
    status:      LoopStatus               = LoopStatus.FAILED
    steps_log:   list[dict]               = field(default_factory=list)
    final_text:  str                      = ""
    duration_ms: int                      = 0
    plan_score:  float                    = 0.0
    input_question: Optional[str]         = None

    def log_step(
        self,
        step_id: str,
        description: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str]  = None,
        duration_ms: int      = 0,
        attempt: int          = 1,
    ) -> None:
        self.steps_log.append({
            "step_id":     step_id,
            "description": description[:100],
            "status":      status,
            "result_len":  len(result) if result else 0,
            "error":       error[:100] if error else None,
            "duration_ms": duration_ms,
            "attempt":     attempt,
        })


@dataclass
class AutonomyResult:
    """Resultado del loop de autonomía."""
    text:          str
    trace:         ExecutionTrace
    plan:          Optional[Plan]  = None
    needs_input:   bool            = False
    input_question: Optional[str] = None


class AutonomyLoop:
    """
    Loop de ejecución autónoma.

    Recibe un objetivo + intent, crea un plan,
    ejecuta cada paso y evalúa el resultado.
    """

    def __init__(
        self,
        planner:    Planner,
        executor=None,
        step_evaluator:  Optional[StepEvaluator]  = None,
        plan_evaluator:  Optional[PlanEvaluator]  = None,
        memory=None,
        model_router=None,
        max_loops:   int = MAX_EXECUTION_LOOPS,
        max_steps:   int = MAX_PLAN_STEPS,
        max_retries: int = MAX_RETRIES_PER_STEP,
    ) -> None:
        self._planner        = planner
        self._executor       = executor
        self._step_evaluator = step_evaluator or StepEvaluator()
        self._plan_evaluator = plan_evaluator or PlanEvaluator()
        self._memory         = memory
        self._router         = model_router
        self._max_loops      = max_loops
        self._max_steps      = max_steps
        self._max_retries    = max_retries

    async def run(
        self,
        goal:        str,
        intent_type: str   = "task",
        context:     str   = "",
        request_id:  str   = "",
        use_llm_plan: bool = False,
    ) -> AutonomyResult:
        """Ejecutar el loop completo."""
        started = time.perf_counter()
        trace   = ExecutionTrace(goal=goal)

        logger.info(
            "[%s] AutonomyLoop START | goal=%s | intent=%s",
            request_id, goal[:80], intent_type,
        )

        # ── 1. Crear plan ─────────────────────────────────────
        try:
            plan = await self._planner.plan(
                goal=goal,
                intent_type=intent_type,
                use_llm=use_llm_plan,
                context=context,
            )
            plan.steps = plan.steps[:self._max_steps]
            trace.plan_id = plan.plan_id

            logger.info(
                "[%s] Plan creado | plan_id=%s | pasos=%d",
                request_id, plan.plan_id, len(plan.steps),
            )
        except Exception as e:
            logger.exception("[%s] Error creando plan", request_id)
            trace.status      = LoopStatus.FAILED
            trace.final_text  = f"No pude crear un plan para: {goal}"
            trace.duration_ms = int((time.perf_counter() - started) * 1000)
            return AutonomyResult(text=trace.final_text, trace=trace)

        # ── 2. Ejecutar ciclo ─────────────────────────────────
        loops = 0

        while not plan.done and loops < self._max_loops:
            step = plan.next_step()
            if step is None:
                break

            loops += 1
            plan.mark_running(step.step_id)
            step_started = time.perf_counter()

            logger.info(
                "[%s] Ejecutando paso %s | %s (intento %d)",
                request_id, step.step_id, step.description[:60], step.retries + 1,
            )

            # Ejecutar paso
            output, error = await self._execute_step(step, context, request_id)
            step_ms = int((time.perf_counter() - step_started) * 1000)

            # Evaluar
            evaluation = self._step_evaluator.evaluate(
                step_description=step.description,
                execution_success=(error is None),
                output=output,
                error=error,
                retries_used=step.retries,
                max_retries=self._max_retries,
            )

            trace.log_step(
                step_id=step.step_id,
                description=step.description,
                status=evaluation.status.value,
                result=output,
                error=error,
                duration_ms=step_ms,
                attempt=step.retries + 1,
            )

            logger.info(
                "[%s] Paso %s → %s (score=%.2f)",
                request_id, step.step_id,
                evaluation.status.value, evaluation.score,
            )

            # Actuar según evaluación
            if evaluation.status == EvalStatus.SUCCESS:
                plan.mark_done(step.step_id, output or "")

            elif evaluation.status == EvalStatus.PARTIAL:
                plan.mark_done(step.step_id, output or "")

            elif evaluation.status == EvalStatus.RETRY:
                step.retries += 1
                step.status = StepStatus.PENDING
                logger.info("[%s] Reintentando paso %s", request_id, step.step_id)

            elif evaluation.status == EvalStatus.NEEDS_USER_INPUT:
                trace.status        = LoopStatus.NEEDS_INPUT
                trace.input_question = evaluation.question
                trace.duration_ms   = int((time.perf_counter() - started) * 1000)
                trace.loops         = loops
                return AutonomyResult(
                    text=evaluation.question or "Necesito más información para continuar.",
                    trace=trace,
                    plan=plan,
                    needs_input=True,
                    input_question=evaluation.question,
                )

            elif evaluation.status == EvalStatus.FAILED:
                plan.mark_failed(step.step_id, error or "Error desconocido.")
                # Continuar con el siguiente paso aunque este haya fallado

        # ── 3. Límite de loops ────────────────────────────────
        if loops >= self._max_loops:
            logger.warning(
                "[%s] Límite de loops alcanzado (%d)", request_id, self._max_loops
            )
            trace.status = LoopStatus.LIMIT_HIT

        # ── 4. Evaluar plan completo ──────────────────────────
        plan_eval = self._plan_evaluator.evaluate_plan(
            plan=plan,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        trace.plan_score  = plan_eval.score
        trace.loops       = loops
        trace.duration_ms = int((time.perf_counter() - started) * 1000)

        # ── 5. Sintetizar respuesta final ─────────────────────
        final_text = self._synthesize(plan, plan_eval, goal)
        trace.final_text = final_text

        if plan.done and plan_eval.score >= 0.8:
            trace.status = LoopStatus.COMPLETED
        elif plan_eval.score > 0:
            trace.status = LoopStatus.PARTIAL
        else:
            trace.status = LoopStatus.FAILED

        # ── 6. Guardar en memoria ─────────────────────────────
        self._save_to_memory(goal, final_text, plan.plan_id)

        logger.info(
            "[%s] AutonomyLoop END | status=%s | loops=%d | score=%.2f | %dms",
            request_id, trace.status.value, loops,
            plan_eval.score, trace.duration_ms,
        )

        return AutonomyResult(text=final_text, trace=trace, plan=plan)

    # ── Private ───────────────────────────────────────────────

    async def _execute_step(
        self,
        step: PlanStep,
        context: str,
        request_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Ejecutar un paso. Retorna (output, error)."""

        # ── Tool execution ────────────────────────────────────
        if step.tool and self._executor:
            result = await self._executor.execute_by_name(
                tool_name=step.tool,
                params={"query": step.description},
                context=context,
                request_id=request_id,
            )
            if result.success and result.tool_result:
                return str(result.tool_result.output), None
            return None, result.error

        # ── LLM execution ─────────────────────────────────────
        if step.requires_llm and self._router:
            try:
                from backend.providers.base import GenerateRequest
                prompt = (
                    f"Ejecuta este paso del plan:\n"
                    f"OBJETIVO GENERAL: {context[:200]}\n"
                    f"PASO A EJECUTAR: {step.description}\n\n"
                    f"Proporciona el resultado concreto de este paso. "
                    f"Sé específico y estructurado."
                )
                gen_result = await self._router.generate(
                    GenerateRequest(
                        prompt=prompt,
                        system="Eres NEXUS ejecutando un paso de plan. Sé concreto y directo.",
                        max_tokens=2000,
                    )
                )
                return gen_result.response.text, None
            except Exception as e:
                return None, str(e)[:200]

        return f"Paso registrado: {step.description}", None

    def _synthesize(self, plan: Plan, plan_eval, goal: str) -> str:
        """Sintetizar respuesta final del plan."""
        results = plan.collect_results()

        if not results:
            return (
                f"Procesé tu solicitud sobre: {goal}\n"
                f"Estado: {plan_eval.summary}"
            )

        if plan_eval.score >= 0.8:
            header = f"**Resultado para: {goal}**\n\n"
        else:
            header = (
                f"**Resultado parcial para: {goal}** "
                f"({plan_eval.steps_done}/{plan_eval.steps_total} pasos)\n\n"
            )

        return header + results

    def _save_to_memory(self, goal: str, result: str, plan_id: str) -> None:
        if self._memory is None:
            return
        try:
            from backend.core.memory import MemoryType
            self._memory.remember(
                content=f"[PLAN:{plan_id}] {goal[:100]} → {result[:200]}",
                memory_type=MemoryType.EPISODIC,
                tags=["autonomy", "plan"],
                importance=0.8,
            )
        except Exception:
            pass
