"""
NEXUS Ω — Autonomy Loop v3.8.0

Usa InferenceLayer para los pasos LLM.
NO conoce ningún proveedor concreto.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import (
    MAX_EXECUTION_LOOPS, MAX_PLAN_STEPS, MAX_RETRIES_PER_STEP, logger,
)
from backend.core.evaluator import EvalStatus, PlanEvaluator, StepEvaluator
from backend.core.planner import Plan, PlanStep, Planner, StepStatus


class LoopStatus(str, Enum):
    COMPLETED   = "completed"
    PARTIAL     = "partial"
    FAILED      = "failed"
    LIMIT_HIT   = "limit_hit"
    NEEDS_INPUT = "needs_input"


@dataclass
class ExecutionTrace:
    run_id:      str        = field(default_factory=lambda: str(uuid.uuid4())[:8])
    plan_id:     str        = ""
    goal:        str        = ""
    loops:       int        = 0
    status:      LoopStatus = LoopStatus.FAILED
    steps_log:   list[dict] = field(default_factory=list)
    final_text:  str        = ""
    duration_ms: int        = 0
    plan_score:  float      = 0.0
    input_question: Optional[str] = None

    def log_step(self, step_id, description, status, result=None,
                 error=None, duration_ms=0, attempt=1) -> None:
        self.steps_log.append({
            "step_id": step_id, "description": description[:100],
            "status": status, "result_len": len(result) if result else 0,
            "error": error[:100] if error else None,
            "duration_ms": duration_ms, "attempt": attempt,
        })


@dataclass
class AutonomyResult:
    text:           str
    trace:          ExecutionTrace
    plan:           Optional[Plan] = None
    needs_input:    bool           = False
    input_question: Optional[str]  = None


class AutonomyLoop:
    """
    Loop autónomo — usa InferenceLayer para los pasos LLM.
    Totalmente desacoplado de proveedores concretos.
    """

    def __init__(
        self,
        planner:        Planner,
        executor=None,
        step_evaluator: Optional[StepEvaluator] = None,
        plan_evaluator: Optional[PlanEvaluator] = None,
        memory=None,
        inference_layer=None,   # InferenceLayer — nueva arquitectura
        model_router=None,      # legacy fallback
        max_loops:   int = MAX_EXECUTION_LOOPS,
        max_steps:   int = MAX_PLAN_STEPS,
        max_retries: int = MAX_RETRIES_PER_STEP,
    ) -> None:
        self._planner        = planner
        self._executor       = executor
        self._step_evaluator = step_evaluator or StepEvaluator()
        self._plan_evaluator = plan_evaluator or PlanEvaluator()
        self._memory         = memory
        self._inference      = inference_layer
        self._legacy_router  = model_router
        self._max_loops      = max_loops
        self._max_steps      = max_steps
        self._max_retries    = max_retries

    async def run(
        self,
        goal:         str,
        intent_type:  str  = "task",
        context:      str  = "",
        request_id:   str  = "",
        use_llm_plan: bool = False,
    ) -> AutonomyResult:
        started = time.perf_counter()
        trace   = ExecutionTrace(goal=goal)

        logger.info("[%s] AutonomyLoop START | goal=%s | intent=%s",
                    request_id, goal[:80], intent_type)

        try:
            plan = await self._planner.plan(
                goal=goal, intent_type=intent_type,
                use_llm=use_llm_plan, context=context,
            )
            plan.steps = plan.steps[:self._max_steps]
            trace.plan_id = plan.plan_id
        except Exception as e:
            logger.exception("[%s] Error creando plan", request_id)
            trace.status     = LoopStatus.FAILED
            trace.final_text = f"No pude crear un plan para: {goal}"
            trace.duration_ms = int((time.perf_counter() - started) * 1000)
            return AutonomyResult(text=trace.final_text, trace=trace)

        loops = 0
        while not plan.done and loops < self._max_loops:
            step = plan.next_step()
            if step is None:
                break

            loops += 1
            plan.mark_running(step.step_id)
            step_started = time.perf_counter()

            output, error = await self._execute_step(step, context, request_id)
            step_ms = int((time.perf_counter() - step_started) * 1000)

            evaluation = self._step_evaluator.evaluate(
                step_description=step.description,
                execution_success=(error is None),
                output=output, error=error,
                retries_used=step.retries, max_retries=self._max_retries,
            )

            trace.log_step(step.step_id, step.description,
                           evaluation.status.value, output, error, step_ms,
                           attempt=step.retries + 1)

            if evaluation.status == EvalStatus.SUCCESS:
                plan.mark_done(step.step_id, output or "")
            elif evaluation.status == EvalStatus.PARTIAL:
                plan.mark_done(step.step_id, output or "")
            elif evaluation.status == EvalStatus.RETRY:
                step.retries += 1
                step.status = StepStatus.PENDING
            elif evaluation.status == EvalStatus.NEEDS_USER_INPUT:
                trace.status        = LoopStatus.NEEDS_INPUT
                trace.input_question = evaluation.question
                trace.duration_ms   = int((time.perf_counter() - started) * 1000)
                trace.loops         = loops
                return AutonomyResult(
                    text=evaluation.question or "Necesito más información.",
                    trace=trace, plan=plan, needs_input=True,
                    input_question=evaluation.question,
                )
            elif evaluation.status == EvalStatus.FAILED:
                plan.mark_failed(step.step_id, error or "Error desconocido.")

        if loops >= self._max_loops:
            logger.warning("[%s] Límite de loops (%d)", request_id, self._max_loops)
            trace.status = LoopStatus.LIMIT_HIT

        plan_eval = self._plan_evaluator.evaluate_plan(
            plan=plan,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        trace.plan_score  = plan_eval.score
        trace.loops       = loops
        trace.duration_ms = int((time.perf_counter() - started) * 1000)

        final_text = self._synthesize(plan, plan_eval, goal)
        trace.final_text = final_text

        if plan.done and plan_eval.score >= 0.8:
            trace.status = LoopStatus.COMPLETED
        elif plan_eval.score > 0:
            trace.status = LoopStatus.PARTIAL
        else:
            trace.status = LoopStatus.FAILED

        self._save_to_memory(goal, final_text, plan.plan_id)

        logger.info("[%s] AutonomyLoop END | status=%s | loops=%d | score=%.2f | %dms",
                    request_id, trace.status.value, loops, plan_eval.score, trace.duration_ms)

        return AutonomyResult(text=final_text, trace=trace, plan=plan)

    async def _execute_step(self, step: PlanStep, context: str, request_id: str):
        """Ejecutar paso — via tool o via InferenceLayer."""
        # Tool execution
        if step.tool and self._executor:
            result = await self._executor.execute_by_name(
                tool_name=step.tool, params={"query": step.description},
                context=context, request_id=request_id,
            )
            if result.success and result.tool_result:
                return str(result.tool_result.output), None
            return None, result.error

        # LLM via InferenceLayer (nueva arquitectura)
        if step.requires_llm and self._inference is not None:
            try:
                from backend.inference.models import InferenceRequest
                prompt = (
                    f"Ejecuta este paso del plan:\n"
                    f"OBJETIVO: {context[:200]}\n"
                    f"PASO: {step.description}\n\n"
                    f"Proporciona el resultado concreto de este paso."
                )
                result = await self._inference.generate(
                    InferenceRequest(
                        prompt=prompt,
                        system="Eres NEXUS ejecutando un paso de plan. Sé concreto.",
                        max_tokens=2000,
                    )
                )
                return result.text, None
            except Exception as e:
                return None, str(e)[:200]

        # LLM via legacy router
        if step.requires_llm and self._legacy_router is not None:
            try:
                from backend.providers.base import GenerateRequest
                prompt = f"Objetivo: {context[:200]}\nPaso: {step.description}\nResultado:"
                gen = await self._legacy_router.generate(
                    GenerateRequest(prompt=prompt,
                                    system="Ejecuta este paso. Sé concreto.",
                                    max_tokens=2000)
                )
                return gen.response.text, None
            except Exception as e:
                return None, str(e)[:200]

        return f"Paso registrado: {step.description}", None

    def _synthesize(self, plan, plan_eval, goal: str) -> str:
        results = plan.collect_results()
        if not results:
            return f"Procesé tu solicitud: {goal}\nEstado: {plan_eval.summary}"
        header = (f"**Resultado para: {goal}**\n\n" if plan_eval.score >= 0.8
                  else f"**Resultado parcial ({plan_eval.steps_done}/{plan_eval.steps_total})**\n\n")
        return header + results

    def _save_to_memory(self, goal, result, plan_id) -> None:
        if not self._memory:
            return
        try:
            from backend.core.memory import MemoryType
            self._memory.remember(
                content=f"[PLAN:{plan_id}] {goal[:100]} → {result[:200]}",
                memory_type=MemoryType.EPISODIC, tags=["autonomy"], importance=0.8,
            )
        except Exception:
            pass
