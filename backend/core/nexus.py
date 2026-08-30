"""
NEXUS Ω — NexusCore v3.6.0

Orquestador principal con Autonomy Core integrado.

Flujo completo:
  mensaje
    → IntentRouter   (¿qué tipo de solicitud es?)
    → Ejecutar según estrategia:
        DIRECT    → respuesta inmediata sin modelo
        TOOL      → invocar herramienta
        LLM       → llamada simple al modelo
        AUTONOMY  → PLAN → EXECUTE → EVALUATE loop
    → Memory         (guardar conversación)
    → NexusResponse

El modelo es un componente del sistema, no el sistema completo.
NEXUS CORE sigue existiendo aunque Gemini esté caído.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import AUTONOMY_ENABLED, DEFAULT_SYSTEM, logger
from backend.providers.base import GenerateRequest, Message
from backend.core.intent import IntentResult, IntentStrategy, IntentType


@dataclass
class NexusResponse:
    """Respuesta unificada — compatible con contrato JSON existente."""
    text:           str
    provider:       str
    model:          str
    fallback:       bool
    local_mode:     bool
    duration_ms:    int
    intent:         str               = "general"
    domain:         str               = "general"
    tools_used:     list[str]         = field(default_factory=list)
    from_memory:    bool              = False
    context_tokens: int               = 0
    plan_id:        Optional[str]     = None
    plan_steps:     int               = 0
    autonomy_loops: int               = 0


class NexusCore:
    """
    Cerebro de NEXUS Ω.

    El Core NO conoce SDKs específicos.
    Solo habla con interfaces abstractas:
      ModelRouter, Memory, IntentRouter, ContextManager,
      Executor, Evaluator, AutonomyLoop.
    """

    def __init__(
        self,
        model_router=None,
        memory=None,
        intent_router=None,
        context_manager=None,
        executor=None,
        evaluator=None,
        autonomy_loop=None,
    ) -> None:
        self._router          = model_router
        self._memory          = memory
        self._intent          = intent_router
        self._context         = context_manager
        self._executor        = executor
        self._evaluator       = evaluator
        self._autonomy_loop   = autonomy_loop

    async def process(
        self,
        message:       str,
        system_prompt: str               = DEFAULT_SYSTEM,
        history:       Optional[list[dict]] = None,
        project:       str               = "",
        request_id:    str               = "",
    ) -> NexusResponse:
        """Procesar mensaje completo."""
        started = time.perf_counter()

        # ── 1. Intent routing ─────────────────────────────────
        intent_result: Optional[IntentResult] = None
        if self._intent:
            intent_result = self._intent.route(message)
            logger.info(
                "[%s] Intent: %s | domain: %s | strategy: %s | conf: %.2f",
                request_id,
                intent_result.intent.value,
                intent_result.domain.value,
                intent_result.strategy.value,
                intent_result.confidence,
            )

            # DIRECT — sin modelo
            if intent_result.strategy == IntentStrategy.DIRECT:
                self._save_to_memory(message, intent_result.direct_response or "")
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(
                    text=intent_result.direct_response or "",
                    provider="system",
                    model="deterministic",
                    fallback=False,
                    local_mode=False,
                    duration_ms=elapsed,
                    intent=intent_result.intent.value,
                    domain=intent_result.domain.value,
                )

            # AUTONOMY — loop multi-paso
            if (
                intent_result.strategy == IntentStrategy.AUTONOMY
                and AUTONOMY_ENABLED
                and self._autonomy_loop
            ):
                return await self._run_autonomy(
                    message, intent_result, system_prompt,
                    history, started, request_id
                )

        # ── 2. Tool execution ─────────────────────────────────
        tool_context_strings: list[str] = []
        tools_used: list[str] = []

        if (
            self._executor
            and intent_result
            and intent_result.requires_tool
            and intent_result.candidate_tools
        ):
            exec_results = await self._executor.execute_candidates(
                candidate_names=intent_result.candidate_tools,
                params={"query": message},
                context=message,
                request_id=request_id,
            )
            tool_context_strings = self._executor.collect_context_strings(exec_results)
            tools_used = [r.tool_name for r in exec_results if r.success]

            if tool_context_strings and intent_result.strategy == IntentStrategy.TOOL:
                combined = "\n".join(tool_context_strings)
                self._save_to_memory(message, combined)
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(
                    text=combined,
                    provider="tool",
                    model=tools_used[0] if tools_used else "tool",
                    fallback=False,
                    local_mode=False,
                    duration_ms=elapsed,
                    intent=intent_result.intent.value if intent_result else "tool",
                    domain=intent_result.domain.value if intent_result else "system",
                    tools_used=tools_used,
                )

        # ── 3. Context assembly ───────────────────────────────
        context_tokens    = 0
        assembled_message = message
        final_history     = history or []

        if self._context:
            bundle = self._context.assemble(
                message=message,
                system_prompt=system_prompt,
                intent=intent_result,
                history=history or [],
                tool_results=tool_context_strings,
                project=project,
            )
            assembled_message = bundle.assembled_prompt
            final_history     = bundle.history
            context_tokens    = bundle.estimated_tokens

        # ── 4. Generate with model ────────────────────────────
        if self._router is None:
            elapsed = int((time.perf_counter() - started) * 1000)
            return NexusResponse(
                text="NEXUS: Sin proveedor de modelo disponible.",
                provider="none",
                model="none",
                fallback=False,
                local_mode=False,
                duration_ms=elapsed,
                intent=intent_result.intent.value if intent_result else "general",
                domain=intent_result.domain.value if intent_result else "general",
            )

        history_messages = [
            Message(role=h["role"], content=h["content"])
            for h in final_history
            if h.get("role") in ("user", "assistant")
        ]

        gen_request = GenerateRequest(
            prompt=assembled_message,
            system=system_prompt,
            history=history_messages,
        )

        result = await self._router.generate(gen_request)

        # ── 5. Evaluate ───────────────────────────────────────
        elapsed = int((time.perf_counter() - started) * 1000)
        if self._evaluator:
            self._evaluator.evaluate_response(
                result.response.text,
                message,
                provider=result.response.provider,
                duration_ms=elapsed,
            )

        # ── 6. Save to memory ─────────────────────────────────
        self._save_to_memory(message, result.response.text)

        return NexusResponse(
            text=result.response.text,
            provider=result.response.provider,
            model=result.response.model,
            fallback=result.fallback,
            local_mode=result.local_mode,
            duration_ms=elapsed,
            intent=intent_result.intent.value if intent_result else "general",
            domain=intent_result.domain.value if intent_result else "general",
            tools_used=tools_used,
            context_tokens=context_tokens,
        )

    async def _run_autonomy(
        self,
        message:       str,
        intent_result: IntentResult,
        system_prompt: str,
        history:       Optional[list[dict]],
        started:       float,
        request_id:    str,
    ) -> NexusResponse:
        """Ejecutar loop de autonomía para intents complejos."""
        logger.info(
            "[%s] Iniciando AutonomyLoop | intent=%s",
            request_id, intent_result.intent.value,
        )

        autonomy_result = await self._autonomy_loop.run(
            goal=message,
            intent_type=intent_result.intent.value,
            context=message,
            request_id=request_id,
            use_llm_plan=True,
        )

        elapsed = int((time.perf_counter() - started) * 1000)

        # Guardar en memoria
        self._save_to_memory(message, autonomy_result.text)

        # Si necesita input del usuario
        if autonomy_result.needs_input:
            return NexusResponse(
                text=autonomy_result.input_question or "Necesito más información.",
                provider="autonomy",
                model="planner",
                fallback=False,
                local_mode=False,
                duration_ms=elapsed,
                intent=intent_result.intent.value,
                domain=intent_result.domain.value,
            )

        plan = autonomy_result.plan
        return NexusResponse(
            text=autonomy_result.text,
            provider=autonomy_result.trace.steps_log[-1]["status"] if autonomy_result.trace.steps_log else "autonomy",
            model=f"autonomy:{autonomy_result.trace.status.value}",
            fallback=False,
            local_mode=False,
            duration_ms=elapsed,
            intent=intent_result.intent.value,
            domain=intent_result.domain.value,
            plan_id=plan.plan_id if plan else None,
            plan_steps=len(plan.steps) if plan else 0,
            autonomy_loops=autonomy_result.trace.loops,
        )

    def _save_to_memory(self, user_msg: str, assistant_msg: str) -> None:
        if self._memory is None:
            return
        try:
            self._memory.conversation.add_user(user_msg)
            self._memory.conversation.add_assistant(assistant_msg)
        except Exception:
            logger.exception("NexusCore: error guardando en memoria.")

    def status(self) -> dict:
        return {
            "model_router":    bool(self._router),
            "memory":          bool(self._memory),
            "intent_router":   bool(self._intent),
            "context_manager": bool(self._context),
            "executor":        bool(self._executor),
            "evaluator":       bool(self._evaluator),
            "autonomy_loop":   bool(self._autonomy_loop),
            "memory_stats":    self._memory.stats() if self._memory else {},
            "autonomy_enabled": AUTONOMY_ENABLED,
        }
