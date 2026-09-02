"""
NEXUS Ω — NexusCore v3.7.0

Integra KnowledgeEngine en el pipeline de procesamiento.

Flujo actualizado:
  mensaje
    → IntentRouter
    → KnowledgeEngine.get_for_context()  ← NUEVO
    → Executor (tools)
    → ContextManager (con knowledge)
    → ModelRouter
    → Memory
    → NexusResponse
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import AUTONOMY_ENABLED, DEFAULT_SYSTEM, logger
from backend.providers.base import GenerateRequest, Message
from backend.core.intent import IntentResult, IntentStrategy


@dataclass
class NexusResponse:
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
    knowledge_used: int               = 0   # entradas de knowledge usadas


class NexusCore:
    """
    Cerebro de NEXUS Ω.
    No conoce SDKs. Solo interfaces abstractas.
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
        knowledge_engine=None,
    ) -> None:
        self._router          = model_router
        self._memory          = memory
        self._intent          = intent_router
        self._context         = context_manager
        self._executor        = executor
        self._evaluator       = evaluator
        self._autonomy_loop   = autonomy_loop
        self._knowledge       = knowledge_engine

    async def process(
        self,
        message:       str,
        system_prompt: str               = DEFAULT_SYSTEM,
        history:       Optional[list[dict]] = None,
        project:       str               = "",
        request_id:    str               = "",
    ) -> NexusResponse:
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

            if intent_result.strategy == IntentStrategy.DIRECT:
                self._save_to_memory(message, intent_result.direct_response or "")
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(
                    text=intent_result.direct_response or "",
                    provider="system", model="deterministic",
                    fallback=False, local_mode=False,
                    duration_ms=elapsed,
                    intent=intent_result.intent.value,
                    domain=intent_result.domain.value,
                )

            if (
                intent_result.strategy == IntentStrategy.AUTONOMY
                and AUTONOMY_ENABLED
                and self._autonomy_loop
            ):
                return await self._run_autonomy(
                    message, intent_result, system_prompt,
                    history, started, request_id
                )

        # ── 2. Knowledge retrieval ────────────────────────────
        knowledge_used = 0
        if self._knowledge:
            try:
                domain_hint = intent_result.domain.value if intent_result else None
                kctx = self._knowledge.get_for_context(
                    query=message,
                    domain=domain_hint if domain_hint != "general" else None,
                    limit=5,
                )
                knowledge_used = kctx.total_found
                if knowledge_used:
                    logger.info(
                        "[%s] Knowledge: %d entradas | dominios: %s",
                        request_id, knowledge_used, kctx.domains_covered,
                    )
            except Exception:
                pass

        # ── 3. Tool execution ─────────────────────────────────
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
                    fallback=False, local_mode=False,
                    duration_ms=elapsed,
                    intent=intent_result.intent.value if intent_result else "tool",
                    domain=intent_result.domain.value if intent_result else "system",
                    tools_used=tools_used,
                    knowledge_used=knowledge_used,
                )

        # ── 4. Context assembly ───────────────────────────────
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
            logger.info(
                "[%s] Context: ~%d tokens | sources: %s",
                request_id, context_tokens, bundle.sources_used,
            )

        # ── 5. Generate ───────────────────────────────────────
        if self._router is None:
            elapsed = int((time.perf_counter() - started) * 1000)
            return NexusResponse(
                text="NEXUS: Sin proveedor de modelo disponible.",
                provider="none", model="none",
                fallback=False, local_mode=False,
                duration_ms=elapsed,
                intent=intent_result.intent.value if intent_result else "general",
                domain=intent_result.domain.value if intent_result else "general",
            )

        history_messages = [
            Message(role=h["role"], content=h["content"])
            for h in final_history
            if h.get("role") in ("user", "assistant")
        ]

        result = await self._router.generate(
            GenerateRequest(
                prompt=assembled_message,
                system=system_prompt,
                history=history_messages,
            )
        )

        elapsed = int((time.perf_counter() - started) * 1000)

        if self._evaluator:
            self._evaluator.evaluate_response(
                result.response.text, message,
                provider=result.response.provider,
                duration_ms=elapsed,
            )

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
            knowledge_used=knowledge_used,
        )

    async def _run_autonomy(
        self, message, intent_result, system_prompt,
        history, started, request_id,
    ) -> NexusResponse:
        logger.info("[%s] AutonomyLoop | intent=%s", request_id, intent_result.intent.value)
        autonomy_result = await self._autonomy_loop.run(
            goal=message,
            intent_type=intent_result.intent.value,
            context=message,
            request_id=request_id,
            use_llm_plan=True,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        self._save_to_memory(message, autonomy_result.text)
        if autonomy_result.needs_input:
            return NexusResponse(
                text=autonomy_result.input_question or "Necesito más información.",
                provider="autonomy", model="planner",
                fallback=False, local_mode=False,
                duration_ms=elapsed,
                intent=intent_result.intent.value,
                domain=intent_result.domain.value,
            )
        plan = autonomy_result.plan
        return NexusResponse(
            text=autonomy_result.text,
            provider="autonomy",
            model=f"autonomy:{autonomy_result.trace.status.value}",
            fallback=False, local_mode=False,
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
            "model_router":     bool(self._router),
            "memory":           bool(self._memory),
            "intent_router":    bool(self._intent),
            "context_manager":  bool(self._context),
            "executor":         bool(self._executor),
            "evaluator":        bool(self._evaluator),
            "autonomy_loop":    bool(self._autonomy_loop),
            "knowledge_engine": bool(self._knowledge),
            "memory_stats":     self._memory.stats() if self._memory else {},
            "knowledge_stats":  self._knowledge.stats() if self._knowledge else {},
            "autonomy_enabled": AUTONOMY_ENABLED,
        }
