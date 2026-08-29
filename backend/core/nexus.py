"""
NEXUS Ω — NexusCore.

Orquestador central. Coordina todos los subsistemas.

Flujo completo:
  mensaje
    → IntentRouter  (¿qué quiere el usuario?)
    → Executor      (¿hay una tool que lo resuelva?)
    → ContextManager (ensamblar contexto para el modelo)
    → ModelRouter   (generar respuesta)
    → Memory        (guardar conversación)
    → NexusResponse

El modelo es un componente del sistema, no el sistema completo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import DEFAULT_SYSTEM, logger
from backend.providers.base import GenerateRequest, Message
from backend.core.intent import IntentResult, IntentStrategy


@dataclass
class NexusResponse:
    """Respuesta unificada del Core — compatible con el contrato JSON existente."""
    text:        str
    provider:    str
    model:       str
    fallback:    bool
    local_mode:  bool
    duration_ms: int
    intent:      str              = "general"
    domain:      str              = "general"
    tools_used:  list[str]        = field(default_factory=list)
    from_memory: bool             = False
    context_tokens: int           = 0


class NexusCore:
    """
    Cerebro de NEXUS Ω.

    Recibe un mensaje y coordina:
      IntentRouter → Executor → ContextManager → ModelRouter → Memory

    El Core NO conoce:
      - SDKs de Gemini, OpenRouter, Groq, Ollama
      - Detalles de HTTP
      - Lógica de FastAPI

    Solo habla con interfaces abstractas.
    """

    def __init__(
        self,
        model_router,
        memory=None,
        intent_router=None,
        context_manager=None,
        executor=None,
        evaluator=None,
    ) -> None:
        self._router   = model_router
        self._memory   = memory
        self._intent   = intent_router
        self._context  = context_manager
        self._executor = executor
        self._evaluator = evaluator

    async def process(
        self,
        message:       str,
        system_prompt: str        = DEFAULT_SYSTEM,
        history:       Optional[list[dict]] = None,
        project:       str        = "",
        request_id:    str        = "",
    ) -> NexusResponse:
        """
        Procesar un mensaje completo.
        Retorna NexusResponse con todos los metadatos.
        """
        started = time.perf_counter()

        # ── 1. Intent routing ─────────────────────────────────
        intent_result: Optional[IntentResult] = None
        if self._intent:
            intent_result = self._intent.route(message)
            logger.info(
                "[%s] Intent: %s | domain: %s | strategy: %s | conf: %.2f",
                request_id,
                intent_result.intent,
                intent_result.domain.value,
                intent_result.strategy.value,
                intent_result.confidence,
            )

            # Respuesta directa sin modelo
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
                    intent=intent_result.intent,
                    domain=intent_result.domain.value,
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
            tools_used = [
                r.tool_name for r in exec_results
                if r.success
            ]

            # Si una tool resuelve completamente la solicitud, retornar directo
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
                    intent=intent_result.intent if intent_result else "tool",
                    domain=intent_result.domain.value if intent_result else "system",
                    tools_used=tools_used,
                )

        # ── 3. Context assembly ───────────────────────────────
        context_tokens = 0
        assembled_message = message
        final_history = history or []

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
                request_id,
                context_tokens,
                bundle.sources_used,
            )

        # ── 4. Generate with model ────────────────────────────
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
            intent=intent_result.intent if intent_result else "general",
            domain=intent_result.domain.value if intent_result else "general",
            tools_used=tools_used,
            context_tokens=context_tokens,
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
        """Estado del Core y sus subsistemas."""
        return {
            "model_router":     bool(self._router),
            "memory":           bool(self._memory),
            "intent_router":    bool(self._intent),
            "context_manager":  bool(self._context),
            "executor":         bool(self._executor),
            "evaluator":        bool(self._evaluator),
            "memory_stats":     self._memory.stats() if self._memory else {},
        }
