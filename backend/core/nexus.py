"""NEXUS Omega -- NexusCore v3.8.0"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
from backend.config import AUTONOMY_ENABLED, DEFAULT_SYSTEM, logger
from backend.inference.models import InferenceRequest
from backend.inference.errors import AllRuntimesFailed, CloudInferenceBlocked
from backend.core.intent import IntentResult, IntentStrategy, IntentType

@dataclass
class NexusResponse:
    text:           str
    provider:       str
    model:          str
    fallback:       bool
    local_mode:     bool
    duration_ms:    int
    intent:         str            = "general"
    domain:         str            = "general"
    tools_used:     list           = field(default_factory=list)
    from_memory:    bool           = False
    context_tokens: int            = 0
    plan_id:        Optional[str]  = None
    plan_steps:     int            = 0
    autonomy_loops: int            = 0
    knowledge_used: int            = 0
    is_local:       bool           = False
    task_id:        Optional[str]  = None

class NexusCore:
    def __init__(self, inference=None, memory=None, intent_router=None,
                 context_manager=None, executor=None, evaluator=None,
                 autonomy_loop=None, knowledge_engine=None,
                 task_state=None, model_router=None):
        self._inference     = inference
        self._legacy_router = model_router
        self._memory        = memory
        self._intent        = intent_router
        self._context       = context_manager
        self._executor      = executor
        self._evaluator     = evaluator
        self._autonomy_loop = autonomy_loop
        self._knowledge     = knowledge_engine
        self._task_state    = task_state

    async def process(self, message, system_prompt=DEFAULT_SYSTEM,
                      history=None, project="", request_id="") -> NexusResponse:
        started = time.perf_counter()

        # 1. Intent routing
        intent_result = None
        if self._intent:
            intent_result = self._intent.route(message)
            logger.info("[%s] Intent: %s | domain: %s | strategy: %s",
                request_id, intent_result.intent.value,
                intent_result.domain.value, intent_result.strategy.value)

            if intent_result.strategy == IntentStrategy.DIRECT:
                self._save_to_memory(message, intent_result.direct_response or "")
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(text=intent_result.direct_response or "",
                    provider="system", model="deterministic",
                    fallback=False, local_mode=True, duration_ms=elapsed,
                    intent=intent_result.intent.value, domain=intent_result.domain.value)

            # Follow-up: buscar tarea pendiente
            if intent_result.intent == IntentType.FOLLOW_UP and self._task_state:
                pending = self._task_state.list_pending()
                if pending:
                    task = pending[0]
                    if task.result:
                        elapsed = int((time.perf_counter() - started) * 1000)
                        return NexusResponse(text=task.result, provider="task_state",
                            model="memory", fallback=False, local_mode=True,
                            duration_ms=elapsed, intent="follow_up", domain=task.domain,
                            task_id=task.task_id)
                    elif task.status.value == "pending" or task.status.value == "in_progress":
                        elapsed = int((time.perf_counter() - started) * 1000)
                        return NexusResponse(
                            text=f"Tarea '{task.description[:80]}' esta en estado: {task.status.value}. Procesando ahora...",
                            provider="task_state", model="memory",
                            fallback=False, local_mode=True, duration_ms=elapsed,
                            intent="follow_up", domain=task.domain, task_id=task.task_id)

            if (intent_result.strategy == IntentStrategy.AUTONOMY
                    and AUTONOMY_ENABLED and self._autonomy_loop):
                return await self._run_autonomy(message, intent_result,
                    system_prompt, history, started, request_id)

        # 2. Knowledge retrieval
        knowledge_used = 0
        if self._knowledge:
            try:
                domain_hint = intent_result.domain.value if intent_result else None
                kctx = self._knowledge.get_for_context(query=message,
                    domain=domain_hint if domain_hint != "general" else None, limit=5)
                knowledge_used = kctx.total_found
            except Exception:
                pass

        # 3. Tool execution
        tool_context_strings = []
        tools_used = []
        if (self._executor and intent_result and intent_result.requires_tool
                and intent_result.candidate_tools):
            exec_results = await self._executor.execute_candidates(
                candidate_names=intent_result.candidate_tools,
                params={"query": message}, context=message, request_id=request_id)
            tool_context_strings = self._executor.collect_context_strings(exec_results)
            tools_used = [r.tool_name for r in exec_results if r.success]
            if tool_context_strings and intent_result.strategy == IntentStrategy.TOOL:
                combined = "\n".join(tool_context_strings)
                self._save_to_memory(message, combined)
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(text=combined, provider="tool",
                    model=tools_used[0] if tools_used else "tool",
                    fallback=False, local_mode=True, duration_ms=elapsed,
                    intent=intent_result.intent.value if intent_result else "tool",
                    domain=intent_result.domain.value if intent_result else "system",
                    tools_used=tools_used, knowledge_used=knowledge_used)

        # 4. Context assembly
        context_tokens    = 0
        assembled_message = message
        final_history     = history or []
        if self._context:
            bundle = self._context.assemble(message=message,
                system_prompt=system_prompt, intent=intent_result,
                history=history or [], tool_results=tool_context_strings, project=project)
            assembled_message = bundle.assembled_prompt
            final_history     = bundle.history
            context_tokens    = bundle.estimated_tokens

        # 5. Inference
        result = await self._infer(assembled_message, system_prompt, final_history, request_id)
        elapsed = int((time.perf_counter() - started) * 1000)
        if self._evaluator:
            self._evaluator.evaluate_response(result["text"], message,
                provider=result["provider"], duration_ms=elapsed)
        self._save_to_memory(message, result["text"])
        return NexusResponse(text=result["text"], provider=result["provider"],
            model=result["model"], fallback=result["fallback"],
            local_mode=result["is_local"], duration_ms=elapsed,
            intent=intent_result.intent.value if intent_result else "general",
            domain=intent_result.domain.value if intent_result else "general",
            tools_used=tools_used, context_tokens=context_tokens,
            knowledge_used=knowledge_used, is_local=result["is_local"])

    async def _infer(self, prompt, system, history, request_id="") -> dict:
        from backend.inference.models import Message as IMessage
        if self._inference is not None:
            try:
                inf_req = InferenceRequest(prompt=prompt, system=system,
                    history=[IMessage(role=h["role"], content=h["content"])
                             for h in history if h.get("role") in ("user","assistant")],
                    request_id=request_id)
                result = await self._inference.generate(inf_req)
                return {"text": result.text, "provider": result.runtime_name,
                        "model": result.model, "fallback": result.fallback_used,
                        "is_local": result.is_local}
            except AllRuntimesFailed as e:
                logger.error("[%s] Todos los runtimes fallaron", request_id)
                info = "; ".join(f"{rt}: {msg[:50]}" for rt, msg in e.runtime_errors[:3])
                return {"text": (
                    f"NEXUS no pudo completar la solicitud ahora mismo.\n"
                    f"Razon: todos los motores de inferencia fallaron.\n"
                    f"Detalle: {info}\n"
                    f"Que puedes hacer: intenta de nuevo en unos momentos, "
                    f"o verifica la conexion a Internet si usas proveedores cloud."),
                    "provider": "none", "model": "none", "fallback": False, "is_local": False}
            except CloudInferenceBlocked:
                return {"text": (
                    "NEXUS esta en modo LOCAL_ONLY. "
                    "No hay motor de inferencia local disponible en este momento. "
                    "Instala Ollama localmente para habilitar respuestas completas."),
                    "provider": "none", "model": "none", "fallback": False, "is_local": True}
        if self._legacy_router is not None:
            try:
                from backend.providers.base import GenerateRequest, Message as PMessage
                gen_req = GenerateRequest(prompt=prompt, system=system,
                    history=[PMessage(role=h["role"], content=h["content"])
                             for h in history if h.get("role") in ("user","assistant")])
                result = await self._legacy_router.generate(gen_req)
                return {"text": result.response.text, "provider": result.response.provider,
                        "model": result.response.model, "fallback": result.fallback,
                        "is_local": result.local_mode}
            except Exception as e:
                return {"text": f"Error de inferencia: {str(e)[:200]}",
                        "provider": "none", "model": "none", "fallback": False, "is_local": False}
        return {"text": "NEXUS: Sin motor de inferencia configurado.",
                "provider": "none", "model": "none", "fallback": False, "is_local": False}

    async def _run_autonomy(self, message, intent_result, system_prompt,
                            history, started, request_id) -> NexusResponse:
        task_id = None
        if self._task_state:
            task = self._task_state.create(description=message,
                intent=intent_result.intent.value, domain=intent_result.domain.value)
            task_id = task.task_id
            self._task_state.update(task_id, __import__("backend.core.task_state",
                fromlist=["TaskStatus"]).TaskStatus.IN_PROGRESS)
        try:
            autonomy_result = await self._autonomy_loop.run(
                goal=message, intent_type=intent_result.intent.value,
                context=message, request_id=request_id, use_llm_plan=True)
            elapsed = int((time.perf_counter() - started) * 1000)
            self._save_to_memory(message, autonomy_result.text)
            if task_id and self._task_state:
                from backend.core.task_state import TaskStatus
                self._task_state.update(task_id, TaskStatus.COMPLETED, result=autonomy_result.text)
            if autonomy_result.needs_input:
                return NexusResponse(text=autonomy_result.input_question or "Necesito mas informacion.",
                    provider="autonomy", model="planner", fallback=False, local_mode=False,
                    duration_ms=elapsed, intent=intent_result.intent.value,
                    domain=intent_result.domain.value, task_id=task_id)
            plan = autonomy_result.plan
            return NexusResponse(text=autonomy_result.text, provider="autonomy",
                model=f"autonomy:{autonomy_result.trace.status.value}",
                fallback=False, local_mode=False, duration_ms=elapsed,
                intent=intent_result.intent.value, domain=intent_result.domain.value,
                plan_id=plan.plan_id if plan else None,
                plan_steps=len(plan.steps) if plan else 0,
                autonomy_loops=autonomy_result.trace.loops, task_id=task_id)
        except Exception as e:
            if task_id and self._task_state:
                from backend.core.task_state import TaskStatus
                self._task_state.update(task_id, TaskStatus.FAILED, error=str(e)[:200])
            elapsed = int((time.perf_counter() - started) * 1000)
            return NexusResponse(text=f"No pude completar la tarea: {str(e)[:200]}",
                provider="error", model="none", fallback=False, local_mode=False,
                duration_ms=elapsed, intent=intent_result.intent.value,
                domain=intent_result.domain.value, task_id=task_id)

    def _save_to_memory(self, user_msg, assistant_msg):
        if not self._memory:
            return
        try:
            self._memory.conversation.add_user(user_msg)
            self._memory.conversation.add_assistant(assistant_msg)
        except Exception:
            pass

    def status(self) -> dict:
        inference_info = {}
        if self._inference:
            inference_info = {"mode": self._inference._policy.mode.value,
                              "cloud_allowed": self._inference._policy.cloud_allowed,
                              "runtimes": self._inference._registry.stats()}
        return {"inference_layer": bool(self._inference), "inference": inference_info,
                "legacy_router": bool(self._legacy_router), "memory": bool(self._memory),
                "intent_router": bool(self._intent), "context_manager": bool(self._context),
                "executor": bool(self._executor), "evaluator": bool(self._evaluator),
                "autonomy_loop": bool(self._autonomy_loop),
                "knowledge_engine": bool(self._knowledge),
                "task_state": bool(self._task_state),
                "memory_stats": self._memory.stats() if self._memory else {},
                "knowledge_stats": self._knowledge.stats() if self._knowledge else {},
                "task_stats": self._task_state.stats() if self._task_state else {},
                "autonomy_enabled": AUTONOMY_ENABLED}
