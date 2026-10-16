"""NEXUS Omega -- NexusCore v3.8.4"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from backend.config import AUTONOMY_ENABLED, DEFAULT_SYSTEM, logger
from backend.inference.models import InferenceRequest
from backend.inference.errors import AllRuntimesFailed, CloudInferenceBlocked
from backend.core.intent import IntentResult, IntentStrategy, IntentType

# Confirmaciones cortas sin contenido propio ("sí", "ok", "dale", "continúa"...).
# Cuando el usuario responde así y existe una tarea pendiente, se interpreta
# como continuación de esa tarea en vez de clasificarse como CHAT genérico.
# No captura frases más largas con verbo propio (esas ya las cubre
# IntentRouter._FOLLOW_UP_PATTERNS, o el patrón de continuación de abajo).
_SHORT_CONFIRMATIONS = re.compile(
    r"^(si|s\u00ed|no|ok|okay|dale|vale|listo|adelante|"
    r"contin\u00faa|continua|ahora|hazlo|va|perfecto|de acuerdo)[\s!\.,]*$",
    re.IGNORECASE,
)

# Marcadores de continuación ("ahora calcula...", "luego compáralo...")
# seguidos de contenido propio (a diferencia de _SHORT_CONFIRMATIONS, que
# solo captura el marcador solo). Señalan una NUEVA operación dentro de
# la tarea activa, no una simple confirmación.
_CONTINUATION_MARKERS = re.compile(
    r"^(ahora|luego|despu\u00e9s|despues|y ahora|tambi\u00e9n|tambien|"
    r"adem\u00e1s|ademas)\s+\S",
    re.IGNORECASE,
)

# Ventana de tiempo (segundos) dentro de la cual una tarea COMPLETED se
# considera "recién terminada" y puede reactivarse ante una continuación
# explícita (v3.8 Bloque 4). Fuera de esta ventana no se reutiliza, para
# no inventar relación con una tarea antigua no relacionada. Valor
# conservador y ajustable; no pretende ser una heurística de contenido.
_TASK_REACTIVATION_WINDOW_SECONDS = 300

# Órdenes explícitas de detener/cancelar la tarea activa.
_STOP_TRIGGERS = re.compile(
    r"^(det[e\u00e9]n(?:te)?\s+la\s+tarea|cancela\s+la\s+tarea|"
    r"para\s+la\s+tarea|detener\s+tarea|cancelar\s+tarea)[\s!\.,]*$",
    re.IGNORECASE,
)

# Mensaje corto que termina en un pronombre demostrativo vago ("...esto",
# "...eso", "...lo") sin más contenido que lo precise. Señal conservadora
# de que falta información esencial para ejecutar la solicitud (v3.8
# Bloque 5, PEDIR_INFO). Deliberadamente estrecho: no intenta detectar
# ambigüedad en general, solo el caso claro de un objeto vago como único
# referente de la acción, en un mensaje corto y sin tarea activa que le
# dé contexto (si hay tarea activa, "esto" ya tiene un referente válido
# y el mensaje se resuelve como CONTINUAR más arriba, no llega aquí).
_VAGUE_OBJECT_PATTERN = re.compile(r"\b(esto|eso|ello|lo)\b\s*[\.\!\?]*\s*$", re.IGNORECASE)
_VAGUE_OBJECT_MAX_WORDS = 8


class Behavior(str, Enum):
    """
    Las 9 categorías de comportamiento (v3.8 Bloque 5). Responden a
    "¿qué debe hacer NEXUS?", no a "¿cómo lo ejecuta?" — esa segunda
    pregunta la sigue resolviendo el código de ejecución ya existente
    (InferenceLayer, AutonomyLoop, Executor, etc.), sin cambios en este
    bloque. Este enum es puramente una etiqueta de clasificación,
    calculada por NexusCore a partir de intent_result y de sus propias
    señales internas (active_task, stop triggers, confirmaciones
    cortas), sin tocar IntentRouter/IntentResult/IntentStrategy.
    """
    CONTINUAR              = "continuar"
    DETENER                = "detener"
    PEDIR_INFO             = "pedir_info"
    RESPONDER              = "responder"
    CONSULTAR_CONOCIMIENTO = "consultar_conocimiento"
    CALCULAR               = "calcular"
    ANALIZAR               = "analizar"
    PLANIFICAR             = "planificar"
    EJECUTAR_HERRAMIENTA   = "ejecutar_herramienta"


def _autonomy_behavior_for_intent(intent_type: IntentType) -> Behavior:
    """Dentro de la estrategia AUTONOMY, qué categoría de negocio le
    corresponde según el IntentType real que la disparó."""
    if intent_type in (IntentType.TASK, IntentType.DESIGN):
        return Behavior.PLANIFICAR
    if intent_type == IntentType.ANALYSIS:
        return Behavior.ANALIZAR
    if intent_type == IntentType.RESEARCH:
        return Behavior.CONSULTAR_CONOCIMIENTO
    # Fallback conservador si AUTONOMY_INTENTS llegara a ampliarse en el
    # futuro con un IntentType no contemplado aquí todavía.
    return Behavior.PLANIFICAR


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
    behavior:       Optional[str]  = None

class NexusCore:
    def __init__(self, inference=None, memory=None, intent_router=None,
                 context_manager=None, executor=None, evaluator=None,
                 autonomy_loop=None, knowledge_engine=None,
                 task_state=None, model_router=None,
                 persistent_memory=None):
        self._persistent_memory = persistent_memory
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

        # 0. Memoria persistente (opcional): comandos explicitos "recuerda / olvida /
        # corrige". Capa aparte: no modifica la tarea activa ni el contexto (solo LEE
        # la tarea activa para conservar su task_id) y no guarda este turno en la
        # memoria de conversacion, para que "olvida" realmente olvide.
        if self._persistent_memory is not None:
            pm_response = self._handle_persistent_memory(message, project, request_id, started)
            if pm_response is not None:
                return pm_response

        # 1. Intent routing
        intent_result = None
        active_task    = None
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
                    intent=intent_result.intent.value, domain=intent_result.domain.value,
                    behavior=Behavior.RESPONDER.value)

            # Detener tarea activa: orden explícita, se resuelve antes que
            # cualquier otra interpretación del mensaje.
            if self._task_state and _STOP_TRIGGERS.match(message.strip()):
                pending = self._task_state.list_pending()
                if pending:
                    task = pending[0]
                    from backend.core.task_state import TaskStatus
                    self._task_state.update(task.task_id, TaskStatus.CANCELLED)
                    self._save_to_memory(message, f"Tarea '{task.description[:80]}' detenida.")
                    elapsed = int((time.perf_counter() - started) * 1000)
                    return NexusResponse(
                        text=f"Tarea '{task.description[:80]}' detenida.",
                        provider="task_state", model="cancelled",
                        fallback=False, local_mode=True, duration_ms=elapsed,
                        intent=intent_result.intent.value, domain=intent_result.domain.value,
                        task_id=task.task_id, behavior=Behavior.DETENER.value)

            # Confirmación corta ("sí", "ok", "dale"...) con tarea pendiente:
            # se trata como FOLLOW_UP para reutilizar la lógica ya existente,
            # en vez de dejar que caiga a clasificación genérica de CHAT.
            if (self._task_state and intent_result.intent != IntentType.FOLLOW_UP
                    and _SHORT_CONFIRMATIONS.match(message.strip())):
                if self._task_state.list_pending():
                    intent_result.intent = IntentType.FOLLOW_UP
                    logger.info("[%s] Confirmación corta detectada, tratando como FOLLOW_UP", request_id)

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
                            task_id=task.task_id, behavior=Behavior.CONTINUAR.value)
                    elif task.status.value == "pending" or task.status.value == "in_progress":
                        elapsed = int((time.perf_counter() - started) * 1000)
                        return NexusResponse(
                            text=f"Tarea '{task.description[:80]}' esta en estado: {task.status.value}. Procesando ahora...",
                            provider="task_state", model="memory",
                            fallback=False, local_mode=True, duration_ms=elapsed,
                            intent="follow_up", domain=task.domain, task_id=task.task_id,
                            behavior=Behavior.CONTINUAR.value)

            # Continuación de tarea con nueva operación (v3.8 Bloque 2/3):
            # un mensaje que arranca con un marcador de continuación
            # ("ahora", "luego", "después"...) seguido de contenido propio,
            # mientras existe una tarea activa, se trata como una NUEVA
            # operación DENTRO de esa tarea: se conserva el task_id y se
            # inyecta el contexto de la tarea anterior, en vez de perderlo
            # o de tratarlo como conversación nueva sin relación.
            #
            # Se calcula ANTES del chequeo de AUTONOMY (a diferencia del
            # Bloque 2, que lo excluía) para que también aplique cuando la
            # nueva operación clasifica como AUTONOMY: en ese caso se le
            # pasa el task_id existente a _run_autonomy, que reutiliza la
            # tarea en vez de crear una segunda (Bloque 3).
            if self._task_state and _CONTINUATION_MARKERS.match(message.strip()):
                pending = self._task_state.list_pending()
                if pending:
                    active_task = pending[0]
                    logger.info(
                        "[%s] Continuación detectada: nueva operación '%s' dentro de tarea %s",
                        request_id, intent_result.intent.value, active_task.task_id)
                else:
                    # Ciclo de vida (v3.8 Bloque 4): no hay ninguna tarea
                    # PENDING/IN_PROGRESS, pero puede existir una recién
                    # COMPLETED que el usuario quiere seguir trabajando.
                    # Solo se reactiva si: (a) hay un marcador de
                    # continuación explícito (ya verificado arriba) y
                    # (b) la tarea completada cae dentro de la ventana de
                    # reactivación. CANCELLED/FAILED nunca se reactivan
                    # (list_recently_completed solo consulta 'completed').
                    recent_done = self._task_state.list_recently_completed(
                        within_seconds=_TASK_REACTIVATION_WINDOW_SECONDS, limit=1)
                    if recent_done:
                        active_task = recent_done[0]
                        from backend.core.task_state import TaskStatus
                        self._task_state.update(active_task.task_id, TaskStatus.IN_PROGRESS)
                        logger.info(
                            "[%s] Tarea %s reactivada (completada recientemente) para nueva operación '%s'",
                            request_id, active_task.task_id, intent_result.intent.value)

            # PEDIR_INFO (v3.8 Bloque 5): el mensaje termina en un objeto
            # vago ("...esto", "...eso") sin más contenido que lo precise,
            # y no hay ninguna tarea activa que le dé referente. En vez de
            # inventar una respuesta o disparar AUTONOMY sobre una
            # solicitud sin sentido claro, se pide la información
            # faltante. Deliberadamente conservador: solo el caso más
            # claro, no un sistema general de detección de ambigüedad.
            if (active_task is None and intent_result.intent != IntentType.FOLLOW_UP
                    and _VAGUE_OBJECT_PATTERN.search(message.strip())
                    and len(message.strip().split()) <= _VAGUE_OBJECT_MAX_WORDS):
                elapsed = int((time.perf_counter() - started) * 1000)
                clarifying = "¿Podrías darme más detalles sobre a qué te refieres exactamente?"
                self._save_to_memory(message, clarifying)
                return NexusResponse(text=clarifying, provider="system", model="pedir_info",
                    fallback=False, local_mode=True, duration_ms=elapsed,
                    intent=intent_result.intent.value, domain=intent_result.domain.value,
                    behavior=Behavior.PEDIR_INFO.value)

            if (intent_result.strategy == IntentStrategy.AUTONOMY
                    and AUTONOMY_ENABLED and self._autonomy_loop):
                autonomy_response = await self._run_autonomy(message, intent_result,
                    system_prompt, history, started, request_id,
                    existing_task_id=active_task.task_id if active_task else None)
                autonomy_response.behavior = _autonomy_behavior_for_intent(intent_result.intent).value
                return autonomy_response

        # Mensaje enriquecido con el contexto de la tarea activa (si aplica)
        # para las etapas de knowledge/tools/context/inferencia. El mensaje
        # original se conserva sin modificar para clasificación de intent
        # (ya hecha arriba) y para guardar en memoria de conversación.
        contextual_message = message
        if active_task is not None:
            context_parts = [f"[Contexto de tarea activa: {active_task.description[:200]}]"]
            if active_task.result:
                context_parts.append(f"[Resultado previo de la tarea: {active_task.result[:300]}]")
            context_parts.append(message)
            contextual_message = "\n".join(context_parts)

        # 2. Knowledge retrieval
        knowledge_used = 0
        if self._knowledge:
            try:
                domain_hint = intent_result.domain.value if intent_result else None
                kctx = self._knowledge.get_for_context(query=contextual_message,
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
                params={"query": contextual_message}, context=contextual_message, request_id=request_id)
            tool_context_strings = self._executor.collect_context_strings(exec_results)
            tools_used = [r.tool_name for r in exec_results if r.success]
            if tool_context_strings and intent_result.strategy == IntentStrategy.TOOL:
                # Respuesta directa al usuario: sin la etiqueta [TOOL:...], que es
                # solo contexto interno para el modelo.
                user_texts = []
                for r in exec_results:
                    tr = getattr(r, "tool_result", None)
                    if getattr(r, "success", False) and tr:
                        fmt = (getattr(tr, "to_user_string", None)
                               or getattr(tr, "to_context_string", None))
                        if fmt:
                            user_texts.append(fmt())
                combined = "\n".join(user_texts) or "\n".join(tool_context_strings)
                self._save_to_memory(message, combined)
                elapsed = int((time.perf_counter() - started) * 1000)
                return NexusResponse(text=combined, provider="tool",
                    model=tools_used[0] if tools_used else "tool",
                    fallback=False, local_mode=True, duration_ms=elapsed,
                    intent=intent_result.intent.value if intent_result else "tool",
                    domain=intent_result.domain.value if intent_result else "system",
                    tools_used=tools_used, knowledge_used=knowledge_used,
                    task_id=active_task.task_id if active_task else None,
                    behavior=Behavior.EJECUTAR_HERRAMIENTA.value)

        # 4. Context assembly
        context_tokens    = 0
        assembled_message = contextual_message
        final_history     = history or []
        if self._context:
            try:
                bundle = self._context.assemble(message=contextual_message,
                    system_prompt=system_prompt, intent=intent_result,
                    history=history or [], tool_results=tool_context_strings, project=project,
                    **self._pm_context_kwargs(active_task))
                assembled_message = bundle.assembled_prompt
                final_history     = bundle.history
                context_tokens    = bundle.estimated_tokens
            except Exception:
                # Si el ensamblaje de contexto falla (memoria corrupta,
                # knowledge engine caído, etc.), no debe tumbar toda la
                # respuesta con un 503: se sigue con el mensaje crudo y
                # el historial sin recortar como fallback mínimo.
                logger.exception(
                    "[%s] Context assembly falló, usando fallback minimo", request_id)
                assembled_message = contextual_message
                final_history     = history or []

        # 5. Inference
        result = await self._infer(assembled_message, system_prompt, final_history, request_id)
        elapsed = int((time.perf_counter() - started) * 1000)
        if self._evaluator:
            self._evaluator.evaluate_response(result["text"], message,
                provider=result["provider"], duration_ms=elapsed)
        self._save_to_memory(message, result["text"])

        # Categoría de comportamiento para la ruta de inferencia normal:
        # CALCULATION -> CALCULAR; cualquier otra cosa que llegue hasta
        # aquí (CHAT, QUESTION, SYSTEM sin match directo, MEMORY_QUERY
        # sin herramienta aplicable...) es una solicitud conversacional
        # normal -> RESPONDER. Si no hay IntentRouter configurado, no hay
        # base para clasificar y se deja en None.
        behavior = None
        if intent_result is not None:
            behavior = (Behavior.CALCULAR if intent_result.intent == IntentType.CALCULATION
                        else Behavior.RESPONDER).value

        return NexusResponse(text=result["text"], provider=result["provider"],
            model=result["model"], fallback=result["fallback"],
            local_mode=result["is_local"], duration_ms=elapsed,
            intent=intent_result.intent.value if intent_result else "general",
            domain=intent_result.domain.value if intent_result else "general",
            tools_used=tools_used, context_tokens=context_tokens,
            knowledge_used=knowledge_used, is_local=result["is_local"],
            task_id=active_task.task_id if active_task else None,
            behavior=behavior)

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
                # Detalle de runtimes (solo logs): al usuario no se le muestra el error crudo.
                raw = [(str(rt), str(msg)) for rt, msg in e.runtime_errors]
                logger.error("[%s] Detalle de runtimes (solo logs): %s", request_id,
                             " | ".join(f"{rt}: {msg[:300]}" for rt, msg in raw))
                reasons = {}
                for rt, msg in raw:
                    up = msg.upper()
                    if "RATE_LIMITED" in up or "RESOURCE_EXHAUSTED" in up or "429" in up:
                        reasons[rt] = "rate_limit"
                    elif "HIGH DEMAND" in up or "OVERLOADED" in up or "503" in up or "529" in up:
                        reasons[rt] = "overloaded"
                    else:
                        reasons[rt] = "error"
                try:
                    from backend.router import degraded_text
                    text = degraded_text(reasons)
                except Exception:
                    text = ("NEXUS está en modo degradado: no hay motores de IA disponibles "
                            "por ahora. Esta respuesta no fue generada por un modelo de IA. "
                            "Intenta de nuevo en un minuto.")
                return {"text": text,
                        "provider": "none", "model": "none", "fallback": False, "is_local": False}
            except CloudInferenceBlocked:
                return {"text": (
                    "NEXUS esta en modo LOCAL_ONLY. "
                    "No hay motor de inferencia local disponible en este momento. "
                    "Instala Ollama localmente para habilitar respuestas completas."),
                    "provider": "none", "model": "none", "fallback": False, "is_local": True}
        if self._legacy_router is not None:
            from backend.router import AllProvidersUnavailable, degraded_text
            try:
                from backend.providers.base import GenerateRequest, Message as PMessage
                gen_req = GenerateRequest(prompt=prompt, system=system,
                    history=[PMessage(role=h["role"], content=h["content"])
                             for h in history if h.get("role") in ("user","assistant")])
                result = await self._legacy_router.generate(gen_req)
                return {"text": result.response.text, "provider": result.response.provider,
                        "model": result.response.model, "fallback": result.fallback,
                        "is_local": result.local_mode}
            except AllProvidersUnavailable as e:
                logger.warning("[%s] Modo degradado, proveedores no disponibles: %s",
                               request_id, e.reasons)
                return {"text": degraded_text(e.reasons),
                        "provider": "none", "model": "none", "fallback": False, "is_local": False}
            except Exception:
                # El detalle va solo a logs; al usuario no se le muestra el error crudo.
                logger.exception("[%s] Error de inferencia (router legacy)", request_id)
                return {"text": degraded_text({}),
                        "provider": "none", "model": "none", "fallback": False, "is_local": False}
        return {"text": "NEXUS: Sin motor de inferencia configurado.",
                "provider": "none", "model": "none", "fallback": False, "is_local": False}

    def _finalize_autonomy_task(self, task_id, autonomy_result) -> None:
        """Estado de la tarea segun el resultado REAL del loop (v3.8 Bloque 4A).

        COMPLETED solo si el plan termino bien. FAILED nunca se marca COMPLETED.
        PARTIAL / LIMIT_HIT / PROVIDER_LIMIT / NEEDS_INPUT quedan IN_PROGRESS para
        que "continua" retome la misma tarea (sin persistencia de pasos, reinicia el plan).
        """
        if not (task_id and self._task_state):
            return
        from backend.core.task_state import TaskStatus
        try:
            trace = getattr(autonomy_result, "trace", None)
            status = getattr(trace, "status", None)
            value = getattr(status, "value", status)
            text = getattr(autonomy_result, "text", None)
            if getattr(autonomy_result, "needs_input", False):
                self._task_state.update(task_id, TaskStatus.IN_PROGRESS, result=text)
            elif value == "completed":
                self._task_state.update(task_id, TaskStatus.COMPLETED, result=text)
            elif value == "failed":
                self._task_state.update(task_id, TaskStatus.FAILED, result=text,
                                        error="El plan no pudo completarse.")
            else:
                self._task_state.update(task_id, TaskStatus.IN_PROGRESS, result=text)
        except Exception:
            logger.exception("No se pudo actualizar el estado de la tarea %s", task_id)

    async def _run_autonomy(self, message, intent_result, system_prompt,
                            history, started, request_id,
                            existing_task_id=None) -> NexusResponse:
        from backend.core.task_state import TaskStatus
        task_id = existing_task_id
        if self._task_state:
            if existing_task_id:
                # Continuidad estructural (v3.8 Bloque 3): ya existe una
                # tarea activa y la nueva operación también resultó en
                # estrategia AUTONOMY. Se reutiliza esa tarea en vez de
                # crear una segunda, para no perder el hilo:
                #   MISMA TAREA + NUEVA OPERACIÓN + NUEVO INTENT = MISMO task_id
                self._task_state.update(task_id, TaskStatus.IN_PROGRESS)
            else:
                task = self._task_state.create(description=message,
                    intent=intent_result.intent.value, domain=intent_result.domain.value)
                task_id = task.task_id
                self._task_state.update(task_id, TaskStatus.IN_PROGRESS)
        try:
            autonomy_result = await self._autonomy_loop.run(
                goal=message, intent_type=intent_result.intent.value,
                context=message, request_id=request_id, use_llm_plan=True)
            elapsed = int((time.perf_counter() - started) * 1000)
            self._save_to_memory(message, autonomy_result.text)
            self._finalize_autonomy_task(task_id, autonomy_result)
            if autonomy_result.needs_input:
                return NexusResponse(text=autonomy_result.text or autonomy_result.input_question or "Necesito mas informacion.",
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
        except Exception:
            # El detalle va solo a logs; ni la tarea ni el usuario reciben el error crudo.
            logger.exception("[%s] Error en AutonomyLoop", request_id)
            if task_id and self._task_state:
                from backend.core.task_state import TaskStatus
                self._task_state.update(task_id, TaskStatus.FAILED, error="Error interno del plan.")
            elapsed = int((time.perf_counter() - started) * 1000)
            return NexusResponse(text="No pude completar la tarea por un error interno. Intenta de nuevo; si persiste, avisa para revisar los registros.",
                provider="error", model="none", fallback=False, local_mode=False,
                duration_ms=elapsed, intent=intent_result.intent.value,
                domain=intent_result.domain.value, task_id=task_id)

    def _handle_persistent_memory(self, message, project, request_id, started):
        """Comandos de memoria persistente. Devuelve NexusResponse, o None si no es un comando."""
        active_task_id = None
        if self._task_state:
            try:
                pending = self._task_state.list_pending()
                active_task_id = pending[0].task_id if pending else None
            except Exception:
                active_task_id = None
        try:
            text = self._persistent_memory.handle_message(
                message, project=project, task_id=active_task_id)
        except Exception:
            logger.exception("[%s] Memoria persistente fallo; el mensaje sigue su curso normal",
                             request_id)
            return None
        if text is None:
            return None
        logger.info("[%s] Comando de memoria persistente atendido", request_id)
        elapsed = int((time.perf_counter() - started) * 1000)
        return NexusResponse(text=text, provider="memory", model="persistent",
            fallback=False, local_mode=True, duration_ms=elapsed,
            intent="memory_command", domain="general", is_local=True,
            task_id=active_task_id, behavior=Behavior.RESPONDER.value)

    def _pm_context_kwargs(self, active_task):
        """kwargs extra para ContextManager.assemble; vacio si no hay memoria persistente."""
        if self._persistent_memory is None:
            return {}
        return {"task_id": getattr(active_task, "task_id", None)}

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
