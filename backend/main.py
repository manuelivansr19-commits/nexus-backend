"""
NEXUS Ω — Servidor principal v3.6.0

Capa HTTP únicamente. Toda la lógica en NexusCore.

Rutas PRESERVADAS (contrato inalterado):
  GET  /                     → Frontend
  HEAD /                     → Health probe
  GET  /health               → Estado del sistema
  GET  /api/nexus/status     → Estado del router
  GET  /api/nexus/config     → Configuración
  POST /api/nexus/chat       → Chat principal
  GET  /sw.js                → Service Worker

Rutas Core v3.5 (sin cambios):
  GET  /api/nexus/memory
  POST /api/nexus/memory/clear
  GET  /api/nexus/tools
  POST /api/nexus/intent

Rutas NUEVAS v3.6.0:
  POST /api/nexus/task       → Tarea autónoma multi-paso
  GET  /api/nexus/autonomy   → Estado del autonomy loop

Rutas AURA Brain (sin cambios):
  GET  /api/aura/status
  POST /api/aura/perceive
  POST /api/aura/simulate
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.config import (
    ALL_SECRETS, APP_VERSION, AUTONOMY_ENABLED, DEFAULT_SYSTEM,
    MAX_EXECUTION_LOOPS, MAX_HISTORY_TURNS, MAX_OUTPUT_TOKENS,
    MAX_PLAN_STEPS, MAX_RETRIES_PER_STEP, MEMORY_DB_PATH,
    NEXUS_LOCAL_ONLY, REQUEST_TIMEOUT_SECONDS, USE_OLLAMA_ONLY,
    logger,
)
from backend.providers import (
    GeminiProvider, GenerateRequest, GroqProvider,
    LocalProvider, Message, OllamaProvider, OpenRouterProvider,
)
from backend.router import ModelRouter

from backend.core.nexus import NexusCore
from backend.core.intent import IntentRouter
from backend.core.context import ContextManager
from backend.core.memory import Memory, MemoryType, SQLiteMemoryStore, RAMMemoryStore
from backend.core.executor import Executor
from backend.core.evaluation import Evaluator
from backend.core.planner import Planner
from backend.core.evaluator import StepEvaluator, PlanEvaluator
from backend.core.autonomy import AutonomyLoop
from backend.core.perception import Modality, Perception, PerceptionEvent
from backend.simulation.engine import SimulationEngine, SimulationScenario
from backend.tools.builtin import create_default_registry
from backend.tools.interfaces import register_future_tools


# ============================================================
# REQUEST MODELS
# ============================================================

class HistoryMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=30000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=30000)
    system:  str = Field(default=DEFAULT_SYSTEM, max_length=12000)
    history: Optional[list[HistoryMessage]] = None
    project: str = Field(default="", max_length=100)


class TaskRequest(BaseModel):
    goal:    str = Field(..., max_length=5000, description="Objetivo de la tarea autónoma")
    system:  str = Field(default=DEFAULT_SYSTEM, max_length=12000)
    project: str = Field(default="", max_length=100)
    use_llm_plan: bool = Field(default=True)


class PerceiveRequest(BaseModel):
    modality:   str = Field(...)
    data:       Any = Field(...)
    source:     str = Field(default="api")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SimulateRequest(BaseModel):
    scenario: str = Field(default="idle")
    ticks:    int = Field(default=1, ge=1, le=20)


class IntentDebugRequest(BaseModel):
    message: str = Field(..., max_length=5000)


# ============================================================
# UTILITIES
# ============================================================

def safe_error_message(error: Exception) -> str:
    text = str(error).strip() or "Proveedor no respondió."
    for s in ALL_SECRETS:
        text = text.replace(s, "[REDACTED]")
    return text[:500]


def new_request_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# GLOBALS
# ============================================================

model_router:    Optional[ModelRouter]       = None
http_client:     Optional[httpx.AsyncClient] = None
nexus_core:      Optional[NexusCore]         = None
autonomy_loop_g: Optional[AutonomyLoop]      = None

perception:  Perception       = Perception()
simulation:  SimulationEngine = SimulationEngine()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    global model_router, http_client, nexus_core, autonomy_loop_g

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=15.0, read=REQUEST_TIMEOUT_SECONDS,
            write=15.0, pool=15.0,
        ),
    )

    providers = [
        LocalProvider(http_client),
        GeminiProvider(),
        OpenRouterProvider(http_client),
        GroqProvider(http_client),
        OllamaProvider(http_client),
    ]
    model_router = ModelRouter(providers)

    # Memory
    try:
        store  = SQLiteMemoryStore(MEMORY_DB_PATH)
        memory = Memory(store)
        logger.info("Memory: SQLite (%s)", MEMORY_DB_PATH)
    except Exception:
        memory = Memory(RAMMemoryStore())
        logger.warning("Memory: RAM fallback")

    # Tools
    registry = create_default_registry(memory)
    register_future_tools(registry)

    # Core components
    intent_router   = IntentRouter(registry=registry)
    context_manager = ContextManager(memory=memory)
    executor        = Executor(registry=registry, memory=memory)
    evaluator       = Evaluator()
    planner         = Planner(model_router=model_router)

    # Autonomy Loop
    autonomy_loop_g = AutonomyLoop(
        planner=planner,
        executor=executor,
        step_evaluator=StepEvaluator(),
        plan_evaluator=PlanEvaluator(),
        memory=memory,
        model_router=model_router,
        max_loops=MAX_EXECUTION_LOOPS,
        max_steps=MAX_PLAN_STEPS,
        max_retries=MAX_RETRIES_PER_STEP,
    )

    nexus_core = NexusCore(
        model_router=model_router,
        memory=memory,
        intent_router=intent_router,
        context_manager=context_manager,
        executor=executor,
        evaluator=evaluator,
        autonomy_loop=autonomy_loop_g,
    )

    status = model_router.provider_status()
    logger.info("=" * 50)
    logger.info("NEXUS Ω v%s | Autonomy: %s", APP_VERSION, AUTONOMY_ENABLED)
    logger.info("Limits: steps=%d loops=%d retries=%d",
                MAX_PLAN_STEPS, MAX_EXECUTION_LOOPS, MAX_RETRIES_PER_STEP)
    logger.info("Tools: %s", [t.name for t in registry.list(enabled_only=False)])
    for name, configured in status.items():
        logger.info("  %-12s %s", name + ":", "✓" if configured else "✗")
    logger.info("=" * 50)

    yield

    await model_router.shutdown()
    await http_client.aclose()
    logger.info("NEXUS Ω detenido.")


# ============================================================
# APP
# ============================================================

app = FastAPI(title="NEXUS Ω", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# RUTAS EXISTENTES — contrato inalterado
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


@app.head("/")
async def home_head():
    return Response(status_code=200)


@app.get("/health")
async def health():
    assert model_router is not None
    return {
        "status":            "healthy",
        "system":            "NEXUS",
        "version":           APP_VERSION,
        "providers":         model_router.provider_status(),
        "models":            model_router.model_names(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "ollama_only":       USE_OLLAMA_ONLY,
        "local_mode":        model_router.local_mode_active(),
    }


@app.get("/api/nexus/status")
async def nexus_status():
    assert model_router is not None
    return {
        "status":            "online",
        "system":            "NEXUS",
        "version":           APP_VERSION,
        "router":            model_router.provider_status(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "local_mode":        model_router.local_mode_active(),
    }


@app.get("/api/nexus/config")
async def nexus_config():
    assert model_router is not None
    return {
        "version":           APP_VERSION,
        **{f"{n}_model": m for n, m in model_router.model_names().items()},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "multi_provider":    True,
        "local_mode":        model_router.local_mode_active(),
        "autonomy_enabled":  AUTONOMY_ENABLED,
        "max_plan_steps":    MAX_PLAN_STEPS,
        "max_execution_loops": MAX_EXECUTION_LOOPS,
    }


@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    assert nexus_core is not None

    request_id = new_request_id()
    started    = time.perf_counter()
    remote_ip  = req.client.host if req.client else "unknown"

    logger.info(
        "[%s] Chat | ip=%s | chars=%d",
        request_id, remote_ip, len(request.message),
    )

    if not request.message.strip():
        return {
            "success":    True,
            "response":   "NEXUS: Escuchando. Adelante con tu consulta.",
            "provider":   "system",
            "fallback":   False,
            "request_id": request_id,
        }

    system_instruction = request.system.strip() or DEFAULT_SYSTEM
    history = [
        {"role": m.role, "content": m.content}
        for m in (request.history or [])[-MAX_HISTORY_TURNS:]
    ]

    try:
        result = await nexus_core.process(
            message=request.message,
            system_prompt=system_instruction,
            history=history,
            project=request.project,
            request_id=request_id,
        )

        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] OK | provider=%s | intent=%s | fallback=%s | %.2fs",
            request_id, result.provider, result.intent, result.fallback, elapsed,
        )

        return {
            "success":      True,
            "response":     result.text,
            "provider":     result.provider,
            "model":        result.model,
            "fallback":     result.fallback,
            "local_mode":   result.local_mode,
            "request_id":   request_id,
            "duration_ms":  result.duration_ms,
            "intent":       result.intent,
            "domain":       result.domain,
            "tools_used":   result.tools_used,
        }

    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception("[%s] FALLARON TODOS | %.2fs", request_id, elapsed)
        raise HTTPException(
            status_code=503,
            detail={
                "message":    "NEXUS no pudo obtener una respuesta.",
                "error":      safe_error_message(error),
                "provider":   "none",
                "request_id": request_id,
            },
        ) from error


@app.get("/sw.js")
async def service_worker():
    return Response(
        content=(
            'self.addEventListener("install",e=>self.skipWaiting());\n'
            'self.addEventListener("activate",e=>self.clients.claim());\n'
        ),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============================================================
# RUTAS Core v3.5 (sin cambios)
# ============================================================

@app.get("/api/nexus/memory")
async def nexus_memory():
    assert nexus_core is not None
    stats = nexus_core._memory.stats() if nexus_core._memory else {}
    return {"success": True, "memory": stats}


@app.post("/api/nexus/memory/clear")
async def nexus_memory_clear(body: dict = {}):
    assert nexus_core is not None
    if nexus_core._memory is None:
        return {"success": False, "error": "Memory no disponible."}
    memory_type_str = body.get("type", "working")
    try:
        mt    = MemoryType(memory_type_str)
        count = nexus_core._memory._store.clear(mt)
    except ValueError:
        count = nexus_core._memory._store.clear()
    return {"success": True, "cleared": count, "type": memory_type_str}


@app.get("/api/nexus/tools")
async def nexus_tools():
    assert nexus_core is not None
    if nexus_core._executor is None or nexus_core._executor._registry is None:
        return {"success": True, "tools": []}
    tools = nexus_core._executor._registry.describe()
    stats = nexus_core._executor._registry.stats()
    return {"success": True, "tools": tools, "stats": stats}


@app.post("/api/nexus/intent")
async def nexus_intent(body: IntentDebugRequest):
    assert nexus_core is not None
    if nexus_core._intent is None:
        return {"success": False, "error": "IntentRouter no disponible."}
    result = nexus_core._intent.route(body.message)
    return {
        "success":          True,
        "intent":           result.intent.value,
        "domain":           result.domain.value,
        "confidence":       result.confidence,
        "strategy":         result.strategy.value,
        "requires_tool":    result.requires_tool,
        "candidate_tools":  result.candidate_tools,
        "requires_memory":  result.requires_memory,
        "requires_planning": result.requires_planning,
    }


# ============================================================
# RUTAS NUEVAS v3.6.0 — Autonomy Core
# ============================================================

@app.post("/api/nexus/task")
async def nexus_task(request: TaskRequest, req: Request):
    """
    Ejecutar una tarea autónoma multi-paso.
    El Planner divide el objetivo en pasos y los ejecuta.
    """
    assert nexus_core is not None and autonomy_loop_g is not None

    request_id = new_request_id()
    started    = time.perf_counter()
    remote_ip  = req.client.host if req.client else "unknown"

    logger.info(
        "[%s] Task | ip=%s | goal=%s",
        request_id, remote_ip, request.goal[:80],
    )

    try:
        result = await autonomy_loop_g.run(
            goal=request.goal,
            intent_type="task",
            context=request.goal,
            request_id=request_id,
            use_llm_plan=request.use_llm_plan,
        )

        elapsed = time.perf_counter() - started
        plan    = result.plan

        return {
            "success":       True,
            "response":      result.text,
            "request_id":    request_id,
            "duration_ms":   int(elapsed * 1000),
            "needs_input":   result.needs_input,
            "input_question": result.input_question,
            "trace": {
                "run_id":    result.trace.run_id,
                "status":    result.trace.status.value,
                "loops":     result.trace.loops,
                "score":     result.trace.plan_score,
                "steps":     result.trace.steps_log,
            },
            "plan": plan.summary() if plan else None,
        }

    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception("[%s] Task error | %.2fs", request_id, elapsed)
        raise HTTPException(
            status_code=503,
            detail={
                "message":    "Error en ejecución autónoma.",
                "error":      safe_error_message(error),
                "request_id": request_id,
            },
        ) from error


@app.get("/api/nexus/autonomy")
async def nexus_autonomy():
    """Estado y configuración del Autonomy Core."""
    return {
        "enabled":           AUTONOMY_ENABLED,
        "max_plan_steps":    MAX_PLAN_STEPS,
        "max_execution_loops": MAX_EXECUTION_LOOPS,
        "max_retries_per_step": MAX_RETRIES_PER_STEP,
        "version":           APP_VERSION,
        "safe_actions":      ["read", "analyze", "plan", "calculate", "search", "remember"],
        "forbidden_actions": ["exec", "shell", "file_write", "file_delete", "network_raw"],
    }


# ============================================================
# RUTAS AURA Brain (sin cambios)
# ============================================================

@app.get("/api/aura/status")
async def aura_status():
    assert model_router is not None
    core_status = nexus_core.status() if nexus_core else {}
    return {
        "system":     "AURA",
        "version":    APP_VERSION,
        "local_mode": model_router.local_mode_active(),
        "providers":  model_router.provider_status(),
        "brain":      core_status,
        "hardware":   {"simulation_scenario": simulation.scenario.value},
    }


@app.post("/api/aura/perceive")
async def aura_perceive(request: PerceiveRequest):
    try:
        modality = Modality(request.modality)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Modality '{request.modality}' inválida.",
        )
    event = PerceptionEvent(
        modality=modality, data=request.data,
        source=request.source, confidence=request.confidence,
    )
    perception.receive(event)
    return {"success": True, "event_id": event.event_id,
            "modality": modality.value, "summary": event.to_text()}


@app.post("/api/aura/simulate")
async def aura_simulate(request: SimulateRequest):
    try:
        scenario = SimulationScenario(request.scenario)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Scenario inválido.")
    simulation.set_scenario(scenario)
    all_events = []
    for _ in range(request.ticks):
        tick_events = simulation.tick()
        for e in tick_events:
            perception.receive(e)
        all_events.extend(tick_events)
    return {
        "success":   True, "scenario": scenario.value,
        "ticks":     request.ticks, "events": len(all_events),
        "summaries": [e.to_text() for e in all_events[:20]],
    }
