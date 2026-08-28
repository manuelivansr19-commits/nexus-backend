"""
NEXUS Ω — Servidor principal v3.4.0

Rutas EXISTENTES (sin cambios de contrato):
  GET  /                     → Frontend
  HEAD /                     → Health probe Render
  GET  /health               → Estado del sistema
  GET  /api/nexus/status     → Estado del router
  GET  /api/nexus/config     → Configuración pública
  POST /api/nexus/chat       → Chat (ahora acepta history)
  GET  /sw.js                → Service Worker

Rutas NUEVAS (AURA Brain):
  GET  /api/aura/status      → Estado del cerebro AURA
  POST /api/aura/perceive    → Inyectar evento de percepción (simulación)
  GET  /api/aura/simulate    → Ejecutar tick de simulación
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
    ALL_SECRETS,
    APP_VERSION,
    DEFAULT_SYSTEM,
    MAX_HISTORY_TURNS,
    MAX_OUTPUT_TOKENS,
    NEXUS_LOCAL_ONLY,
    REQUEST_TIMEOUT_SECONDS,
    USE_OLLAMA_ONLY,
    logger,
)
from backend.providers import (
    GeminiProvider,
    GenerateRequest,
    GroqProvider,
    LocalProvider,
    Message,
    OllamaProvider,
    OpenRouterProvider,
)
from backend.router import ModelRouter

# AURA Brain
from backend.core.perception import Modality, Perception, PerceptionEvent
from backend.core.memory import Memory, MemoryType
from backend.core.reasoning import Reasoning
from backend.core.planning import Planner
from backend.core.action import ActionExecutor
from backend.core.evaluation import Evaluator
from backend.simulation.engine import SimulationEngine, SimulationScenario


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class HistoryMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=30000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=30000)
    system:  str = Field(default=DEFAULT_SYSTEM, max_length=12000)
    history: Optional[list[HistoryMessage]] = None


class PerceiveRequest(BaseModel):
    modality: str = Field(..., description="text|audio|imu|vision|lidar|system")
    data:     Any = Field(...)
    source:   str = Field(default="api")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SimulateRequest(BaseModel):
    scenario: str = Field(default="idle",
                          description="idle|exploring|conversation|obstacle")
    ticks:    int = Field(default=1, ge=1, le=20)


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

router:      Optional[ModelRouter] = None
http_client: Optional[httpx.AsyncClient] = None

# AURA Brain components (singletons)
perception:  Perception     = Perception()
memory:      Memory         = Memory()
reasoning:   Reasoning      = Reasoning()
planner:     Planner        = Planner()
executor:    ActionExecutor = ActionExecutor()
evaluator:   Evaluator      = Evaluator()
simulation:  SimulationEngine = SimulationEngine()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    global router, http_client

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=15.0, read=REQUEST_TIMEOUT_SECONDS,
            write=15.0, pool=15.0,
        ),
    )

    providers = [
        LocalProvider(http_client),         # Motor local (AURA)
        GeminiProvider(),                   # External primary
        OpenRouterProvider(http_client),    # External fallback 1
        GroqProvider(http_client),          # External fallback 2
        OllamaProvider(http_client),        # External fallback 3
    ]

    router = ModelRouter(providers)

    status = router.provider_status()
    logger.info("=" * 50)
    logger.info("NEXUS Ω v%s iniciado", APP_VERSION)
    logger.info("Modo local: %s", NEXUS_LOCAL_ONLY)
    for name, configured in status.items():
        logger.info("  %-12s %s", name + ":", "✓" if configured else "✗")
    logger.info("=" * 50)

    yield

    await router.shutdown()
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
# RUTAS EXISTENTES (contrato inalterado)
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


@app.head("/")
async def home_head():
    return Response(status_code=200)


@app.get("/health")
async def health():
    assert router is not None
    return {
        "status":            "healthy",
        "system":            "NEXUS",
        "version":           APP_VERSION,
        "providers":         router.provider_status(),
        "models":            router.model_names(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "ollama_only":       USE_OLLAMA_ONLY,
        "local_mode":        router.local_mode_active(),
    }


@app.get("/api/nexus/status")
async def nexus_status():
    assert router is not None
    return {
        "status":            "online",
        "system":            "NEXUS",
        "version":           APP_VERSION,
        "router":            router.provider_status(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "local_mode":        router.local_mode_active(),
    }


@app.get("/api/nexus/config")
async def nexus_config():
    assert router is not None
    return {
        "version":        APP_VERSION,
        **{f"{n}_model": m for n, m in router.model_names().items()},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "multi_provider": True,
        "local_mode":     router.local_mode_active(),
    }


@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    assert router is not None

    request_id = new_request_id()
    started    = time.perf_counter()
    remote_ip  = req.client.host if req.client else "unknown"

    logger.info(
        "[%s] Solicitud recibida | ip=%s | chars=%d",
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

    history: list[Message] = []
    if request.history:
        history = [
            Message(role=m.role, content=m.content)
            for m in request.history[-MAX_HISTORY_TURNS:]
        ]

    system_instruction = request.system.strip() or DEFAULT_SYSTEM

    gen_request = GenerateRequest(
        prompt=request.message,
        system=system_instruction,
        history=history,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    try:
        result  = await router.generate(gen_request)
        elapsed = time.perf_counter() - started

        # Evaluar respuesta
        eval_result = evaluator.evaluate_response(
            result.response.text,
            request.message,
            provider=result.response.provider,
            duration_ms=int(elapsed * 1000),
        )

        logger.info(
            "[%s] OK | provider=%s | fallback=%s | score=%.2f | %.2fs",
            request_id, result.response.provider,
            result.fallback, eval_result.score, elapsed,
        )

        return {
            "success":     True,
            "response":    result.response.text,
            "provider":    result.response.provider,
            "model":       result.response.model,
            "fallback":    result.fallback,
            "local_mode":  result.local_mode,
            "request_id":  request_id,
            "duration_ms": int(elapsed * 1000),
        }

    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception(
            "[%s] TODOS LOS PROVEEDORES FALLARON | %.2fs",
            request_id, elapsed,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message":    "NEXUS no pudo obtener una respuesta de ningún proveedor.",
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
# RUTAS NUEVAS — AURA Brain
# ============================================================

@app.get("/api/aura/status")
async def aura_status():
    """Estado del cerebro AURA y sus subsistemas."""
    assert router is not None
    return {
        "system":      "AURA",
        "version":     APP_VERSION,
        "local_mode":  router.local_mode_active(),
        "providers":   router.provider_status(),
        "brain": {
            "memory":     memory.stats(),
            "evaluation": evaluator.stats(),
            "perception_queue": len(perception.pending()),
        },
        "hardware": {
            "simulation_scenario": simulation.scenario.value,
        },
    }


@app.post("/api/aura/perceive")
async def aura_perceive(request: PerceiveRequest):
    """
    Inyectar un evento de percepción externo.
    Útil para testing, simulación manual, o integración futura.
    """
    try:
        modality = Modality(request.modality)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Modality '{request.modality}' inválida. "
                   f"Opciones: {[m.value for m in Modality]}",
        )

    event = PerceptionEvent(
        modality=modality,
        data=request.data,
        source=request.source,
        confidence=request.confidence,
    )
    perception.receive(event)

    return {
        "success":  True,
        "event_id": event.event_id,
        "modality": modality.value,
        "summary":  event.to_text(),
    }


@app.post("/api/aura/simulate")
async def aura_simulate(request: SimulateRequest):
    """
    Ejecutar N ticks de simulación con el escenario elegido.
    Retorna los eventos generados.
    """
    try:
        scenario = SimulationScenario(request.scenario)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Scenario '{request.scenario}' inválido. "
                   f"Opciones: {[s.value for s in SimulationScenario]}",
        )

    simulation.set_scenario(scenario)
    all_events = []
    for _ in range(request.ticks):
        tick_events = simulation.tick()
        for e in tick_events:
            perception.receive(e)
        all_events.extend(tick_events)

    return {
        "success":   True,
        "scenario":  scenario.value,
        "ticks":     request.ticks,
        "events":    len(all_events),
        "summaries": [e.to_text() for e in all_events[:20]],
    }
