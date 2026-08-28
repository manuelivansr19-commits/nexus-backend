"""
NEXUS Ω — Servidor principal.

Endpoints:
  GET  /                → Frontend (index.html)
  HEAD /                → Health probe Render
  GET  /health          → Estado del sistema
  GET  /api/nexus/status → Estado del router
  GET  /api/nexus/config → Configuración pública
  POST /api/nexus/chat  → Chat principal
  GET  /sw.js           → Service Worker
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.config import (
    APP_VERSION,
    DEFAULT_SYSTEM,
    MAX_HISTORY_TURNS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    USE_OLLAMA_ONLY,
    ALL_SECRETS,
    logger,
)
from backend.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenRouterProvider,
)
from backend.providers.base import GenerateRequest, Message
from backend.router import ModelRouter


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class HistoryMessage(BaseModel):
    role: str = Field(
        ...,
        pattern="^(user|assistant)$",
        description="Rol del mensaje: 'user' o 'assistant'",
    )
    content: str = Field(
        ...,
        max_length=30000,
    )


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=30000)
    system: str = Field(default=DEFAULT_SYSTEM, max_length=12000)
    history: Optional[list[HistoryMessage]] = Field(
        default=None,
        description="Historial de conversación (opcional). Máximo configurable por MAX_HISTORY_TURNS.",
    )


# ============================================================
# UTILITIES
# ============================================================

def safe_error_message(error: Exception) -> str:
    text = str(error).strip() or "Proveedor no respondió."
    for secret in ALL_SECRETS:
        text = text.replace(secret, "[REDACTED]")
    return text[:500]


def new_request_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# GLOBALS
# ============================================================

router: Optional[ModelRouter] = None
http_client: Optional[httpx.AsyncClient] = None


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    global router, http_client

    # Crear cliente HTTP compartido
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=15.0,
            read=REQUEST_TIMEOUT_SECONDS,
            write=15.0,
            pool=15.0,
        ),
    )

    # Inicializar providers
    providers = [
        GeminiProvider(),
        OpenRouterProvider(http_client),
        GroqProvider(http_client),
        OllamaProvider(http_client),
    ]

    router = ModelRouter(providers)

    status = router.provider_status()
    logger.info("=" * 50)
    logger.info("NEXUS Ω iniciado | v%s", APP_VERSION)
    for name, configured in status.items():
        logger.info("  %s: %s", name, "✓" if configured else "✗")
    logger.info("=" * 50)

    yield

    # Shutdown
    await router.shutdown()
    await http_client.aclose()
    logger.info("NEXUS Ω detenido.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NEXUS Ω",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
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
        "status": "healthy",
        "system": "NEXUS",
        "version": APP_VERSION,
        "providers": router.provider_status(),
        "models": router.model_names(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "ollama_only": USE_OLLAMA_ONLY,
    }


@app.get("/api/nexus/status")
async def nexus_status():
    assert router is not None
    return {
        "status": "online",
        "system": "NEXUS",
        "version": APP_VERSION,
        "router": router.provider_status(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


@app.get("/api/nexus/config")
async def nexus_config():
    assert router is not None
    return {
        "version": APP_VERSION,
        **{
            f"{name}_model": model
            for name, model in router.model_names().items()
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "multi_provider": True,
    }


@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    assert router is not None

    request_id = new_request_id()
    started = time.perf_counter()
    remote_ip = req.client.host if req.client else "unknown"

    logger.info(
        "[%s] Solicitud recibida | ip=%s | chars=%d",
        request_id, remote_ip, len(request.message),
    )

    # Mensaje vacío → respuesta estática
    if not request.message.strip():
        return {
            "success": True,
            "response": "NEXUS: Escuchando. Adelante con tu consulta.",
            "provider": "system",
            "fallback": False,
            "request_id": request_id,
        }

    # Construir historial
    history: list[Message] = []
    if request.history:
        # Limitar turnos para no exceder contexto
        trimmed = request.history[-MAX_HISTORY_TURNS:]
        history = [
            Message(role=m.role, content=m.content)
            for m in trimmed
        ]

    system_instruction = request.system.strip() or DEFAULT_SYSTEM

    gen_request = GenerateRequest(
        prompt=request.message,
        system=system_instruction,
        history=history,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    try:
        result = await router.generate(gen_request)

        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] RESPUESTA OK | provider=%s | fallback=%s | duración=%.2fs",
            request_id,
            result.response.provider,
            result.fallback,
            elapsed,
        )

        return {
            "success": True,
            "response": result.response.text,
            "provider": result.response.provider,
            "model": result.response.model,
            "fallback": result.fallback,
            "request_id": request_id,
            "duration_ms": int(elapsed * 1000),
        }

    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception(
            "[%s] TODOS LOS PROVEEDORES FALLARON | duración=%.2fs",
            request_id, elapsed,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "NEXUS no pudo obtener una respuesta de ningún proveedor.",
                "error": safe_error_message(error),
                "provider": "none",
                "request_id": request_id,
            },
        ) from error


@app.get("/sw.js")
async def service_worker():
    return Response(
        content=(
            'self.addEventListener("install", e => self.skipWaiting());\n'
            'self.addEventListener("activate", e => self.clients.claim());\n'
        ),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
