import asyncio
import logging
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURACIÓN
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("NEXUS-BACKEND")

app = FastAPI(
    title="NEXUS AI",
    version="2.5.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VARIABLES DE ENTORNO
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-1.5-flash",
).strip()

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "",
).strip()

OLLAMA_API_KEY = os.getenv(
    "OLLAMA_API_KEY",
    "",
).strip()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3",
).strip()

USE_OLLAMA_ONLY = (
    os.getenv(
        "USE_OLLAMA_ONLY",
        "false",
    ).lower()
    == "true"
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "600",
    )
)

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        "120",
    )
)


# ============================================================
# CLIENTE GEMINI
# ============================================================

gemini_client: Optional[genai.Client] = None

if GEMINI_API_KEY and not USE_OLLAMA_ONLY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
else:
    if USE_OLLAMA_ONLY:
        logger.info(
            "Modo exclusivo Ollama activado."
        )
    elif not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY no está configurada."
        )


# ============================================================
# MODELOS DE DATOS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        default="",
        max_length=10000,
    )

    system: str = Field(
        default=(
            "Eres NEXUS, un sistema avanzado de "
            "inteligencia artificial enfocado en "
            "estrategia, psicología aplicada, PNL "
            "y desarrollo de proyectos. "
            "Responde siempre en español, de forma "
            "analítica, clara, directa y estructurada. "
            "Nunca dejes frases a medias y completa "
            "tus explicaciones de manera profesional."
        ),
        max_length=5000,
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def is_retryable_gemini_error(error: Exception) -> bool:
    code = getattr(error, "code", None)

    if code in {
        429,
        500,
        502,
        503,
        504,
    }:
        return True

    error_text = str(error).upper()

    retryable_markers = (
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA",
        "RATE LIMIT",
        "INTERNAL",
        "UNAVAILABLE",
        "TIMEOUT",
    )

    return any(
        marker in error_text
        for marker in retryable_markers
    )


def safe_error_message(error: Exception) -> str:
    code = getattr(error, "code", None)

    if code:
        return f"Proveedor rechazó la solicitud con código {code}."

    return "El proveedor de inteligencia artificial no respondió."


# ============================================================
# PROVEEDOR GEMINI
# ============================================================

async def call_gemini(
    prompt: str,
    system_instruction: str,
) -> str:
    if not gemini_client:
        raise RuntimeError(
            "Cliente Gemini no inicializado."
        )

    start_time = time.perf_counter()

    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.6,
        ),
    )

    text = getattr(response, "text", None)

    if not text or not text.strip():
        raise RuntimeError(
            "Gemini devolvió una respuesta vacía."
        )

    duration = time.perf_counter() - start_time

    logger.info(
        "Gemini exitoso | modelo=%s | duración=%.2fs",
        GEMINI_MODEL,
        duration,
    )

    return text.strip()


# ============================================================
# PROVEEDOR OLLAMA
# ============================================================

async def call_ollama(
    prompt: str,
    system_instruction: str,
) -> str:
    if not OLLAMA_URL:
        raise RuntimeError(
            "OLLAMA_URL no está configurada."
        )

    headers = {
        "Content-Type": "application/json",
    }

    if OLLAMA_API_KEY:
        headers["Authorization"] = (
            f"Bearer {OLLAMA_API_KEY}"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System: {system_instruction}\n\nUser: {prompt}\n\nNEXUS:",
        "stream": False,
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=REQUEST_TIMEOUT_SECONDS,
        write=10.0,
        pool=10.0,
    )

    start_time = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as http_client:
        response = await http_client.post(
            OLLAMA_URL,
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        data = response.json()

    text = data.get(
        "response",
        "",
    ).strip()

    if not text:
        raise RuntimeError(
            "Ollama devolvió una respuesta vacía."
        )

    duration = time.perf_counter() - start_time

    logger.info(
        "Ollama exitoso | modelo=%s | duración=%.2fs",
        OLLAMA_MODEL,
        duration,
    )

    return text


# ============================================================
# RUTAS DEL SISTEMA
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


@app.head("/")
async def home_head():
    return Response(status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "NEXUS",
        "gemini_active": gemini_client is not None,
        "ollama_configured": bool(OLLAMA_URL),
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


@app.get("/api/nexus/status")
async def nexus_status():
    return {
        "status": "online",
        "system": "NEXUS",
        "gemini_active": gemini_client is not None,
        "ollama_active": bool(OLLAMA_URL),
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


@app.get("/api/nexus/config")
async def nexus_config():
    return {
        "model": GEMINI_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "use_ollama_only": USE_OLLAMA_ONLY,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


# ============================================================
# CHAT PRINCIPAL
# ============================================================

@app.post("/api/nexus/chat")
async def chat(
    request: ChatRequest,
    req: Request,
):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "[%s] Solicitud recibida | ip=%s",
        request_id,
        req.client.host if req.client else "unknown",
    )

    if not request.message.strip():
        return {
            "success": True,
            "response": (
                "NEXUS: Escuchando. "
                "Adelante con tu consulta."
            ),
            "provider": "system",
            "fallback": False,
            "request_id": request_id,
        }

    # --------------------------------------------------------
    # MODO EXCLUSIVO OLLAMA
    # --------------------------------------------------------

    if USE_OLLAMA_ONLY:
        try:
            text = await call_ollama(
                request.message,
                request.system,
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "[%s] Ollama exclusivo exitoso | duración=%.2fs",
                request_id,
                duration,
            )

            return {
                "success": True,
                "response": text,
                "provider": "ollama",
                "fallback": False,
                "request_id": request_id,
            }

        except Exception as error:
            logger.exception(
                "[%s] Ollama exclusivo falló",
                request_id,
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Ollama no está disponible."
                    ),
                    "provider": "ollama",
                    "request_id": request_id,
                },
            ) from error

    # --------------------------------------------------------
    # GEMINI PRINCIPAL
    # --------------------------------------------------------

    try:
        text = await call_gemini(
            request.message,
            request.system,
        )

        duration = time.perf_counter() - start_time

        logger.info(
            "[%s] Gemini exitoso | duración=%.2fs",
            request_id,
            duration,
        )

        return {
            "success": True,
            "response": text,
            "provider": "gemini",
            "fallback": False,
            "request_id": request_id,
        }

    except Exception as gemini_error:
        logger.warning(
            "[%s] Gemini falló | motivo=%s",
            request_id,
            safe_error_message(gemini_error),
        )

        if not is_retryable_gemini_error(
            gemini_error
        ):
            logger.error(
                "[%s] Error de Gemini no reintentable",
                request_id,
            )

    # --------------------------------------------------------
    # OLLAMA FALLBACK
    # --------------------------------------------------------

    if OLLAMA_URL:
        try:
            text = await call_ollama(
                request.message,
                request.system,
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "[%s] Ollama fallback exitoso | duración=%.2fs",
                request_id,
                duration,
            )

            return {
                "success": True,
                "response": text,
                "provider": "ollama",
                "fallback": True,
                "request_id": request_id,
            }

        except Exception:
            logger.exception(
                "[%s] Ollama fallback también falló",
                request_id,
            )

    # --------------------------------------------------------
    # AMBOS PROVEEDORES FALLARON
    # --------------------------------------------------------

    logger.error(
        "[%s] Gemini y Ollama no están disponibles",
        request_id,
    )

    raise HTTPException(
        status_code=503,
        detail={
            "message": (
                "Gemini y Ollama no están disponibles "
                "en este momento."
            ),
            "provider": "none",
            "fallback": True,
            "request_id": request_id,
        },
    )
    
