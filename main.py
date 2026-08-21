import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
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

APP_VERSION = "2.7.0"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("NEXUS-BACKEND")


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return max(1, value)
    except ValueError:
        logger.warning(
            "Variable %s inválida. Usando %d.",
            name,
            default,
        )
        return default


def get_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return max(1.0, value)
    except ValueError:
        logger.warning(
            "Variable %s inválida. Usando %.1f.",
            name,
            default,
        )
        return default


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
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

USE_OLLAMA_ONLY = get_bool_env(
    "USE_OLLAMA_ONLY",
    False,
)

MAX_OUTPUT_TOKENS = get_int_env(
    "MAX_OUTPUT_TOKENS",
    8192,
)

REQUEST_TIMEOUT_SECONDS = get_float_env(
    "REQUEST_TIMEOUT_SECONDS",
    180.0,
)


# ============================================================
# MODELO DE SOLICITUD
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        default="",
        max_length=20000,
    )

    system: str = Field(
        default=(
            "Eres NEXUS, un sistema avanzado de "
            "inteligencia artificial personal. "
            "Estás enfocado en estrategia, análisis, "
            "psicología aplicada, PNL, tecnología y "
            "desarrollo de proyectos. "
            "Responde siempre en español. "
            "Sé analítico, claro, directo y estructurado. "
            "No inventes información. "
            "Cuando no tengas certeza, dilo claramente. "
            "Completa tus explicaciones y nunca dejes "
            "frases a medias."
        ),
        max_length=10000,
    )


# ============================================================
# CLIENTE GEMINI
# ============================================================

gemini_client: Optional[genai.Client] = None


def initialize_gemini() -> Optional[genai.Client]:
    if USE_OLLAMA_ONLY:
        logger.info(
            "NEXUS está en modo exclusivo Ollama."
        )
        return None

    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY no está configurada."
        )
        return None

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
        )

        logger.info(
            "Gemini inicializado | modelo=%s",
            GEMINI_MODEL,
        )

        return client

    except Exception:
        logger.exception(
            "No se pudo inicializar Gemini."
        )
        return None


gemini_client = initialize_gemini()


# ============================================================
# CICLO DE VIDA DE LA APLICACIÓN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info(
        "NEXUS iniciado | versión=%s",
        APP_VERSION,
    )

    yield

    if gemini_client is not None:
        try:
            await gemini_client.aio.aclose()
            logger.info(
                "Cliente Gemini cerrado correctamente."
            )
        except Exception:
            logger.exception(
                "Error al cerrar el cliente Gemini."
            )

    logger.info("NEXUS detenido.")


app = FastAPI(
    title="NEXUS AI",
    version=APP_VERSION,
    lifespan=lifespan,
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
# UTILIDADES
# ============================================================

def safe_error_message(error: Exception) -> str:
    code = getattr(error, "code", None)

    if code is not None:
        return (
            f"Proveedor rechazó la solicitud "
            f"con código {code}."
        )

    text = str(error).strip()

    if not text:
        return (
            "El proveedor no respondió."
        )

    secrets = [
        GEMINI_API_KEY,
        OLLAMA_API_KEY,
    ]

    for secret in secrets:
        if secret:
            text = text.replace(
                secret,
                "[REDACTED]",
            )

    return text[:500]


def new_request_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# GEMINI
# ============================================================

async def call_gemini(
    prompt: str,
    system_instruction: str,
) -> str:
    if gemini_client is None:
        raise RuntimeError(
            "Cliente Gemini no inicializado."
        )

    started_at = time.perf_counter()

    try:
        response = (
            await gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.5,
                ),
            )
        )

    except Exception as error:
        elapsed = time.perf_counter() - started_at

        logger.error(
            "Gemini ERROR | modelo=%s | duración=%.2fs "
            "| tipo=%s | error=%s",
            GEMINI_MODEL,
            elapsed,
            type(error).__name__,
            safe_error_message(error),
        )

        raise

    text = getattr(
        response,
        "text",
        None,
    )

    if not text or not text.strip():
        raise RuntimeError(
            "Gemini devolvió una respuesta vacía."
        )

    elapsed = time.perf_counter() - started_at

    logger.info(
        "Gemini exitoso | modelo=%s | duración=%.2fs",
        GEMINI_MODEL,
        elapsed,
    )

    return text.strip()


# ============================================================
# OLLAMA
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
        "prompt": (
            f"System: {system_instruction}

"
            f"User: {prompt}

"
            "NEXUS:"
        ),
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

    started_at = time.perf_counter()

    try:
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

    except Exception as error:
        elapsed = time.perf_counter() - started_at

        logger.error(
            "Ollama ERROR | modelo=%s | duración=%.2fs "
            "| tipo=%s | error=%s",
            OLLAMA_MODEL,
            elapsed,
            type(error).__name__,
            safe_error_message(error),
        )

        raise

    text = data.get(
        "response",
        "",
    ).strip()

    if not text:
        raise RuntimeError(
            "Ollama devolvió una respuesta vacía."
        )

    elapsed = time.perf_counter() - started_at

    logger.info(
        "Ollama exitoso | modelo=%s | duración=%.2fs",
        OLLAMA_MODEL,
        elapsed,
    )

    return text


# ============================================================
# RUTAS PRINCIPALES
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
        "version": APP_VERSION,
        "model": GEMINI_MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "gemini_active": gemini_client is not None,
        "ollama_configured": bool(OLLAMA_URL),
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


@app.get("/api/nexus/status")
async def nexus_status():
    return {
        "status": "online",
        "system": "NEXUS",
        "version": APP_VERSION,
        "model": GEMINI_MODEL,
        "gemini_active": gemini_client is not None,
        "ollama_active": bool(OLLAMA_URL),
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


@app.get("/api/nexus/config")
async def nexus_config():
    return {
        "model": GEMINI_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/nexus/chat")
async def chat(
    request: ChatRequest,
    req: Request,
):
    request_id = new_request_id()
    started_at = time.perf_counter()

    remote_ip = (
        req.client.host
        if req.client
        else "unknown"
    )

    logger.info(
        "[%s] Solicitud recibida | ip=%s | chars=%d",
        request_id,
        remote_ip,
        len(request.message),
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

            elapsed = (
                time.perf_counter()
                - started_at
            )

            logger.info(
                "[%s] Ollama exclusivo exitoso "
                "| duración=%.2fs",
                request_id,
                elapsed,
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
                "[%s] Ollama exclusivo falló | %s",
                request_id,
                safe_error_message(error),
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

        elapsed = (
            time.perf_counter()
            - started_at
        )

        logger.info(
            "[%s] Gemini exitoso | duración=%.2fs",
            request_id,
            elapsed,
        )

        return {
            "success": True,
            "response": text,
            "provider": "gemini",
            "fallback": False,
            "request_id": request_id,
        }

    except Exception as error:
        logger.warning(
            "[%s] Gemini falló | %s",
            request_id,
            safe_error_message(error),
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

            elapsed = (
                time.perf_counter()
                - started_at
            )

            logger.info(
                "[%s] Ollama fallback exitoso "
                "| duración=%.2fs",
                request_id,
                elapsed,
            )

            return {
                "success": True,
                "response": text,
                "provider": "ollama",
                "fallback": True,
                "request_id": request_id,
            }

        except Exception as error:
            logger.exception(
                "[%s] Ollama fallback falló | %s",
                request_id,
                safe_error_message(error),
            )

    # --------------------------------------------------------
    # AMBOS PROVEEDORES FALLARON
    # --------------------------------------------------------

    logger.error(
        "[%s] Gemini y Ollama no están disponibles.",
        request_id,
    )

    raise HTTPException(
        status_code=503,
        detail={
            "message": (
                "NEXUS no pudo obtener una respuesta "
                "de sus proveedores de IA."
            ),
            "provider": "none",
            "fallback": True,
            "request_id": request_id,
        },
    )
