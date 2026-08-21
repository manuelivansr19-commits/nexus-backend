import asyncio
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
# NEXUS AI
# ============================================================

APP_VERSION = "3.1.0"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper(),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("NEXUS-BACKEND")


# ============================================================
# ENV HELPERS
# ============================================================

def get_bool_env(
    name: str,
    default: bool = False,
) -> bool:

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_int_env(
    name: str,
    default: int,
) -> int:

    try:
        value = int(
            os.getenv(
                name,
                str(default),
            )
        )

        return max(1, value)

    except ValueError:

        logger.warning(
            "Variable %s inválida. Usando %d.",
            name,
            default,
        )

        return default


def get_float_env(
    name: str,
    default: float,
) -> float:

    try:
        value = float(
            os.getenv(
                name,
                str(default),
            )
        )

        return max(1.0, value)

    except ValueError:

        logger.warning(
            "Variable %s inválida. Usando %.1f.",
            name,
            default,
        )

        return default


# ============================================================
# CONFIGURACIÓN GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()


# MOTOR PRINCIPAL
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
).strip()


# FALLBACKS
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.6-flash",
).strip()

GEMINI_SECONDARY_MODEL = os.getenv(
    "GEMINI_SECONDARY_MODEL",
    "gemini-2.5-flash",
).strip()


# ============================================================
# OLLAMA
# ============================================================

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


# ============================================================
# GENERACIÓN
# ============================================================

MAX_OUTPUT_TOKENS = get_int_env(
    "MAX_OUTPUT_TOKENS",
    8192,
)


# ============================================================
# THINKING
# ============================================================

THINKING_LEVEL = os.getenv(
    "THINKING_LEVEL",
    "high",
).strip().lower()

if THINKING_LEVEL not in {
    "low",
    "medium",
    "high",
}:

    THINKING_LEVEL = "high"


# ============================================================
# GOOGLE SEARCH
# ============================================================

ENABLE_GOOGLE_SEARCH = get_bool_env(
    "ENABLE_GOOGLE_SEARCH",
    True,
)


# ============================================================
# TIMEOUT / RETRIES
# ============================================================

REQUEST_TIMEOUT_SECONDS = get_float_env(
    "REQUEST_TIMEOUT_SECONDS",
    180.0,
)

GEMINI_MAX_RETRIES = get_int_env(
    "GEMINI_MAX_RETRIES",
    3,
)

RETRY_BASE_SECONDS = get_float_env(
    "RETRY_BASE_SECONDS",
    2.0,
)


# ============================================================
# CLIENTE GEMINI
# ============================================================

gemini_client: Optional[genai.Client] = None


def initialize_gemini():

    if USE_OLLAMA_ONLY:

        logger.info(
            "NEXUS en modo exclusivo Ollama."
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
            "Gemini inicializado | modelo=%s "
            "| fallback=%s "
            "| secondary=%s "
            "| thinking=%s",
            GEMINI_MODEL,
            GEMINI_FALLBACK_MODEL,
            GEMINI_SECONDARY_MODEL,
            THINKING_LEVEL,
        )

        return client

    except Exception:

        logger.exception(
            "No se pudo inicializar Gemini."
        )

        return None


gemini_client = initialize_gemini()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
):

    logger.info(
        "============================================"
    )

    logger.info(
        "NEXUS AI iniciado"
    )

    logger.info(
        "Versión: %s",
        APP_VERSION,
    )

    logger.info(
        "Modelo principal: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Fallback: %s",
        GEMINI_FALLBACK_MODEL,
    )

    logger.info(
        "Thinking: %s",
        THINKING_LEVEL,
    )

    logger.info(
        "Google Search: %s",
        ENABLE_GOOGLE_SEARCH,
    )

    logger.info(
        "============================================"
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
                "Error cerrando cliente Gemini."
            )

    logger.info(
        "NEXUS detenido."
    )


# ============================================================
# FASTAPI
# ============================================================

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
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str = Field(
        default="",
        max_length=30000,
    )

    system: str = Field(
        default=(
            "Eres NEXUS, un sistema avanzado de "
            "inteligencia artificial personal. "

            "Tu función principal es ayudar al usuario "
            "a pensar, analizar, investigar, diseñar "
            "estrategias y resolver problemas complejos. "

            "Responde siempre en español. "

            "Para preguntas simples responde de forma "
            "directa. "

            "Para preguntas complejas realiza un análisis "
            "profundo antes de responder. "

            "Descompón los problemas en partes cuando sea "
            "necesario. "

            "Compara escenarios, identifica riesgos, "
            "ventajas, desventajas y consecuencias. "

            "Cuando corresponda proporciona una "
            "recomendación concreta y un plan de acción. "

            "No inventes información. "

            "Si una afirmación depende de información "
            "actual, utiliza búsqueda web cuando esté "
            "disponible. "

            "Distingue claramente entre hechos, "
            "inferencias y recomendaciones. "

            "No muestres razonamientos internos privados. "
            "Entrega únicamente conclusiones y "
            "explicaciones útiles. "

            "Nunca dejes una respuesta incompleta."
        ),
        max_length=15000,
    )


# ============================================================
# UTILIDADES
# ============================================================

def safe_error_message(
    error: Exception,
) -> str:

    code = getattr(
        error,
        "code",
        None,
    )

    if code is not None:

        return (
            f"Proveedor rechazó la solicitud "
            f"con código {code}."
        )

    text = str(error).strip()

    if not text:

        return "El proveedor no respondió."

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


def is_rate_limit_error(
    error: Exception,
) -> bool:

    code = getattr(
        error,
        "code",
        None,
    )

    if code == 429:
        return True

    text = str(error).upper()

    return any(
        marker in text
        for marker in (
            "429",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "RATE LIMIT",
            "QUOTA",
        )
    )


def is_retryable_error(
    error: Exception,
) -> bool:

    if is_rate_limit_error(error):
        return True

    code = getattr(
        error,
        "code",
        None,
    )

    if code in {
        408,
        500,
        502,
        503,
        504,
    }:

        return True

    text = str(error).upper()

    return any(
        marker in text
        for marker in (
            "TIMEOUT",
            "DEADLINE",
            "UNAVAILABLE",
            "INTERNAL",
            "CONNECTION",
        )
    )


def new_request_id() -> str:

    return str(uuid.uuid4())


# ============================================================
# GEMINI CONFIG
# ============================================================

def build_gemini_config():

    config_kwargs = {
        "system_instruction": None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    thinking_config = types.ThinkingConfig(
        thinking_level=THINKING_LEVEL,
    )

    config_kwargs[
        "thinking_config"
    ] = thinking_config

    if ENABLE_GOOGLE_SEARCH:

        config_kwargs[
            "tools"
        ] = [
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]

    return types.GenerateContentConfig(
        **config_kwargs
    )


# ============================================================
# GEMINI CALL
# ============================================================

async def call_gemini_once(
    prompt: str,
    system_instruction: str,
    model: str,
) -> str:

    if gemini_client is None:

        raise RuntimeError(
            "Cliente Gemini no inicializado."
        )

    started_at = time.perf_counter()

    config = build_gemini_config()

    config.system_instruction = (
        system_instruction
    )

    try:

        response = (
            await gemini_client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        )

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - started_at
        )

        logger.error(
            "Gemini ERROR | modelo=%s "
            "| duración=%.2fs "
            "| tipo=%s "
            "| error=%s",
            model,
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

    elapsed = (
        time.perf_counter()
        - started_at
    )

    logger.info(
        "Gemini exitoso | modelo=%s "
        "| duración=%.2fs "
        "| thinking=%s "
        "| search=%s",
        model,
        elapsed,
        THINKING_LEVEL,
        ENABLE_GOOGLE_SEARCH,
    )

    return text.strip()


# ============================================================
# GEMINI ROBUSTO
# ============================================================

async def call_gemini(
    prompt: str,
    system_instruction: str,
) -> tuple[str, str]:

    models = [
        GEMINI_MODEL,
        GEMINI_FALLBACK_MODEL,
        GEMINI_SECONDARY_MODEL,
    ]

    # Elimina duplicados conservando orden
    models = list(
        dict.fromkeys(
            model
            for model in models
            if model
        )
    )

    last_error = None

    for model_index, model in enumerate(models):

        for attempt in range(
            GEMINI_MAX_RETRIES
        ):

            try:

                text = await call_gemini_once(
                    prompt,
                    system_instruction,
                    model,
                )

                return text, model

            except Exception as error:

                last_error = error

                retryable = (
                    is_retryable_error(error)
                )

                logger.warning(
                    "Gemini intento fallido "
                    "| modelo=%s "
                    "| intento=%d/%d "
                    "| retryable=%s "
                    "| error=%s",
                    model,
                    attempt + 1,
                    GEMINI_MAX_RETRIES,
                    retryable,
                    safe_error_message(error),
                )

                if not retryable:
                    break

                if (
                    attempt
                    < GEMINI_MAX_RETRIES - 1
                ):

                    delay = (
                        RETRY_BASE_SECONDS
                        * (
                            2 ** attempt
                        )
                    )

                    logger.info(
                        "Esperando %.1fs antes "
                        "del siguiente intento.",
                        delay,
                    )

                    await asyncio.sleep(
                        delay
                    )

        # Si agotó el modelo actual
        # pasa al siguiente.

        if model_index < len(models) - 1:

            logger.warning(
                "Cambiando de modelo "
                "%s -> %s",
                model,
                models[
                    model_index + 1
                ],
            )

    if last_error:

        raise last_error

    raise RuntimeError(
        "No existen modelos Gemini configurados."
    )


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

        headers[
            "Authorization"
        ] = (
            f"Bearer {OLLAMA_API_KEY}"
        )

    payload = {
        "model": OLLAMA_MODEL,

        "prompt": (
            f"System:\n"
            f"{system_instruction}\n\n"
            f"User:\n"
            f"{prompt}\n\n"
            f"NEXUS:"
        ),

        "stream": False,

        "options": {
            "num_predict":
                MAX_OUTPUT_TOKENS,
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

        logger.error(
            "Ollama ERROR | modelo=%s "
            "| error=%s",
            OLLAMA_MODEL,
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

    elapsed = (
        time.perf_counter()
        - started_at
    )

    logger.info(
        "Ollama exitoso | modelo=%s "
        "| duración=%.2fs",
        OLLAMA_MODEL,
        elapsed,
    )

    return text


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        "index.html"
    )


@app.head("/")
async def home_head():

    return Response(
        status_code=200
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "system": "NEXUS",
        "version": APP_VERSION,

        "primary_model":
            GEMINI_MODEL,

        "fallback_model":
            GEMINI_FALLBACK_MODEL,

        "secondary_model":
            GEMINI_SECONDARY_MODEL,

        "thinking":
            THINKING_LEVEL,

        "google_search":
            ENABLE_GOOGLE_SEARCH,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "gemini_active":
            gemini_client is not None,

        "ollama_configured":
            bool(OLLAMA_URL),

        "use_ollama_only":
            USE_OLLAMA_ONLY,
    }


# ============================================================
# STATUS
# ============================================================

@app.get(
    "/api/nexus/status"
)
async def nexus_status():

    return {
        "status": "online",
        "system": "NEXUS",
        "version": APP_VERSION,

        "primary_model":
            GEMINI_MODEL,

        "thinking":
            THINKING_LEVEL,

        "google_search":
            ENABLE_GOOGLE_SEARCH,

        "gemini_active":
            gemini_client is not None,

        "ollama_active":
            bool(OLLAMA_URL),

        "use_ollama_only":
            USE_OLLAMA_ONLY,
    }


# ============================================================
# CONFIG
# ============================================================

@app.get(
    "/api/nexus/config"
)
async def nexus_config():

    return {
        "version":
            APP_VERSION,

        "model":
            GEMINI_MODEL,

        "fallback_model":
            GEMINI_FALLBACK_MODEL,

        "secondary_model":
            GEMINI_SECONDARY_MODEL,

        "thinking":
            THINKING_LEVEL,

        "google_search":
            ENABLE_GOOGLE_SEARCH,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "ollama_model":
            OLLAMA_MODEL,

        "use_ollama_only":
            USE_OLLAMA_ONLY,
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/nexus/chat"
)
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
        "[%s] Solicitud recibida "
        "| ip=%s "
        "| chars=%d",
        request_id,
        remote_ip,
        len(request.message),
    )

    # --------------------------------------------------------
    # VACÍO
    # --------------------------------------------------------

    if not request.message.strip():

        return {
            "success": True,

            "response": (
                "NEXUS: Escuchando. "
                "Adelante con tu consulta."
            ),

            "provider":
                "system",

            "fallback":
                False,

            "request_id":
                request_id,
        }

    # --------------------------------------------------------
    # OLLAMA ONLY
    # --------------------------------------------------------

    if USE_OLLAMA_ONLY:

        try:

            text = await call_ollama(
                request.message,
                request.system,
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
                    "message":
                        "Ollama no está disponible.",

                    "provider":
                        "ollama",

                    "request_id":
                        request_id,
                },
            ) from error

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    try:

        text, used_model = (
            await call_gemini(
                request.message,
                request.system,
            )
        )

        elapsed = (
            time.perf_counter()
            - started_at
        )

        logger.info(
            "[%s] NEXUS respondió "
            "| provider=gemini "
            "| modelo=%s "
            "| duración=%.2fs",
            request_id,
            used_model,
            elapsed,
        )

        return {
            "success":
                True,

            "response":
                text,

            "provider":
                "gemini",

            "model":
                used_model,

            "fallback":
                used_model != GEMINI_MODEL,

            "thinking":
                THINKING_LEVEL,

            "google_search":
                ENABLE_GOOGLE_SEARCH,

            "request_id":
                request_id,
        }

    except Exception as gemini_error:

        logger.warning(
            "[%s] Todos los modelos Gemini fallaron "
            "| error=%s",
            request_id,
            safe_error_message(
                gemini_error
            ),
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
                "success":
                    True,

                "response":
                    text,

                "provider":
                    "ollama",

                "fallback":
                    True,

                "request_id":
                    request_id,
            }

        except Exception:

            logger.exception(
                "[%s] Ollama fallback falló",
                request_id,
            )

    # --------------------------------------------------------
    # TODO FALLÓ
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - started_at
    )

    logger.error(
        "[%s] Todos los proveedores fallaron "
        "| duración=%.2fs",
        request_id,
        elapsed,
    )

    raise HTTPException(
        status_code=503,
        detail={
            "message": (
                "NEXUS no pudo obtener una "
                "respuesta de sus proveedores "
                "de inteligencia artificial."
            ),

            "provider":
                "none",

            "fallback":
                True,

            "request_id":
                request_id,
        },
    )
