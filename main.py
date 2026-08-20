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
# NEXUS AI 2.6.4
# MOTOR DE RAZONAMIENTO
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("NEXUS-BACKEND")

app = FastAPI(
    title="NEXUS AI",
    version="2.6.4",
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
# CONFIGURACIÓN
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
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


# ============================================================
# RAZONAMIENTO
# ============================================================

THINKING_BUDGET = int(
    os.getenv(
        "THINKING_BUDGET",
        "4096",
    )
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "8192",
    )
)

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        "180",
    )
)


# ============================================================
# CLIENTE GEMINI
# ============================================================

gemini_client: Optional[genai.Client] = None

if GEMINI_API_KEY and not USE_OLLAMA_ONLY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
        )

        logger.info(
            "Gemini inicializado | modelo=%s",
            GEMINI_MODEL,
        )

    except Exception:

        logger.exception(
            "No se pudo inicializar Gemini."
        )

        gemini_client = None

else:

    if USE_OLLAMA_ONLY:

        logger.info(
            "NEXUS está funcionando en modo exclusivo Ollama."
        )

    elif not GEMINI_API_KEY:

        logger.warning(
            "GEMINI_API_KEY no está configurada."
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
            "\n\n"
            "Tu función principal es ayudar al usuario "
            "a pensar, analizar, investigar, comparar, "
            "planear y tomar decisiones. "
            "\n\n"
            "Cuando la pregunta sea compleja debes "
            "analizarla profundamente antes de responder. "
            "Descompón el problema en variables, identifica "
            "supuestos, riesgos, relaciones causa-efecto, "
            "escenarios y consecuencias. "
            "\n\n"
            "No simplifiques una pregunta compleja solo "
            "para responder rápido. "
            "\n\n"
            "Distingue claramente entre hechos, inferencias "
            "y opiniones. No inventes datos. "
            "\n\n"
            "Si necesitas información actualizada y tienes "
            "acceso a búsqueda web, utilízala. "
            "\n\n"
            "Responde siempre en español. "
            "Sé directo, inteligente, estructurado y "
            "profesional. "
            "\n\n"
            "Para preguntas sencillas responde brevemente. "
            "Para análisis complejos proporciona el nivel "
            "de profundidad necesario. "
            "\n\n"
            "Nunca cortes una explicación a la mitad."
        ),
        max_length=10000,
    )


# ============================================================
# UTILIDADES
# ============================================================

def is_retryable_gemini_error(
    error: Exception,
) -> bool:

    code = getattr(
        error,
        "code",
        None,
    )

    if code in {
        408,
        429,
        500,
        502,
        503,
        504,
    }:

        return True

    error_text = str(error).upper()

    markers = (
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "RESOURCE_EXHAUSTED",
        "QUOTA",
        "RATE LIMIT",
        "INTERNAL",
        "UNAVAILABLE",
        "TIMEOUT",
        "DEADLINE",
    )

    return any(
        marker in error_text
        for marker in markers
    )


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

    if text:

        return text[:500]

    return (
        "El proveedor de inteligencia artificial "
        "no respondió."
    )


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

    start_time = time.perf_counter()

    try:

        response = await gemini_client.aio.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=system_instruction,

                # ------------------------------------------------
                # RAZONAMIENTO PROFUNDO
                # ------------------------------------------------

                thinking_config=types.ThinkingConfig(
                    thinking_budget=THINKING_BUDGET,
                    include_thoughts=False,
                ),

                # ------------------------------------------------
                # RESPUESTA
                # ------------------------------------------------

                max_output_tokens=MAX_OUTPUT_TOKENS,

                temperature=0.4,

                # ------------------------------------------------
                # BÚSQUEDA WEB
                # ------------------------------------------------

                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ],
            ),
        )

    except Exception as error:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.error(
            "Gemini ERROR | modelo=%s | duración=%.2fs | tipo=%s | error=%s",
            GEMINI_MODEL,
            duration,
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

        logger.error(
            "Gemini devolvió respuesta vacía."
        )

        raise RuntimeError(
            "Gemini devolvió una respuesta vacía."
        )

    # ========================================================
    # MÉTRICAS DE RAZONAMIENTO
    # ========================================================

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage:

        thoughts = getattr(
            usage,
            "thoughts_token_count",
            None,
        )

        output_tokens = getattr(
            usage,
            "candidates_token_count",
            None,
        )

        total_tokens = getattr(
            usage,
            "total_token_count",
            None,
        )

        logger.info(
            "Gemini uso | thoughts=%s | output=%s | total=%s",
            thoughts,
            output_tokens,
            total_tokens,
        )

    duration = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Gemini exitoso | modelo=%s | duración=%.2fs",
        GEMINI_MODEL,
        duration,
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
            f"System: {system_instruction}\n\n"
            f"User: {prompt}\n\n"
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

    start_time = time.perf_counter()

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
            "Ollama ERROR | tipo=%s | error=%s",
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

    duration = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Ollama exitoso | modelo=%s | duración=%.2fs",
        OLLAMA_MODEL,
        duration,
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
        "version": "2.6.4",
        "model": GEMINI_MODEL,
        "thinking_budget": THINKING_BUDGET,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "gemini_active": (
            gemini_client is not None
        ),
        "ollama_configured": bool(
            OLLAMA_URL
        ),
        "use_ollama_only": USE_OLLAMA_ONLY,
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/nexus/status")
async def nexus_status():

    return {
        "status": "online",
        "system": "NEXUS",
        "version": "2.6.4",
        "model": GEMINI_MODEL,
        "reasoning": True,
        "web_search": True,
        "gemini_active": (
            gemini_client is not None
        ),
        "ollama_active": bool(
            OLLAMA_URL
        ),
    }


# ============================================================
# CONFIG
# ============================================================

@app.get("/api/nexus/config")
async def nexus_config():

    return {

        "model": GEMINI_MODEL,

        "ollama_model": OLLAMA_MODEL,

        "thinking_budget": THINKING_BUDGET,

        "max_output_tokens": MAX_OUTPUT_TOKENS,

        "reasoning": True,

        "web_search": True,

        "use_ollama_only": (
            USE_OLLAMA_ONLY
        ),
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/nexus/chat")
async def chat(
    request: ChatRequest,
    req: Request,
):

    request_id = str(
        uuid.uuid4()
    )

    start_time = time.perf_counter()

    logger.info(
        "[%s] Solicitud recibida | ip=%s | chars=%d",
        request_id,
        req.client.host
        if req.client
        else "unknown",
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

            "provider": "system",

            "fallback": False,

            "request_id": request_id,
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
                    "message": (
                        "Ollama no está disponible."
                    ),
                    "provider": "ollama",
                    "request_id": request_id,
                    "error": safe_error_message(
                        error
                    ),
                },
            ) from error

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    gemini_error = None

    try:

        text = await call_gemini(
            request.message,
            request.system,
        )

        duration = (
            time.perf_counter()
            - start_time
        )

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

            "reasoning": True,

            "request_id": request_id,
        }

    except Exception as error:

        gemini_error = error

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

            duration = (
                time.perf_counter()
                - start_time
            )

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
                "[%s] Ollama fallback falló",
                request_id,
            )

    # --------------------------------------------------------
    # TODO FALLÓ
    # --------------------------------------------------------

    detail = {

        "message": (
            "NEXUS no pudo obtener una respuesta."
        ),

        "provider": "none",

        "request_id": request_id,
    }

    if gemini_error:

        detail["gemini_error"] = (
            safe_error_message(
                gemini_error
            )
        )

    raise HTTPException(
        status_code=503,
        detail=detail,
)
