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
# NEXUS AI v3.2
# MULTI-PROVIDER INTELLIGENCE ROUTER
# ============================================================

APP_VERSION = "3.2.0"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
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

    except (ValueError, TypeError):

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

    except (ValueError, TypeError):

        logger.warning(
            "Variable %s inválida. Usando %.1f.",
            name,
            default,
        )

        return default


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
).strip()


OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "",
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/auto",
).strip()

OPENROUTER_FALLBACK_MODELS = os.getenv(
    "OPENROUTER_FALLBACK_MODELS",
    "",
).strip()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
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
# DEFAULT SYSTEM
# ============================================================

DEFAULT_SYSTEM = """
Eres NEXUS, un sistema avanzado de inteligencia artificial
personal orientado a análisis profundo, estrategia,
tecnología, negocios, operaciones, psicología aplicada,
investigación y desarrollo de proyectos.

Tu función no es simplemente contestar preguntas.

Debes:

1. Comprender el problema antes de responder.
2. Separar hechos, inferencias y recomendaciones.
3. Analizar causas y consecuencias.
4. Identificar riesgos, oportunidades y variables ocultas.
5. Proponer alternativas cuando existan.
6. Explicar tu razonamiento de manera clara y estructurada,
   sin inventar información.
7. Cuando una afirmación dependa de información actual,
   reconocer que requiere verificación.
8. Si existe incertidumbre, indicarla explícitamente.
9. No abandonar una respuesta compleja a mitad.
10. No responder con frases genéricas cuando el problema
    requiere análisis.

Responde siempre en español.

Estilo:

- directo
- estratégico
- analítico
- estructurado
- preciso
- práctico

Cuando sea útil utiliza:

DIAGNÓSTICO
ANÁLISIS
RIESGOS
ESCENARIOS
RECOMENDACIÓN
PLAN DE ACCIÓN

No inventes datos, fuentes, resultados ni capacidades.
"""


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str = Field(
        default="",
        max_length=30000,
    )

    system: str = Field(
        default=DEFAULT_SYSTEM,
        max_length=12000,
    )


# ============================================================
# GLOBAL CLIENTS
# ============================================================

gemini_client: Optional[genai.Client] = None


# ============================================================
# GEMINI INITIALIZATION
# ============================================================

def initialize_gemini():

    global gemini_client

    if USE_OLLAMA_ONLY:

        logger.info(
            "Gemini desactivado: USE_OLLAMA_ONLY=true"
        )

        return

    if not GEMINI_API_KEY:

        logger.warning(
            "GEMINI_API_KEY no configurada."
        )

        return

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


initialize_gemini()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):

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
        "Gemini: %s",
        bool(gemini_client),
    )

    logger.info(
        "OpenRouter: %s",
        bool(OPENROUTER_API_KEY),
    )

    logger.info(
        "Groq: %s",
        bool(GROQ_API_KEY),
    )

    logger.info(
        "Ollama: %s",
        bool(OLLAMA_URL),
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
                "Error cerrando Gemini."
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
# UTILITIES
# ============================================================

def safe_error_message(
    error: Exception,
) -> str:

    text = str(error).strip()

    if not text:

        text = "Proveedor no respondió."

    secrets = [
        GEMINI_API_KEY,
        OPENROUTER_API_KEY,
        GROQ_API_KEY,
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
            "RATE LIMIT",
            "TOO MANY REQUESTS",
            "QUOTA",
        )
    )


def new_request_id():

    return str(
        uuid.uuid4()
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
            "Gemini no está disponible."
        )

    started = time.perf_counter()

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

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.error(
            "Gemini ERROR | modelo=%s | "
            "duración=%.2fs | error=%s",
            GEMINI_MODEL,
            elapsed,
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
        - started
    )

    logger.info(
        "Gemini OK | %.2fs",
        elapsed,
    )

    return text.strip()


# ============================================================
# OPENROUTER
# ============================================================

async def call_openrouter(
    prompt: str,
    system_instruction: str,
) -> str:

    if not OPENROUTER_API_KEY:

        raise RuntimeError(
            "OPENROUTER_API_KEY no configurada."
        )

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": (
            "https://nexus-backend-wduu.onrender.com"
        ),
        "X-Title": "NEXUS AI",
    }

    models = []

    if OPENROUTER_MODEL:

        models.append(
            OPENROUTER_MODEL
        )

    if OPENROUTER_FALLBACK_MODELS:

        models.extend(
            [
                model.strip()
                for model
                in OPENROUTER_FALLBACK_MODELS.split(",")
                if model.strip()
            ]
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.5,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }

    if len(models) > 1:

        payload["models"] = models

    started = time.perf_counter()

    timeout = httpx.Timeout(
        connect=15.0,
        read=REQUEST_TIMEOUT_SECONDS,
        write=15.0,
        pool=15.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: "
            f"{response.text[:400]}"
        )

    data = response.json()

    choices = data.get(
        "choices",
        [],
    )

    if not choices:

        raise RuntimeError(
            "OpenRouter no devolvió choices."
        )

    message = choices[0].get(
        "message",
        {},
    )

    text = message.get(
        "content",
        "",
    )

    if not text:

        raise RuntimeError(
            "OpenRouter devolvió contenido vacío."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info(
        "OpenRouter OK | modelo=%s | %.2fs",
        data.get(
            "model",
            OPENROUTER_MODEL,
        ),
        elapsed,
    )

    return text.strip()


# ============================================================
# GROQ
# ============================================================

async def call_groq(
    prompt: str,
    system_instruction: str,
) -> str:

    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY no configurada."
        )

    headers = {
        "Authorization": (
            f"Bearer {GROQ_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.5,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
    }

    timeout = httpx.Timeout(
        connect=15.0,
        read=REQUEST_TIMEOUT_SECONDS,
        write=15.0,
        pool=15.0,
    )

    started = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"Groq HTTP {response.status_code}: "
            f"{response.text[:400]}"
        )

    data = response.json()

    choices = data.get(
        "choices",
        [],
    )

    if not choices:

        raise RuntimeError(
            "Groq no devolvió choices."
        )

    text = (
        choices[0]
        .get("message", {})
        .get("content", "")
    )

    if not text:

        raise RuntimeError(
            "Groq devolvió contenido vacío."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info(
        "Groq OK | modelo=%s | %.2fs",
        GROQ_MODEL,
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
            "OLLAMA_URL no configurada."
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
            f"{system_instruction}\n\n"
            f"USUARIO:\n{prompt}\n\n"
            "NEXUS:"
        ),
        "stream": False,
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS,
            "temperature": 0.5,
        },
    }

    timeout = httpx.Timeout(
        connect=15.0,
        read=REQUEST_TIMEOUT_SECONDS,
        write=15.0,
        pool=15.0,
    )

    started = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(
            OLLAMA_URL,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"Ollama HTTP {response.status_code}"
        )

    data = response.json()

    text = data.get(
        "response",
        "",
    ).strip()

    if not text:

        raise RuntimeError(
            "Ollama devolvió contenido vacío."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info(
        "Ollama OK | modelo=%s | %.2fs",
        OLLAMA_MODEL,
        elapsed,
    )

    return text


# ============================================================
# PROVIDER ROUTER
# ============================================================

async def generate_response(
    prompt: str,
    system_instruction: str,
):

    providers = []

    if not USE_OLLAMA_ONLY:

        if gemini_client is not None:

            providers.append(
                (
                    "gemini",
                    call_gemini,
                )
            )

        if OPENROUTER_API_KEY:

            providers.append(
                (
                    "openrouter",
                    call_openrouter,
                )
            )

        if GROQ_API_KEY:

            providers.append(
                (
                    "groq",
                    call_groq,
                )
            )

    if OLLAMA_URL:

        providers.append(
            (
                "ollama",
                call_ollama,
            )
        )

    if not providers:

        raise RuntimeError(
            "Ningún proveedor de IA está configurado."
        )

    errors = []

    for provider_name, provider_function in providers:

        started = time.perf_counter()

        logger.info(
            "Router intentando proveedor=%s",
            provider_name,
        )

        try:

            text = await provider_function(
                prompt,
                system_instruction,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            logger.info(
                "Router éxito | proveedor=%s | %.2fs",
                provider_name,
                elapsed,
            )

            return (
                text,
                provider_name,
                False,
            )

        except Exception as error:

            errors.append(
                f"{provider_name}: "
                f"{safe_error_message(error)}"
            )

            if is_rate_limit_error(error):

                logger.warning(
                    "Proveedor %s limitado (429). "
                    "Saltando inmediatamente al siguiente.",
                    provider_name,
                )

            else:

                logger.warning(
                    "Proveedor %s falló: %s",
                    provider_name,
                    safe_error_message(error),
                )

    raise RuntimeError(
        "Todos los proveedores fallaron | "
        + " | ".join(errors)
    )


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

        "providers": {
            "gemini": bool(
                gemini_client
            ),
            "openrouter": bool(
                OPENROUTER_API_KEY
            ),
            "groq": bool(
                GROQ_API_KEY
            ),
            "ollama": bool(
                OLLAMA_URL
            ),
        },

        "models": {
            "gemini": GEMINI_MODEL,
            "openrouter": OPENROUTER_MODEL,
            "groq": GROQ_MODEL,
            "ollama": OLLAMA_MODEL,
        },

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "ollama_only":
            USE_OLLAMA_ONLY,
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/nexus/status")
async def nexus_status():

    return {
        "status": "online",
        "system": "NEXUS",
        "version": APP_VERSION,

        "router": {
            "gemini": bool(
                gemini_client
            ),
            "openrouter": bool(
                OPENROUTER_API_KEY
            ),
            "groq": bool(
                GROQ_API_KEY
            ),
            "ollama": bool(
                OLLAMA_URL
            ),
        },

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,
    }


# ============================================================
# CONFIG
# ============================================================

@app.get("/api/nexus/config")
async def nexus_config():

    return {
        "version": APP_VERSION,

        "gemini_model":
            GEMINI_MODEL,

        "openrouter_model":
            OPENROUTER_MODEL,

        "groq_model":
            GROQ_MODEL,

        "ollama_model":
            OLLAMA_MODEL,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "multi_provider":
            True,
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

    started = time.perf_counter()

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

    system_instruction = (
        request.system.strip()
        or DEFAULT_SYSTEM
    )

    try:

        (
            text,
            provider,
            fallback,
        ) = await generate_response(
            request.message,
            system_instruction,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(
            "[%s] RESPUESTA OK | "
            "provider=%s | fallback=%s | "
            "duración=%.2fs",
            request_id,
            provider,
            fallback,
            elapsed,
        )

        return {
            "success": True,
            "response": text,
            "provider": provider,
            "fallback": fallback,
            "request_id": request_id,
            "duration_ms": int(
                elapsed * 1000
            ),
        }

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.exception(
            "[%s] TODOS LOS PROVEEDORES FALLARON "
            "| duración=%.2fs",
            request_id,
            elapsed,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "NEXUS no pudo obtener una "
                    "respuesta de ningún proveedor."
                ),
                "provider": "none",
                "request_id": request_id,
            },
        ) from error


# ============================================================
# SERVICE WORKER
# ============================================================

@app.get("/sw.js")
async def service_worker():

    return Response(
        content=(
            """
self.addEventListener(
    "install",
    event => self.skipWaiting()
);

self.addEventListener(
    "activate",
    event => self.clients.claim()
);
"""
        ),
        media_type="application/javascript",
        headers={
            "Cache-Control":
                "no-cache, no-store, must-revalidate"
        },
    )
