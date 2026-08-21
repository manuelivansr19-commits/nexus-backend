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
from pydantic import BaseModel, Field


# ============================================================
# NEXUS AI 3.1
# Gemini 3.7 Flash + Thinking HIGH + Google Search
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

logger = logging.getLogger(
    "NEXUS-BACKEND"
)


# ============================================================
# ENVIRONMENT
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()


# IMPORTANTE:
# Esta variable de Render puede sobrescribir este valor.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
).strip()


THINKING_LEVEL = os.getenv(
    "THINKING_LEVEL",
    "high",
).strip().lower()


MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "16384",
    )
)


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
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        "300",
    )
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

DEFAULT_SYSTEM = """
Eres NEXUS, una inteligencia artificial personal
avanzada orientada al razonamiento, análisis profundo,
estrategia, investigación, tecnología, negocios,
psicología aplicada y desarrollo de proyectos.

NO eres un chatbot superficial.

Tu trabajo es ayudar al usuario a comprender problemas
complejos, descubrir patrones, evaluar escenarios y
convertir análisis en acciones concretas.

MODO DE OPERACIÓN:

1. Comprende primero la intención real de la consulta.

2. Analiza el problema antes de responder.

3. Divide los problemas complejos en componentes.

4. Identifica:
   - hechos
   - supuestos
   - variables
   - riesgos
   - oportunidades
   - consecuencias
   - escenarios

5. Diferencia hechos comprobados de inferencias.

6. No inventes información.

7. Cuando una información pueda haber cambiado,
   utiliza búsqueda web si está disponible.

8. Cuando hagas análisis estratégico utiliza,
   cuando corresponda:

   DIAGNÓSTICO
   ANÁLISIS
   ESCENARIOS
   RIESGOS
   OPORTUNIDADES
   RECOMENDACIÓN
   SIGUIENTE ACCIÓN

9. Cuando analices tecnología o código:

   PROBLEMA
   CAUSA
   SOLUCIÓN
   IMPLEMENTACIÓN
   VERIFICACIÓN

10. Cuando existan varias alternativas,
    compáralas y recomienda una.

11. No reduzcas una consulta compleja
    a una respuesta superficial.

12. Puedes responder extensamente cuando
    la complejidad lo requiera.

13. Responde siempre en español salvo
    que el usuario solicite otro idioma.

14. Sé directo y evita relleno.

15. No repitas innecesariamente la pregunta
    del usuario.

16. Si falta información crítica,
    indícalo claramente y trabaja con
    los supuestos disponibles.

17. No expongas cadenas internas de pensamiento
    privadas. Entrega conclusiones, análisis,
    evidencia y justificaciones útiles.

PRIORIDADES:

PRECISIÓN
>
RAZONAMIENTO
>
CONTEXTO
>
EVIDENCIA
>
ACCIÓN
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
        max_length=15000,
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client: Optional[genai.Client] = None


def initialize_gemini():

    global gemini_client

    if USE_OLLAMA_ONLY:

        logger.info(
            "NEXUS configurado en modo exclusivo Ollama."
        )

        return

    if not GEMINI_API_KEY:

        logger.error(
            "GEMINI_API_KEY no está configurada."
        )

        return

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
        )

        logger.info(
            "Gemini inicializado | modelo=%s | thinking=%s",
            GEMINI_MODEL,
            THINKING_LEVEL,
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
        "=============================================="
    )

    logger.info(
        "NEXUS AI iniciado"
    )

    logger.info(
        "Versión: %s",
        APP_VERSION,
    )

    logger.info(
        "Modelo: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Thinking: %s",
        THINKING_LEVEL,
    )

    logger.info(
        "Google Search: habilitado"
    )

    logger.info(
        "Output tokens: %s",
        MAX_OUTPUT_TOKENS,
    )

    logger.info(
        "=============================================="
    )

    yield

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
# ERROR SANITIZATION
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

    text = str(
        error
    ).strip()

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

    return text[:1000]


# ============================================================
# GEMINI 3.7 FLASH
# INTERACTIONS API
# ============================================================

async def call_gemini(
    prompt: str,
    system_instruction: str,
) -> str:

    if gemini_client is None:

        raise RuntimeError(
            "Cliente Gemini no inicializado."
        )

    started = time.perf_counter()

    try:

        interaction = (
            await gemini_client.aio.interactions.create(

                model=GEMINI_MODEL,

                input=(
                    f"{system_instruction}\n\n"
                    f"CONSULTA DEL USUARIO:\n"
                    f"{prompt}"
                ),

                generation_config={
                    "thinking_level": THINKING_LEVEL,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },

                tools=[
                    {
                        "type": "google_search"
                    }
                ],
            )
        )

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.error(
            "Gemini ERROR | modelo=%s | "
            "thinking=%s | duración=%.2fs | "
            "tipo=%s | error=%s",

            GEMINI_MODEL,

            THINKING_LEVEL,

            elapsed,

            type(error).__name__,

            safe_error_message(
                error
            ),
        )

        raise

    # ========================================================
    # EXTRAER TEXTO
    # ========================================================

    text = getattr(
        interaction,
        "output_text",
        None,
    )

    if not text:

        # Compatibilidad adicional
        # con versiones del SDK que devuelven
        # outputs estructurados.

        output = getattr(
            interaction,
            "outputs",
            None,
        )

        if output:

            fragments = []

            for item in output:

                item_text = getattr(
                    item,
                    "text",
                    None,
                )

                if item_text:

                    fragments.append(
                        item_text
                    )

            if fragments:

                text = "\n".join(
                    fragments
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
        "Gemini OK | modelo=%s | "
        "thinking=%s | duración=%.2fs",

        GEMINI_MODEL,

        THINKING_LEVEL,

        elapsed,
    )

    return text.strip()


# ============================================================
# OLLAMA FALLBACK
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
            f"System:\n"
            f"{system_instruction}\n\n"
            f"User:\n"
            f"{prompt}\n\n"
            f"NEXUS:\n"
        ),

        "stream": False,

        "options": {

            "num_predict":
                MAX_OUTPUT_TOKENS,
        },
    }

    timeout = httpx.Timeout(

        connect=15.0,

        read=REQUEST_TIMEOUT_SECONDS,

        write=15.0,

        pool=15.0,
    )

    started = time.perf_counter()

    try:

        async with httpx.AsyncClient(
            timeout=timeout,
        ) as client:

            response = await client.post(

                OLLAMA_URL,

                json=payload,

                headers=headers,
            )

            response.raise_for_status()

            data = response.json()

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.error(

            "Ollama ERROR | modelo=%s | "
            "duración=%.2fs | error=%s",

            OLLAMA_MODEL,

            elapsed,

            safe_error_message(
                error
            ),
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
        - started
    )

    logger.info(

        "Ollama OK | modelo=%s | "
        "duración=%.2fs",

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

        "model": GEMINI_MODEL,

        "thinking_level":
            THINKING_LEVEL,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "gemini_active":
            gemini_client is not None,

        "google_search": True,

        "ollama_configured":
            bool(OLLAMA_URL),

        "use_ollama_only":
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

        "model": GEMINI_MODEL,

        "thinking_level":
            THINKING_LEVEL,

        "gemini_active":
            gemini_client is not None,

        "google_search": True,

        "ollama_active":
            bool(OLLAMA_URL),

        "use_ollama_only":
            USE_OLLAMA_ONLY,
    }


# ============================================================
# CONFIG
# ============================================================

@app.get("/api/nexus/config")
async def nexus_config():

    return {

        "system": "NEXUS",

        "version": APP_VERSION,

        "model": GEMINI_MODEL,

        "thinking_level":
            THINKING_LEVEL,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "google_search": True,

        "ollama_model":
            OLLAMA_MODEL,

        "fallback_enabled":
            bool(OLLAMA_URL),
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

    started = time.perf_counter()

    remote_ip = (

        req.client.host

        if req.client

        else "unknown"
    )

    logger.info(

        "[%s] Consulta recibida | "
        "ip=%s | chars=%d",

        request_id,

        remote_ip,

        len(request.message),
    )


    # ========================================================
    # EMPTY MESSAGE
    # ========================================================

    if not request.message.strip():

        return {

            "success": True,

            "response": (
                "NEXUS: Estoy listo. "
                "Plantea el problema."
            ),

            "provider": "system",

            "request_id": request_id,
        }


    # ========================================================
    # OLLAMA ONLY
    # ========================================================

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

                "request_id":
                    request_id,
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

                    "error":
                        safe_error_message(
                            error
                        ),
                },

            ) from error


    # ========================================================
    # GEMINI PRINCIPAL
    # ========================================================

    try:

        text = await call_gemini(

            request.message,

            request.system,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(

            "[%s] Gemini completo | "
            "duración=%.2fs",

            request_id,

            elapsed,
        )

        return {

            "success": True,

            "response": text,

            "provider": "gemini",

            "fallback": False,

            "model":
                GEMINI_MODEL,

            "thinking":
                THINKING_LEVEL,

            "web_search": True,

            "request_id":
                request_id,
        }


    except Exception as gemini_error:

        logger.warning(

            "[%s] Gemini falló | %s",

            request_id,

            safe_error_message(
                gemini_error
            ),
        )


    # ========================================================
    # OLLAMA FALLBACK
    # ========================================================

    if OLLAMA_URL:

        try:

            text = await call_ollama(

                request.message,

                request.system,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            logger.info(

                "[%s] Ollama fallback | "
                "duración=%.2fs",

                request_id,

                elapsed,
            )

            return {

                "success": True,

                "response": text,

                "provider": "ollama",

                "fallback": True,

                "request_id":
                    request_id,
            }

        except Exception as ollama_error:

            logger.exception(

                "[%s] Ollama fallback falló | %s",

                request_id,

                safe_error_message(
                    ollama_error
                ),
            )


    # ========================================================
    # TODO FALLÓ
    # ========================================================

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.error(

        "[%s] Todos los proveedores fallaron | "
        "duración=%.2fs",

        request_id,

        elapsed,
    )

    raise HTTPException(

        status_code=503,

        detail={

            "message": (
                "NEXUS no pudo obtener "
                "una respuesta."
            ),

            "provider": "none",

            "fallback": True,

            "request_id":
                request_id,
        },
        )
