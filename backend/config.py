"""
NEXUS Ω — Configuración centralizada.

Ningún otro módulo debe leer os.getenv() directamente.
"""

import logging
import os

# ============================================================
# VERSION
# ============================================================

APP_VERSION = "3.4.0"

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("NEXUS")


# ============================================================
# ENV HELPERS
# ============================================================

def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (ValueError, TypeError):
        logger.warning("Variable %s inválida. Usando %d.", name, default)
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(name, str(default))))
    except (ValueError, TypeError):
        logger.warning("Variable %s inválida. Usando %.1f.", name, default)
        return default


# ============================================================
# EXTERNAL PROVIDERS
# ============================================================

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_MAX_RETRIES = _int_env("GEMINI_MAX_RETRIES", 3)

OPENROUTER_API_KEY        = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL          = os.getenv("OPENROUTER_MODEL", "openrouter/auto").strip()
OPENROUTER_FALLBACK_MODELS = os.getenv("OPENROUTER_FALLBACK_MODELS", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

OLLAMA_URL     = os.getenv("OLLAMA_URL", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3").strip()

# ============================================================
# LOCAL ENGINE (AURA Brain)
# ============================================================

# llama.cpp path to a local GGUF model file
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "").strip()
# Which local runtime to prefer: "llama_cpp" | "ollama"
LOCAL_RUNTIME    = os.getenv("LOCAL_RUNTIME", "llama_cpp").strip().lower()
# Local Ollama URL (defaults to localhost — distinct from cloud OLLAMA_URL)
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434").strip()
LOCAL_OLLAMA_MODEL = os.getenv("LOCAL_OLLAMA_MODEL", OLLAMA_MODEL).strip()

# ============================================================
# ROUTING MODE
# ============================================================

# Legacy compat
USE_OLLAMA_ONLY   = _bool_env("USE_OLLAMA_ONLY", False)
# New: run ONLY on local engine — no external calls at all
NEXUS_LOCAL_ONLY  = _bool_env("NEXUS_LOCAL_ONLY", False)

# ============================================================
# GENERATION
# ============================================================

MAX_OUTPUT_TOKENS       = _int_env("MAX_OUTPUT_TOKENS", 8192)
REQUEST_TIMEOUT_SECONDS = _float_env("REQUEST_TIMEOUT_SECONDS", 180.0)
DEFAULT_TEMPERATURE     = 0.5

# ============================================================
# CONVERSATION
# ============================================================

MAX_HISTORY_TURNS = _int_env("MAX_HISTORY_TURNS", 20)

# ============================================================
# SYSTEM PROMPT
# ============================================================

DEFAULT_SYSTEM = """
Eres NEXUS Ω, un sistema avanzado de inteligencia artificial
personal orientado a análisis profundo, estrategia,
tecnología, negocios, operaciones, psicología aplicada,
investigación y desarrollo de proyectos.

Debes:
1. Comprender el problema antes de responder.
2. Separar hechos, inferencias y recomendaciones.
3. Analizar causas y consecuencias.
4. Identificar riesgos, oportunidades y variables ocultas.
5. Proponer alternativas cuando existan.
6. Explicar tu razonamiento de manera clara y estructurada.
7. Si existe incertidumbre, indicarla explícitamente.
8. No abandonar una respuesta compleja a mitad.

Responde siempre en español.
Estilo: directo, estratégico, analítico, estructurado, preciso.
No inventes datos, fuentes, resultados ni capacidades.
""".strip()

# ============================================================
# SECRETS (for log redaction)
# ============================================================

ALL_SECRETS = [
    s for s in [
        GEMINI_API_KEY,
        OPENROUTER_API_KEY,
        GROQ_API_KEY,
        OLLAMA_API_KEY,
    ]
    if s
]


# ============================================================
# PHASE 2 — NEXUS CORE
# ============================================================

import os as _os

# SQLite path (Render filesystem is ephemeral — for dev only)
MEMORY_DB_PATH = _os.getenv("MEMORY_DB_PATH", "nexus_memory.db").strip()

# Context window budget (approx tokens: chars / 4)
CONTEXT_TOKEN_LIMIT = _int_env("CONTEXT_TOKEN_LIMIT", 6000)

# Max memory entries returned per search
MEMORY_SEARCH_LIMIT = _int_env("MEMORY_SEARCH_LIMIT", 8)

# Intent confidence threshold below which we fall back to LLM classification
INTENT_CONFIDENCE_THRESHOLD = 0.75

# Tool execution timeout (seconds)
TOOL_TIMEOUT_SECONDS = _float_env("TOOL_TIMEOUT_SECONDS", 30.0)
