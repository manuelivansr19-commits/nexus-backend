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


# ------------------------------------------------------------
# Soberanía de inferencia (Fase 1)
#
# Las funciones siguientes son PURAS (reciben el valor crudo y devuelven
# resultado + advertencias) para poder probarlas sin recargar el módulo.
# Las advertencias NUNCA repiten el valor recibido: una variable mal
# escrita podría contener un secreto pegado por error.
# ------------------------------------------------------------

CONFIG_WARNINGS: list = []

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}

VALID_INFERENCE_MODES = ("local_first", "local_only", "cloud_first", "cloud_only")
KNOWN_CLOUD_RUNTIMES = ("gemini", "groq", "openrouter")
DEFAULT_CLOUD_RUNTIMES = ("gemini",)
DEFAULT_TASKS_DB_PATH = "nexus_tasks.db"


def parse_strict_bool(name: str, raw, default: bool = False):
    """Lectura restrictiva: un valor no reconocido cuenta como ACTIVADO.

    Pensada para interruptores de seguridad (NEXUS_LOCAL_ONLY): equivocarse al
    escribirlos no debe abrir el acceso a servicios externos.
    """
    if raw is None:
        return default, []
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True, []
    if value in _FALSE_VALUES:
        return False, []
    return True, [f"{name} tiene un valor no reconocido; se trata como activado (restrictivo)."]


def parse_cloud_runtimes(raw):
    """Lista de runtimes cloud permitidos. Sin variable -> solo Gemini; vacía -> ninguno."""
    if raw is None:
        return DEFAULT_CLOUD_RUNTIMES, []
    names, unknown = [], 0
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in KNOWN_CLOUD_RUNTIMES:
            unknown += 1
        elif item not in names:
            names.append(item)
    warnings = []
    if unknown:
        warnings.append(
            f"CLOUD_RUNTIMES_ENABLED contiene {unknown} nombre(s) desconocido(s); se ignoran.")
    return tuple(names), warnings


def resolve_inference_settings(mode_raw: str, allow_cloud: bool, local_only: bool):
    """Modo de inferencia efectivo. La política más restrictiva siempre prevalece.

    - Modo inválido -> local_only con cloud bloqueado (no se abre la nube por una errata).
    - NEXUS_LOCAL_ONLY activo -> local_only con cloud bloqueado, sea cual sea el modo pedido.
    - ALLOW_CLOUD_INFERENCE=false nunca se relaja.
    """
    warnings = []
    mode = (mode_raw or "").strip().lower()
    allow = bool(allow_cloud)
    if mode not in VALID_INFERENCE_MODES:
        warnings.append(
            "INFERENCE_MODE no reconocido; se usa local_only con la nube bloqueada.")
        mode, allow = "local_only", False
    if local_only:
        mode, allow = "local_only", False
    return mode, allow, warnings


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
# New: run ONLY on local engine — no external calls at all.
# Lectura restrictiva: un valor no reconocido cuenta como activado (+ advertencia).
NEXUS_LOCAL_ONLY, _w = parse_strict_bool("NEXUS_LOCAL_ONLY", os.getenv("NEXUS_LOCAL_ONLY"), False)
CONFIG_WARNINGS.extend(_w)

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

# Base de tareas (antes fija en main.py). Por defecto conserva el comportamiento anterior.
TASKS_DB_PATH = _os.getenv("TASKS_DB_PATH", DEFAULT_TASKS_DB_PATH).strip() or DEFAULT_TASKS_DB_PATH

# Context window budget (approx tokens: chars / 4)
CONTEXT_TOKEN_LIMIT = _int_env("CONTEXT_TOKEN_LIMIT", 6000)

# Max memory entries returned per search
MEMORY_SEARCH_LIMIT = _int_env("MEMORY_SEARCH_LIMIT", 8)

# Memoria persistente (opcional): guarda solo ante una instruccion explicita del usuario.
# Aislamiento por proyecto unicamente (no hay aislamiento por identidad de usuario).
PERSISTENT_MEMORY_ENABLED       = _os.getenv("PERSISTENT_MEMORY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
PERSISTENT_MEMORY_MAX_PER_SCOPE = _int_env("PERSISTENT_MEMORY_MAX_PER_SCOPE", 500)
PERSISTENT_MEMORY_MAX_CHARS     = _int_env("PERSISTENT_MEMORY_MAX_CHARS", 500)
PERSISTENT_MEMORY_CONTEXT_LIMIT = _int_env("PERSISTENT_MEMORY_CONTEXT_LIMIT", 5)

# Intent confidence threshold below which we fall back to LLM classification
INTENT_CONFIDENCE_THRESHOLD = 0.75

# Tool execution timeout (seconds)
TOOL_TIMEOUT_SECONDS = _float_env("TOOL_TIMEOUT_SECONDS", 30.0)


# ============================================================
# PHASE 3 — AUTONOMY CORE
# ============================================================

APP_VERSION = "3.6.0"   # override anterior

MAX_PLAN_STEPS       = _int_env("MAX_PLAN_STEPS", 15)
MAX_EXECUTION_LOOPS  = _int_env("MAX_EXECUTION_LOOPS", 20)
MAX_RETRIES_PER_STEP = _int_env("MAX_RETRIES_PER_STEP", 3)
AUTONOMY_ENABLED     = _bool_env("AUTONOMY_ENABLED", True)

# System prompt del Planner (no expone secretos)
PLANNER_SYSTEM = """
Eres NEXUS Planner. Tu única función es crear planes estructurados en JSON.
Responde SOLO con JSON válido. Sin markdown, sin explicaciones, sin texto extra.
Los pasos deben ser concretos, verificables y ordenados.
Nunca incluyas acciones irreversibles, destructivas o que requieran acceso externo no disponible.
""".strip()


# ============================================================
# PHASE 4 — KNOWLEDGE ENGINE
# ============================================================

APP_VERSION = "3.7.0"   # override

KNOWLEDGE_DB_PATH      = _os.getenv("KNOWLEDGE_DB_PATH", "nexus_knowledge.db").strip()
KNOWLEDGE_MAX_RESULTS  = _int_env("KNOWLEDGE_MAX_RESULTS", 10)
KNOWLEDGE_MIN_CONFIDENCE = 0.5
KNOWLEDGE_CONTEXT_LIMIT  = _int_env("KNOWLEDGE_CONTEXT_LIMIT", 5)


# ============================================================
# PHASE 5 — LOCAL-FIRST / INFERENCE ABSTRACTION
# ============================================================

APP_VERSION = "3.8.0"

# Modo de inferencia: local_first | local_only | cloud_first (deprecated) | cloud_only
# Interruptor de cloud — false = NUNCA llamar servicios externos de inferencia
#
# Los dos valores EFECTIVOS se resuelven aquí, una sola vez; todo el sistema
# (política, /health, /api/nexus/*) debe leer estas constantes o la política
# construida con ellas. La política más restrictiva siempre prevalece:
# NEXUS_LOCAL_ONLY, un modo inválido o ALLOW_CLOUD_INFERENCE=false cierran la nube.
INFERENCE_MODE, ALLOW_CLOUD_INFERENCE, _w = resolve_inference_settings(
    _os.getenv("INFERENCE_MODE", "local_first"),
    _bool_env("ALLOW_CLOUD_INFERENCE", True),
    NEXUS_LOCAL_ONLY,
)
CONFIG_WARNINGS.extend(_w)

# Runtimes cloud habilitados (lista blanca). Sin variable: solo Gemini.
# Groq y OpenRouter quedan desactivados salvo habilitación explícita; ningún
# proveedor se activa solo porque otro falle. Vacía = ningún runtime cloud.
CLOUD_RUNTIMES_ENABLED, _w = parse_cloud_runtimes(_os.getenv("CLOUD_RUNTIMES_ENABLED"))
CONFIG_WARNINGS.extend(_w)
del _w

for _msg in CONFIG_WARNINGS:
    logger.warning("CONFIG: %s", _msg)


# PHASE 5 - LOCAL-FIRST
APP_VERSION = '3.8.0'
