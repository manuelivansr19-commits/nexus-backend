"""
NEXUS Ω — Intent Router v3.6.0

Clasifica la intención del mensaje en 9 tipos:

  CHAT         → conversación simple, saludos
  QUESTION     → pregunta factual ("¿qué es X?")
  ANALYSIS     → análisis profundo de un tema/situación
  RESEARCH     → investigación, buscar información
  TASK         → tarea multi-paso que requiere planificación
  DESIGN       → diseñar sistemas, arquitecturas, productos
  CALCULATION  → cálculos numéricos, estimaciones
  SYSTEM       → comandos del sistema NEXUS
  MEMORY_QUERY → consulta o escritura en memoria

Capas de clasificación:
  1. Reglas determinísticas (regex / comandos exactos)
  2. Keyword scoring por intent y dominio
  3. Heurística de complejidad (longitud + verbos de acción)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IntentType(str, Enum):
    CHAT         = "chat"
    QUESTION     = "question"
    ANALYSIS     = "analysis"
    RESEARCH     = "research"
    TASK         = "task"
    DESIGN       = "design"
    CALCULATION  = "calculation"
    SYSTEM       = "system"
    MEMORY_QUERY = "memory_query"


class Domain(str, Enum):
    SYSTEM      = "system"
    TIME        = "time"
    MEMORY      = "memory"
    STRATEGY    = "strategy"
    ANALYSIS    = "analysis"
    TECHNOLOGY  = "technology"
    PSYCHOLOGY  = "psychology"
    ECONOMICS   = "economics"
    LAW         = "law"
    ROBOTICS    = "robotics"
    SCIENCE     = "science"
    GENERAL     = "general"


class IntentStrategy(str, Enum):
    DIRECT    = "direct"    # respuesta sin LLM
    TOOL      = "tool"      # invocar herramienta
    LLM       = "llm"       # llamada simple al modelo
    AUTONOMY  = "autonomy"  # loop multi-paso


# Intents que requieren planificación autónoma
AUTONOMY_INTENTS = {IntentType.TASK, IntentType.DESIGN, IntentType.ANALYSIS}
# Intents que se resuelven con una sola llamada LLM
SINGLE_LLM_INTENTS = {IntentType.QUESTION, IntentType.RESEARCH, IntentType.CALCULATION, IntentType.CHAT}


@dataclass
class IntentResult:
    intent:           IntentType
    domain:           Domain
    confidence:       float
    strategy:         IntentStrategy
    requires_tool:    bool             = False
    candidate_tools:  list[str]        = field(default_factory=list)
    requires_memory:  bool             = False
    requires_planning: bool            = False
    direct_response:  Optional[str]    = None
    metadata:         dict             = field(default_factory=dict)


# ── Reglas determinísticas ────────────────────────────────────

_DIRECT_RULES: list[tuple[list[str], IntentType, str]] = [
    (
        [r"^(hola|hi|hello|buenas|saludos|hey)\s*[!.]*$"],
        IntentType.CHAT,
        "Manuel, como estas? Todo operativo. En que trabajamos hoy?",
    ),
    (
        [r"^(ping|test)\s*$"],
        IntentType.SYSTEM,
        "pong",
    ),
    (
        [r"^(gracias|thanks|thank you|ok gracias)\s*[!.]*$"],
        IntentType.CHAT,
        "De acuerdo.",
    ),
    (
        [r"^(ayuda|help|\?|que puedes hacer|qué puedes hacer)$"],
        IntentType.SYSTEM,
        (
            "NEXUS Ω puede analizar, diseñar, calcular, investigar y planificar. "
            "Intents activos: CHAT, QUESTION, ANALYSIS, RESEARCH, TASK, DESIGN, "
            "CALCULATION, SYSTEM, MEMORY_QUERY. ¿Qué necesitas?"
        ),
    ),
    (
        [r"^(estado|status)\s*$"],
        IntentType.SYSTEM,
        "NEXUS Ω v3.6.0 operativo. Autonomy Core activo.",
    ),
]

# ── Keyword scoring por IntentType ────────────────────────────

_INTENT_KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.QUESTION: [
        "qué es", "que es", "cómo funciona", "como funciona",
        "explica", "define", "qué significa", "cuál es",
        "por qué", "para qué", "diferencia entre",
    ],
    IntentType.ANALYSIS: [
        "analiza", "análisis", "evalúa", "diagnóstico",
        "examina", "revisa", "identifica riesgos",
        "ventajas y desventajas", "pros y contras",
        "es viable", "factibilidad",
    ],
    IntentType.RESEARCH: [
        "investiga", "busca información", "encuentra",
        "qué se sabe sobre", "últimas noticias", "estado del arte",
        "tendencias", "investigación sobre",
    ],
    IntentType.TASK: [
        "implementa", "crea", "desarrolla", "construye",
        "planifica", "organiza", "ejecuta", "produce",
        "genera", "elabora", "prepara",
    ],
    IntentType.DESIGN: [
        "diseña", "arquitectura", "sistema para", "propón",
        "estructura", "modelo de", "framework", "esquema",
        "plano", "blueprint",
    ],
    IntentType.CALCULATION: [
        "calcula", "cuánto", "cuántos", "estima",
        "proyecta", "presupuesto", "costo", "precio",
        "porcentaje", "tasa", "roi", "consumo",
    ],
    IntentType.MEMORY_QUERY: [
        "recuerda", "recuerdo", "antes dijiste", "anteriormente",
        "memoriza", "guarda", "olvidar", "mencionaste",
        "discutimos", "acordamos", "decidimos",
    ],
    IntentType.SYSTEM: [
        "nexus", "sistema", "configuración", "estado",
        "reinicia", "versión", "proveedor", "herramientas",
    ],
}

# ── Domain keywords ───────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.TIME:       ["hora", "tiempo", "fecha", "día", "hoy", "mañana"],
    Domain.MEMORY:     ["recuerda", "memoria", "guarda", "historial"],
    Domain.STRATEGY:   ["estrategia", "negocio", "empresa", "mercado", "competencia"],
    Domain.ANALYSIS:   ["analiza", "análisis", "diagnóstico", "evalúa"],
    Domain.TECHNOLOGY: ["código", "programar", "api", "sistema", "software", "ia"],
    Domain.PSYCHOLOGY: ["psicología", "comportamiento", "persuasión", "sesgos"],
    Domain.ECONOMICS:  ["economía", "finanzas", "inversión", "mercado", "precio"],
    Domain.LAW:        ["derecho", "legal", "contrato", "ley", "normativa"],
    Domain.ROBOTICS:   ["robot", "robótica", "servo", "motor", "sensor", "aura"],
    Domain.SCIENCE:    ["ciencia", "física", "química", "biología", "matemáticas"],
    Domain.SYSTEM:     ["nexus", "sistema", "estado", "status"],
}

# ── Tool mappings ─────────────────────────────────────────────

_INTENT_TOOLS: dict[str, list[str]] = {
    "time_query":    ["clock"],
    "memory_search": ["memory_search"],
    "system_status": ["system_status"],
}


class IntentRouter:
    """
    Clasifica mensajes en 9 tipos de intent.

    Capa 1: reglas determinísticas
    Capa 2: keyword scoring
    Capa 3: heurística de complejidad
    """

    def __init__(self, registry=None) -> None:
        self._registry = registry

    def route(self, message: str, context: str = "") -> IntentResult:
        stripped = message.strip()
        lower    = stripped.lower()

        # ── Capa 1: determinística ────────────────────────────
        for patterns, intent_type, response in _DIRECT_RULES:
            for pattern in patterns:
                if re.match(pattern, lower, re.IGNORECASE):
                    return IntentResult(
                        intent=intent_type,
                        domain=Domain.SYSTEM,
                        confidence=1.0,
                        strategy=IntentStrategy.DIRECT,
                        direct_response=response,
                    )

        # ── Capa 2: keyword scoring ───────────────────────────
        intent_type, intent_conf = self._score_intent(lower)
        domain, _                = self._score_domain(lower)

        requires_memory = self._needs_memory(lower)
        requires_planning = intent_type in AUTONOMY_INTENTS

        # Determinar estrategia
        strategy = self._determine_strategy(intent_type, lower)

        # Tool candidates
        requires_tool   = False
        candidate_tools: list[str] = []

        key = f"{intent_type.value}_query" if intent_type == IntentType.SYSTEM else ""
        if intent_type == IntentType.MEMORY_QUERY:
            candidate_tools = ["memory_search"]
            requires_tool   = True
            strategy        = IntentStrategy.TOOL
        elif domain == Domain.TIME or any(w in lower for w in ["hora", "fecha"]):
            candidate_tools = ["clock"]
            requires_tool   = True
            strategy        = IntentStrategy.TOOL
        elif self._registry and not requires_planning:
            found = self._registry.find_by_intent(lower, max_results=2)
            if found:
                candidate_tools = [t.name for t in found]
                requires_tool   = True
                if strategy == IntentStrategy.LLM:
                    strategy = IntentStrategy.TOOL

        return IntentResult(
            intent=intent_type,
            domain=domain,
            confidence=intent_conf,
            strategy=strategy,
            requires_tool=requires_tool,
            candidate_tools=candidate_tools,
            requires_memory=requires_memory,
            requires_planning=requires_planning,
        )

    # ── Private ───────────────────────────────────────────────

    def _score_intent(self, lower: str) -> tuple[IntentType, float]:
        scores: dict[IntentType, int] = {}
        for intent_type, keywords in _INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[intent_type] = score

        if not scores:
            # Heurística de longitud: mensajes largos → ANALYSIS o TASK
            words = len(lower.split())
            if words > 20:
                return IntentType.ANALYSIS, 0.55
            if words > 10:
                return IntentType.QUESTION, 0.55
            return IntentType.CHAT, 0.5

        best  = max(scores, key=scores.__getitem__)
        total = sum(scores.values())
        conf  = min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45)
        return best, conf

    def _score_domain(self, lower: str) -> tuple[Domain, float]:
        scores: dict[Domain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return Domain.GENERAL, 0.5
        best  = max(scores, key=scores.__getitem__)
        total = sum(scores.values())
        conf  = min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45)
        return best, conf

    def _determine_strategy(self, intent_type: IntentType, lower: str) -> IntentStrategy:
        if intent_type in AUTONOMY_INTENTS:
            return IntentStrategy.AUTONOMY
        if intent_type == IntentType.MEMORY_QUERY:
            return IntentStrategy.TOOL
        return IntentStrategy.LLM

    def _needs_memory(self, lower: str) -> bool:
        return any(t in lower for t in [
            "recuerda", "antes", "anteriormente", "mencionaste",
            "dijiste", "hablamos", "discutimos", "acordamos",
            "guarda", "memoriza",
        ])
