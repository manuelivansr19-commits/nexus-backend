"""
NEXUS Ω — Intent Router.

Clasifica la intención del mensaje.

Capas (en orden):
  1. Determinística — comandos exactos, patrones regex
  2. Keyword scoring — puntuación por palabras clave por dominio
  3. Fallback LLM — (futuro) clasificación por modelo ligero

Salida estructurada:
  intent           — string identificando la intención
  domain           — área temática
  confidence       — 0.0 – 1.0
  requires_tool    — si necesita una herramienta
  candidate_tools  — nombres de tools candidatas
  requires_memory  — si debe consultar memoria
  strategy         — DIRECT | TOOL | LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Domain(str, Enum):
    SYSTEM       = "system"       # estado, ayuda, configuración
    TIME         = "time"         # hora, fecha, calendarios
    MEMORY       = "memory"       # recordar, buscar, olvidar
    STRATEGY     = "strategy"     # planes, negocios, estrategia
    ANALYSIS     = "analysis"     # análisis, diagnóstico
    TECHNOLOGY   = "technology"   # IA, código, sistemas
    PSYCHOLOGY   = "psychology"   # comportamiento, persuasión
    ECONOMICS    = "economics"    # finanzas, mercados
    LAW          = "law"          # derecho, legal
    ROBOTICS     = "robotics"     # robótica, hardware
    GENERAL      = "general"      # todo lo demás


class IntentStrategy(str, Enum):
    DIRECT = "direct"   # respuesta sin LLM
    TOOL   = "tool"     # invocar herramienta
    LLM    = "llm"      # generar con modelo


@dataclass
class IntentResult:
    intent:          str
    domain:          Domain
    confidence:      float
    strategy:        IntentStrategy
    requires_tool:   bool            = False
    candidate_tools: list[str]       = field(default_factory=list)
    requires_memory: bool            = False
    direct_response: Optional[str]   = None
    metadata:        dict            = field(default_factory=dict)


# ── Reglas determinísticas ────────────────────────────────────

_DIRECT_RULES: list[tuple[list[str], str, str]] = [
    # (patrones, intent, respuesta directa)
    (
        [r"^(hola|hi|hello|buenas|saludos|hey)\s*[!.]*$"],
        "greeting",
        "Manuel, como estas? Todo operativo. En que trabajamos hoy?",
    ),
    (
        [r"^(ping|test)\s*$"],
        "ping",
        "pong",
    ),
    (
        [r"^(gracias|thanks|thank you|ok gracias)\s*[!.]*$"],
        "thanks",
        "De acuerdo.",
    ),
    (
        [r"^(ayuda|help|\?|qué puedes hacer|que puedes hacer)$"],
        "help",
        (
            "NEXUS Ω puede analizar estrategia, negocios, tecnología, "
            "psicología aplicada, derecho y economía. "
            "También gestiona memoria de conversación y herramientas. "
            "¿Cuál es tu objetivo?"
        ),
    ),
]

# ── Keyword scoring por dominio ────────────────────────────────

_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.TIME: [
        "hora", "tiempo", "fecha", "día", "hoy", "mañana", "cuando",
        "time", "date", "clock", "ahora", "semana", "mes", "año",
    ],
    Domain.MEMORY: [
        "recuerda", "recuerdo", "guarda", "memoriza", "olvidar",
        "antes dijiste", "anteriormente", "mencionaste", "busca en",
        "historial", "memoria",
    ],
    Domain.STRATEGY: [
        "estrategia", "plan", "negocio", "empresa", "mercado",
        "competencia", "objetivo", "meta", "kpi", "roadmap",
        "pivot", "escalar", "modelo de negocio", "ventaja competitiva",
    ],
    Domain.ANALYSIS: [
        "analiza", "análisis", "diagnóstico", "evalúa", "diagnoza",
        "revisa", "examina", "causa", "efecto", "problema", "solución",
        "riesgo", "oportunidad",
    ],
    Domain.TECHNOLOGY: [
        "código", "programar", "implementar", "api", "arquitectura",
        "sistema", "software", "algoritmo", "ia", "machine learning",
        "modelo", "datos", "database", "servidor",
    ],
    Domain.PSYCHOLOGY: [
        "psicología", "comportamiento", "persuasión", "manipulación",
        "sesgos", "influencia", "motivación", "liderazgo", "negociación",
        "ventas", "marketing", "consumer",
    ],
    Domain.ECONOMICS: [
        "economía", "finanzas", "inversión", "mercado", "precio",
        "inflación", "bolsa", "cripto", "dinero", "presupuesto",
        "flujo de caja", "valoración",
    ],
    Domain.LAW: [
        "derecho", "legal", "contrato", "ley", "normativa", "regulación",
        "compliance", "gdpr", "propiedad intelectual", "patente",
        "corporativo", "fiscalidad",
    ],
    Domain.ROBOTICS: [
        "robot", "robótica", "servo", "motor", "sensor", "lidar",
        "cámara", "hardware", "arduino", "raspberry", "jetson",
        "actuador", "aura",
    ],
    Domain.SYSTEM: [
        "estado", "status", "nexus", "sistema", "reinicia",
        "configuración", "versión", "proveedor", "gemini",
    ],
}

# ── Tools por intent ──────────────────────────────────────────

_INTENT_TOOLS: dict[str, list[str]] = {
    "time_query":    ["clock"],
    "memory_search": ["memory_search"],
    "system_status": ["system_status"],
}


class IntentRouter:
    """
    Clasifica la intención del mensaje en dos capas:
    1. Reglas determinísticas (exactas/regex)
    2. Keyword scoring por dominio
    """

    def __init__(self, registry=None) -> None:
        self._registry = registry   # ToolRegistry opcional

    def route(self, message: str, context: str = "") -> IntentResult:
        """Clasificar mensaje y retornar IntentResult."""
        stripped = message.strip()
        lower    = stripped.lower()

        # ── Capa 1: reglas determinísticas ───────────────────
        for patterns, intent, response in _DIRECT_RULES:
            for pattern in patterns:
                if re.match(pattern, lower, re.IGNORECASE):
                    return IntentResult(
                        intent=intent,
                        domain=Domain.SYSTEM,
                        confidence=1.0,
                        strategy=IntentStrategy.DIRECT,
                        direct_response=response,
                    )

        # ── Capa 2: keyword scoring ───────────────────────────
        domain, domain_conf = self._score_domain(lower)

        # Detectar necesidad de tools específicas
        requires_tool   = False
        candidate_tools: list[str] = []
        strategy        = IntentStrategy.LLM
        intent          = self._intent_from_domain(lower, domain)
        requires_memory = self._needs_memory(lower)

        # Tools por intent conocido
        if intent in _INTENT_TOOLS:
            candidate_tools = _INTENT_TOOLS[intent]
            requires_tool   = True
            strategy        = IntentStrategy.TOOL

        # Discovery adicional desde registry
        if self._registry and not requires_tool:
            found = self._registry.find_by_intent(lower, max_results=2)
            if found:
                candidate_tools = [t.name for t in found]
                requires_tool   = True
                strategy        = IntentStrategy.TOOL

        return IntentResult(
            intent=intent,
            domain=domain,
            confidence=domain_conf,
            strategy=strategy,
            requires_tool=requires_tool,
            candidate_tools=candidate_tools,
            requires_memory=requires_memory,
        )

    # ── Private ───────────────────────────────────────────────

    def _score_domain(self, lower: str) -> tuple[Domain, float]:
        scores: dict[Domain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return Domain.GENERAL, 0.5

        best = max(scores, key=scores.__getitem__)
        total = sum(scores.values())
        conf  = min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45)
        return best, conf

    def _intent_from_domain(self, lower: str, domain: Domain) -> str:
        # Time
        if domain == Domain.TIME or any(
            w in lower for w in ["hora", "fecha", "qué hora", "que hora"]
        ):
            return "time_query"

        # Memory ops
        if domain == Domain.MEMORY or any(
            w in lower for w in ["recuerda", "busca en memoria", "olvidar"]
        ):
            return "memory_search"

        # System status
        if domain == Domain.SYSTEM and any(
            w in lower for w in ["estado", "status", "funcionando"]
        ):
            return "system_status"

        # Analysis
        if domain == Domain.ANALYSIS:
            return "analysis"

        # Strategy
        if domain == Domain.STRATEGY:
            return "strategy"

        # Code/Tech
        if domain == Domain.TECHNOLOGY:
            return "technology"

        return f"{domain.value}_query"

    def _needs_memory(self, lower: str) -> bool:
        triggers = [
            "recuerda", "antes", "anteriormente", "mencionaste",
            "dijiste", "hablamos", "discutimos", "guarda", "memoriza",
        ]
        return any(t in lower for t in triggers)



