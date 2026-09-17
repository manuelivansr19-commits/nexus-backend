"""NEXUS Omega -- Intent Router v3.8.0 patch"""
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
    FOLLOW_UP    = "follow_up"

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
    MARKETING   = "marketing"
    GENERAL     = "general"

class IntentStrategy(str, Enum):
    DIRECT   = "direct"
    TOOL     = "tool"
    LLM      = "llm"
    AUTONOMY = "autonomy"

AUTONOMY_INTENTS = {IntentType.TASK, IntentType.DESIGN, IntentType.ANALYSIS, IntentType.RESEARCH}
SINGLE_LLM_INTENTS = {IntentType.QUESTION, IntentType.CALCULATION, IntentType.CHAT}

_FOLLOW_UP_PATTERNS = [
    r"^(dame|dime|muestra|entrega|da me)\s+(el|la|lo|los|las|ese|esa|eso)",
    r"^(y el|y la|y lo|y eso|y ese)",
    r"(que te pedi|que te solicite|que pediste|lo que pedi|lo anterior|el estudio|el analisis|el reporte|el informe|el resultado)",
    r"^(ya|ahora|entonces|bueno|ok|okay)\s+(dame|dime|hazlo|ejecuta)",
    r"(sigue|continua|termina|completa|finaliza)\s+(con|el|la|lo|eso)",
]

_DIRECT_RULES = [
    ([r"^(hola|hi|hello|buenas|saludos|hey)\s*[!.]*$"], IntentType.CHAT,
     "Manuel, como estas? Todo operativo. En que trabajamos hoy?"),
    ([r"^(ping|test)\s*$"], IntentType.SYSTEM, "pong"),
    ([r"^(ayuda|help|\?|que puedes hacer)$"], IntentType.SYSTEM,
     "NEXUS puede analizar, investigar, planificar y ejecutar tareas complejas. Describe tu objetivo."),
    ([r"^(estado|status)\s*$"], IntentType.SYSTEM, "NEXUS Omega v3.8.0 operativo."),
]

_INTENT_KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.RESEARCH: [
        "estudio de mercado","investigacion","investiga","busca informacion",
        "encuentra datos","analiza el mercado","mercado en","competencia en",
        "oportunidades en","tendencias","sector","industria","demografica",
        "canales de","alternativas a","formas de llegar","sin google ads",
        "marketing digital","posicionamiento","estrategia de marketing",
        "publico objetivo","target","segmento","lanzamiento","lanza",
        "hazlo ahora","hazlo ahorita","ejecuta","genera el","dame el estudio",
        "dame el reporte","dame el analisis","dame el informe",
    ],
    IntentType.ANALYSIS: [
        "analiza","analisis","evalua","diagnostico","examina","revisa",
        "es viable","factibilidad","ventajas","desventajas","pros","contras",
        "riesgos","oportunidades","diagnostica",
    ],
    IntentType.TASK: [
        "implementa","crea","desarrolla","construye","planifica","organiza",
        "ejecuta","produce","genera","elabora","prepara","diseña",
        "quiero que","necesito que","puedes hacer","haz me","hazme",
    ],
    IntentType.DESIGN: [
        "diseña","arquitectura","sistema para","propone","estructura",
        "modelo de","framework","esquema","blueprint",
    ],
    IntentType.QUESTION: [
        "que es","como funciona","explica","define","que significa","cual es",
        "por que","para que","diferencia entre","cuantos","cuando fue",
    ],
    IntentType.CALCULATION: [
        "calcula","cuanto","cuantos","estima","proyecta","presupuesto",
        "costo","precio","porcentaje","tasa","roi",
    ],
    IntentType.MEMORY_QUERY: [
        "recuerda","recuerdo","antes dijiste","anteriormente","mencionaste",
        "memoriza","guarda","olvidar","discutimos","acordamos",
    ],
    IntentType.SYSTEM: [
        "nexus","sistema","configuracion","estado","version","herramientas",
    ],
}

_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.TIME:       ["hora","tiempo","fecha","dia","hoy","manana"],
    Domain.MEMORY:     ["recuerda","memoria","guarda","historial"],
    Domain.STRATEGY:   ["estrategia","negocio","empresa","mercado","competencia","lanzamiento"],
    Domain.MARKETING:  ["marketing","publicidad","ads","campana","canal","cliente","ventas",
                        "estudio de mercado","google ads","facebook ads","redes sociales",
                        "posicionamiento","branding","leads","conversion"],
    Domain.ANALYSIS:   ["analiza","analisis","diagnostico","evalua"],
    Domain.TECHNOLOGY: ["codigo","programar","api","sistema","software","ia"],
    Domain.PSYCHOLOGY: ["psicologia","comportamiento","persuasion","sesgos"],
    Domain.ECONOMICS:  ["economia","finanzas","inversion","precio"],
    Domain.LAW:        ["derecho","legal","contrato","ley"],
    Domain.ROBOTICS:   ["robot","robotica","servo","motor","sensor","aura"],
    Domain.SCIENCE:    ["ciencia","fisica","quimica","biologia"],
    Domain.SYSTEM:     ["nexus","sistema","estado","status"],
}

_INTENT_TOOLS: dict[str, list[str]] = {
    "time_query":    ["clock"],
    "memory_search": ["memory_search"],
    "system_status": ["system_status"],
}

@dataclass
class IntentResult:
    intent:            IntentType
    domain:            Domain
    confidence:        float
    strategy:          IntentStrategy
    requires_tool:     bool             = False
    candidate_tools:   list             = field(default_factory=list)
    requires_memory:   bool             = False
    requires_planning: bool             = False
    is_follow_up:      bool             = False
    direct_response:   Optional[str]    = None
    metadata:          dict             = field(default_factory=dict)

class IntentRouter:
    def __init__(self, registry=None):
        self._registry = registry

    def route(self, message: str, context: str = "") -> IntentResult:
        stripped = message.strip()
        lower    = stripped.lower()

        # Capa 1: reglas directas
        for patterns, intent_type, response in _DIRECT_RULES:
            for pattern in patterns:
                if re.match(pattern, lower, re.IGNORECASE):
                    return IntentResult(intent=intent_type, domain=Domain.SYSTEM,
                        confidence=1.0, strategy=IntentStrategy.DIRECT, direct_response=response)

        # Capa 2: follow-up detection
        for pattern in _FOLLOW_UP_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                return IntentResult(intent=IntentType.FOLLOW_UP, domain=Domain.MEMORY,
                    confidence=0.9, strategy=IntentStrategy.TOOL,
                    requires_tool=True, candidate_tools=["memory_search"],
                    requires_memory=True, is_follow_up=True)

        # Capa 3: keyword scoring
        intent_type, intent_conf = self._score_intent(lower)
        domain,      _           = self._score_domain(lower)

        requires_memory   = self._needs_memory(lower)
        requires_planning = intent_type in AUTONOMY_INTENTS
        strategy          = self._determine_strategy(intent_type, lower)

        requires_tool  = False
        candidate_tools = []

        if intent_type == IntentType.MEMORY_QUERY:
            candidate_tools = ["memory_search"]
            requires_tool   = True
            strategy        = IntentStrategy.TOOL
        elif domain == Domain.TIME or any(w in lower for w in ["hora","fecha"]):
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

        return IntentResult(intent=intent_type, domain=domain, confidence=intent_conf,
            strategy=strategy, requires_tool=requires_tool, candidate_tools=candidate_tools,
            requires_memory=requires_memory, requires_planning=requires_planning)

    def _score_intent(self, lower):
        scores = {}
        for intent_type, keywords in _INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[intent_type] = score
        if not scores:
            words = len(lower.split())
            if words > 15: return IntentType.RESEARCH, 0.6
            if words > 8:  return IntentType.ANALYSIS, 0.55
            if words > 3:  return IntentType.QUESTION, 0.55
            return IntentType.CHAT, 0.5
        best  = max(scores, key=scores.__getitem__)
        total = sum(scores.values())
        conf  = min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45)
        return best, conf

    def _score_domain(self, lower):
        scores = {}
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

    def _determine_strategy(self, intent_type, lower):
        if intent_type in AUTONOMY_INTENTS:
            return IntentStrategy.AUTONOMY
        if intent_type == IntentType.MEMORY_QUERY:
            return IntentStrategy.TOOL
        return IntentStrategy.LLM

    def _needs_memory(self, lower):
        return any(t in lower for t in [
            "recuerda","antes","anteriormente","mencionaste",
            "dijiste","hablamos","discutimos","acordamos","guarda","memoriza"])
