"""
NEXUS Ω — Reasoning.

Interfaz de razonamiento. Decide si una solicitud necesita LLM
o puede resolverse con lógica local (reglas, herramientas).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ReasoningStrategy(str, Enum):
    DIRECT      = "direct"       # respuesta directa sin LLM
    LLM         = "llm"          # necesita modelo de lenguaje
    TOOL        = "tool"         # necesita una herramienta
    MULTI_STEP  = "multi_step"   # requiere planificación


@dataclass
class ReasoningResult:
    strategy:    ReasoningStrategy
    confidence:  float = 1.0
    context:     str   = ""         # contexto ensamblado para el LLM
    tool_name:   Optional[str] = None
    direct_response: Optional[str] = None
    metadata:    dict = field(default_factory=dict)


class Reasoning:
    """
    Clasificador de intención + ensamblador de contexto.

    Fase actual: reglas simples.
    Fase futura: clasificador local (embeddings + intent model).
    """

    # Patrones que NO necesitan LLM
    _DIRECT_PATTERNS = {
        "hora":    lambda: time.strftime("%H:%M:%S"),
        "ping":    lambda: "pong",
        "version": lambda: "NEXUS Ω v3.4.0",
        "estado":  lambda: "Sistemas operativos.",
    }

    def analyze(
        self,
        prompt: str,
        perception_context: str = "",
        memory_context: str = "",
    ) -> ReasoningResult:
        """
        Analiza el prompt y decide la estrategia.
        """
        lower = prompt.strip().lower()

        # ── Respuestas directas sin LLM ──────────────────────
        for keyword, fn in self._DIRECT_PATTERNS.items():
            if lower == keyword or lower.startswith(keyword + " "):
                return ReasoningResult(
                    strategy=ReasoningStrategy.DIRECT,
                    direct_response=fn(),
                    confidence=1.0,
                )

        # ── Necesita herramienta específica ─────────────────
        if any(w in lower for w in ("busca", "buscar", "search", "web")):
            return ReasoningResult(
                strategy=ReasoningStrategy.TOOL,
                tool_name="web_search",
                confidence=0.85,
                context=self._assemble_context(prompt, perception_context, memory_context),
            )

        # ── Default: LLM ─────────────────────────────────────
        return ReasoningResult(
            strategy=ReasoningStrategy.LLM,
            confidence=0.95,
            context=self._assemble_context(prompt, perception_context, memory_context),
        )

    def _assemble_context(
        self,
        prompt: str,
        perception: str,
        memory: str,
    ) -> str:
        """Ensambla el contexto completo para el LLM."""
        parts = []
        if perception and perception != "Sin percepción activa.":
            parts.append(f"[PERCEPCIÓN]\n{perception}")
        if memory and memory != "Sin contexto de trabajo.":
            parts.append(f"[MEMORIA]\n{memory}")
        parts.append(f"[SOLICITUD]\n{prompt}")
        return "\n\n".join(parts)
