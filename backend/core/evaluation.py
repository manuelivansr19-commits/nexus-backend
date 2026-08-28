"""
NEXUS Ω — Evaluation.

Evalúa la calidad de respuestas y resultados de acciones.
Permite al sistema aprender qué funciona y qué no.

Fase actual: evaluación heurística simple.
Fase futura: evaluador LLM + feedback humano + RL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvalScore(str, Enum):
    EXCELLENT = "excellent"   # 0.9 – 1.0
    GOOD      = "good"        # 0.7 – 0.9
    ACCEPTABLE = "acceptable" # 0.5 – 0.7
    POOR      = "poor"        # 0.0 – 0.5


@dataclass
class EvaluationResult:
    score:    float          # 0.0 – 1.0
    label:    EvalScore
    reasons:  list[str] = field(default_factory=list)
    provider: Optional[str] = None
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_score(cls, score: float, **kwargs) -> "EvaluationResult":
        if score >= 0.9:
            label = EvalScore.EXCELLENT
        elif score >= 0.7:
            label = EvalScore.GOOD
        elif score >= 0.5:
            label = EvalScore.ACCEPTABLE
        else:
            label = EvalScore.POOR
        return cls(score=score, label=label, **kwargs)


class Evaluator:
    """
    Evalúa respuestas del sistema.
    Registra evaluaciones para análisis posterior.
    """

    def __init__(self) -> None:
        self._history: list[EvaluationResult] = []

    def evaluate_response(
        self,
        response_text: str,
        prompt: str,
        provider: str = "",
        duration_ms: int = 0,
    ) -> EvaluationResult:
        """Heurísticas básicas para evaluar una respuesta."""
        score = 1.0
        reasons: list[str] = []

        # Respuesta vacía
        if not response_text.strip():
            result = EvaluationResult.from_score(
                0.0,
                reasons=["Respuesta vacía"],
                provider=provider,
                duration_ms=duration_ms,
            )
            self._history.append(result)
            return result

        # Longitud mínima
        if len(response_text) < 20:
            score -= 0.3
            reasons.append("Respuesta muy corta")

        # Latencia alta (> 15s)
        if duration_ms > 15_000:
            score -= 0.1
            reasons.append(f"Latencia alta ({duration_ms}ms)")

        # Indicadores de error en texto
        for marker in ("error", "no puedo", "lo siento, no"):
            if marker in response_text.lower()[:100]:
                score -= 0.2
                reasons.append(f"Posible error en respuesta: '{marker}'")
                break

        score = max(0.0, min(1.0, score))
        if not reasons:
            reasons.append("OK")

        result = EvaluationResult.from_score(
            score,
            reasons=reasons,
            provider=provider,
            duration_ms=duration_ms,
        )
        self._history.append(result)
        return result

    def stats(self) -> dict:
        if not self._history:
            return {"total": 0}
        scores = [e.score for e in self._history]
        return {
            "total": len(self._history),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
        }

    def recent(self, limit: int = 10) -> list[EvaluationResult]:
        return self._history[-limit:]
