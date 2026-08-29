"""
NEXUS Ω — Context Manager.

Ensambla el contexto completo que se envía al modelo.
Respeta un límite de tokens para no saturar la ventana de contexto.

El contexto combina (en orden de prioridad):
  1. System prompt
  2. Contexto de proyecto activo
  3. Hechos relevantes de memoria
  4. Historial de conversación reciente
  5. Resultado de tools ejecutadas
  6. Mensaje actual del usuario
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.config import CONTEXT_TOKEN_LIMIT


def _estimate_tokens(text: str) -> int:
    """Estimación rápida: 1 token ≈ 4 caracteres."""
    return max(1, len(text) // 4)


@dataclass
class ContextBundle:
    """Contexto ensamblado listo para el modelo."""
    system_prompt:    str
    assembled_prompt: str          # mensaje enriquecido para el modelo
    history:          list[dict]   # [{"role": ..., "content": ...}]
    estimated_tokens: int = 0
    sources_used:     list[str] = field(default_factory=list)


class ContextManager:
    """
    Ensambla contexto respetando el límite de tokens.

    Estrategia de recorte (cuando el budget se agota):
      1. Reducir historial (mantener los más recientes)
      2. Reducir memoria de hechos
      3. Reducir contexto de proyecto
      4. Mantener siempre: system prompt + mensaje actual
    """

    def __init__(
        self,
        memory=None,
        token_limit: int = CONTEXT_TOKEN_LIMIT,
    ) -> None:
        self._memory      = memory
        self._token_limit = token_limit

    def assemble(
        self,
        message:       str,
        system_prompt: str,
        intent=None,
        history:       Optional[list[dict]] = None,
        tool_results:  Optional[list[str]]  = None,
        project:       str = "",
    ) -> ContextBundle:
        """
        Ensambla el contexto completo.

        Retorna ContextBundle con:
        - system_prompt final
        - assembled_prompt (mensaje enriquecido)
        - history (recortado si necesario)
        - estimated_tokens
        - sources_used
        """
        sources: list[str] = []
        budget   = self._token_limit
        sections: list[str] = []

        # Reservar presupuesto para componentes fijos
        system_tokens = _estimate_tokens(system_prompt)
        message_tokens = _estimate_tokens(message)
        budget -= system_tokens + message_tokens

        # ── Contexto de proyecto ──────────────────────────────
        project_ctx = ""
        if self._memory and project:
            project_ctx = self._memory.projects.get_context(project, limit=3)
            if project_ctx:
                cost = _estimate_tokens(project_ctx)
                if budget - cost > 500:
                    sections.append(project_ctx)
                    sources.append("project_memory")
                    budget -= cost

        # ── Hechos relevantes ─────────────────────────────────
        fact_ctx = ""
        if self._memory and intent and budget > 500:
            facts = self._memory.facts.search(message, limit=3)
            if facts:
                fact_lines = "\n".join(f"• {e.content}" for e in facts)
                fact_ctx   = f"[HECHOS RELEVANTES]\n{fact_lines}"
                cost       = _estimate_tokens(fact_ctx)
                if budget - cost > 300:
                    sections.append(fact_ctx)
                    sources.append("fact_memory")
                    budget -= cost

        # ── Tool results ──────────────────────────────────────
        if tool_results:
            tool_ctx = "\n".join(tool_results)
            cost     = _estimate_tokens(tool_ctx)
            if budget - cost > 200:
                sections.append(f"[RESULTADOS DE HERRAMIENTAS]\n{tool_ctx}")
                sources.append("tool_results")
                budget -= cost

        # ── Historial de conversación (recortado) ─────────────
        final_history = self._trim_history(
            history or [],
            budget=max(0, budget - 200),
        )
        if final_history:
            sources.append("conversation_history")

        # ── Ensamblar prompt final ────────────────────────────
        if sections:
            context_block = "\n\n".join(sections)
            assembled = f"{context_block}\n\n[MENSAJE ACTUAL]\n{message}"
        else:
            assembled = message

        total_tokens = (
            system_tokens
            + message_tokens
            + _estimate_tokens("\n\n".join(sections))
            + sum(_estimate_tokens(h["content"]) for h in final_history)
        )

        return ContextBundle(
            system_prompt=system_prompt,
            assembled_prompt=assembled,
            history=final_history,
            estimated_tokens=total_tokens,
            sources_used=sources,
        )

    def _trim_history(
        self,
        history: list[dict],
        budget: int,
    ) -> list[dict]:
        """Recortar historial respetando el budget de tokens."""
        if not history:
            return []

        # Empezar desde el más reciente
        result: list[dict] = []
        used   = 0
        for entry in reversed(history):
            cost = _estimate_tokens(entry.get("content", ""))
            if used + cost > budget:
                break
            result.insert(0, entry)
            used += cost

        return result

    def budget_info(self, text: str) -> dict:
        """Información de uso de tokens para un texto."""
        tokens = _estimate_tokens(text)
        return {
            "estimated_tokens": tokens,
            "limit":            self._token_limit,
            "remaining":        self._token_limit - tokens,
            "usage_pct":        round(tokens / self._token_limit * 100, 1),
        }
