"""
NEXUS Ω — Context Manager v3.7.0

Ensambla el contexto completo para el modelo.
Ahora incluye conocimiento del KnowledgeEngine.

Orden de prioridad (dentro del budget):
  1. System prompt
  2. Conocimiento relevante (KnowledgeEngine)
  3. Contexto de proyecto
  4. Hechos de memoria
  5. Historial de conversación
  6. Resultados de tools
  7. Mensaje actual
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.config import CONTEXT_TOKEN_LIMIT


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class ContextBundle:
    system_prompt:    str
    assembled_prompt: str
    history:          list[dict]
    estimated_tokens: int       = 0
    sources_used:     list[str] = field(default_factory=list)


class ContextManager:

    def __init__(
        self,
        memory=None,
        knowledge_engine=None,
        token_limit: int = CONTEXT_TOKEN_LIMIT,
    ) -> None:
        self._memory           = memory
        self._knowledge_engine = knowledge_engine
        self._token_limit      = token_limit

    def assemble(
        self,
        message:       str,
        system_prompt: str,
        intent=None,
        history:       Optional[list[dict]] = None,
        tool_results:  Optional[list[str]]  = None,
        project:       str = "",
    ) -> ContextBundle:
        sources: list[str] = []
        budget   = self._token_limit
        sections: list[str] = []

        system_tokens  = _estimate_tokens(system_prompt)
        message_tokens = _estimate_tokens(message)
        budget -= system_tokens + message_tokens

        # ── Conocimiento relevante (KnowledgeEngine) ──────────
        if self._knowledge_engine and budget > 800:
            try:
                ctx = self._knowledge_engine.get_for_context(
                    query=message,
                    limit=5,
                    min_confidence=0.5,
                    exclude_outdated=True,
                )
                if not ctx.is_empty():
                    knowledge_text = ctx.to_prompt_string(max_entries=5)
                    cost = _estimate_tokens(knowledge_text)
                    if budget - cost > 600:
                        sections.append(knowledge_text)
                        sources.append("knowledge_engine")
                        budget -= cost
            except Exception:
                pass

        # ── Contexto de proyecto ──────────────────────────────
        if self._memory and project and budget > 500:
            project_ctx = self._memory.projects.get_context(project, limit=3)
            if project_ctx:
                cost = _estimate_tokens(project_ctx)
                if budget - cost > 400:
                    sections.append(project_ctx)
                    sources.append("project_memory")
                    budget -= cost

        # ── Hechos relevantes ─────────────────────────────────
        if self._memory and budget > 400:
            facts = self._memory.facts.search(message, limit=3)
            if facts:
                fact_lines = "\n".join(f"• {e.content}" for e in facts)
                fact_ctx   = f"[MEMORIA FACTUAL]\n{fact_lines}"
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
                sections.append(f"[HERRAMIENTAS]\n{tool_ctx}")
                sources.append("tool_results")
                budget -= cost

        # ── Historial recortado ───────────────────────────────
        final_history = self._trim_history(
            history or [],
            budget=max(0, budget - 200),
        )
        if final_history:
            sources.append("conversation_history")

        # ── Ensamblar ─────────────────────────────────────────
        if sections:
            assembled = "\n\n".join(sections) + f"\n\n[MENSAJE]\n{message}"
        else:
            assembled = message

        total_tokens = (
            system_tokens + message_tokens
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

    def _trim_history(self, history: list[dict], budget: int) -> list[dict]:
        if not history:
            return []
        result: list[dict] = []
        used = 0
        for entry in reversed(history):
            cost = _estimate_tokens(entry.get("content", ""))
            if used + cost > budget:
                break
            result.insert(0, entry)
            used += cost
        return result

    def budget_info(self, text: str) -> dict:
        tokens = _estimate_tokens(text)
        return {
            "estimated_tokens": tokens,
            "limit":            self._token_limit,
            "remaining":        self._token_limit - tokens,
            "usage_pct":        round(tokens / self._token_limit * 100, 1),
        }
