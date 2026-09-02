"""
NEXUS Ω — Knowledge Retrieval v3.7.0

Búsqueda y recuperación de conocimiento.
Arquitectura preparada para semantic/vector search.
Implementación actual: FTS5 + filtros SQL.
"""

from __future__ import annotations

import time
from typing import Optional

from backend.config import logger
from backend.knowledge.models import KnowledgeContext, KnowledgeEntry, KnowledgeStatus


class KnowledgeRetriever:
    """
    Recupera conocimiento relevante para una consulta.

    Estrategias:
      1. FTS (Full-Text Search) — búsqueda por texto
      2. Domain filter — filtrar por dominio específico
      3. Combined — FTS + filtros

    Preparado para:
      - Semantic search (embeddings) — interfaz definida
      - Vector store — migración sin cambios en el engine
    """

    def __init__(self, store) -> None:
        self._store = store

    def search(
        self,
        query:          str,
        domain:         Optional[str]  = None,
        subdomain:      Optional[str]  = None,
        knowledge_type: Optional[str]  = None,
        status:         Optional[str]  = None,
        min_confidence: float          = 0.0,
        since_days:     Optional[float] = None,
        limit:          int            = 10,
        exclude_outdated: bool         = False,
    ) -> KnowledgeContext:
        """
        Buscar conocimiento relevante.
        Retorna KnowledgeContext estructurado.
        """
        started = time.perf_counter()

        # Excluir outdated si se solicita
        if exclude_outdated and status is None:
            status = None  # no filtrar por status, pero filtrar después

        entries = self._store.search_fts(
            query=query,
            domain=domain,
            subdomain=subdomain,
            knowledge_type=knowledge_type,
            status=status,
            min_confidence=min_confidence,
            since_days=since_days,
            limit=limit * 2,   # traer más para filtrar
        )

        if exclude_outdated:
            entries = [
                e for e in entries
                if e.status != KnowledgeStatus.OUTDATED
            ]

        entries = entries[:limit]
        domains_covered = list({e.domain.value for e in entries})
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "KnowledgeRetrieval: query='%s' → %d resultados | %dms",
            query[:50], len(entries), elapsed_ms,
        )

        return KnowledgeContext(
            query=query,
            entries=entries,
            domains_covered=domains_covered,
            total_found=len(entries),
            search_time_ms=elapsed_ms,
        )

    def search_by_domain(
        self,
        domain:    str,
        subdomain: Optional[str] = None,
        limit:     int           = 20,
    ) -> KnowledgeContext:
        """Recuperar todo el conocimiento de un dominio."""
        started = time.perf_counter()
        entries = self._store.filter(
            domain=domain,
            subdomain=subdomain,
            limit=limit,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return KnowledgeContext(
            query=f"domain:{domain}",
            entries=entries,
            domains_covered=[domain],
            total_found=len(entries),
            search_time_ms=elapsed_ms,
        )

    def get_recent(
        self,
        days:   float = 7.0,
        domain: Optional[str] = None,
        limit:  int   = 10,
    ) -> KnowledgeContext:
        """Recuperar conocimiento reciente."""
        started = time.perf_counter()
        entries = self._store.filter(
            domain=domain,
            since_days=days,
            limit=limit,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        domains = list({e.domain.value for e in entries})
        return KnowledgeContext(
            query=f"recent:{days}d",
            entries=entries,
            domains_covered=domains,
            total_found=len(entries),
            search_time_ms=elapsed_ms,
        )

    # ── Semantic search interface (futuro) ────────────────────

    def semantic_search(
        self,
        query:  str,
        limit:  int = 10,
    ) -> KnowledgeContext:
        """
        Búsqueda semántica por embeddings.

        ESTADO: Interfaz definida. Implementación pendiente.
        PREREQUISITO: Vector store (Chroma, Qdrant, pgvector).
        FALLBACK: FTS search.
        """
        logger.info(
            "KnowledgeRetrieval: semantic_search no implementada, "
            "usando FTS fallback."
        )
        return self.search(query=query, limit=limit)
