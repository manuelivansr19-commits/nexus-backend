"""
NEXUS Ω — Knowledge Engine v3.7.0

Orquestador principal del sistema de conocimiento.

Responsabilidades:
  - Coordinar Store, Retriever, Ingestion, Classifier
  - Proveer interfaz unificada al NexusCore
  - Actualizar statuses automáticamente
  - Estadísticas y health del sistema

Provider agnostic — no conoce Gemini ni ningún LLM.
"""

from __future__ import annotations

from typing import Optional

from backend.config import KNOWLEDGE_DB_PATH, logger
from backend.knowledge.classifier import KnowledgeClassifier
from backend.knowledge.ingestion import IngestionResult, KnowledgeIngestion
from backend.knowledge.models import (
    Domain, KnowledgeContext, KnowledgeEntry,
    KnowledgeStatus, KnowledgeType,
)
from backend.knowledge.retrieval import KnowledgeRetriever
from backend.knowledge.store import KnowledgeStore


class KnowledgeEngine:
    """
    Cerebro del Knowledge Engine.

    Interfaz pública:
      add()            → ingerir conocimiento
      search()         → buscar por texto + filtros
      get()            → obtener por ID
      get_for_context() → recuperar para contexto del LLM
      stats()          → estadísticas
      refresh()        → actualizar statuses por edad
    """

    def __init__(
        self,
        store:      Optional[KnowledgeStore]    = None,
        db_path:    str                         = KNOWLEDGE_DB_PATH,
    ) -> None:
        self._store      = store or KnowledgeStore(db_path)
        self._classifier = KnowledgeClassifier()
        self._retriever  = KnowledgeRetriever(self._store)
        self._ingestion  = KnowledgeIngestion(self._store, self._classifier)
        logger.info("KnowledgeEngine inicializado.")

    # ── WRITE ─────────────────────────────────────────────────

    def add(
        self,
        title:          str,
        content:        str,
        domain:         Optional[str] = None,
        subdomain:      Optional[str] = None,
        source:         str           = "user",
        source_url:     str           = "",
        knowledge_type: Optional[str] = None,
        confidence:     float         = 0.8,
        tags:           Optional[list[str]] = None,
        date_source:    str           = "",
        metadata:       Optional[dict] = None,
        allow_duplicate: bool         = False,
    ) -> IngestionResult:
        """Agregar conocimiento al engine."""
        return self._ingestion.ingest(
            title=title,
            content=content,
            domain=domain,
            subdomain=subdomain,
            source=source,
            source_url=source_url,
            knowledge_type=knowledge_type,
            confidence=confidence,
            tags=tags or [],
            date_source=date_source,
            metadata=metadata or {},
            allow_duplicate=allow_duplicate,
        )

    def add_batch(self, items: list[dict], source: str = "batch") -> list[IngestionResult]:
        return self._ingestion.ingest_batch(items, source)

    def delete(self, entry_id: str) -> bool:
        return self._store.delete(entry_id)

    def mark_review(self, entry_id: str) -> bool:
        return self._store.update_status(entry_id, KnowledgeStatus.REQUIRES_REVIEW)

    # ── READ ──────────────────────────────────────────────────

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        return self._store.get(entry_id)

    def search(
        self,
        query:          str,
        domain:         Optional[str]   = None,
        subdomain:      Optional[str]   = None,
        knowledge_type: Optional[str]   = None,
        status:         Optional[str]   = None,
        min_confidence: float           = 0.0,
        since_days:     Optional[float] = None,
        limit:          int             = 10,
        exclude_outdated: bool          = False,
    ) -> KnowledgeContext:
        """Buscar conocimiento."""
        return self._retriever.search(
            query=query,
            domain=domain,
            subdomain=subdomain,
            knowledge_type=knowledge_type,
            status=status,
            min_confidence=min_confidence,
            since_days=since_days,
            limit=limit,
            exclude_outdated=exclude_outdated,
        )

    def search_by_domain(
        self, domain: str, subdomain: Optional[str] = None, limit: int = 20
    ) -> KnowledgeContext:
        return self._retriever.search_by_domain(domain, subdomain, limit)

    # ── CONTEXT INTEGRATION ───────────────────────────────────

    def get_for_context(
        self,
        query:          str,
        domain:         Optional[str] = None,
        limit:          int           = 5,
        min_confidence: float         = 0.5,
        exclude_outdated: bool        = True,
    ) -> KnowledgeContext:
        """
        Recuperar conocimiento listo para inyectar en el contexto del LLM.
        Filtra entradas de baja confianza y obsoletas por defecto.
        """
        return self._retriever.search(
            query=query,
            domain=domain,
            min_confidence=min_confidence,
            limit=limit,
            exclude_outdated=exclude_outdated,
        )

    def detect_domain_from_text(self, text: str) -> Domain:
        return self._classifier.classify_domain(text)

    # ── MAINTENANCE ───────────────────────────────────────────

    def refresh_statuses(self) -> int:
        """Actualizar statuses de todas las entradas según edad."""
        updated = self._classifier.refresh_statuses(self._store)
        if updated:
            logger.info("KnowledgeEngine: %d entradas actualizadas.", updated)
        return updated

    def stats(self) -> dict:
        base = self._store.stats()
        base["engine"] = "KnowledgeEngine v3.7.0"
        return base

    def domains(self) -> list[str]:
        return [d.value for d in Domain if d != Domain.GENERAL]
