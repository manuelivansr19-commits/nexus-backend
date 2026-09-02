"""
NEXUS Ω — Knowledge Ingestion v3.7.0

Ingestión de conocimiento con validación y deduplicación.

Fuentes soportadas:
  - Manual (usuario directo)
  - API (POST)
  - Archivos (interfaz definida, no implementada)
  - Web (interfaz definida, no implementada)

El Ingestion pipeline:
  Input → Validate → Classify → Deduplicate → Store
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.config import logger
from backend.knowledge.classifier import KnowledgeClassifier
from backend.knowledge.models import (
    Domain, KnowledgeEntry, KnowledgeStatus, KnowledgeType,
)


@dataclass
class IngestionResult:
    success:    bool
    entry_id:   Optional[str] = None
    duplicate:  bool          = False
    errors:     list[str]     = None
    action:     str           = "created"   # created | updated | skipped

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class KnowledgeIngestion:
    """
    Pipeline de ingestión de conocimiento.

    Valida → Clasifica → Deduplica → Guarda.
    El Ingestion NO ejecuta código ni accede a sistemas externos.
    """

    def __init__(self, store, classifier: Optional[KnowledgeClassifier] = None) -> None:
        self._store      = store
        self._classifier = classifier or KnowledgeClassifier()

    def ingest(
        self,
        title:          str,
        content:        str,
        domain:         Optional[str]  = None,
        subdomain:      Optional[str]  = None,
        source:         str            = "user",
        source_url:     str            = "",
        knowledge_type: Optional[str]  = None,
        confidence:     float          = 0.8,
        tags:           Optional[list[str]] = None,
        date_source:    str            = "",
        metadata:       Optional[dict] = None,
        allow_duplicate: bool          = False,
    ) -> IngestionResult:
        """
        Ingerir una entrada de conocimiento.

        Retorna IngestionResult con el resultado de la operación.
        """

        # ── 1. Construir entrada ──────────────────────────────
        # Clasificar dominio si no se especifica
        detected_domain = Domain.GENERAL
        if domain:
            try:
                detected_domain = Domain(domain)
            except ValueError:
                detected_domain = self._classifier.classify_domain(
                    f"{title} {content}"
                )
        else:
            detected_domain = self._classifier.classify_domain(
                f"{title} {content}"
            )

        # Clasificar tipo si no se especifica
        detected_type = KnowledgeType.FACT
        if knowledge_type:
            try:
                detected_type = KnowledgeType(knowledge_type)
            except ValueError:
                detected_type = self._classifier.classify_type(content)
        else:
            detected_type = self._classifier.classify_type(content)

        # Detectar subdominio si no se especifica
        detected_subdomain = subdomain or self._classifier.classify_subdomain(
            f"{title} {content}", detected_domain
        )

        entry = KnowledgeEntry(
            title          = title.strip(),
            content        = content.strip(),
            domain         = detected_domain,
            subdomain      = detected_subdomain,
            source         = source,
            source_url     = source_url,
            knowledge_type = detected_type,
            status         = KnowledgeStatus.CURRENT,
            confidence     = max(0.0, min(1.0, confidence)),
            tags           = tags or [],
            date_source    = date_source,
            metadata       = metadata or {},
        )

        # ── 2. Validar ────────────────────────────────────────
        errors = self._classifier.validate(entry)
        if errors:
            logger.warning("KnowledgeIngestion: validación falló: %s", errors)
            return IngestionResult(success=False, errors=errors)

        # ── 3. Deduplicar ─────────────────────────────────────
        if not allow_duplicate and self._classifier.is_duplicate(self._store, entry):
            logger.info(
                "KnowledgeIngestion: duplicado detectado (hash=%s)",
                entry.content_hash,
            )
            existing = self._store.get_by_hash(entry.content_hash)
            return IngestionResult(
                success=True,
                entry_id=existing.entry_id if existing else None,
                duplicate=True,
                action="skipped",
            )

        # ── 4. Guardar ────────────────────────────────────────
        entry_id = self._store.save(entry)
        logger.info(
            "KnowledgeIngestion: guardado | id=%s | domain=%s | type=%s",
            entry_id, detected_domain.value, detected_type.value,
        )

        return IngestionResult(
            success=True,
            entry_id=entry_id,
            duplicate=False,
            action="created",
        )

    def ingest_batch(
        self,
        items: list[dict],
        source: str = "batch",
    ) -> list[IngestionResult]:
        """Ingerir múltiples entradas."""
        results = []
        for item in items:
            result = self.ingest(
                title=item.get("title", "Sin título"),
                content=item.get("content", ""),
                domain=item.get("domain"),
                source=source,
                confidence=item.get("confidence", 0.8),
                tags=item.get("tags", []),
            )
            results.append(result)
        return results


# ── Source interfaces (futuras) ───────────────────────────────

class SourceInterface:
    """Interfaz base para fuentes de conocimiento."""
    name: str = "base"
    enabled: bool = False

    async def fetch(self, query: str) -> list[dict]:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False


class WebSourceInterface(SourceInterface):
    """Interfaz para ingestión desde web. NO implementada."""
    name    = "web"
    enabled = False

    async def fetch(self, query: str) -> list[dict]:
        raise NotImplementedError(
            "WebSourceInterface: no implementada. "
            "Prerequisito: API de búsqueda web."
        )


class FileSourceInterface(SourceInterface):
    """Interfaz para ingestión desde archivos. NO implementada."""
    name    = "file"
    enabled = False

    async def fetch(self, path: str) -> list[dict]:
        raise NotImplementedError(
            "FileSourceInterface: no implementada. "
            "Prerequisito: DocumentReaderTool."
        )


class UserSourceInterface(SourceInterface):
    """Fuente de usuario — siempre disponible."""
    name    = "user"
    enabled = True

    async def fetch(self, content: str) -> list[dict]:
        return [{"content": content, "source": "user"}]
