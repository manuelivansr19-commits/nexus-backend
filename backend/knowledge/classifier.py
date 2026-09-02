"""
NEXUS Ω — Knowledge Classifier v3.7.0

Clasifica y valida entradas de conocimiento.
Detecta dominios, tipos, y actualiza status según edad.

NO llama a LLMs directamente.
Clasificación determinística en primera instancia.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from backend.knowledge.models import (
    Domain, KnowledgeStatus, KnowledgeType,
    DOMAIN_AGING_DAYS, DOMAIN_OUTDATED_DAYS, SUBDOMAINS,
)


# ── Keyword mapping para clasificación de dominio ────────────

_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.ARTIFICIAL_INTELLIGENCE: [
        "ia", "ai", "llm", "modelo", "neural", "machine learning",
        "deep learning", "transformer", "embeddings", "gpt", "gemini",
        "ollama", "inferencia", "fine-tuning",
    ],
    Domain.ROBOTICS: [
        "robot", "robótica", "servo", "motor", "actuador",
        "cinemática", "manipulador", "navegación", "ros", "aura",
        "lidar", "slam",
    ],
    Domain.AEROSPACE: [
        "cohete", "satélite", "órbita", "propulsión", "aeronave",
        "reentrada", "misión espacial", "nasa", "esa",
    ],
    Domain.ENERGY: [
        "energía", "batería", "solar", "eólico", "nuclear",
        "hidrógeno", "almacenamiento", "red eléctrica", "lifepo4",
        "fotovoltaico", "turbina",
    ],
    Domain.AGRICULTURE: [
        "hidropónico", "aeropónico", "acuapónico", "cultivo",
        "nutrientes", "ph", "invernadero", "cosecha", "riego",
        "sensores agrícolas",
    ],
    Domain.MATERIALS: [
        "material", "compuesto", "metal", "polímero", "cerámica",
        "nanomaterial", "semiconductor", "fibra de carbono",
    ],
    Domain.ELECTRONICS: [
        "circuito", "microcontrolador", "arduino", "esp32", "raspberry",
        "pcb", "fpga", "sensor", "comunicación", "protocolo",
        "i2c", "spi", "uart",
    ],
    Domain.ENGINEERING: [
        "ingeniería", "diseño", "sistema", "control", "mecánica",
        "eléctrica", "software", "embebido", "protocolo",
    ],
}

# ── Keyword mapping para tipo de conocimiento ─────────────────

_TYPE_KEYWORDS: dict[KnowledgeType, list[str]] = {
    KnowledgeType.FACT: [
        "es", "son", "fue", "tiene", "consiste en", "se define",
        "compuesto por", "funciona mediante",
    ],
    KnowledgeType.ESTIMATE: [
        "aproximadamente", "estimado", "alrededor de", "cerca de",
        "se estima", "según estimaciones",
    ],
    KnowledgeType.FORECAST: [
        "se prevé", "se espera", "proyección", "para 2025", "para 2030",
        "en el futuro", "se proyecta", "crecerá",
    ],
    KnowledgeType.SCENARIO: [
        "si", "podría", "en caso de", "escenario", "hipotéticamente",
        "bajo condiciones",
    ],
    KnowledgeType.HYPOTHESIS: [
        "hipótesis", "se cree que", "posiblemente", "sugiere que",
        "podría indicar", "sin confirmar",
    ],
    KnowledgeType.OBSERVATION: [
        "se observó", "se detectó", "en la práctica", "en pruebas",
        "experimentalmente",
    ],
}


class KnowledgeClassifier:
    """
    Clasifica entradas de conocimiento.
    Actualiza status según antigüedad.
    """

    def classify_domain(self, text: str) -> Domain:
        """Detectar dominio por keywords."""
        lower  = text.lower()
        scores: dict[Domain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return Domain.GENERAL
        return max(scores, key=scores.__getitem__)

    def classify_type(self, text: str) -> KnowledgeType:
        """Detectar tipo de conocimiento por keywords."""
        lower  = text.lower()
        scores: dict[KnowledgeType, int] = {}
        for ktype, keywords in _TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[ktype] = score
        if not scores:
            return KnowledgeType.FACT
        return max(scores, key=scores.__getitem__)

    def classify_subdomain(self, text: str, domain: Domain) -> str:
        """Detectar subdominio dentro del dominio dado."""
        subs = SUBDOMAINS.get(domain, [])
        if not subs:
            return ""
        lower = text.lower()
        for sub in subs:
            if sub.replace("_", " ") in lower or sub in lower:
                return sub
        return ""

    def compute_status(
        self,
        domain: Domain,
        date_acquired: float,
        manual_override: Optional[KnowledgeStatus] = None,
    ) -> KnowledgeStatus:
        """Calcular status según edad y dominio."""
        if manual_override == KnowledgeStatus.REQUIRES_REVIEW:
            return KnowledgeStatus.REQUIRES_REVIEW
        days = (time.time() - date_acquired) / 86400
        aging    = DOMAIN_AGING_DAYS.get(domain, 60)
        outdated = DOMAIN_OUTDATED_DAYS.get(domain, 180)
        if days > outdated:
            return KnowledgeStatus.OUTDATED
        if days > aging:
            return KnowledgeStatus.AGING
        return KnowledgeStatus.CURRENT

    def validate(self, entry) -> list[str]:
        """
        Validar una entrada antes de guardar.
        Retorna lista de errores (vacía = OK).
        """
        errors: list[str] = []
        if not entry.title or not entry.title.strip():
            errors.append("title: no puede estar vacío.")
        if not entry.content or not entry.content.strip():
            errors.append("content: no puede estar vacío.")
        if len(entry.content.strip()) < 10:
            errors.append("content: demasiado corto (mínimo 10 caracteres).")
        if not (0.0 <= entry.confidence <= 1.0):
            errors.append("confidence: debe estar entre 0.0 y 1.0.")
        # No secretos en contenido
        _forbidden = ["api_key", "password", "secret", "token", "bearer"]
        content_lower = entry.content.lower()
        for f in _forbidden:
            if f in content_lower:
                errors.append(f"content: contiene término prohibido '{f}'.")
        return errors

    def is_duplicate(self, store, entry) -> bool:
        """Verificar si ya existe una entrada con el mismo hash."""
        existing = store.get_by_hash(entry.content_hash)
        return existing is not None

    def refresh_statuses(self, store) -> int:
        """
        Actualizar status de todas las entradas según su edad.
        Retorna cantidad de entradas actualizadas.
        """
        from backend.knowledge.models import KnowledgeStatus
        entries = store.filter(limit=10000)
        updated = 0
        for entry in entries:
            new_status = self.compute_status(entry.domain, entry.date_acquired)
            if new_status != entry.status:
                store.update_status(entry.entry_id, new_status)
                updated += 1
        return updated
