"""
NEXUS Ω — Knowledge Models v3.7.0

Tipos de datos del Knowledge Engine.
Ninguna clase aquí accede a base de datos ni LLMs.
Solo define los contratos de datos.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================
# ENUMS
# ============================================================

class KnowledgeType(str, Enum):
    FACT        = "fact"        # hecho verificado
    OBSERVATION = "observation" # observación registrada
    ESTIMATE    = "estimate"    # estimación con incertidumbre
    FORECAST    = "forecast"    # proyección futura
    SCENARIO    = "scenario"    # escenario posible
    HYPOTHESIS  = "hypothesis"  # hipótesis sin confirmar


class KnowledgeStatus(str, Enum):
    CURRENT         = "current"          # < AGING_DAYS días
    AGING           = "aging"            # entre AGING y OUTDATED
    OUTDATED        = "outdated"         # > OUTDATED_DAYS días
    REQUIRES_REVIEW = "requires_review"  # marcado para revisión


class Domain(str, Enum):
    ENGINEERING          = "engineering"
    ROBOTICS             = "robotics"
    AEROSPACE            = "aerospace"
    ENERGY               = "energy"
    AGRICULTURE          = "agriculture"
    MATERIALS            = "materials"
    ELECTRONICS          = "electronics"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    GENERAL              = "general"


# Subdominios por dominio
SUBDOMAINS: dict[Domain, list[str]] = {
    Domain.ENERGY: [
        "renewable", "non_renewable", "nuclear", "storage",
        "solar", "wind", "hydrogen", "grid",
    ],
    Domain.AGRICULTURE: [
        "hydroponics", "aeroponics", "aquaponics",
        "controlled_environment", "sensors", "automation",
    ],
    Domain.ROBOTICS: [
        "kinematics", "actuators", "sensors", "navigation",
        "manipulation", "swarm", "soft_robotics",
    ],
    Domain.ARTIFICIAL_INTELLIGENCE: [
        "llm", "computer_vision", "rl", "nlp",
        "embeddings", "fine_tuning", "inference",
    ],
    Domain.ENGINEERING: [
        "mechanical", "electrical", "software", "systems",
        "control", "embedded",
    ],
    Domain.ELECTRONICS: [
        "microcontrollers", "fpga", "pcb", "power",
        "sensors", "communication",
    ],
    Domain.AEROSPACE: [
        "propulsion", "structures", "guidance", "communications",
        "re_entry", "orbital",
    ],
    Domain.MATERIALS: [
        "composites", "metals", "polymers", "ceramics",
        "semiconductors", "nanomaterials",
    ],
}

# Días antes de considerar obsoleto — más corto para dominios que cambian rápido
DOMAIN_AGING_DAYS: dict[Domain, int] = {
    Domain.ARTIFICIAL_INTELLIGENCE: 14,
    Domain.ELECTRONICS:             30,
    Domain.ENERGY:                  45,
    Domain.ROBOTICS:                60,
    Domain.AEROSPACE:               90,
    Domain.ENGINEERING:             180,
    Domain.AGRICULTURE:             90,
    Domain.MATERIALS:               180,
    Domain.GENERAL:                 60,
}

DOMAIN_OUTDATED_DAYS: dict[Domain, int] = {
    Domain.ARTIFICIAL_INTELLIGENCE: 45,
    Domain.ELECTRONICS:             90,
    Domain.ENERGY:                  120,
    Domain.ROBOTICS:                180,
    Domain.AEROSPACE:               365,
    Domain.ENGINEERING:             730,
    Domain.AGRICULTURE:             180,
    Domain.MATERIALS:               365,
    Domain.GENERAL:                 180,
}


# ============================================================
# KNOWLEDGE ENTRY
# ============================================================

@dataclass
class KnowledgeEntry:
    """Unidad atómica de conocimiento."""

    title:          str
    content:        str
    domain:         Domain             = Domain.GENERAL
    subdomain:      str                = ""
    source:         str                = "user"        # origen del conocimiento
    source_url:     str                = ""
    knowledge_type: KnowledgeType      = KnowledgeType.FACT
    status:         KnowledgeStatus    = KnowledgeStatus.CURRENT
    confidence:     float              = 0.8           # 0.0 – 1.0
    tags:           list[str]          = field(default_factory=list)
    date_source:    str                = ""            # fecha del documento fuente
    metadata:       dict               = field(default_factory=dict)

    # Auto-generados
    entry_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    date_acquired:  float = field(default_factory=time.time)
    updated_at:     float = field(default_factory=time.time)
    content_hash:   str   = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA256 del contenido normalizado para deduplicación."""
        normalized = " ".join(self.content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def age_days(self) -> float:
        return (time.time() - self.date_acquired) / 86400

    def compute_status(self) -> KnowledgeStatus:
        """Calcular status basado en edad y dominio."""
        days = self.age_days()
        aging_threshold   = DOMAIN_AGING_DAYS.get(self.domain, 60)
        outdated_threshold = DOMAIN_OUTDATED_DAYS.get(self.domain, 180)

        if days > outdated_threshold:
            return KnowledgeStatus.OUTDATED
        if days > aging_threshold:
            return KnowledgeStatus.AGING
        return KnowledgeStatus.CURRENT

    def to_dict(self) -> dict:
        return {
            "entry_id":       self.entry_id,
            "title":          self.title,
            "content":        self.content,
            "domain":         self.domain.value,
            "subdomain":      self.subdomain,
            "source":         self.source,
            "source_url":     self.source_url,
            "knowledge_type": self.knowledge_type.value,
            "status":         self.status.value,
            "confidence":     self.confidence,
            "tags":           self.tags,
            "date_source":    self.date_source,
            "date_acquired":  self.date_acquired,
            "updated_at":     self.updated_at,
            "content_hash":   self.content_hash,
            "metadata":       self.metadata,
            "age_days":       round(self.age_days(), 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        return cls(
            entry_id       = d.get("entry_id", str(uuid.uuid4())[:12]),
            title          = d["title"],
            content        = d["content"],
            domain         = Domain(d.get("domain", "general")),
            subdomain      = d.get("subdomain", ""),
            source         = d.get("source", "user"),
            source_url     = d.get("source_url", ""),
            knowledge_type = KnowledgeType(d.get("knowledge_type", "fact")),
            status         = KnowledgeStatus(d.get("status", "current")),
            confidence     = float(d.get("confidence", 0.8)),
            tags           = d.get("tags", []),
            date_source    = d.get("date_source", ""),
            date_acquired  = float(d.get("date_acquired", time.time())),
            updated_at     = float(d.get("updated_at", time.time())),
            content_hash   = d.get("content_hash", ""),
            metadata       = d.get("metadata", {}),
        )

    def to_context_string(self) -> str:
        """Formato estructurado para contexto del LLM."""
        parts = [
            f"[{self.domain.value.upper()}/{self.subdomain or 'general'}]",
            f"Título: {self.title}",
            f"Tipo: {self.knowledge_type.value} | Confianza: {self.confidence:.0%}",
            f"Fuente: {self.source}" + (f" ({self.date_source})" if self.date_source else ""),
            f"Estado: {self.status.value}",
            f"",
            self.content,
        ]
        return "\n".join(parts)


# ============================================================
# KNOWLEDGE CONTEXT (resultado de retrieval)
# ============================================================

@dataclass
class KnowledgeContext:
    """Resultado estructurado de una búsqueda de conocimiento."""
    query:           str
    entries:         list[KnowledgeEntry]
    domains_covered: list[str]
    total_found:     int
    search_time_ms:  int = 0

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def to_prompt_string(self, max_entries: int = 5) -> str:
        """Formato estructurado para el LLM."""
        if self.is_empty():
            return ""
        entries = self.entries[:max_entries]
        sections = [
            f"[CONOCIMIENTO RELEVANTE — {self.total_found} entradas encontradas]",
            f"Consulta: {self.query}",
            f"Dominios: {', '.join(self.domains_covered)}",
            "",
        ]
        for i, entry in enumerate(entries, 1):
            sections.append(f"--- Entrada {i} ---")
            sections.append(entry.to_context_string())
            sections.append("")
        return "\n".join(sections)

    def has_outdated(self) -> bool:
        return any(e.status == KnowledgeStatus.OUTDATED for e in self.entries)

    def avg_confidence(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.confidence for e in self.entries) / len(self.entries)
