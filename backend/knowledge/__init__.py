"""NEXUS Ω — Knowledge Engine package v3.7.0"""

from backend.knowledge.models import (
    KnowledgeEntry, KnowledgeType, KnowledgeStatus,
    Domain, KnowledgeContext, SUBDOMAINS,
    DOMAIN_AGING_DAYS, DOMAIN_OUTDATED_DAYS,
)
from backend.knowledge.store import KnowledgeStore
from backend.knowledge.classifier import KnowledgeClassifier
from backend.knowledge.retrieval import KnowledgeRetriever
from backend.knowledge.ingestion import KnowledgeIngestion, IngestionResult
from backend.knowledge.engine import KnowledgeEngine

__all__ = [
    "KnowledgeEntry", "KnowledgeType", "KnowledgeStatus",
    "Domain", "KnowledgeContext", "SUBDOMAINS",
    "KnowledgeStore", "KnowledgeClassifier",
    "KnowledgeRetriever", "KnowledgeIngestion", "IngestionResult",
    "KnowledgeEngine",
]
