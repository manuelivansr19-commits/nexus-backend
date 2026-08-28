"""
NEXUS Ω — Memory.

Abstracción de memoria del sistema AURA.

Tipos:
  working  → contexto inmediato de la conversación actual
  episodic → eventos y experiencias pasadas
  semantic → conocimiento factual persistente

El backend concreto (RAM, SQLite, vector store) se elige
vía MemoryStore. Por ahora: implementación en memoria.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MemoryType(str, Enum):
    WORKING  = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    content:     str
    memory_type: MemoryType = MemoryType.WORKING
    entry_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   float = field(default_factory=time.time)
    tags:        list[str] = field(default_factory=list)
    importance:  float = 0.5        # 0.0 – 1.0
    metadata:    dict  = field(default_factory=dict)

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class MemoryStore:
    """
    Almacén en memoria (RAM).

    Interfaz preparada para futura migración a:
    - SQLite (persistencia local en AURA)
    - PostgreSQL (producción)
    - Vector store (búsqueda semántica)
    """

    def __init__(self, max_working: int = 50, max_episodic: int = 500) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._max = {
            MemoryType.WORKING:  max_working,
            MemoryType.EPISODIC: max_episodic,
            MemoryType.SEMANTIC: 10_000,
        }

    def save(self, entry: MemoryEntry) -> str:
        self._evict(entry.memory_type)
        self._store[entry.entry_id] = entry
        return entry.entry_id

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._store.get(entry_id)

    def recent(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        entries = list(self._store.values())
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Búsqueda simple por substring. Override para vector search."""
        q = query.lower()
        results = [
            e for e in self._store.values()
            if q in e.content.lower() or any(q in t for t in e.tags)
        ]
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]

    def delete(self, entry_id: str) -> bool:
        return bool(self._store.pop(entry_id, None))

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        if memory_type is None:
            count = len(self._store)
            self._store.clear()
            return count
        ids = [k for k, v in self._store.items() if v.memory_type == memory_type]
        for i in ids:
            del self._store[i]
        return len(ids)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for e in self._store.values():
            counts[e.memory_type.value] = counts.get(e.memory_type.value, 0) + 1
        return {"total": len(self._store), "by_type": counts}

    def _evict(self, memory_type: MemoryType) -> None:
        entries = [e for e in self._store.values() if e.memory_type == memory_type]
        if len(entries) >= self._max[memory_type]:
            entries.sort(key=lambda e: (e.importance, e.timestamp))
            del self._store[entries[0].entry_id]


class Memory:
    """Fachada de alto nivel sobre MemoryStore."""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self._store = store or MemoryStore()

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.WORKING,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> str:
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            importance=importance,
        )
        return self._store.save(entry)

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        return self._store.search(query, limit)

    def working_context(self, limit: int = 10) -> str:
        """Contexto de trabajo para el LLM."""
        entries = self._store.recent(MemoryType.WORKING, limit)
        if not entries:
            return "Sin contexto de trabajo."
        return "\n".join(f"- {e.content}" for e in reversed(entries))

    def stats(self) -> dict:
        return self._store.stats()
