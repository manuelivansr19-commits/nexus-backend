"""
NEXUS Ω — Memory System.

Tipos de memoria:
  ConversationMemory → historial de mensajes del chat actual
  FactMemory         → hechos y datos persistentes
  ProjectMemory      → contexto de proyectos y objetivos

Backends:
  RAMMemoryStore     → desarrollo (volátil)
  SQLiteMemoryStore  → producción local / AURA (persistente)

La interfaz MemoryStore permite cambiar backend sin tocar el Core.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class MemoryType(str, Enum):
    CONVERSATION = "conversation"   # mensajes user/assistant
    FACT         = "fact"           # hechos persistentes
    PROJECT      = "project"        # contexto de proyectos
    WORKING      = "working"        # contexto inmediato (corta vida)
    EPISODIC     = "episodic"       # eventos pasados
    SEMANTIC     = "semantic"       # conocimiento general


@dataclass
class MemoryEntry:
    content:     str
    memory_type: MemoryType       = MemoryType.WORKING
    entry_id:    str              = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp:   float            = field(default_factory=time.time)
    tags:        list[str]        = field(default_factory=list)
    importance:  float            = 0.5
    metadata:    dict             = field(default_factory=dict)
    role:        str              = ""   # "user" | "assistant" | ""

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        return {
            "entry_id":    self.entry_id,
            "content":     self.content,
            "memory_type": self.memory_type.value,
            "timestamp":   self.timestamp,
            "tags":        self.tags,
            "importance":  self.importance,
            "metadata":    self.metadata,
            "role":        self.role,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            entry_id    = d.get("entry_id", str(uuid.uuid4())[:12]),
            content     = d["content"],
            memory_type = MemoryType(d.get("memory_type", "working")),
            timestamp   = d.get("timestamp", time.time()),
            tags        = d.get("tags", []),
            importance  = d.get("importance", 0.5),
            metadata    = d.get("metadata", {}),
            role        = d.get("role", ""),
        )


# ============================================================
# ABSTRACT STORE INTERFACE
# ============================================================

class MemoryStore:
    """Interfaz base. Override para cambiar backend."""

    def save(self, entry: MemoryEntry) -> str:
        raise NotImplementedError

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        raise NotImplementedError

    def recent(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        raise NotImplementedError

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        raise NotImplementedError

    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


# ============================================================
# RAM STORE (desarrollo, volátil)
# ============================================================

class RAMMemoryStore(MemoryStore):
    """Almacenamiento en RAM. Rápido, pero se pierde al reiniciar."""

    def __init__(
        self,
        max_conversation: int = 100,
        max_fact: int = 1000,
    ) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._limits = {
            MemoryType.CONVERSATION: max_conversation,
            MemoryType.WORKING:      50,
            MemoryType.FACT:         max_fact,
            MemoryType.PROJECT:      200,
            MemoryType.EPISODIC:     500,
            MemoryType.SEMANTIC:     5000,
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
        q = query.lower()
        results = [
            e for e in self._store.values()
            if q in e.content.lower()
            or any(q in t.lower() for t in e.tags)
        ]
        results.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
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
        return {"total": len(self._store), "by_type": counts, "backend": "ram"}

    def _evict(self, memory_type: MemoryType) -> None:
        limit   = self._limits.get(memory_type, 1000)
        entries = [e for e in self._store.values() if e.memory_type == memory_type]
        if len(entries) >= limit:
            entries.sort(key=lambda e: (e.importance, e.timestamp))
            del self._store[entries[0].entry_id]


# ============================================================
# SQLITE STORE (persistente)
# ============================================================

class SQLiteMemoryStore(MemoryStore):
    """
    Almacenamiento SQLite. Persiste entre reinicios.

    Diseñado para:
    - Desarrollo local
    - Hardware físico de AURA

    Adapter-compatible: implementa la misma interfaz que RAMMemoryStore.
    Para migrar a PostgreSQL/Supabase: crear PostgreSQLMemoryStore
    con la misma interfaz.

    NOTA: Render free tier tiene filesystem efímero.
    Para producción en Render usar PostgreSQL/Supabase.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        entry_id    TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        tags        TEXT DEFAULT '[]',
        importance  REAL DEFAULT 0.5,
        metadata    TEXT DEFAULT '{}',
        role        TEXT DEFAULT '',
        created_at  REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_created_at  ON memories(created_at DESC);
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(entry_id UNINDEXED, content, tokenize='unicode61');
    """

    def __init__(self, db_path: str = "nexus_memory.db") -> None:
        self._db_path = db_path
        self._conn    = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def save(self, entry: MemoryEntry) -> str:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (entry_id, content, memory_type, tags, importance, metadata, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.content,
                entry.memory_type.value,
                json.dumps(entry.tags),
                entry.importance,
                json.dumps(entry.metadata),
                entry.role,
                entry.timestamp,
            ),
        )
        # FTS sync
        self._conn.execute(
            "INSERT OR REPLACE INTO memories_fts(entry_id, content) VALUES (?, ?)",
            (entry.entry_id, entry.content),
        )
        self._conn.commit()
        return entry.entry_id

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def recent(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        if memory_type:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?",
                (memory_type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        # FTS search first
        try:
            rows = self._conn.execute(
                """
                SELECT m.* FROM memories m
                JOIN memories_fts f ON m.entry_id = f.entry_id
                WHERE memories_fts MATCH ?
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS fallback: LIKE search
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, entry_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE entry_id = ?", (entry_id,)
        )
        self._conn.execute(
            "DELETE FROM memories_fts WHERE entry_id = ?", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        if memory_type is None:
            cur = self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM memories_fts")
        else:
            ids = [
                r[0] for r in self._conn.execute(
                    "SELECT entry_id FROM memories WHERE memory_type = ?",
                    (memory_type.value,),
                ).fetchall()
            ]
            cur = self._conn.execute(
                "DELETE FROM memories WHERE memory_type = ?",
                (memory_type.value,),
            )
            for i in ids:
                self._conn.execute(
                    "DELETE FROM memories_fts WHERE entry_id = ?", (i,)
                )
        self._conn.commit()
        return cur.rowcount

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        rows  = self._conn.execute(
            "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
        ).fetchall()
        return {
            "total":   total,
            "by_type": {r[0]: r[1] for r in rows},
            "backend": "sqlite",
            "db_path": self._db_path,
        }

    def close(self) -> None:
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            entry_id    = row["entry_id"],
            content     = row["content"],
            memory_type = MemoryType(row["memory_type"]),
            timestamp   = row["created_at"],
            tags        = json.loads(row["tags"] or "[]"),
            importance  = row["importance"],
            metadata    = json.loads(row["metadata"] or "{}"),
            role        = row["role"] or "",
        )


# ============================================================
# SPECIALIZED MEMORY FACADES
# ============================================================

class ConversationMemory:
    """Memoria de conversación: guarda mensajes user/assistant."""

    def __init__(self, store: MemoryStore, max_turns: int = 20) -> None:
        self._store    = store
        self._max_turns = max_turns

    def add_user(self, content: str) -> str:
        return self._store.save(MemoryEntry(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            role="user",
            importance=0.6,
        ))

    def add_assistant(self, content: str, importance: float = 0.7) -> str:
        return self._store.save(MemoryEntry(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            role="assistant",
            importance=importance,
        ))

    def recent_turns(self, limit: int = 10) -> list[MemoryEntry]:
        entries = self._store.recent(MemoryType.CONVERSATION, limit=limit * 2)
        entries.sort(key=lambda e: e.timestamp)
        return entries[-limit:]

    def to_history_list(self, limit: int = 10) -> list[dict]:
        """Formato compatible con providers (role + content)."""
        return [
            {"role": e.role, "content": e.content}
            for e in self.recent_turns(limit)
            if e.role in ("user", "assistant")
        ]

    def clear(self) -> int:
        return self._store.clear(MemoryType.CONVERSATION)


class FactMemory:
    """Memoria de hechos: datos persistentes y conocimiento."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def save_fact(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.8,
    ) -> str:
        return self._store.save(MemoryEntry(
            content=content,
            memory_type=MemoryType.FACT,
            tags=tags or [],
            importance=importance,
        ))

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        results = self._store.search(query, limit * 2)
        return [e for e in results if e.memory_type == MemoryType.FACT][:limit]

    def recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._store.recent(MemoryType.FACT, limit)


class ProjectMemory:
    """Memoria de proyectos: contexto y objetivos de largo plazo."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def save(self, content: str, project: str = "", tags: Optional[list[str]] = None) -> str:
        meta = {"project": project} if project else {}
        return self._store.save(MemoryEntry(
            content=content,
            memory_type=MemoryType.PROJECT,
            tags=tags or ([project] if project else []),
            importance=0.9,
            metadata=meta,
        ))

    def get_context(self, project: str = "", limit: int = 5) -> str:
        entries = self._store.recent(MemoryType.PROJECT, limit=limit)
        if project:
            entries = [e for e in entries if project in e.tags or e.metadata.get("project") == project]
        if not entries:
            return ""
        return "\n".join(f"[PROYECTO] {e.content}" for e in entries)


# ============================================================
# UNIFIED MEMORY INTERFACE
# ============================================================

class Memory:
    """
    Fachada unificada. El Core usa esta clase.
    No importa el backend concreto.
    """

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self._store      = store or RAMMemoryStore()
        self.conversation = ConversationMemory(self._store)
        self.facts        = FactMemory(self._store)
        self.projects     = ProjectMemory(self._store)

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.WORKING,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        role: str = "",
    ) -> str:
        return self._store.save(MemoryEntry(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            importance=importance,
            role=role,
        ))

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        return self._store.search(query, limit)

    def working_context(self, limit: int = 8) -> str:
        entries = self._store.recent(MemoryType.WORKING, limit)
        if not entries:
            return ""
        return "\n".join(f"• {e.content}" for e in reversed(entries))

    def stats(self) -> dict:
        return self._store.stats()

    def clear_working(self) -> int:
        return self._store.clear(MemoryType.WORKING)
