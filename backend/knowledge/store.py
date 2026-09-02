"""
NEXUS Ω — Knowledge Store v3.7.0

Almacenamiento persistente de conocimiento con SQLite + FTS5.
Base de datos separada de la memoria conversacional.

Características:
  - Búsqueda full-text (FTS5)
  - Deduplicación por content_hash
  - Índices por dominio, tipo, confianza
  - Actualización sin pérdida de metadata
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

from backend.config import logger
from backend.knowledge.models import (
    Domain, KnowledgeEntry, KnowledgeStatus, KnowledgeType,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    entry_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    domain          TEXT NOT NULL DEFAULT 'general',
    subdomain       TEXT DEFAULT '',
    source          TEXT DEFAULT 'user',
    source_url      TEXT DEFAULT '',
    knowledge_type  TEXT NOT NULL DEFAULT 'fact',
    status          TEXT NOT NULL DEFAULT 'current',
    confidence      REAL DEFAULT 0.8,
    content_hash    TEXT DEFAULT '',
    tags            TEXT DEFAULT '[]',
    date_source     TEXT DEFAULT '',
    date_acquired   REAL NOT NULL,
    updated_at      REAL NOT NULL,
    metadata        TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_k_domain     ON knowledge(domain);
CREATE INDEX IF NOT EXISTS idx_k_type       ON knowledge(knowledge_type);
CREATE INDEX IF NOT EXISTS idx_k_status     ON knowledge(status);
CREATE INDEX IF NOT EXISTS idx_k_confidence ON knowledge(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_k_acquired   ON knowledge(date_acquired DESC);
CREATE INDEX IF NOT EXISTS idx_k_hash       ON knowledge(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
    USING fts5(
        entry_id UNINDEXED,
        title,
        content,
        tags,
        tokenize='unicode61'
    );
"""


class KnowledgeStore:
    """
    Almacén SQLite para el Knowledge Engine.

    Interfaz compatible con futura migración a
    PostgreSQL/Supabase — mismos métodos, diferente backend.
    """

    def __init__(self, db_path: str = "nexus_knowledge.db") -> None:
        self._db_path = db_path
        self._conn    = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("KnowledgeStore: SQLite (%s)", db_path)

    # ── WRITE ─────────────────────────────────────────────────

    def save(self, entry: KnowledgeEntry) -> str:
        """Guardar o actualizar una entrada."""
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO knowledge
                (entry_id, title, content, domain, subdomain,
                 source, source_url, knowledge_type, status,
                 confidence, content_hash, tags, date_source,
                 date_acquired, updated_at, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                entry.entry_id, entry.title, entry.content,
                entry.domain.value, entry.subdomain,
                entry.source, entry.source_url,
                entry.knowledge_type.value, entry.status.value,
                entry.confidence, entry.content_hash,
                json.dumps(entry.tags), entry.date_source,
                entry.date_acquired, now,
                json.dumps(entry.metadata),
            ),
        )
        # FTS sync
        self._conn.execute(
            "DELETE FROM knowledge_fts WHERE entry_id = ?",
            (entry.entry_id,),
        )
        self._conn.execute(
            "INSERT INTO knowledge_fts(entry_id, title, content, tags) VALUES (?,?,?,?)",
            (entry.entry_id, entry.title, entry.content,
             " ".join(entry.tags)),
        )
        self._conn.commit()
        return entry.entry_id

    def update_status(self, entry_id: str, status: KnowledgeStatus) -> bool:
        cur = self._conn.execute(
            "UPDATE knowledge SET status=?, updated_at=? WHERE entry_id=?",
            (status.value, time.time(), entry_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, entry_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM knowledge WHERE entry_id=?", (entry_id,)
        )
        self._conn.execute(
            "DELETE FROM knowledge_fts WHERE entry_id=?", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── READ ──────────────────────────────────────────────────

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE entry_id=?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_by_hash(self, content_hash: str) -> Optional[KnowledgeEntry]:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE content_hash=? LIMIT 1",
            (content_hash,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def search_fts(
        self,
        query:          str,
        domain:         Optional[str] = None,
        subdomain:      Optional[str] = None,
        knowledge_type: Optional[str] = None,
        status:         Optional[str] = None,
        min_confidence: float         = 0.0,
        since_days:     Optional[float] = None,
        limit:          int           = 10,
    ) -> list[KnowledgeEntry]:
        """Búsqueda full-text con filtros opcionales."""
        try:
            base = """
                SELECT k.* FROM knowledge k
                JOIN knowledge_fts f ON k.entry_id = f.entry_id
                WHERE knowledge_fts MATCH ?
            """
            params: list = [query]
        except Exception:
            # FTS fallback a LIKE
            base   = "SELECT * FROM knowledge WHERE content LIKE ? OR title LIKE ?"
            params = [f"%{query}%", f"%{query}%"]

        filters, extra = self._build_filters(
            domain, subdomain, knowledge_type, status,
            min_confidence, since_days,
        )
        sql = base + filters + f" ORDER BY k.confidence DESC, k.date_acquired DESC LIMIT {limit}"
        try:
            rows = self._conn.execute(sql, params + extra).fetchall()
        except sqlite3.OperationalError:
            sql  = (
                f"SELECT * FROM knowledge WHERE (content LIKE ? OR title LIKE ?)"
                + filters.replace("k.", "")
                + f" ORDER BY confidence DESC LIMIT {limit}"
            )
            rows = self._conn.execute(
                sql, [f"%{query}%", f"%{query}%"] + extra
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def filter(
        self,
        domain:         Optional[str]   = None,
        subdomain:      Optional[str]   = None,
        knowledge_type: Optional[str]   = None,
        status:         Optional[str]   = None,
        min_confidence: float           = 0.0,
        since_days:     Optional[float] = None,
        limit:          int             = 20,
    ) -> list[KnowledgeEntry]:
        """Filtrar sin búsqueda de texto."""
        filters, params = self._build_filters(
            domain, subdomain, knowledge_type, status,
            min_confidence, since_days,
        )
        sql  = f"SELECT * FROM knowledge WHERE 1=1 {filters} ORDER BY confidence DESC LIMIT {limit}"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        by_domain = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT domain, COUNT(*) FROM knowledge GROUP BY domain"
            ).fetchall()
        }
        by_type = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT knowledge_type, COUNT(*) FROM knowledge GROUP BY knowledge_type"
            ).fetchall()
        }
        by_status = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT status, COUNT(*) FROM knowledge GROUP BY status"
            ).fetchall()
        }
        avg_conf = self._conn.execute(
            "SELECT AVG(confidence) FROM knowledge"
        ).fetchone()[0] or 0.0

        return {
            "total":      total,
            "by_domain":  by_domain,
            "by_type":    by_type,
            "by_status":  by_status,
            "avg_confidence": round(avg_conf, 3),
            "backend":    "sqlite",
            "db_path":    self._db_path,
        }

    def close(self) -> None:
        self._conn.close()

    # ── Private ───────────────────────────────────────────────

    def _build_filters(
        self,
        domain, subdomain, knowledge_type, status,
        min_confidence, since_days,
    ) -> tuple[str, list]:
        filters = ""
        params: list = []
        if domain:
            filters += " AND domain=?"
            params.append(domain)
        if subdomain:
            filters += " AND subdomain=?"
            params.append(subdomain)
        if knowledge_type:
            filters += " AND knowledge_type=?"
            params.append(knowledge_type)
        if status:
            filters += " AND status=?"
            params.append(status)
        if min_confidence > 0:
            filters += " AND confidence>=?"
            params.append(min_confidence)
        if since_days is not None:
            since = time.time() - since_days * 86400
            filters += " AND date_acquired>=?"
            params.append(since)
        return filters, params

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            entry_id       = row["entry_id"],
            title          = row["title"],
            content        = row["content"],
            domain         = Domain(row["domain"]),
            subdomain      = row["subdomain"] or "",
            source         = row["source"] or "user",
            source_url     = row["source_url"] or "",
            knowledge_type = KnowledgeType(row["knowledge_type"]),
            status         = KnowledgeStatus(row["status"]),
            confidence     = row["confidence"],
            content_hash   = row["content_hash"] or "",
            tags           = json.loads(row["tags"] or "[]"),
            date_source    = row["date_source"] or "",
            date_acquired  = row["date_acquired"],
            updated_at     = row["updated_at"],
            metadata       = json.loads(row["metadata"] or "{}"),
        )
