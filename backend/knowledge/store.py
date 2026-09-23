"""
NEXUS Ω — Knowledge Store v3.8.1
"""
import sqlite3
import json
from typing import List, Optional
from backend.config import logger
from backend.knowledge.models import (
    KnowledgeEntry, KnowledgeStatus, Domain, KnowledgeType,
)


class KnowledgeStore:
    def __init__(self, db_path: str = "nexus_knowledge.db") -> None:
        self.db_path = db_path
        self._init_db()

    # ── internal ──────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row_to_entry(self, row) -> KnowledgeEntry:
        (
            entry_id, domain, subdomain, title, content, source, source_url,
            knowledge_type, status, confidence, tags, date_source,
            date_acquired, updated_at, content_hash, metadata,
        ) = row
        return KnowledgeEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            domain=Domain(domain) if domain else Domain.GENERAL,
            subdomain=subdomain or "",
            source=source or "user",
            source_url=source_url or "",
            knowledge_type=KnowledgeType(knowledge_type) if knowledge_type else KnowledgeType.FACT,
            status=KnowledgeStatus(status) if status else KnowledgeStatus.CURRENT,
            confidence=confidence if confidence is not None else 0.8,
            tags=json.loads(tags) if tags else [],
            date_source=date_source or "",
            date_acquired=date_acquired if date_acquired is not None else 0.0,
            updated_at=updated_at if updated_at is not None else 0.0,
            content_hash=content_hash or "",
            metadata=json.loads(metadata) if metadata else {},
        )

    def close(self) -> None:
        """
        Compatibilidad con callers que cierran el store explícitamente.
        No hay conexión persistente que cerrar: cada método abre y cierra
        su propia conexión SQLite, así que esto es un no-op seguro.
        """
        pass

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    entry_id        TEXT PRIMARY KEY,
                    domain          TEXT,
                    subdomain       TEXT,
                    title           TEXT,
                    content         TEXT,
                    source          TEXT,
                    source_url      TEXT,
                    knowledge_type  TEXT,
                    status          TEXT,
                    confidence      REAL,
                    tags            TEXT,
                    date_source     TEXT,
                    date_acquired   REAL,
                    updated_at      REAL,
                    content_hash    TEXT,
                    metadata        TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_hash "
                "ON knowledge(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_domain "
                "ON knowledge(domain)"
            )
            conn.commit()
        except Exception as e:
            logger.error("KnowledgeStore: error inicializando DB — %s", e)
        finally:
            conn.close()

    # ── write ─────────────────────────────────────────────────

    def save(self, entry: KnowledgeEntry) -> Optional[str]:
        """Guardar (o reemplazar) una entrada. Retorna el entry_id o None si falla."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO knowledge
                   (entry_id, domain, subdomain, title, content, source,
                    source_url, knowledge_type, status, confidence, tags,
                    date_source, date_acquired, updated_at, content_hash, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.entry_id,
                    entry.domain.value if hasattr(entry.domain, "value") else str(entry.domain),
                    entry.subdomain,
                    entry.title,
                    entry.content,
                    entry.source,
                    entry.source_url,
                    entry.knowledge_type.value if hasattr(entry.knowledge_type, "value") else str(entry.knowledge_type),
                    entry.status.value if hasattr(entry.status, "value") else str(entry.status),
                    entry.confidence,
                    json.dumps(entry.tags),
                    entry.date_source,
                    entry.date_acquired,
                    entry.updated_at,
                    entry.content_hash,
                    json.dumps(entry.metadata),
                ),
            )
            conn.commit()
            return entry.entry_id
        except Exception as e:
            logger.error("KnowledgeStore: error guardando — %s", e)
            return None
        finally:
            conn.close()

    def delete(self, entry_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM knowledge WHERE entry_id = ?", (entry_id,))
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.error("KnowledgeStore: error eliminando — %s", e)
            return False
        finally:
            conn.close()

    def update_status(self, entry_id: str, status: KnowledgeStatus) -> bool:
        conn = self._connect()
        try:
            status_val = status.value if hasattr(status, "value") else str(status)
            cur = conn.execute(
                "UPDATE knowledge SET status = ? WHERE entry_id = ?",
                (status_val, entry_id),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.error("KnowledgeStore: error actualizando status — %s", e)
            return False
        finally:
            conn.close()

    # ── read ──────────────────────────────────────────────────

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT entry_id, domain, subdomain, title, content, source, "
                "source_url, knowledge_type, status, confidence, tags, "
                "date_source, date_acquired, updated_at, content_hash, metadata "
                "FROM knowledge WHERE entry_id = ?",
                (entry_id,),
            )
            row = cursor.fetchone()
            return self._row_to_entry(row) if row else None
        except Exception as e:
            logger.error("KnowledgeStore: error en get — %s", e)
            return None
        finally:
            conn.close()

    def get_by_hash(self, content_hash: str) -> Optional[KnowledgeEntry]:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT entry_id, domain, subdomain, title, content, source, "
                "source_url, knowledge_type, status, confidence, tags, "
                "date_source, date_acquired, updated_at, content_hash, metadata "
                "FROM knowledge WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            row = cursor.fetchone()
            return self._row_to_entry(row) if row else None
        except Exception as e:
            logger.error("KnowledgeStore: error en get_by_hash — %s", e)
            return None
        finally:
            conn.close()

    def search(self, query: str, domain: Optional[str] = None) -> List[KnowledgeEntry]:
        """Búsqueda simple por texto (compatibilidad hacia atrás)."""
        return self.search_fts(query=query, domain=domain, limit=50)

    def search_fts(
        self,
        query: str = "",
        domain: Optional[str] = None,
        subdomain: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        status: Optional[str] = None,
        min_confidence: float = 0.0,
        since_days: Optional[float] = None,
        limit: int = 10,
    ) -> List[KnowledgeEntry]:
        """Búsqueda por texto + filtros (LIKE-based; FTS5 real es mejora futura)."""
        results: List[KnowledgeEntry] = []
        conn = self._connect()
        try:
            cleaned = query.strip() if query else ""
            sql = (
                "SELECT entry_id, domain, subdomain, title, content, source, "
                "source_url, knowledge_type, status, confidence, tags, "
                "date_source, date_acquired, updated_at, content_hash, metadata "
                "FROM knowledge WHERE 1=1"
            )
            params: list = []

            if cleaned:
                # Búsqueda por palabras: basta con que aparezca alguna
                # de las palabras de la consulta (OR entre términos y
                # entre campos), como un buscador normal. Evita tanto
                # el problema de frase-exacta-contigua (que casi nunca
                # ocurre en texto real) como el de exigir TODAS las
                # palabras (que falla cuando el término solo está
                # implícito en el dominio, no en el texto literal).
                words = [w for w in cleaned.split() if w]
                if words:
                    or_clauses = []
                    for word in words:
                        or_clauses.append("title LIKE ? OR content LIKE ?")
                        params += [f"%{word}%", f"%{word}%"]
                    sql += " AND (" + " OR ".join(or_clauses) + ")"
            if domain:
                sql += " AND domain = ?"
                params.append(domain)
            if subdomain:
                sql += " AND subdomain = ?"
                params.append(subdomain)
            if knowledge_type:
                sql += " AND knowledge_type = ?"
                params.append(knowledge_type)
            if status:
                sql += " AND status = ?"
                params.append(status)
            if min_confidence:
                sql += " AND confidence >= ?"
                params.append(min_confidence)
            if since_days is not None:
                import time
                cutoff = time.time() - (since_days * 86400)
                sql += " AND date_acquired >= ?"
                params.append(cutoff)

            sql += " ORDER BY date_acquired DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                results.append(self._row_to_entry(row))
        except Exception as e:
            logger.error("KnowledgeStore: error en búsqueda — %s", e)
        finally:
            conn.close()
        return results

    def filter(
        self,
        domain: Optional[str] = None,
        subdomain: Optional[str] = None,
        since_days: Optional[float] = None,
        limit: int = 100,
    ) -> List[KnowledgeEntry]:
        """Recuperar entradas por filtros simples, sin texto de búsqueda."""
        return self.search_fts(
            query="",
            domain=domain,
            subdomain=subdomain,
            since_days=since_days,
            limit=limit,
        )

    # ── stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = self._connect()
        result = {
            "total_entries": 0,
            "by_domain": {},
            "by_status": {},
        }
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
            result["total_entries"] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT domain, COUNT(*) FROM knowledge GROUP BY domain"
            )
            result["by_domain"] = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM knowledge GROUP BY status"
            )
            result["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error("KnowledgeStore: error en stats — %s", e)
        finally:
            conn.close()
        return result
