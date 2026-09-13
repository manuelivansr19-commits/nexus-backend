"""
NEXUS Î© â€” Knowledge Store v3.8.0
"""
import sqlite3
import json
from typing import List, Optional
from backend.config import logger
from backend.knowledge.models import KnowledgeEntry, KnowledgeStatus

class KnowledgeStore:
    def __init__(self, db_path: str = "nexus_knowledge.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id TEXT PRIMARY KEY,
                        domain TEXT,
                        subdomain TEXT,
                        title TEXT,
                        content TEXT,
                        status TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error("KnowledgeStore: error inicializando DB â€” %s", e)

    def search(self, query: str, domain: Optional[str] = None) -> List[KnowledgeEntry]:
        results = []
        try:
            cleaned = query.strip() if query else ""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if domain:
                    cursor.execute(
                        "SELECT id, domain, subdomain, title, content, status, created_at, updated_at, metadata FROM knowledge WHERE domain = ? AND (title LIKE ? OR content LIKE ?)",
                        (domain, f"%{cleaned}%", f"%{cleaned}%")
                    )
                else:
                    cursor.execute(
                        "SELECT id, domain, subdomain, title, content, status, created_at, updated_at, metadata FROM knowledge WHERE title LIKE ? OR content LIKE ?",
                        (f"%{cleaned}%", f"%{cleaned}%")
                    )
                for row in cursor.fetchall():
                    results.append(
                        KnowledgeEntry(
                            id=row[0],
                            domain=row[1],
                            subdomain=row[2],
                            title=row[3],
                            content=row[4],
                            status=KnowledgeStatus(row[5]) if row[5] else KnowledgeStatus.ACTIVE,
                            created_at=row[6],
                            updated_at=row[7],
                            metadata=json.loads(row[8]) if row[8] else {}
                        )
                    )
        except Exception as e:
            logger.error("KnowledgeStore: error en bÃºsqueda â€” %s", e)
        return results

    def save(self, entry: KnowledgeEntry) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO knowledge (id, domain, subdomain, title, content, status, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.domain.value if hasattr(entry.domain, 'value') else str(entry.domain),
                        entry.subdomain,
                        entry.title,
                        entry.content,
                        entry.status.value if hasattr(entry.status, 'value') else str(entry.status),
                        entry.created_at,
                        entry.updated_at,
                        json.dumps(entry.metadata)
                    )
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("KnowledgeStore: error guardando â€” %s", e)
            return False

