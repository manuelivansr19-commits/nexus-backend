"""NEXUS Omega -- Task State Manager v3.8.0"""
import json, sqlite3, threading, time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from backend.config import logger

class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"

@dataclass
class Task:
    task_id:     str
    description: str
    intent:      str          = ""
    domain:      str          = "general"
    status:      TaskStatus   = TaskStatus.PENDING
    created_at:  float        = field(default_factory=time.time)
    updated_at:  float        = field(default_factory=time.time)
    result:      Optional[str] = None
    error:       Optional[str] = None
    metadata:    dict          = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id":     self.task_id,
            "description": self.description,
            "intent":      self.intent,
            "domain":      self.domain,
            "status":      self.status.value,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "result":      self.result,
            "error":       self.error,
            "age_minutes": round((time.time() - self.created_at) / 60, 1),
        }

_KEEP = object()                 # "no tocar este campo" en transition()
_META_LOCK = threading.RLock()   # lectura-modificacion-escritura atomica de metadata


class TaskStateManager:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id     TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        intent      TEXT DEFAULT '',
        domain      TEXT DEFAULT 'general',
        status      TEXT DEFAULT 'pending',
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        result      TEXT,
        error       TEXT,
        metadata    TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_status  ON tasks(status);
    CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
    """

    def __init__(self, db_path="nexus_tasks.db"):
        self._db_path = db_path
        self._conn    = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def create(self, description, intent="", domain="general") -> Task:
        task = Task(task_id=str(uuid.uuid4())[:12], description=description[:500],
                    intent=intent, domain=domain)
        self._conn.execute(
            "INSERT INTO tasks (task_id,description,intent,domain,status,created_at,updated_at,metadata) VALUES (?,?,?,?,?,?,?,?)",
            (task.task_id, task.description, task.intent, task.domain,
             task.status.value, task.created_at, task.updated_at, "{}"))
        self._conn.commit()
        logger.info("TaskState: creada %s | %s", task.task_id, intent)
        return task

    def get(self, task_id) -> Optional[Task]:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def update(self, task_id, status: TaskStatus, result=None, error=None) -> bool:
        cur = self._conn.execute(
            "UPDATE tasks SET status=?,result=?,error=?,updated_at=? WHERE task_id=?",
            (status.value, result, error, time.time(), task_id))
        self._conn.commit()
        return cur.rowcount > 0

    def list_pending(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('pending','in_progress') ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        return [self._row(r) for r in rows]

    def list_recently_completed(self, within_seconds: float = 300, limit: int = 5) -> list:
        """
        Tareas con status='completed' cuya última actualización
        (updated_at, no created_at) cae dentro de los últimos
        `within_seconds`. Ordenadas de más a menos reciente.

        Uso previsto (v3.8 Bloque 4): permitir que NexusCore reactive
        una tarea recién terminada cuando el usuario continúa
        explícitamente sobre ella, sin reutilizar tareas completadas
        hace mucho tiempo ni tareas CANCELLED/FAILED (excluidas por el
        filtro de status, igual que en list_pending()).
        """
        cutoff = time.time() - within_seconds
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'completed' AND updated_at >= ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [self._row(r) for r in rows]

    def list_recent(self, limit=10) -> list:
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def find_by_description(self, query) -> Optional[Task]:
        words = [w for w in query.lower().split() if len(w) > 3][:4]
        if not words:
            return None
        like  = "%" + "%".join(words[:2]) + "%"
        row   = self._conn.execute(
            "SELECT * FROM tasks WHERE description LIKE ? ORDER BY created_at DESC LIMIT 1",
            (like,)).fetchone()
        return self._row(row) if row else None

    def stats(self) -> dict:
        total    = self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        by_status = {r[0]: r[1] for r in self._conn.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()}
        return {"total": total, "by_status": by_status}

    # ------------------------------------------------------------------
    # v3.8 Bloque 4B: metadata y transiciones protegidas (metodos nuevos)
    # ------------------------------------------------------------------
    def get_metadata(self, task_id) -> dict:
        """metadata de la tarea como dict; {} si no existe, esta vacia o es invalida."""
        row = self._conn.execute(
            "SELECT metadata FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row["metadata"] or "{}")
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def merge_metadata(self, task_id, patch=None, *, mutator=None, allowed_status=None):
        """
        Actualiza metadata de forma atomica (no cambia status, result ni updated_at).
        - patch: dict que se fusiona (dict.update).
        - mutator: funcion(dict) -> dict | None; devolver None aborta sin escribir.
        - allowed_status: iterable de TaskStatus; si el estado actual no esta, no escribe.
        Devuelve el metadata escrito, o None si no se escribio.
        """
        with _META_LOCK:
            row = self._conn.execute(
                "SELECT status, metadata FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row:
                return None
            if allowed_status is not None and row["status"] not in {
                    getattr(s, "value", s) for s in allowed_status}:
                return None
            try:
                data = json.loads(row["metadata"] or "{}")
            except (ValueError, TypeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            if mutator is not None:
                data = mutator(dict(data))
                if data is None:
                    return None
            if patch:
                data.update(patch)
            self._conn.execute("UPDATE tasks SET metadata=? WHERE task_id=?",
                               (json.dumps(data, ensure_ascii=False), task_id))
            self._conn.commit()
            return data

    def transition(self, task_id, new_status: TaskStatus, allowed_from,
                   result=_KEEP, error=_KEEP) -> bool:
        """
        Cambia el estado SOLO si el actual esta en allowed_from (UPDATE condicional).
        Asi una tarea CANCELLED/COMPLETED/FAILED no se reactiva ni se pisa por accidente.
        result/error: si no se indican, se conservan. Devuelve True si cambio una fila.
        """
        allowed = [getattr(s, "value", s) for s in allowed_from]
        if not allowed:
            return False
        sets, params = ["status=?", "updated_at=?"], [new_status.value, time.time()]
        if result is not _KEEP:
            sets.append("result=?")
            params.append(result)
        if error is not _KEEP:
            sets.append("error=?")
            params.append(error)
        params.append(task_id)
        params.extend(allowed)
        marks = ",".join("?" * len(allowed))
        cur = self._conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE task_id=? AND status IN ({marks})",
            params)
        self._conn.commit()
        return cur.rowcount > 0

    def close(self):
        self._conn.close()

    def _row(self, row) -> Task:
        return Task(
            task_id=row["task_id"], description=row["description"],
            intent=row["intent"] or "", domain=row["domain"] or "general",
            status=TaskStatus(row["status"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
            result=row["result"], error=row["error"],
            metadata=json.loads(row["metadata"] or "{}"))
