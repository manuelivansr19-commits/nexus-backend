# -*- coding: utf-8 -*-
"""
NEXUS Omega -- Plan checkpoint (v3.8 Bloque 4B: continuidad real).

Guarda el plan y el resultado de cada paso en la columna `metadata` de la tarea
(SQLite, sin cambiar el esquema) para que "continua" pueda retomar el plan desde
el primer paso pendiente en vez de reiniciarlo o de devolver texto guardado.

Reglas de seguridad (ver diseno aprobado):
  - Un paso DONE nunca se vuelve a ejecutar: se reutiliza su resultado guardado.
  - Un paso de texto, o de una herramienta de solo lectura (riesgo LOW), que quedo
    RUNNING o FAILED se puede repetir sin efectos.
  - Un paso RUNNING con herramienta de riesgo > LOW (o no reconocida) NO se repite
    solo: el plan se pausa y exige confirmacion explicita del usuario.
  - Un checkpoint corrupto o invalido nunca se usa: se reinicia con aviso.
  - Una tarea que ya no esta PENDING/IN_PROGRESS (p.ej. CANCELLED) nunca se escribe.

Este modulo NO conoce proveedores ni NexusCore; solo planner.py y task_state.py.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from backend.config import logger
from backend.core.planner import Plan, PlanStep, StepStatus
from backend.core.task_state import TaskStatus

PLAN_KEY = "plan_checkpoint"
CHECKPOINT_VERSION = 1
MAX_STEP_RESULT_CHARS = 12000      # tope por resultado de paso guardado
MAX_STEPS = 50                     # un checkpoint con mas pasos se considera invalido
RUNNING_STALE_SECONDS = 300        # un "running" sin latido mas viejo que esto = interrumpido
RESUMABLE_REASONS = ("provider_limit", "loop_limit", "partial")   # pausas que "si" puede retomar
ACTIVE_STATUSES = (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)

_STEP_STATUSES = {s.value for s in StepStatus}


# ---------------------------------------------------------------------------
# Serializacion / validacion / reconstruccion
# ---------------------------------------------------------------------------
def _cap(text, limit):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit]


def serialize_plan(plan: Plan, goal: str, intent_type: str, run_state: str,
                   pause_reason: str = "", run_token: str = "",
                   resume_count: int = 0, now: Optional[float] = None) -> dict:
    steps = []
    for s in plan.steps:
        steps.append({
            "step_id": str(s.step_id),
            "description": _cap(s.description, 500) or "",
            "tool": s.tool,
            "requires_llm": bool(s.requires_llm),
            "expected_output": _cap(s.expected_output, 300) or "",
            "depends_on": [str(d) for d in (s.depends_on or [])],
            "status": s.status.value,
            "result": _cap(s.result, MAX_STEP_RESULT_CHARS),
            "result_truncated": bool(s.result and len(str(s.result)) > MAX_STEP_RESULT_CHARS),
            "error": _cap(s.error, 300),
            "retries": int(s.retries or 0),
        })
    return {
        "v": CHECKPOINT_VERSION,
        "goal": _cap(goal, 2000) or "",
        "intent_type": str(intent_type or "task"),
        "plan_id": str(plan.plan_id),
        "source": str(plan.source or "local"),
        "plan_created_at": float(plan.created_at or 0.0),
        "run_state": run_state,            # running | paused | finished | cancelled
        "pause_reason": pause_reason,      # provider_limit | loop_limit | partial | needs_input | ...
        "run_token": run_token,
        "resume_count": int(resume_count),
        "saved_at": float(now if now is not None else time.time()),
        "steps": steps,
    }


def validate_checkpoint(cp) -> bool:
    """True solo si el checkpoint tiene la estructura completa y coherente."""
    try:
        if not isinstance(cp, dict) or cp.get("v") != CHECKPOINT_VERSION:
            return False
        if not isinstance(cp.get("goal"), str) or not cp["goal"].strip():
            return False
        if not isinstance(cp.get("intent_type"), str) or not isinstance(cp.get("plan_id"), str):
            return False
        steps = cp.get("steps")
        if not isinstance(steps, list) or not (1 <= len(steps) <= MAX_STEPS):
            return False
        ids = set()
        for sd in steps:
            if not isinstance(sd, dict):
                return False
            sid = sd.get("step_id")
            if not isinstance(sid, str) or not sid or sid in ids:
                return False
            ids.add(sid)
            if not isinstance(sd.get("description"), str) or not sd["description"].strip():
                return False
            if sd.get("status") not in _STEP_STATUSES:
                return False
            if sd.get("tool") is not None and not isinstance(sd.get("tool"), str):
                return False
            if sd.get("result") is not None and not isinstance(sd.get("result"), str):
                return False
            if not isinstance(sd.get("depends_on", []), list):
                return False
        for sd in steps:
            if any(d not in ids for d in sd.get("depends_on", [])):
                return False
        return True
    except Exception:
        return False


def restore_plan(cp: dict, repeatable: Optional[Callable[[PlanStep], bool]] = None):
    """
    Reconstruye el Plan desde un checkpoint valido.

    Devuelve (plan, uncertain, reset):
      plan       Plan con los pasos terminados conservados (status DONE + result).
      uncertain  pasos que quedaron RUNNING con una herramienta que NO es repetible
                 con seguridad: siguen RUNNING (el loop no los ejecuta) hasta que el
                 usuario confirme repetirlos.
      reset      cuantos pasos FAILED/RUNNING se devolvieron a PENDING para reintentarlos.
    Si el checkpoint es invalido devuelve (None, [], 0).
    """
    if not validate_checkpoint(cp):
        return None, [], 0
    repeatable = repeatable or (lambda step: not step.tool)
    steps, uncertain, reset = [], [], 0
    for sd in cp["steps"]:
        step = PlanStep(
            description=sd["description"], step_id=sd["step_id"], tool=sd.get("tool"),
            requires_llm=bool(sd.get("requires_llm", True)),
            expected_output=sd.get("expected_output", "") or "",
            depends_on=list(sd.get("depends_on", [])),
            status=StepStatus(sd["status"]), result=sd.get("result"),
            error=sd.get("error"), retries=int(sd.get("retries", 0) or 0))
        if step.status == StepStatus.RUNNING:
            if repeatable(step):
                step.status, step.result, step.error = StepStatus.PENDING, None, None
                reset += 1
            else:
                uncertain.append(step)
        elif step.status == StepStatus.FAILED:
            step.status, step.result, step.error, step.retries = StepStatus.PENDING, None, None, 0
            reset += 1
        steps.append(step)
    plan = Plan(goal=cp["goal"], plan_id=cp["plan_id"], steps=steps,
                source=cp.get("source", "local"),
                created_at=float(cp.get("plan_created_at") or time.time()))
    plan.done = all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in steps)
    return plan, uncertain, reset


def tool_is_repeatable(executor, tool_name) -> bool:
    """
    True solo si la herramienta es de riesgo LOW (solo lectura). Cualquier duda
    (registro inaccesible, herramienta desconocida, riesgo superior) -> False,
    para que un paso interrumpido NO se repita sin confirmacion.
    """
    try:
        registry = getattr(executor, "_registry", None)
        if registry is None:
            registry = getattr(executor, "registry", None)
        if registry is None:
            return False
        tool = None
        for attr in ("get", "get_tool"):
            fn = getattr(registry, attr, None)
            if callable(fn):
                tool = fn(tool_name)
                if tool is not None:
                    break
        risk = getattr(tool, "risk_level", None)
        return str(getattr(risk, "value", risk)).strip().lower() == "low"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lectura de estado
# ---------------------------------------------------------------------------
def load_checkpoint(task_state, task_id) -> Optional[dict]:
    """Checkpoint crudo (dict) de la tarea, o None. Nunca lanza."""
    try:
        data = task_state.get_metadata(task_id)
        cp = data.get(PLAN_KEY) if isinstance(data, dict) else None
        return cp if isinstance(cp, dict) else None
    except Exception:
        logger.exception("No se pudo leer el checkpoint de la tarea %s", task_id)
        return None


def is_running(cp, now: Optional[float] = None) -> bool:
    """True si el checkpoint dice 'running' con latido reciente (plan en ejecucion)."""
    if not isinstance(cp, dict) or cp.get("run_state") != "running":
        return False
    now = time.time() if now is None else now
    try:
        return (now - float(cp.get("saved_at") or 0.0)) < RUNNING_STALE_SECONDS
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Almacen por tarea (lo usa AutonomyLoop de forma duck-typed)
# ---------------------------------------------------------------------------
class CheckpointStore:
    """Guarda/lee el checkpoint de UNA tarea. Escribe solo si la tarea sigue activa."""

    def __init__(self, task_state, task_id: str, goal: str, intent_type: str):
        self._tm = task_state
        self.task_id = task_id
        self.goal = goal
        self.intent_type = intent_type
        self.token = uuid.uuid4().hex[:8]
        self.claim_reason = ""

    # -- API usada por AutonomyLoop -------------------------------------------
    def save(self, plan: Plan, run_state: str = "running", pause_reason: str = "") -> bool:
        """Guarda el checkpoint. False si no se pudo (tarea cancelada, otro run activo...)."""
        def mutate(data):
            cur = data.get(PLAN_KEY)
            now = time.time()
            if (isinstance(cur, dict) and cur.get("run_token") not in (None, "", self.token)
                    and is_running(cur, now)):
                return None            # otro run vivo es dueno del checkpoint
            resume_count = int(cur.get("resume_count", 0)) if isinstance(cur, dict) else 0
            data[PLAN_KEY] = serialize_plan(
                plan, self.goal, self.intent_type, run_state, pause_reason,
                run_token=self.token, resume_count=resume_count, now=now)
            return data
        try:
            return self._tm.merge_metadata(self.task_id, mutator=mutate,
                                           allowed_status=ACTIVE_STATUSES) is not None
        except Exception:
            logger.exception("No se pudo guardar el checkpoint de la tarea %s", self.task_id)
            return False

    def should_stop(self) -> bool:
        """True si la tarea ya no esta activa (cancelada, terminada o borrada)."""
        try:
            task = self._tm.get(self.task_id)
            return task is None or task.status not in ACTIVE_STATUSES
        except Exception:
            return False

    # -- reclamo atomico (evita doble ejecucion con "continua" repetido) --------
    def claim(self) -> bool:
        """
        Marca la tarea como 'running' para este run. False si otro run vivo ya la tiene
        (latido reciente) o la tarea no esta activa; el motivo queda en claim_reason.
        """
        self.claim_reason = ""

        def mutate(data):
            cur = data.get(PLAN_KEY)
            now = time.time()
            if (isinstance(cur, dict) and cur.get("run_token") not in (None, "", self.token)
                    and is_running(cur, now)):
                self.claim_reason = "running"
                return None
            new = dict(cur) if isinstance(cur, dict) else {}
            new["run_state"] = "running"
            new["run_token"] = self.token
            new["saved_at"] = now
            new["pause_reason"] = ""
            new["resume_count"] = int(new.get("resume_count", 0) or 0) + (1 if cur else 0)
            data[PLAN_KEY] = new
            return data
        try:
            ok = self._tm.merge_metadata(self.task_id, mutator=mutate,
                                         allowed_status=ACTIVE_STATUSES) is not None
        except Exception:
            logger.exception("No se pudo reclamar la tarea %s", self.task_id)
            self.claim_reason = "error"
            return False
        if not ok and not self.claim_reason:
            self.claim_reason = "inactive"
        return ok
