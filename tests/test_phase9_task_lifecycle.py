"""
NEXUS Ω — Phase 9 Tests: ciclo de vida de reactivación de tareas
completadas (v3.8 Bloque 4).

Antes de este bloque, en cuanto una tarea pasaba a COMPLETED
desaparecía por completo de list_pending() y cualquier continuación
posterior creaba una tarea nueva, rompiendo la continuidad. Este
archivo cubre los 6 escenarios obligatorios del bloque:

  1. IN_PROGRESS -> continuación -> mismo task_id (sin cambios,
     regresión del comportamiento de Bloques 2/3).
  2. COMPLETED reciente + continuación explícita -> se reactiva y se
     reutiliza el mismo task_id.
  3. COMPLETED fuera de la ventana de reactivación -> NO se reutiliza.
  4. CANCELLED -> nunca se reactiva, con o sin continuación.
  5. COMPLETED + solicitud nueva SIN marcador de continuación -> no
     hay relación asumida, se crea una tarea nueva.
  6. Dos continuaciones encadenadas (COMPLETED -> reactivar ->
     COMPLETED -> reactivar de nuevo) siguen usando el mismo task_id.
"""

import time

import pytest

from backend.core.nexus import NexusCore
from backend.core.intent import IntentResult, IntentType, IntentStrategy, Domain
from backend.core.task_state import TaskStateManager, TaskStatus


class ScriptedIntentRouter:
    def __init__(self, mapping):
        self._mapping = mapping

    def route(self, message):
        lower = message.strip().lower()
        for pattern, (intent, strategy) in self._mapping.items():
            if pattern in lower:
                return IntentResult(intent=intent, domain=Domain.GENERAL,
                                     confidence=0.9, strategy=strategy)
        return IntentResult(intent=IntentType.CHAT, domain=Domain.GENERAL,
                             confidence=0.5, strategy=IntentStrategy.LLM)


class FakeInferenceLayer:
    async def generate(self, request):
        class R:
            text = "[llm]"
            runtime_name = "fake"
            model = "fake-model"
            fallback_used = False
            is_local = True
        return R()


class FakeAutonomyLoop:
    async def run(self, goal, intent_type, context, request_id, use_llm_plan):
        class Trace:
            status = type("S", (), {"value": "completed"})()
            loops = 1

        class Result:
            text = f"[resultado autonomy para: {goal[:40]}]"
            plan = None
            needs_input = False
            trace = Trace()

        return Result()


def _make_task_state(tmp_path, name="phase9_tasks.db"):
    return TaskStateManager(str(tmp_path / name))


_ROUTER = ScriptedIntentRouter({
    "analiza mis ventas":       (IntentType.ANALYSIS, IntentStrategy.AUTONOMY),
    "diseña una alternativa":   (IntentType.DESIGN, IntentStrategy.AUTONOMY),
    "calcula el costo":         (IntentType.CALCULATION, IntentStrategy.LLM),
    "algo completamente nuevo": (IntentType.ANALYSIS, IntentStrategy.AUTONOMY),
})


class TestTaskLifecycleReactivation:

    @pytest.mark.asyncio
    async def test_in_progress_continuation_keeps_same_task_id(self, tmp_path):
        """Escenario 1: una tarea IN_PROGRESS se reutiliza igual que
        en los bloques anteriores (regresión, no debe romperse)."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza mis ventas", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            resp = await core.process("Ahora calcula el costo")

            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_recently_completed_task_reactivated_on_continuation(self, tmp_path):
        """Escenario 2: una tarea recién COMPLETED se reactiva ante una
        continuación explícita y conserva el task_id."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            assert ts.get(r1.task_id).status == TaskStatus.COMPLETED

            r2 = await core.process("Ahora calcula el costo")

            assert r2.task_id == r1.task_id
            assert ts.get(r1.task_id).status == TaskStatus.IN_PROGRESS
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_old_completed_task_not_reused(self, tmp_path):
        """Escenario 3: una tarea COMPLETED fuera de la ventana de
        reactivación (simulada como muy antigua) NO se reutiliza."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")

            # Forzar que la tarea se vea completada hace mucho tiempo,
            # fuera de la ventana de reactivación de 300s.
            old_time = time.time() - 9999
            ts._conn.execute(
                "UPDATE tasks SET updated_at = ? WHERE task_id = ?",
                (old_time, r1.task_id))
            ts._conn.commit()

            r2 = await core.process("Ahora calcula el costo")

            assert r2.task_id != r1.task_id
            assert r2.task_id is None
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_cancelled_task_never_reactivated(self, tmp_path):
        """Escenario 4: una tarea CANCELLED nunca se reactiva, aunque
        el mensaje traiga un marcador de continuación explícito."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Tarea cancelada", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.CANCELLED)

            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            resp = await core.process("Ahora calcula el costo")

            assert resp.task_id is None
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_completed_task_not_reused_without_continuation_marker(self, tmp_path):
        """Escenario 5: una solicitud AUTONOMY nueva, SIN marcador de
        continuación, no asume relación con una tarea completada
        reciente; crea una tarea nueva."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            r2 = await core.process("algo completamente nuevo")

            assert r2.task_id != r1.task_id
            assert r2.task_id is not None
            assert ts.stats()["total"] == 2
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_chained_continuations_keep_reusing_same_task(self, tmp_path):
        """Escenario 6 (el más importante): dos continuaciones
        encadenadas, cada una pasando por COMPLETED entre medio,
        siguen reutilizando el mismo task_id ambas veces."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            assert ts.get(r1.task_id).status == TaskStatus.COMPLETED

            r2 = await core.process("Ahora diseña una alternativa")
            assert ts.get(r1.task_id).status == TaskStatus.COMPLETED  # vuelve a completarse

            r3 = await core.process("Ahora calcula el costo")

            assert r1.task_id == r2.task_id == r3.task_id
            assert ts.stats()["total"] == 1
        finally:
            ts.close()
