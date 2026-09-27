"""
NEXUS Ω — Phase 7 Tests: continuidad de tarea ante cambio de intent
(v3.8 Bloque 2).

Cubre exactamente los 6 escenarios obligatorios del bloque:

  1. Análisis -> cálculo: nueva operación, mismo task_id.
  2. Análisis -> nueva suboperación (comparación): mismo task_id.
  3. Planificación -> continuación (pedir pasos): mismo task_id.
  4. Sin tarea activa: NO se inventa un task_id.
  5. Detención de tarea: se cancela y no se reutiliza después.
  6. Las confirmaciones cortas del Bloque 1 (sí/ok/dale/continúa/
     ahora/listo) se conservan y no se rompen con este cambio.

Usa un IntentRouter de prueba controlado (no el real) para aislar el
comportamiento nuevo de NexusCore de los detalles de clasificación de
palabras clave de intent.py, tal como ya se hizo en el Bloque 1.
"""

import pytest

from backend.core.nexus import NexusCore
from backend.core.intent import IntentResult, IntentType, IntentStrategy, Domain
from backend.core.task_state import TaskStateManager, TaskStatus


class ScriptedIntentRouter:
    """Router de prueba: clasifica según qué substring reconozca en el
    mensaje, para controlar exactamente el intent/estrategia de cada
    escenario sin depender de las keywords reales de IntentRouter."""

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
            text = f"[respuesta LLM: {request.prompt[:60]}]"
            runtime_name = "fake"
            model = "fake-model"
            fallback_used = False
            is_local = True
        return R()


def _make_task_state(tmp_path, name="phase7_tasks.db"):
    return TaskStateManager(str(tmp_path / name))


# Router compartido: reconoce las frases de los escenarios obligatorios.
# Todo se clasifica con strategy=LLM (no AUTONOMY) a propósito, para
# aislar la lógica de continuidad de tarea de _run_autonomy, que crea
# tareas nuevas por diseño (ver limitación documentada en el reporte).
_ROUTER = ScriptedIntentRouter({
    "analiza mis ventas":            (IntentType.ANALYSIS, IntentStrategy.LLM),
    "calcula el costo":              (IntentType.CALCULATION, IntentStrategy.LLM),
    "compáralo con el mes pasado":   (IntentType.ANALYSIS, IntentStrategy.LLM),
    "planea una estrategia":         (IntentType.TASK, IntentStrategy.LLM),
    "dame los pasos":                (IntentType.QUESTION, IntentStrategy.LLM),
})


class TestTaskContinuityAcrossIntentChange:

    @pytest.mark.asyncio
    async def test_analysis_then_calculation_same_task(self, tmp_path):
        """Escenario 1: 'Analiza mis ventas' -> 'Ahora calcula el costo'
        debe conservar el mismo task_id."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza mis ventas de este mes",
                              intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            resp = await core.process("Ahora calcula el costo")

            assert resp.intent == "calculation"
            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_analysis_then_new_suboperation_same_task(self, tmp_path):
        """Escenario 2: 'Analiza mis ventas' -> 'Ahora compáralo con el
        mes pasado' debe conservar el mismo task_id."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza mis ventas de este mes",
                              intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            resp = await core.process("Ahora compáralo con el mes pasado")

            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_planning_then_ask_for_steps_same_task(self, tmp_path):
        """Escenario 3: 'Planea una estrategia...' -> 'Ahora dame los
        pasos' debe conservar el mismo task_id."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Planea una estrategia para reducir costos",
                              intent="task", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            resp = await core.process("Ahora dame los pasos")

            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_continuation_marker_without_active_task_no_fake_id(self, tmp_path):
        """Escenario 4: 'Ahora calcula el costo' sin ninguna tarea
        activa NO debe inventar un task_id; se procesa como solicitud
        independiente."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            resp = await core.process("Ahora calcula el costo")

            assert resp.task_id is None
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_stop_task_cancels_it(self, tmp_path):
        """Escenario 5 (parte 1): 'Detén la tarea' cancela la tarea
        activa usando el mecanismo existente (TaskStatus.CANCELLED)."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Tarea de prueba a detener",
                              intent="task", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            resp = await core.process("Detén la tarea")

            assert resp.provider == "task_state"
            assert resp.task_id == task.task_id
            updated = ts.get(task.task_id)
            assert updated.status == TaskStatus.CANCELLED
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_stopped_task_not_reused_afterwards(self, tmp_path):
        """Escenario 5 (parte 2): tras detener la tarea, un mensaje de
        continuación posterior NO debe reutilizar el task_id cancelado."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Tarea de prueba a detener",
                              intent="task", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            await core.process("Detén la tarea")
            resp = await core.process("Ahora calcula el costo")

            assert resp.task_id is None
            assert resp.task_id != task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_short_confirmations_still_work_after_block2(self, tmp_path):
        """Escenario 6: las confirmaciones cortas del Bloque 1 no se
        rompen con los cambios de este bloque."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Tarea de prueba", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=_ROUTER, task_state=ts)
            for word in ["sí", "ok", "dale", "continúa", "ahora", "listo"]:
                resp = await core.process(word)
                assert resp.provider == "task_state", (
                    f"'{word}' debió seguir tratándose como follow-up (Bloque 1)")
        finally:
            ts.close()
