"""
NEXUS Ω — Phase 8 Tests: continuidad estructural de tarea cuando la
nueva operación también clasifica como AUTONOMY (v3.8 Bloque 3).

Antes de este bloque, cualquier continuación cuyo nuevo intent cayera
en AUTONOMY (TASK/DESIGN/ANALYSIS/RESEARCH) creaba una tarea nueva en
_run_autonomy(), perdiendo la relación con la tarea activa. Este
archivo cubre exactamente los 8 escenarios obligatorios del bloque.

Usa un IntentRouter y un AutonomyLoop de prueba controlados (no los
reales) para aislar el comportamiento nuevo de NexusCore/_run_autonomy
de los detalles de clasificación de intent.py y de la ejecución real
del loop de autonomía.
"""

import pytest

from backend.core.nexus import NexusCore
from backend.core.intent import IntentResult, IntentType, IntentStrategy, Domain
from backend.core.task_state import TaskStateManager, TaskStatus


class ScriptedIntentRouter:
    """Router de prueba: clasifica según qué substring reconozca."""

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
    """Simula AutonomyLoop.run() devolviendo siempre un resultado
    exitoso simple, sin ejecutar planificación real."""

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


def _make_task_state(tmp_path, name="phase8_tasks.db"):
    return TaskStateManager(str(tmp_path / name))


# Router compartido: cada frase de los escenarios obligatorios clasifica
# con strategy=AUTONOMY, a propósito, para probar exactamente el caso
# que este bloque resuelve.
_ROUTER = ScriptedIntentRouter({
    "analiza mis ventas":     (IntentType.ANALYSIS, IntentStrategy.AUTONOMY),
    "diseña una alternativa": (IntentType.DESIGN, IntentStrategy.AUTONOMY),
    "investiga el mercado":   (IntentType.RESEARCH, IntentStrategy.AUTONOMY),
    "planea una estrategia":  (IntentType.TASK, IntentStrategy.AUTONOMY),
})


class TestAutonomyTaskContinuity:

    @pytest.mark.asyncio
    async def test_analysis_then_design_same_task(self, tmp_path):
        """Escenario 1: ANALYSIS -> continuación AUTONOMY (DESIGN)."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            assert r1.task_id is not None
            # La tarea queda COMPLETED al terminar _run_autonomy; la
            # reactivamos para simular que el usuario la sigue tratando
            # como la tarea "en curso" con la que quiere continuar.
            ts.update(r1.task_id, TaskStatus.IN_PROGRESS)

            r2 = await core.process("Ahora diseña una alternativa")

            assert r2.task_id == r1.task_id
            assert r2.intent == "design"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_design_then_research_same_task(self, tmp_path):
        """Escenario 2: DESIGN -> continuación AUTONOMY (RESEARCH)."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Diseña una alternativa")
            ts.update(r1.task_id, TaskStatus.IN_PROGRESS)

            r2 = await core.process("Ahora investiga el mercado")

            assert r2.task_id == r1.task_id
            assert r2.intent == "research"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_research_then_task_same_task(self, tmp_path):
        """Escenario 3: RESEARCH -> continuación AUTONOMY (TASK)."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Investiga el mercado")
            ts.update(r1.task_id, TaskStatus.IN_PROGRESS)

            r2 = await core.process("Ahora planea una estrategia")

            assert r2.task_id == r1.task_id
            assert r2.intent == "task"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_task_then_design_same_task(self, tmp_path):
        """Escenario 4: TASK -> continuación AUTONOMY (DESIGN)."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Planea una estrategia")
            ts.update(r1.task_id, TaskStatus.IN_PROGRESS)

            r2 = await core.process("Ahora diseña una alternativa")

            assert r2.task_id == r1.task_id
            assert r2.intent == "design"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_no_second_task_created(self, tmp_path):
        """Escenario 6: la continuación AUTONOMY no debe crear una
        segunda tarea en TaskStateManager (verificado vía stats())."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            ts.update(r1.task_id, TaskStatus.IN_PROGRESS)
            await core.process("Ahora diseña una alternativa")

            assert ts.stats()["total"] == 1
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_cancelled_task_not_reused(self, tmp_path):
        """Escenario 7: una tarea detenida/cancelada NO se reutiliza;
        la siguiente solicitud AUTONOMY crea una tarea nueva."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r1 = await core.process("Analiza mis ventas")
            ts.update(r1.task_id, TaskStatus.CANCELLED)

            r2 = await core.process("Ahora diseña una alternativa")

            assert r2.task_id != r1.task_id
            assert ts.stats()["total"] == 2
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_autonomy_without_active_task_creates_new_task(self, tmp_path):
        """Escenario 8: una solicitud AUTONOMY sin ninguna tarea activa
        sigue creando una tarea nueva con normalidad (comportamiento
        preexistente, no debe romperse)."""
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                              task_state=ts, autonomy_loop=FakeAutonomyLoop())
            r = await core.process("Analiza mis ventas")

            assert r.task_id is not None
            assert ts.stats()["total"] == 1
        finally:
            ts.close()
