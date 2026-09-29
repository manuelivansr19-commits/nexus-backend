"""
NEXUS Ω — Phase 10 Tests: capa de clasificación de comportamiento
(v3.8 Bloque 5).

Cubre la primera capa de las 9 categorías (RESPONDER, CONTINUAR,
PEDIR_INFO, EJECUTAR_HERRAMIENTA, CONSULTAR_CONOCIMIENTO, CALCULAR,
ANALIZAR, PLANIFICAR, DETENER): responde a "¿qué debe hacer NEXUS con
la solicitud?", NO a "¿cómo ejecuta toda esa capacidad?".

La categoría se deriva DENTRO de NexusCore a partir del intent_result
(IntentRouter permanece stateless y sin cambios) y de sus señales
internas (tarea activa, stop, confirmaciones cortas). Se expone como
NexusResponse.behavior.

La mayoría de los tests usan un IntentRouter controlado para aislar la
derivación de comportamiento de las listas de palabras clave de
intent.py (las mismas que ya se usaron como técnica en los Bloques 1-4).
Solo dos tests usan el IntentRouter REAL, con frases cuya clasificación
se conoce con certeza por lectura directa del código ("hola" -> regla
directa; "Necesito que hagas esto" -> PEDIR_INFO sin importar qué
intent asigne el router).

Además incluye un test de observabilidad: una solicitud conversacional
normal debe pasar por el motor de inferencia y NO terminar en una
respuesta determinista de estado del sistema.
"""

import pytest

from backend.core.nexus import NexusCore, Behavior
from backend.core.intent import (
    IntentResult, IntentRouter, IntentType, IntentStrategy, Domain,
)
from backend.core.task_state import TaskStateManager, TaskStatus


class ScriptedIntentRouter:
    """Router de prueba: clasifica según qué substring reconozca. Cada
    valor es (intent, strategy) o (intent, strategy, extras_dict)."""

    def __init__(self, mapping):
        self._mapping = mapping

    def route(self, message):
        lower = message.strip().lower()
        for pattern, spec in self._mapping.items():
            if pattern in lower:
                intent, strategy = spec[0], spec[1]
                extra = spec[2] if len(spec) > 2 else {}
                return IntentResult(intent=intent, domain=Domain.GENERAL,
                                     confidence=0.9, strategy=strategy, **extra)
        return IntentResult(intent=IntentType.CHAT, domain=Domain.GENERAL,
                             confidence=0.5, strategy=IntentStrategy.LLM)


class FakeInferenceLayer:
    async def generate(self, request):
        class R:
            text = "[respuesta del motor de inferencia]"
            runtime_name = "fake"
            model = "fake-model"
            fallback_used = False
            is_local = True
        return R()


class FakeAutonomyLoop:
    """AutonomyLoop falso que cuenta cuántas veces fue invocado."""

    def __init__(self):
        self.calls = 0

    async def run(self, goal, intent_type, context, request_id, use_llm_plan):
        self.calls += 1

        class Trace:
            status = type("S", (), {"value": "completed"})()
            loops = 1

        class Result:
            text = f"[autonomy: {goal[:30]}]"
            plan = None
            needs_input = False
            trace = Trace()

        return Result()


class FakeExecutor:
    async def execute_candidates(self, candidate_names, params, context, request_id):
        class R:
            success = True
            tool_name = candidate_names[0]
        return [R()]

    def collect_context_strings(self, results):
        return ["[resultado de herramienta]"]


_ROUTER = ScriptedIntentRouter({
    "qué es la fotosíntesis":    (IntentType.QUESTION, IntentStrategy.LLM),
    "raíz cuadrada":             (IntentType.CALCULATION, IntentStrategy.LLM),
    "analiza estos datos":       (IntentType.ANALYSIS, IntentStrategy.AUTONOMY),
    "investiga la fotosíntesis": (IntentType.RESEARCH, IntentStrategy.AUTONOMY),
    "planea un proyecto":        (IntentType.TASK, IntentStrategy.AUTONOMY),
    "diseña una estructura":     (IntentType.DESIGN, IntentStrategy.AUTONOMY),
    "necesito que":              (IntentType.TASK, IntentStrategy.AUTONOMY),
    "usa la herramienta":        (IntentType.TASK, IntentStrategy.TOOL,
                                   {"requires_tool": True, "candidate_tools": ["calc_tool"]}),
})


def _make_core(tmp_path, with_executor=False):
    ts = TaskStateManager(str(tmp_path / "phase10_tasks.db"))
    loop = FakeAutonomyLoop()
    core = NexusCore(inference=FakeInferenceLayer(), intent_router=_ROUTER,
                      task_state=ts, autonomy_loop=loop,
                      executor=FakeExecutor() if with_executor else None)
    return core, ts, loop


class TestBehaviorCategories:

    @pytest.mark.asyncio
    async def test_question_is_responder(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            resp = await core.process("¿Qué es la fotosíntesis?")
            assert resp.behavior == Behavior.RESPONDER.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_calculation_is_calcular(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            resp = await core.process("¿Cuál es la raíz cuadrada de 144?")
            assert resp.behavior == Behavior.CALCULAR.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_analysis_is_analizar(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            resp = await core.process("Analiza estos datos")
            assert resp.behavior == Behavior.ANALIZAR.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_research_is_consultar_conocimiento(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            resp = await core.process("Investiga la fotosíntesis")
            assert resp.behavior == Behavior.CONSULTAR_CONOCIMIENTO.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_task_and_design_are_planificar(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            r1 = await core.process("Planea un proyecto de energía solar")
            r2 = await core.process("Diseña una estructura robótica")
            assert r1.behavior == Behavior.PLANIFICAR.value
            assert r2.behavior == Behavior.PLANIFICAR.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_tool_strategy_is_ejecutar_herramienta(self, tmp_path):
        core, ts, _ = _make_core(tmp_path, with_executor=True)
        try:
            resp = await core.process("Usa la herramienta calculadora para sumar")
            assert resp.provider == "tool"
            assert resp.behavior == Behavior.EJECUTAR_HERRAMIENTA.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_autonomy_route_still_runs(self, tmp_path):
        """Regresión: etiquetar el comportamiento NO debe impedir que la
        ruta AUTONOMY existente siga ejecutándose y creando su tarea."""
        core, ts, loop = _make_core(tmp_path)
        try:
            resp = await core.process("Analiza estos datos")
            assert loop.calls == 1
            assert resp.task_id is not None
            assert ts.stats()["total"] == 1
        finally:
            ts.close()


class TestPedirInfo:

    @pytest.mark.asyncio
    async def test_vague_request_asks_for_info_instead_of_autonomy(self, tmp_path):
        """'Necesito que hagas esto' (objeto vago, sin tarea activa) debe
        pedir información en lugar de disparar AUTONOMY a ciegas."""
        core, ts, loop = _make_core(tmp_path)
        try:
            resp = await core.process("Necesito que hagas esto")
            assert resp.behavior == Behavior.PEDIR_INFO.value
            assert resp.model == "pedir_info"
            assert loop.calls == 0
            assert ts.stats()["total"] == 0
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_specific_request_does_not_ask_for_info(self, tmp_path):
        """Un pedido con objeto concreto no debe tratarse como PEDIR_INFO."""
        core, ts, _ = _make_core(tmp_path)
        try:
            resp = await core.process("Necesito que analices las ventas de marzo")
            assert resp.behavior != Behavior.PEDIR_INFO.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_vague_object_with_active_task_is_not_pedir_info(self, tmp_path):
        """Si hay una tarea activa y el mensaje es una continuación
        explícita, 'esto' tiene un referente válido: no se pide info."""
        core, ts, _ = _make_core(tmp_path)
        try:
            task = ts.create(description="Analiza mis ventas", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            resp = await core.process("Ahora hazlo con esto")
            assert resp.behavior != Behavior.PEDIR_INFO.value
            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_real_intent_router_vague_request_asks_for_info(self, tmp_path):
        """Con el IntentRouter REAL: 'Necesito que hagas esto' contiene
        'necesito que' (keyword de TASK -> AUTONOMY), pero PEDIR_INFO
        debe preemptar antes de disparar el loop de autonomía."""
        ts = TaskStateManager(str(tmp_path / "phase10_real_router.db"))
        loop = FakeAutonomyLoop()
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=IntentRouter(),
                              task_state=ts, autonomy_loop=loop)
            resp = await core.process("Necesito que hagas esto")
            assert resp.behavior == Behavior.PEDIR_INFO.value
            assert loop.calls == 0
        finally:
            ts.close()


class TestContinuarYDetener:

    @pytest.mark.asyncio
    async def test_short_confirmations_are_continuar(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            task = ts.create(description="Tarea en curso", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            for word in ["sí", "continúa"]:
                resp = await core.process(word)
                assert resp.behavior == Behavior.CONTINUAR.value, word
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_stop_is_detener(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            task = ts.create(description="Tarea a detener", intent="task", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            resp = await core.process("Detén la tarea")
            assert resp.behavior == Behavior.DETENER.value
            assert ts.get(task.task_id).status == TaskStatus.CANCELLED
        finally:
            ts.close()


class TestObservability:
    """Mecanismo de observabilidad: una solicitud conversacional normal
    debe pasar por el motor de inferencia y NO terminar por accidente en
    una respuesta determinista de estado del sistema."""

    @pytest.mark.asyncio
    async def test_normal_requests_use_inference_not_system_fallback(self, tmp_path):
        core, ts, _ = _make_core(tmp_path)
        try:
            for msg in ["¿Qué es la fotosíntesis?",
                        "cuéntame algo interesante",
                        "¿Cuál es la raíz cuadrada de 144?"]:
                resp = await core.process(msg)
                assert resp.provider == "fake", msg
                assert resp.model != "deterministic", msg
                assert "operativo" not in resp.text.lower(), msg
                assert resp.behavior in (Behavior.RESPONDER.value, Behavior.CALCULAR.value), msg
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_real_router_greeting_is_deterministic_responder(self, tmp_path):
        """Con el IntentRouter REAL, un saludo simple sí se resuelve por
        la regla directa (determinista) y se etiqueta como RESPONDER;
        esto documenta cuál es la ÚNICA vía legítima a una respuesta
        determinista, para distinguirla de un fallback accidental."""
        ts = TaskStateManager(str(tmp_path / "phase10_greeting.db"))
        try:
            core = NexusCore(inference=FakeInferenceLayer(), intent_router=IntentRouter(),
                              task_state=ts)
            resp = await core.process("hola")
            assert resp.provider == "system"
            assert resp.model == "deterministic"
            assert resp.behavior == Behavior.RESPONDER.value
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_no_intent_router_leaves_behavior_unset(self, tmp_path):
        """Sin IntentRouter no hay base para clasificar: behavior None."""
        core = NexusCore(inference=FakeInferenceLayer())
        resp = await core.process("cualquier mensaje")
        assert resp.behavior is None
