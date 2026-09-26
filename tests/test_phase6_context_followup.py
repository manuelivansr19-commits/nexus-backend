"""
NEXUS Ω — Phase 6 Tests: continuidad de tarea y resiliencia de contexto.

Cubre el primer bloque incremental de v3.8 (ver reporte de sesión):

  A. Confirmaciones cortas ("sí", "ok", "dale"...) se tratan como
     continuación de una tarea pendiente en vez de clasificarse como
     CHAT genérico, cuando existe una tarea pendiente real.
  B. Un fallo en ContextManager.assemble() ya no tumba la respuesta
     con un error no manejado (antes causaba el 503 / "No pude
     procesar eso" que se veía en el frontend); se usa un fallback
     mínimo y la conversación sigue respondiendo con normalidad.

No modifica ni depende de IntentRouter real (se usa un router estático
de prueba) para aislar exactamente el comportamiento nuevo de NexusCore,
sin acoplarse a los patrones de palabras clave de intent.py.
"""

import pytest

from backend.core.nexus import NexusCore
from backend.core.intent import IntentResult, IntentType, IntentStrategy, Domain
from backend.core.task_state import TaskStateManager, TaskStatus


class StaticIntentRouter:
    """Siempre clasifica como CHAT/LLM, como haría el IntentRouter real
    ante un mensaje corto sin keywords reconocidas (p.ej. "ok")."""
    def route(self, message):
        return IntentResult(intent=IntentType.CHAT, domain=Domain.GENERAL,
                             confidence=0.5, strategy=IntentStrategy.LLM)


class FakeInferenceLayer:
    """Motor de inferencia falso: siempre responde con éxito, para
    aislar el comportamiento de NexusCore de la disponibilidad real
    de proveedores."""
    async def generate(self, request):
        class R:
            text = "respuesta normal del LLM"
            runtime_name = "fake"
            model = "fake-model"
            fallback_used = False
            is_local = True
        return R()


class BrokenContextManager:
    """Simula un ContextManager que revienta al ensamblar (memoria
    corrupta, knowledge engine caído, etc.)."""
    def assemble(self, **kwargs):
        raise RuntimeError("fallo simulado de contexto")


def _make_task_state(tmp_path):
    return TaskStateManager(str(tmp_path / "phase6_tasks.db"))


class TestShortConfirmationFollowsPendingTask:

    @pytest.mark.asyncio
    async def test_short_confirmation_with_pending_task_uses_followup(self, tmp_path):
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza la empresa X", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=StaticIntentRouter(),
                              task_state=ts)
            resp = await core.process("ok")

            assert resp.provider == "task_state"
            assert resp.intent == "follow_up"
            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_short_confirmation_without_pending_task_goes_to_llm(self, tmp_path):
        ts = _make_task_state(tmp_path)
        try:
            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=StaticIntentRouter(),
                              task_state=ts)
            resp = await core.process("ok")

            assert resp.provider == "fake"
            assert resp.text == "respuesta normal del LLM"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_normal_message_not_treated_as_confirmation(self, tmp_path):
        """Un mensaje con contenido propio ('analiza la competencia...')
        no debe engancharse a la lógica de confirmación corta, aunque
        exista una tarea pendiente."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza la empresa X", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=StaticIntentRouter(),
                              task_state=ts)
            resp = await core.process("analiza la competencia de mi empresa")

            assert resp.provider == "fake"
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_confirmation_returns_stored_result_when_available(self, tmp_path):
        """Si la tarea en curso ya tiene un resultado (por ejemplo, un
        proceso en segundo plano lo escribió antes de marcarla como
        completada), la confirmación corta debe devolver ese resultado
        directamente en vez de ir al LLM."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza la empresa X", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS, result="Resultado final del analisis.")

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=StaticIntentRouter(),
                              task_state=ts)
            resp = await core.process("perfecto")

            assert resp.text == "Resultado final del analisis."
            assert resp.task_id == task.task_id
        finally:
            ts.close()

    @pytest.mark.asyncio
    async def test_various_confirmation_words_recognized(self, tmp_path):
        """Cobertura de varias palabras de confirmación distintas, no
        solo 'ok', para no depender de un único caso feliz."""
        ts = _make_task_state(tmp_path)
        try:
            task = ts.create(description="Analiza la empresa X", intent="analysis", domain="general")
            ts.update(task.task_id, TaskStatus.IN_PROGRESS)

            core = NexusCore(inference=FakeInferenceLayer(),
                              intent_router=StaticIntentRouter(),
                              task_state=ts)
            for word in ["sí", "dale", "continúa", "ahora", "listo"]:
                resp = await core.process(word)
                assert resp.provider == "task_state", f"'{word}' debió tratarse como follow-up"
        finally:
            ts.close()


class TestContextAssemblyResilience:

    @pytest.mark.asyncio
    async def test_broken_context_manager_does_not_crash_response(self):
        """Antes de este fix, una excepción en assemble() se propagaba
        sin capturar hasta el handler de main.py, que respondía 503 y
        el frontend mostraba 'No pude procesar eso.'."""
        core = NexusCore(inference=FakeInferenceLayer(),
                          intent_router=StaticIntentRouter(),
                          context_manager=BrokenContextManager())
        resp = await core.process("cualquier mensaje normal")

        assert resp.provider == "fake"
        assert resp.text == "respuesta normal del LLM"

    @pytest.mark.asyncio
    async def test_working_context_manager_still_used_normally(self):
        """El fix no debe romper el camino feliz: un ContextManager
        real que funciona correctamente sigue usándose sin cambios."""
        from backend.core.context import ContextManager
        core = NexusCore(inference=FakeInferenceLayer(),
                          intent_router=StaticIntentRouter(),
                          context_manager=ContextManager())
        resp = await core.process("cualquier mensaje normal")

        assert resp.provider == "fake"
        assert resp.text == "respuesta normal del LLM"
