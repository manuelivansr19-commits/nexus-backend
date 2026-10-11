# -*- coding: utf-8 -*-
"""
Tests del Bloque 2: seleccion de `clock` y respuesta directa de herramientas.

Colocar en tests/ y correr:
    python -m pytest tests/test_tool_selection_output.py -v

Cubre lo que test_phase12 NO cubre: ese archivo usa IntentRouter(registry=None),
asi que nunca ejercita la ruta find_by_intent (subcadenas) que si corre en
produccion. Aqui se prueba con y sin registro.

Ademas cubre el objetivo del test #4 de test_degraded_mode.py (una peticion
determinista responde sin tocar el LLM), con fakes sobre las firmas reales:
NexusCore(intent_router=, executor=, model_router=), process(message, ...),
IntentResult(...) y NexusResponse.
"""
import asyncio
import types
import unittest

from backend.core.intent import (
    Domain,
    IntentResult,
    IntentRouter,
    IntentStrategy,
    IntentType,
)
from backend.core.nexus import NexusCore
from backend.tools.base import ToolInput, ToolResult
from backend.tools.builtin import ClockTool, create_default_registry


def _is_clock(result):
    return "clock" in result.candidate_tools


def _routers():
    """Sin registro (como test_phase12) y con registro (como produccion)."""
    return [
        ("sin registro", IntentRouter(registry=None)),
        ("con registro", IntentRouter(registry=create_default_registry())),
    ]


class TestClockSelection(unittest.TestCase):

    EXPLICIT = [
        "¿Qué hora es?",
        "¿Qué hora tenemos ahora?",
        "¿Qué hora es actualmente?",
        "que hora es",
        "¿Qué horas son?",
    ]

    NOT_CLOCK = [
        # "ahora" contiene "hora": era el bug
        "ahora crea un plan de marketing",
        "Ahora quiero que analices el mercado",
        "ahora dime algo sobre estrategia",
        # otras subcadenas / palabras comunes
        "¿Cuál es el horario de atención?",
        "A la hora de decidir, ¿qué conviene más?",
        "¿Cuánto tiempo toma lanzar una campaña?",
        "Buen día, hablemos de estrategia",
        "Cuando puedas, revisa el candidate",
    ]

    def test_consultas_explicitas_de_hora_activan_clock(self):
        for label, router in _routers():
            for message in self.EXPLICIT:
                with self.subTest(router=label, message=message):
                    result = router.route(message)
                    self.assertTrue(_is_clock(result), f"{message!r} deberia activar clock")
                    self.assertEqual(result.strategy, IntentStrategy.TOOL)

    def test_mensajes_normales_no_activan_clock(self):
        for label, router in _routers():
            for message in self.NOT_CLOCK:
                with self.subTest(router=label, message=message):
                    result = router.route(message)
                    self.assertFalse(_is_clock(result), f"{message!r} NO deberia activar clock")

    def test_clock_no_pisa_la_estrategia_autonomy(self):
        # Un mensaje que requiere planificacion nunca se convierte en clock.
        for label, router in _routers():
            with self.subTest(router=label):
                result = router.route("ahora crea un plan de marketing")
                self.assertFalse(_is_clock(result))
                if result.requires_planning:
                    self.assertEqual(result.strategy, IntentStrategy.AUTONOMY)


class TestToolResultUserString(unittest.TestCase):

    def test_string_output_sin_etiqueta(self):
        r = ToolResult(success=True, output="NEXUS operativo.", tool_name="system_status")
        self.assertEqual(r.to_user_string(), "NEXUS operativo.")
        self.assertIn("[TOOL:system_status]", r.to_context_string())  # contexto interno intacto

    def test_dict_sin_user_text_no_lleva_etiqueta(self):
        r = ToolResult(success=True, output={"a": 1}, tool_name="x")
        self.assertNotIn("[TOOL", r.to_user_string())

    def test_user_text_tiene_prioridad(self):
        r = ToolResult(success=True, output={"a": 1}, tool_name="x",
                       metadata={"user_text": "Hola."})
        self.assertEqual(r.to_user_string(), "Hola.")

    def test_clock_tool_texto_para_usuario(self):
        result = asyncio.run(ClockTool().execute(
            ToolInput(params={}, context="", request_id="t")))
        # el output estructurado no cambia
        self.assertEqual(set(["time", "date", "iso"]), set(result.output))
        text = result.to_user_string()
        self.assertTrue(text.startswith("Son las "))
        self.assertNotIn("[TOOL", text)
        self.assertNotIn("{", text)
        # sin nombres en ingles (no depende del locale)
        for ingles in ("Sunday", "Monday", "October", "January"):
            self.assertNotIn(ingles, text)


class FakeIntent:
    def route(self, message, context=""):
        return IntentResult(
            intent=IntentType.QUESTION, domain=Domain.SYSTEM, confidence=1.0,
            strategy=IntentStrategy.TOOL, requires_tool=True,
            candidate_tools=["system_status"],
        )


class FakeExecutor:
    def __init__(self, tool_result):
        self._tool_result = tool_result

    async def execute_candidates(self, candidate_names, params, context="", request_id=""):
        return [types.SimpleNamespace(success=True, tool_name="system_status",
                                      tool_result=self._tool_result)]

    def collect_context_strings(self, results):
        return [r.tool_result.to_context_string() for r in results
                if r.success and r.tool_result]


class CountingRouter:
    def __init__(self):
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        raise AssertionError("el LLM no debe llamarse para una peticion determinista")


class TestDirectToolResponse(unittest.TestCase):

    def _process(self, tool_result):
        router = CountingRouter()
        core = NexusCore(intent_router=FakeIntent(),
                         executor=FakeExecutor(tool_result),
                         model_router=router)
        resp = asyncio.run(core.process("estado del sistema", request_id="t-1"))
        return resp, router

    def test_peticion_determinista_no_toca_el_llm_ni_filtra_etiqueta(self):
        resp, router = self._process(
            ToolResult(success=True, output="NEXUS operativo.", tool_name="system_status"))
        self.assertEqual(resp.text, "NEXUS operativo.")
        self.assertNotIn("[TOOL", resp.text)
        self.assertEqual(resp.provider, "tool")
        self.assertEqual(resp.tools_used, ["system_status"])
        self.assertEqual(router.calls, 0)

    def test_resultado_sin_atributo_tool_result_no_rompe(self):
        # Regresion (tests/test_phase10): un resultado de ejecucion que solo tiene
        # success/tool_name, sin tool_result. Antes: AttributeError.
        class R:
            success = True
            tool_name = "calculadora"

        class Ejecutor:
            async def execute_candidates(self, candidate_names, params, context="", request_id=""):
                return [R()]

            def collect_context_strings(self, results):
                return ["[TOOL:calculadora] 4"]

        router = CountingRouter()
        core = NexusCore(intent_router=FakeIntent(), executor=Ejecutor(), model_router=router)
        resp = asyncio.run(core.process("suma", request_id="t-2"))
        self.assertEqual(resp.text, "[TOOL:calculadora] 4")   # comportamiento anterior intacto
        self.assertEqual(router.calls, 0)

    def test_tool_result_sin_to_user_string_usa_el_contexto(self):
        # Compatibilidad: un resultado "viejo" sin to_user_string no rompe process().
        legacy = types.SimpleNamespace(
            to_context_string=lambda: "[TOOL:system_status] viejo")
        resp, router = self._process(legacy)
        self.assertEqual(resp.text, "[TOOL:system_status] viejo")
        self.assertEqual(router.calls, 0)


if __name__ == "__main__":
    unittest.main()
