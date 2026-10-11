# -*- coding: utf-8 -*-
"""
Tests del Bloque 1: modo degradado (ruta legacy ModelRouter + NexusCore._infer).

Escritos con unittest (pytest los recoge igual): colocar en tests/ y correr
    python -m pytest tests/test_degraded_mode.py -v
"""
import asyncio
import time as _time
import types
import unittest
from unittest import mock

import backend.router as router_mod
from backend.router import (
    COOLDOWN_MAX_S,
    AllProvidersUnavailable,
    ModelRouter,
    degraded_text,
    is_overload_error,
    is_rate_limit_error,
    retry_after_seconds,
)
from backend.core.nexus import NexusCore

REQ = types.SimpleNamespace(prompt="hola", system="", history=[])
RAW_503 = ("503 UNAVAILABLE. This model is currently experiencing high demand. "
           "Spikes in demand are usually temporary.")
RAW_429 = "429 RESOURCE_EXHAUSTED. You exceeded your current quota."


def run(coro):
    return asyncio.run(coro)


class ProviderError(Exception):
    def __init__(self, message, code=None, retry_after=None, headers=None):
        super().__init__(message)
        self.code = code
        if retry_after is not None:
            self.retry_after = retry_after
        if headers is not None:
            self.response = types.SimpleNamespace(headers=headers)


class FakeProvider:
    """Duck-typed: el router solo usa name, is_configured, is_local, generate, shutdown."""
    is_local = False

    def __init__(self, name, error=None, text="respuesta ok", configured=True):
        self.name = name
        self.model = f"{name}-model"
        self.is_configured = configured
        self.error = error
        self.text = text
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(text=self.text, provider=self.name, model=self.model)

    async def shutdown(self):
        pass


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RaisingRouter:
    def __init__(self, exc):
        self.exc = exc

    async def generate(self, request):
        raise self.exc


def make_core(router):
    # __new__ evita depender de la firma del constructor real.
    core = NexusCore.__new__(NexusCore)
    core._inference = None
    core._legacy_router = router
    return core


def infer(core):
    return run(core._infer("hola", "", [], "t-1"))


class DegradedModeBase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        patches = [
            mock.patch.object(router_mod, "NEXUS_LOCAL_ONLY", False),
            mock.patch.object(router_mod, "USE_OLLAMA_ONLY", False),
            mock.patch.object(router_mod, "COOLDOWN_RATE_LIMIT_S", 60.0),
            mock.patch.object(router_mod, "COOLDOWN_OVERLOAD_S", 30.0),
            mock.patch.object(router_mod, "time", types.SimpleNamespace(
                monotonic=self.clock.monotonic, perf_counter=_time.perf_counter)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestDegradedMode(DegradedModeBase):

    # 1
    def test_429_sin_otro_proveedor_lanza_all_providers_unavailable(self):
        gem = FakeProvider("gemini", ProviderError(RAW_429, code=429))
        with self.assertRaises(AllProvidersUnavailable) as ctx:
            run(ModelRouter([gem]).generate(REQ))
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertEqual(ctx.exception.reasons, {"gemini": "rate_limit"})

    # 2
    def test_503_high_demand_se_clasifica_overloaded(self):
        err = ProviderError(RAW_503, code=503)
        self.assertTrue(is_overload_error(err))
        self.assertFalse(is_rate_limit_error(err))
        self.assertTrue(is_overload_error(Exception("Gemini error 503: high demand")))
        self.assertFalse(is_overload_error(Exception("connection reset on port 15032")))

        gem = FakeProvider("gemini", err)
        with self.assertRaises(AllProvidersUnavailable) as ctx:
            run(ModelRouter([gem]).generate(REQ))
        self.assertEqual(ctx.exception.reasons, {"gemini": "overloaded"})

    # 3
    def test_todos_fallan_infer_devuelve_texto_degradado_sin_error_crudo(self):
        router = ModelRouter([
            FakeProvider("gemini", ProviderError(RAW_503, code=503)),
            FakeProvider("groq", ProviderError(RAW_429, code=429)),
        ])
        out = infer(make_core(router))
        self.assertIn("modo degradado", out["text"])
        self.assertIn("cuota", out["text"])
        for crudo in ("high demand", "RESOURCE_EXHAUSTED", "Todos los proveedores"):
            self.assertNotIn(crudo, out["text"])
        self.assertEqual(out["provider"], "none")
        # las tres causas se distinguen
        self.assertIn("cuota", degraded_text({"g": "rate_limit"}))
        self.assertIn("saturado", degraded_text({"g": "overloaded"}))
        self.assertIn("fallo interno", degraded_text({"g": "error"}))

    # 4
    @unittest.skip(
        "PENDIENTE, no implementado a proposito: hay que inspeccionar el codigo real. "
        "Objetivo: con IntentStrategy.TOOL y un resultado de herramienta exitoso, "
        "NexusCore.process() debe responder provider='tool' SIN llamar a _infer. "
        "DESCONOCIDO (no se inventa): firma de NexusCore.process(), como se invoca "
        "IntentRouter, y los constructores de NexusResponse e IntentResult. "
        "CONOCIDO (solo del paso 3 pegado de process()): usa "
        "self._executor.execute_candidates(candidate_names, params, context, request_id), "
        "self._executor.collect_context_strings(...) e intent_result.requires_tool / "
        "candidate_tools / strategy."
    )
    def test_peticion_determinista_no_toca_el_llm(self):
        pass

    # 5
    def test_peticion_que_requiere_llm_sin_proveedor(self):
        for router in (ModelRouter([]),
                       ModelRouter([FakeProvider("gemini", configured=False)])):
            out = infer(make_core(router))
            self.assertIn("modo degradado", out["text"])
            self.assertIn("no fue generada por un modelo", out["text"])
            self.assertNotIn("Ningún proveedor", out["text"])

    # 6
    def test_ningun_runtimeerror_escapa(self):
        excepciones = [
            RuntimeError("boom-interno"),
            AllProvidersUnavailable(["x: y"], {"x": "error"}),
            ValueError("v-interno"),
            KeyError("k-interno"),
        ]
        for exc in excepciones:
            out = infer(make_core(RaisingRouter(exc)))
            self.assertIsInstance(out["text"], str)
            for crudo in ("boom-interno", "v-interno", "k-interno"):
                self.assertNotIn(crudo, out["text"])

    # 7
    def test_enfriamiento_evita_llamadas_y_expira(self):
        gem = FakeProvider("gemini", ProviderError(RAW_429, code=429))
        groq = FakeProvider("groq", text="ok groq")
        router = ModelRouter([gem, groq])

        res = run(router.generate(REQ))
        self.assertEqual(res.response.text, "ok groq")
        self.assertEqual((gem.calls, groq.calls), (1, 1))   # un intento, sin reintentos

        self.clock.advance(59)                               # dentro de los 60 s
        run(router.generate(REQ))
        self.assertEqual((gem.calls, groq.calls), (1, 2))    # gemini no se llamo

        self.clock.advance(2)                                # 61 s: vencido
        run(router.generate(REQ))
        self.assertEqual(gem.calls, 2)                       # se reintenta solo

        # 503: ventana de 30 s
        gem2 = FakeProvider("gemini", ProviderError(RAW_503, code=503))
        r2 = ModelRouter([gem2])
        with self.assertRaises(AllProvidersUnavailable):
            run(r2.generate(REQ))
        self.clock.advance(29)
        with self.assertRaises(AllProvidersUnavailable) as ctx:
            run(r2.generate(REQ))
        self.assertEqual(gem2.calls, 1)
        self.assertEqual(ctx.exception.reasons, {"gemini": "overloaded"})  # conserva la causa
        self.clock.advance(2)
        with self.assertRaises(AllProvidersUnavailable):
            run(r2.generate(REQ))
        self.assertEqual(gem2.calls, 2)

    # 8
    def test_retry_after_se_respeta_y_se_topa(self):
        self.assertEqual(retry_after_seconds(ProviderError("x", retry_after=12.5)), 12.5)
        self.assertEqual(retry_after_seconds(
            ProviderError("x", headers={"Retry-After": "20"})), 20.0)
        self.assertEqual(retry_after_seconds(
            ProviderError("quota exceeded. Please retry in 7.5s.")), 7.5)
        self.assertIsNone(retry_after_seconds(ProviderError("sin pista")))

        gem = FakeProvider("gemini", ProviderError(RAW_429, code=429, retry_after=12.5))
        router = ModelRouter([gem])
        with self.assertRaises(AllProvidersUnavailable):
            run(router.generate(REQ))
        self.clock.advance(12)
        with self.assertRaises(AllProvidersUnavailable):
            run(router.generate(REQ))
        self.assertEqual(gem.calls, 1)                       # 12 s < 12.5 s
        self.clock.advance(1)
        with self.assertRaises(AllProvidersUnavailable):
            run(router.generate(REQ))
        self.assertEqual(gem.calls, 2)                       # 13 s > 12.5 s

        # tope: un Retry-After absurdo nunca bloquea mas de COOLDOWN_MAX_S
        gem2 = FakeProvider("gemini", ProviderError(RAW_429, code=429, retry_after=99999))
        r2 = ModelRouter([gem2])
        with self.assertRaises(AllProvidersUnavailable):
            run(r2.generate(REQ))
        self.clock.advance(COOLDOWN_MAX_S - 1)
        with self.assertRaises(AllProvidersUnavailable):
            run(r2.generate(REQ))
        self.assertEqual(gem2.calls, 1)
        self.clock.advance(2)
        with self.assertRaises(AllProvidersUnavailable):
            run(r2.generate(REQ))
        self.assertEqual(gem2.calls, 2)


if __name__ == "__main__":
    unittest.main()
