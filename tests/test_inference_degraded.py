# -*- coding: utf-8 -*-
"""Bloque 3: la rama InferenceLayer no filtra el error crudo de los runtimes.
Colocar en tests/ y correr: python -m pytest tests/test_inference_degraded.py -v"""
import asyncio
import logging
import unittest

from backend.core.nexus import NexusCore
from backend.inference.errors import AllRuntimesFailed


def _failed(errors):
    # No dependemos de la firma del constructor de AllRuntimesFailed.
    e = AllRuntimesFailed.__new__(AllRuntimesFailed)
    e.runtime_errors = errors
    return e


class FailingInference:
    def __init__(self, errors):
        self._errors = errors

    async def generate(self, request):
        raise _failed(self._errors)


REAL = [
    ("ollama", "RUNTIME_UNAVAILABLE: Ollama no responde en http://"),
    ("llama_cpp", "CONFIGURATION_ERROR: SCAFFOLDED: llama-cpp-python"),
    ("gemini", "RATE_LIMITED: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429}}"),
]


class TestInferenceDegraded(unittest.TestCase):

    def _infer(self, errors):
        core = NexusCore(inference=FailingInference(errors))
        return asyncio.run(core._infer("hola", "sys", [], request_id="t"))

    def test_no_filtra_detalle_crudo(self):
        text = self._infer(REAL)["text"]
        for crudo in ("RUNTIME_UNAVAILABLE", "SCAFFOLDED", "RESOURCE_EXHAUSTED",
                      "429", "http://", "{'error'", "Detalle:"):
            self.assertNotIn(crudo, text)

    def test_mensaje_de_cuota_y_honestidad(self):
        r = self._infer(REAL)
        self.assertIn("cuota", r["text"])
        self.assertIn("no fue generada por un modelo", r["text"])
        self.assertEqual(r["provider"], "none")

    def test_sobrecarga_y_error_generico(self):
        self.assertIn("saturado", self._infer([("gemini", "UNAVAILABLE 503 high demand")])["text"])
        t = self._infer([("ollama", "RUNTIME_UNAVAILABLE: x")])["text"]
        self.assertIn("fallo interno", t)
        self.assertNotIn("RUNTIME_UNAVAILABLE", t)

    def test_detalle_completo_va_a_logs(self):
        with self.assertLogs(level="ERROR") as cm:
            self._infer(REAL)
        joined = "\n".join(cm.output)
        self.assertIn("RESOURCE_EXHAUSTED", joined)
        self.assertIn("ollama", joined)


if __name__ == "__main__":
    unittest.main()
