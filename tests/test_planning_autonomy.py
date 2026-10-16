# -*- coding: utf-8 -*-
"""
Bloque 4A (Planificacion): honestidad y continuidad del plan.

Cubre: estados de la tarea, needs_input, cuota/sobrecarga, falsos exitos,
continuidad entre pasos (objetivo completo + resultados previos) y sintesis.
Solo usa la API publica existente (AutonomyLoop, Planner, TaskStateManager,
NexusCore._run_autonomy). No necesita red, proveedores ni servidor.
"""
import time
import unittest
from types import SimpleNamespace

from backend.core.autonomy import (
    AutonomyLoop, AutonomyResult, ExecutionTrace, LoopStatus, RESTART_NOTE,
)
from backend.core.nexus import NexusCore
from backend.core.planner import Planner
from backend.core.task_state import TaskStateManager, TaskStatus


class FakeAllRuntimesFailed(Exception):
    """Imita AllRuntimesFailed sin depender de su constructor."""
    def __init__(self, runtime_errors):
        super().__init__("all runtimes failed")
        self.runtime_errors = runtime_errors


class FakeInference:
    def __init__(self, responder):
        self._responder = responder
        self.prompts = []

    async def generate(self, request):
        self.prompts.append(request.prompt)
        out = self._responder(len(self.prompts), request)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out)


def _ok(n, _req):
    return f"Resultado del paso {n}: contenido suficiente para ser valido."


def _loop(inference=None, **kw):
    return AutonomyLoop(planner=Planner(), inference_layer=inference, **kw)


def _intent():
    return SimpleNamespace(intent=SimpleNamespace(value="task"),
                           domain=SimpleNamespace(value="general"))


class TestAutonomyFalseSuccess(unittest.IsolatedAsyncioTestCase):
    async def test_sin_motor_no_es_exito(self):
        res = await _loop(inference=None).run("Analiza mi negocio", intent_type="task",
                                              context="Analiza mi negocio")
        self.assertEqual(res.trace.status, LoopStatus.FAILED)
        self.assertNotIn("Paso registrado", res.text)
        self.assertNotIn("Resultado para", res.text)
        self.assertIn("No pude completar", res.text)

    async def test_flujo_ok_sigue_completando(self):
        inf = FakeInference(_ok)
        res = await _loop(inf).run("Analiza mi negocio", context="Analiza mi negocio")
        self.assertEqual(res.trace.status, LoopStatus.COMPLETED)
        self.assertIn("Resultado para: Analiza mi negocio", res.text)
        self.assertNotIn(RESTART_NOTE, res.text)


class TestAutonomyProviderLimit(unittest.IsolatedAsyncioTestCase):
    async def _run_with(self, exc):
        inf = FakeInference(lambda n, r: exc)
        res = await _loop(inf).run("Disena una arquitectura", context="Disena una arquitectura")
        return res, inf

    async def test_cuota_detiene_el_plan_en_la_primera_llamada(self):
        res, inf = await self._run_with(
            Exception("429 RESOURCE_EXHAUSTED: quota exceeded for model gemini-x"))
        self.assertEqual(len(inf.prompts), 1)          # sin reintentos ni pasos extra
        self.assertEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT)
        self.assertFalse(res.needs_input)

    async def test_no_expone_error_crudo(self):
        res, _ = await self._run_with(
            Exception("429 RESOURCE_EXHAUSTED: quota exceeded project=secret-proj-123"))
        for raw in ("RESOURCE_EXHAUSTED", "429", "secret-proj-123", "quota exceeded"):
            self.assertNotIn(raw, res.text)
        self.assertIn("límite", res.text)
        self.assertIn(RESTART_NOTE, res.text)
        self.assertNotIn("Resultado para", res.text)
        for entry in res.trace.steps_log:
            self.assertNotIn("RESOURCE_EXHAUSTED", str(entry))

    async def test_sobrecarga_503_y_high_demand(self):
        for msg in ("503 Service Unavailable", "This model is currently experiencing HIGH DEMAND",
                    "overloaded_error 529"):
            res, inf = await self._run_with(Exception(msg))
            self.assertEqual(len(inf.prompts), 1, msg)
            self.assertEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT, msg)

    async def test_all_runtimes_failed_con_detalle_en_runtime_errors(self):
        exc = FakeAllRuntimesFailed({"gemini": "RATE_LIMITED retry in 30s", "ollama": "no url"})
        res, inf = await self._run_with(exc)
        self.assertEqual(len(inf.prompts), 1)
        self.assertEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT)
        self.assertNotIn("RATE_LIMITED", res.text)
        self.assertNotIn("ollama", res.text)

    async def test_error_transitorio_no_de_cuota_sigue_reintentando(self):
        res, inf = await self._run_with(Exception("TIMEOUT contacting upstream"))
        self.assertGreater(len(inf.prompts), 1)
        self.assertNotEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT)

    async def test_cuota_a_mitad_conserva_lo_completado(self):
        def responder(n, _r):
            return _ok(n, _r) if n <= 2 else Exception("RESOURCE_EXHAUSTED")
        inf = FakeInference(responder)
        res = await _loop(inf).run("Investiga el mercado", context="Investiga el mercado")
        self.assertEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT)
        self.assertIn("Resultado parcial (2/", res.text)
        self.assertIn("Resultado del paso 1", res.text)
        self.assertIn("Pasos sin completar", res.text)
        self.assertNotIn("RESOURCE_EXHAUSTED", res.text)


class TestAutonomyNeedsInput(unittest.IsolatedAsyncioTestCase):
    async def test_needs_input_incluye_nota_de_reinicio(self):
        inf = FakeInference(lambda n, r: Exception("necesito saber el presupuesto"))
        res = await _loop(inf).run("Planifica el proyecto", context="Planifica el proyecto")
        self.assertTrue(res.needs_input)
        self.assertEqual(res.trace.status, LoopStatus.NEEDS_INPUT)
        self.assertIn("presupuesto", res.text)
        self.assertIn(RESTART_NOTE, res.text)
        self.assertIn("presupuesto", res.input_question)
        self.assertNotIn(RESTART_NOTE, res.input_question)


class TestAutonomyContinuity(unittest.IsolatedAsyncioTestCase):
    async def test_objetivo_completo_no_truncado_a_200(self):
        goal = "Objetivo " + ("detalle " * 80) + "FIN-DEL-OBJETIVO"
        self.assertGreater(len(goal), 300)
        inf = FakeInference(_ok)
        await _loop(inf).run(goal, context=goal)
        self.assertIn("FIN-DEL-OBJETIVO", inf.prompts[0])

    async def test_paso_recibe_resultados_anteriores(self):
        inf = FakeInference(_ok)
        await _loop(inf).run("Analiza", context="Analiza")
        self.assertNotIn("PASOS ANTERIORES", inf.prompts[0])
        self.assertIn("Resultado del paso 1", inf.prompts[1])
        self.assertIn("Resultado del paso 1", inf.prompts[2])
        self.assertIn("Resultado del paso 2", inf.prompts[2])
        self.assertIn("no instrucciones", inf.prompts[1])

    async def test_resultados_previos_acotados(self):
        big = "X" * 5000
        inf = FakeInference(lambda n, r: big + f" fin{n}")
        await _loop(inf).run("Analiza", context="Analiza")
        for prompt in inf.prompts:
            self.assertLess(len(prompt), 3000 + 2000 + 1200)
        # el ultimo prompt no arrastra los 5000 x 4 caracteres
        self.assertLess(inf.prompts[-1].count("X"), 3100)

    async def test_paso_no_recibe_su_propio_resultado(self):
        inf = FakeInference(_ok)
        await _loop(inf).run("Analiza", context="Analiza")
        self.assertNotIn("Resultado del paso 2", inf.prompts[1])


class TestAutonomyLimitHitAndSynthesis(unittest.IsolatedAsyncioTestCase):
    async def test_limit_hit_se_conserva(self):
        inf = FakeInference(_ok)
        res = await _loop(inf, max_loops=2).run("Investiga", context="Investiga")
        self.assertEqual(res.trace.status, LoopStatus.LIMIT_HIT)
        self.assertIn("límite de ejecución", res.text)
        self.assertIn("Pasos sin completar", res.text)
        self.assertIn(RESTART_NOTE, res.text)
        self.assertNotIn("Resultado para", res.text)

    async def test_plan_que_termina_justo_en_el_limite_es_completado(self):
        inf = FakeInference(_ok)
        # plantilla "task" = 5 pasos; 5 loops exactos
        res = await _loop(inf, max_loops=5).run("Haz algo", context="Haz algo")
        self.assertEqual(res.trace.status, LoopStatus.COMPLETED)
        self.assertIn("Resultado para", res.text)

    async def test_paso_fallido_no_se_presenta_como_exito(self):
        def responder(n, _r):
            return Exception("boom detalle-interno-xyz") if n == 2 else _ok(n, _r)
        inf = FakeInference(responder)
        res = await _loop(inf).run("Analiza", context="Analiza")
        self.assertEqual(res.trace.status, LoopStatus.PARTIAL)
        self.assertNotIn("Resultado para", res.text)
        self.assertIn("Resultado parcial", res.text)
        self.assertIn("Pasos sin completar (1)", res.text)
        self.assertNotIn("detalle-interno-xyz", res.text)
        self.assertIn(RESTART_NOTE, res.text)

    async def test_todo_falla_sin_cuota_es_failed_sin_nota(self):
        inf = FakeInference(lambda n, r: Exception("error raro permanente"))
        res = await _loop(inf).run("Analiza", context="Analiza")
        self.assertEqual(res.trace.status, LoopStatus.FAILED)
        self.assertIn("No pude completar el plan", res.text)
        self.assertNotIn("error raro permanente", res.text)
        self.assertNotIn(RESTART_NOTE, res.text)


class TestNexusTaskState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = TaskStateManager(":memory:")

    def tearDown(self):
        self.tm.close()

    def _core(self, loop):
        return NexusCore(task_state=self.tm, autonomy_loop=loop)

    @staticmethod
    def _fake_loop(status, text="texto", needs_input=False, question=None):
        trace = ExecutionTrace(status=status)

        class _L:
            async def run(self, **kw):
                return AutonomyResult(text=text, trace=trace, plan=None,
                                      needs_input=needs_input, input_question=question)
        return _L()

    async def _run(self, loop, msg="Haz un plan"):
        core = self._core(loop)
        resp = await core._run_autonomy(msg, _intent(), "sys", None, time.perf_counter(), "rid")
        return resp, self.tm.get(resp.task_id)

    async def test_completed_marca_completed(self):
        resp, task = await self._run(self._fake_loop(LoopStatus.COMPLETED))
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    async def test_failed_nunca_queda_completed(self):
        resp, task = await self._run(self._fake_loop(LoopStatus.FAILED, "No pude completar"))
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertNotEqual(task.status, TaskStatus.COMPLETED)

    async def test_partial_limit_y_provider_limit_quedan_in_progress(self):
        for st in (LoopStatus.PARTIAL, LoopStatus.LIMIT_HIT, LoopStatus.PROVIDER_LIMIT):
            resp, task = await self._run(self._fake_loop(st))
            self.assertEqual(task.status, TaskStatus.IN_PROGRESS, st)
            self.assertIn(task.task_id, [t.task_id for t in self.tm.list_pending()])

    async def test_needs_input_queda_in_progress_y_conserva_el_hilo(self):
        loop = self._fake_loop(LoopStatus.NEEDS_INPUT, text="¿Presupuesto?\n\n" + RESTART_NOTE,
                               needs_input=True, question="¿Presupuesto?")
        resp, task = await self._run(loop)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(resp.task_id, task.task_id)
        self.assertIn("¿Presupuesto?", resp.text)
        self.assertIn(RESTART_NOTE, resp.text)

    async def test_estado_desconocido_nunca_se_marca_completed(self):
        trace = SimpleNamespace(status=SimpleNamespace(value="algo_nuevo"), loops=1)

        class _L:
            async def run(self, **kw):
                return SimpleNamespace(text="t", trace=trace, plan=None,
                                       needs_input=False, input_question=None)
        resp, task = await self._run(_L())
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)

    async def test_excepcion_no_expone_error_crudo(self):
        class _L:
            async def run(self, **kw):
                raise RuntimeError("SECRET-TOKEN-123 detalle interno")
        resp, task = await self._run(_L())
        self.assertNotIn("SECRET-TOKEN-123", resp.text)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertNotIn("SECRET-TOKEN-123", task.error or "")
        self.assertEqual(resp.provider, "error")

    async def test_tarea_existente_se_reutiliza_y_se_actualiza(self):
        existing = self.tm.create("tarea previa", intent="task")
        core = self._core(self._fake_loop(LoopStatus.PROVIDER_LIMIT))
        resp = await core._run_autonomy("sigue", _intent(), "sys", None,
                                        time.perf_counter(), "rid",
                                        existing_task_id=existing.task_id)
        self.assertEqual(resp.task_id, existing.task_id)
        self.assertEqual(self.tm.get(existing.task_id).status, TaskStatus.IN_PROGRESS)
        self.assertEqual(self.tm.stats()["total"], 1)

    async def test_extremo_a_extremo_cuota_agotada(self):
        inf = FakeInference(lambda n, r: Exception("429 RESOURCE_EXHAUSTED quota"))
        resp, task = await self._run(_loop(inf), msg="Analiza mi negocio")
        self.assertEqual(len(inf.prompts), 1)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(resp.model, "autonomy:provider_limit")
        self.assertNotIn("RESOURCE_EXHAUSTED", resp.text)
        self.assertIn(RESTART_NOTE, resp.text)

    async def test_extremo_a_extremo_exito(self):
        resp, task = await self._run(_loop(FakeInference(_ok)), msg="Analiza mi negocio")
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(resp.model, "autonomy:completed")


if __name__ == "__main__":
    unittest.main()
