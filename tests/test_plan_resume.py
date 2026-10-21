# -*- coding: utf-8 -*-
"""
Bloque 4B (Continuidad real): "continua" retoma de verdad el plan guardado.

Usa NexusCore.process() real, AutonomyLoop real, Planner local, IntentRouter real y
TaskStateManager real sobre bases temporales. El "proveedor" es un doble que cuenta
llamadas. No necesita red, servidor ni datos de produccion.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.core import plan_checkpoint as pc
from backend.core.autonomy import AutonomyLoop, LoopStatus, RESTART_NOTE
from backend.core.intent import IntentRouter
from backend.core.nexus import NexusCore
from backend.core.planner import Plan, PlanStep, Planner, StepStatus
from backend.core.task_state import TaskStateManager, TaskStatus

GOAL = "Analiza el mercado de tiendas online para una cafeteria"


class Crash(BaseException):
    """Imita una interrupcion brusca (el proceso muere): no la captura ningun except Exception."""


class FakeInference:
    """Doble del proveedor: cuenta llamadas y permite fallar / ejecutar un gancho."""
    def __init__(self, fail_at=None, fail_with="429 RESOURCE_EXHAUSTED quota", hook=None,
                 crash_at=None):
        self.fail_at, self.fail_with, self.hook, self.crash_at = fail_at, fail_with, hook, crash_at
        self.prompts = []

    @property
    def calls(self):
        return len(self.prompts)

    def steps_run(self):
        out = []
        for p in self.prompts:
            marker = "PASO ACTUAL: "
            out.append(p.split(marker, 1)[1].split("\n", 1)[0] if marker in p else p[:60])
        return out

    async def generate(self, request):
        self.prompts.append(request.prompt)
        n = len(self.prompts)
        if self.hook is not None:
            await self.hook(n)
        if self.crash_at is not None and n == self.crash_at:
            raise Crash("proceso interrumpido")
        if self.fail_at is not None and n >= self.fail_at:
            raise Exception(self.fail_with)
        return SimpleNamespace(text=f"Resultado del paso {n}: contenido suficiente y valido.")


def _intent(value="analysis"):
    return SimpleNamespace(intent=SimpleNamespace(value=value),
                           domain=SimpleNamespace(value="general"))


def _core(tm, inference, executor=None):
    loop = AutonomyLoop(planner=Planner(), inference_layer=inference, executor=executor)
    return NexusCore(intent_router=IntentRouter(registry=None), task_state=tm,
                     autonomy_loop=loop, executor=executor)


async def _start(core, goal=GOAL):
    """Lanza un plan (como lo haria process) y devuelve la respuesta."""
    return await core._run_autonomy(goal, _intent(), "sys", None, time.perf_counter(), "rid")


def _snap(tm, task_id):
    t = tm.get(task_id)
    return (t.status, t.result, t.error, t.description, t.updated_at,
            json.dumps(tm.get_metadata(task_id), sort_keys=True))


def _age(tm, task_id, seconds):
    """Envejece updated_at y el latido del checkpoint (simula tiempo transcurrido)."""
    tm._conn.execute("UPDATE tasks SET updated_at=updated_at-? WHERE task_id=?",
                     (seconds, task_id))
    tm._conn.commit()

    def mut(d):
        if isinstance(d.get(pc.PLAN_KEY), dict):
            d[pc.PLAN_KEY]["saved_at"] = float(d[pc.PLAN_KEY]["saved_at"]) - seconds
        return d
    tm.merge_metadata(task_id, mutator=mut)


def _plan_with(steps_spec, goal=GOAL):
    steps = []
    for i, (desc, status, tool, result) in enumerate(steps_spec, 1):
        steps.append(PlanStep(description=desc, step_id=f"s{i}", tool=tool,
                              requires_llm=tool is None, status=status, result=result))
    return Plan(goal=goal, plan_id="plan0001", steps=steps)


def _put_checkpoint(tm, task_id, plan, run_state="paused", reason="partial",
                    goal=GOAL, intent_type="analysis", saved_at=None):
    cp = pc.serialize_plan(plan, goal, intent_type, run_state, reason, run_token="old",
                           now=saved_at if saved_at is not None else time.time())
    tm.merge_metadata(task_id, {pc.PLAN_KEY: cp})
    return cp


def _make_task(tm, desc=GOAL, status=TaskStatus.IN_PROGRESS, result=None, intent="analysis"):
    t = tm.create(desc, intent=intent)
    tm.update(t.task_id, status, result=result)
    return t


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = TaskStateManager(":memory:")
        p = mock.patch("backend.core.nexus.AUTONOMY_ENABLED", True)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        self.tm.close()

    async def _paused(self, fail_at=3, goal=GOAL):
        """Plan pausado por cuota en la llamada `fail_at`; devuelve (task_id, n_pasos)."""
        inf = FakeInference(fail_at=fail_at)
        resp = await _start(_core(self.tm, inf), goal)
        self.assertEqual(resp.model, "autonomy:provider_limit")
        return resp.task_id, resp.plan_steps


# =====================================================================
# Reanudacion real
# =====================================================================
class TestResumeReal(_Base):
    async def test_continua_retoma_desde_el_primer_paso_pendiente(self):
        tid, n = await self._paused(fail_at=3)
        self.assertEqual(self.tm.get(tid).status, TaskStatus.IN_PROGRESS)
        inf2 = FakeInference()
        r = await _core(self.tm, inf2).process("continúa")
        self.assertEqual(r.task_id, tid)
        self.assertEqual(r.provider, "autonomy")
        self.assertEqual(r.model, "autonomy:completed")
        self.assertEqual(r.behavior, "continuar")
        self.assertEqual(inf2.calls, n - 2)               # solo los pasos pendientes
        self.assertEqual(self.tm.get(tid).status, TaskStatus.COMPLETED)
        self.assertEqual(self.tm.stats()["total"], 1)     # misma tarea, no crea otra

    async def test_pasos_terminados_no_se_repiten_y_el_resultado_los_incluye(self):
        tid, n = await self._paused(fail_at=3)
        before = pc.load_checkpoint(self.tm, tid)
        done_desc = [s["description"] for s in before["steps"] if s["status"] == "done"]
        done_res = [s["result"] for s in before["steps"] if s["status"] == "done"]
        self.assertEqual(len(done_desc), 2)
        inf2 = FakeInference()
        r = await _core(self.tm, inf2).process("continúa")
        ran = " | ".join(inf2.steps_run())
        for d in done_desc:
            self.assertNotIn(d, ran)                      # nunca se vuelven a ejecutar
        for res in done_res:
            self.assertIn(res, r.text)                    # su resultado guardado se conserva
        self.assertIn("2 de", r.text)                     # aviso de retoma

    async def test_el_plan_no_se_replanifica_al_reanudar(self):
        tid, n = await self._paused(fail_at=3)
        pid = pc.load_checkpoint(self.tm, tid)["plan_id"]
        r = await _core(self.tm, FakeInference()).process("continúa")
        self.assertEqual(r.plan_id, pid)
        self.assertEqual(r.plan_steps, n)

    async def test_plan_incompleto_nunca_queda_completed(self):
        tid, n = await self._paused(fail_at=3)
        inf2 = FakeInference(fail_at=1)                   # cuota otra vez en el primer paso
        r = await _core(self.tm, inf2).process("continúa")
        self.assertEqual(inf2.calls, 1)                   # se detiene, no insiste
        self.assertEqual(r.model, "autonomy:provider_limit")
        self.assertEqual(self.tm.get(tid).status, TaskStatus.IN_PROGRESS)
        cp = pc.load_checkpoint(self.tm, tid)
        self.assertEqual(sum(1 for s in cp["steps"] if s["status"] == "done"), 2)  # no pierde avance
        # y una tercera vez, con proveedor sano, termina
        r3 = await _core(self.tm, FakeInference()).process("continúa")
        self.assertEqual(r3.model, "autonomy:completed")
        self.assertEqual(self.tm.get(tid).status, TaskStatus.COMPLETED)

    async def test_nota_de_pausa_es_veraz(self):
        inf = FakeInference(fail_at=3)
        resp = await _start(_core(self.tm, inf))
        self.assertIn("retomo desde el primer paso pendiente", resp.text)
        self.assertNotIn(RESTART_NOTE, resp.text)         # ya no dice "se reinicia"
        self.assertIn("2 paso(s) completado(s)", resp.text)

    async def test_sin_checkpoint_reinicia_con_aviso_honesto(self):
        t = _make_task(self.tm, GOAL, result="texto viejo")
        inf = FakeInference()
        r = await _core(self.tm, inf).process("continúa")
        self.assertIn("No tengo un avance guardado válido", r.text)
        self.assertGreaterEqual(inf.calls, 3)             # corrio el plan completo de nuevo
        self.assertEqual(r.task_id, t.task_id)
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.COMPLETED)
        self.assertEqual(self.tm.stats()["total"], 1)

    async def test_checkpoint_corrupto_no_falla_y_reinicia_con_aviso(self):
        for bad in ({"v": 1, "goal": "x"},                              # incompleto
                    {"v": 99, "goal": GOAL, "steps": []},               # version futura
                    "esto no es un dict", [1, 2, 3]):
            with self.subTest(bad=bad):
                tm = TaskStateManager(":memory:")
                try:
                    t = _make_task(tm)
                    tm.merge_metadata(t.task_id, {pc.PLAN_KEY: bad})
                    inf = FakeInference()
                    with mock.patch("backend.core.nexus.AUTONOMY_ENABLED", True):
                        r = await _core(tm, inf).process("continúa")
                    self.assertIn("No tengo un avance guardado válido", r.text)
                    self.assertGreater(inf.calls, 0)
                finally:
                    tm.close()

    async def test_metadata_con_json_invalido_en_la_base(self):
        t = _make_task(self.tm)
        self.tm._conn.execute("UPDATE tasks SET metadata='{no es json' WHERE task_id=?",
                              (t.task_id,))
        self.tm._conn.commit()
        self.assertEqual(self.tm.get_metadata(t.task_id), {})   # lectura segura de 4B
        # LIMITACION PREEXISTENTE (fuera de 4B): TaskStateManager._row() hace json.loads sin
        # proteger, asi que list_pending()/get() fallan con una fila asi. No se modifica aqui.

    async def test_checkpoint_sobrevive_cerrar_y_reabrir_la_base(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            db = str(Path(d) / "tasks_test.db")
            tm = TaskStateManager(db)
            with mock.patch("backend.core.nexus.AUTONOMY_ENABLED", True):
                resp = await _start(_core(tm, FakeInference(fail_at=3)))
                tid, n = resp.task_id, resp.plan_steps
                tm.close()
                tm2 = TaskStateManager(db)
                try:
                    inf2 = FakeInference()
                    r = await _core(tm2, inf2).process("continúa")
                    self.assertEqual(r.model, "autonomy:completed")
                    self.assertEqual(inf2.calls, n - 2)
                    self.assertEqual(tm2.get(tid).status, TaskStatus.COMPLETED)
                finally:
                    tm2.close()


# =====================================================================
# Interrupcion y acciones externas
# =====================================================================
class TestInterruption(_Base):
    async def test_interrupcion_a_mitad_de_paso_de_texto_se_reintenta_al_reanudar(self):
        inf = FakeInference(crash_at=3)
        core = _core(self.tm, inf)
        with self.assertRaises(Crash):
            await _start(core)
        task = self.tm.list_pending()[0]
        cp = pc.load_checkpoint(self.tm, task.task_id)
        self.assertEqual(cp["run_state"], "running")
        self.assertEqual([s["status"] for s in cp["steps"]][:3], ["done", "done", "running"])
        # con latido reciente se considera aun en ejecucion: no se lanza otra copia
        inf2 = FakeInference()
        r = await _core(self.tm, inf2).process("continúa")
        self.assertIn("sigue en ejecución", r.text)
        self.assertEqual(inf2.calls, 0)
        # sin latido (proceso muerto) se reanuda: pasos 1-2 intactos, paso 3 se repite
        _age(self.tm, task.task_id, 1000)
        r = await _core(self.tm, inf2).process("continúa")
        self.assertEqual(r.model, "autonomy:completed")
        self.assertEqual(inf2.calls, len(cp["steps"]) - 2)
        self.assertNotIn(cp["steps"][0]["description"], " | ".join(inf2.steps_run()))

    def _tool_task(self, tool_status=StepStatus.RUNNING, tool="web_search"):
        t = _make_task(self.tm, "Investiga precios en la web")
        plan = _plan_with([
            ("Definir el alcance", StepStatus.DONE, None, "Alcance definido con detalle suficiente."),
            ("Buscar precios en la web", tool_status, tool, None),
            ("Resumir hallazgos", StepStatus.PENDING, None, None)])
        _put_checkpoint(self.tm, t.task_id, plan, run_state="running", reason="",
                        goal="Investiga precios en la web", intent_type="research",
                        saved_at=time.time() - 1000)
        return t

    @staticmethod
    def _executor(risk):
        calls = []

        class _Reg:
            def get(self, name):
                return None if risk is None else SimpleNamespace(
                    risk_level=SimpleNamespace(value=risk))

        class _Ex:
            _registry = _Reg()

            async def execute_by_name(self, tool_name, params, context, request_id):
                calls.append(tool_name)
                return SimpleNamespace(success=True, error=None, tool_result=SimpleNamespace(
                    output="Resultado de la busqueda web con detalles suficientes."))
        ex = _Ex()
        ex.calls = calls
        return ex

    async def test_herramienta_high_a_medias_no_se_repite_sola(self):
        t = self._tool_task()
        ex, inf = self._executor("high"), FakeInference()
        before = _snap(self.tm, t.task_id)
        r = await _core(self.tm, inf, executor=ex).process("continúa")
        self.assertEqual(ex.calls, [])                    # la herramienta NO se ejecuto
        self.assertEqual(inf.calls, 0)
        self.assertIn("de todos modos", r.text)
        self.assertIn("web_search", r.text)
        self.assertEqual(_snap(self.tm, t.task_id), before)   # nada cambio
        # un "si" aislado tampoco la repite
        r2 = await _core(self.tm, inf, executor=ex).process("sí")
        self.assertEqual(ex.calls, [])

    async def test_herramienta_high_se_repite_solo_con_confirmacion_explicita(self):
        t = self._tool_task()
        ex, inf = self._executor("high"), FakeInference()
        r = await _core(self.tm, inf, executor=ex).process("continúa de todos modos")
        self.assertEqual(ex.calls, ["web_search"])        # exactamente una vez
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.COMPLETED)

    async def test_herramienta_desconocida_se_trata_como_no_repetible(self):
        for risk in (None, "medium", "critical"):
            with self.subTest(risk=risk):
                tm = TaskStateManager(":memory:")
                try:
                    self.tm = tm
                    self._tool_task()
                    ex = self._executor(risk)
                    with mock.patch("backend.core.nexus.AUTONOMY_ENABLED", True):
                        r = await _core(tm, FakeInference(), executor=ex).process("continúa")
                    self.assertEqual(ex.calls, [], risk)
                    self.assertIn("de todos modos", r.text)
                finally:
                    tm.close()

    async def test_herramienta_low_a_medias_se_repite_sin_confirmacion(self):
        t = self._tool_task(tool="clock")
        ex = self._executor("low")
        r = await _core(self.tm, FakeInference(), executor=ex).process("continúa")
        self.assertEqual(ex.calls, ["clock"])
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.COMPLETED)

    async def test_herramienta_fallida_no_dudosa_se_reintenta(self):
        t = self._tool_task(tool_status=StepStatus.FAILED)
        ex = self._executor("high")
        await _core(self.tm, FakeInference(), executor=ex).process("continúa")
        self.assertEqual(ex.calls, ["web_search"])        # fallo reportado != efecto dudoso

    async def test_continua_repetido_no_ejecuta_el_plan_dos_veces(self):
        tid, n = await self._paused(fail_at=3)
        replies = []
        holder = {}

        async def hook(call_no):
            if call_no == 1:                               # segundo "continua" en pleno run
                replies.append(await holder["core"].process("continúa"))
        inf = FakeInference(hook=hook)
        holder["core"] = _core(self.tm, inf)
        r = await holder["core"].process("continúa")
        self.assertEqual(inf.calls, n - 2)                 # cada paso pendiente UNA sola vez
        self.assertEqual(len(set(inf.steps_run())), n - 2)
        self.assertIn("sigue en ejecución", replies[0].text)
        self.assertEqual(r.model, "autonomy:completed")


# =====================================================================
# Cancelacion
# =====================================================================
class TestCancelDuringRun(_Base):
    async def test_cancelar_durante_la_ejecucion_detiene_el_plan_sin_pisar_el_estado(self):
        holder, seen = {}, {}

        async def hook(call_no):
            if call_no == 2:
                c = await holder["core"].process("cancela la tarea")
                self.assertEqual(c.model, "cancelled")
                tid = c.task_id
                seen["meta"] = json.dumps(self.tm.get_metadata(tid), sort_keys=True)
        inf = FakeInference(hook=hook)
        holder["core"] = _core(self.tm, inf)
        resp = await _start(holder["core"])
        self.assertEqual(inf.calls, 2)                     # no se ejecuta ningun paso mas
        self.assertEqual(resp.model, "autonomy:cancelled")
        self.assertIn("detenida", resp.text)
        task = self.tm.get(resp.task_id)
        self.assertEqual(task.status, TaskStatus.CANCELLED)   # no vuelve a IN_PROGRESS/COMPLETED
        self.assertEqual(json.dumps(self.tm.get_metadata(task.task_id), sort_keys=True),
                         seen["meta"])                     # el checkpoint no se reescribe

    async def test_cancelada_no_se_reactiva_con_continua_si_ni_ok(self):
        tid, n = await self._paused(fail_at=3)
        core = _core(self.tm, FakeInference())
        c = await core.process("cancela la tarea")
        self.assertEqual(c.model, "cancelled")
        snap = _snap(self.tm, tid)
        total = self.tm.stats()["total"]
        inf = FakeInference()
        core2 = _core(self.tm, inf)
        for msg in ("continúa", "sí", "si", "ok", "dale", "sigue con el plan", "no",
                    "continúa de todos modos", f"continúa {tid}"):
            with self.subTest(msg=msg):
                r = await core2.process(msg)
                self.assertNotEqual(r.provider, "autonomy")
                self.assertEqual(self.tm.get(tid).status, TaskStatus.CANCELLED)
        self.assertEqual(inf.calls, 0)
        self.assertEqual(_snap(self.tm, tid), snap)        # fila y metadata intactas
        self.assertEqual(self.tm.stats()["total"], total)  # no crea tareas nuevas

    async def test_cancel_por_id_de_una_tarea_ya_cancelada_no_la_reactiva_al_reanudar(self):
        tid, n = await self._paused(fail_at=3)
        self.tm.update(tid, TaskStatus.CANCELLED)
        resp = await _core(self.tm, FakeInference())._run_autonomy(
            GOAL, _intent(), "sys", None, time.perf_counter(), "rid", existing_task_id=tid)
        self.assertEqual(resp.model, "inactive")
        self.assertEqual(self.tm.get(tid).status, TaskStatus.CANCELLED)


# =====================================================================
# Definicion de "continua", "si", "no"
# =====================================================================
class TestMessageMatrix(_Base):
    async def test_si_aislado_con_una_candidata_clara_retoma(self):
        tid, n = await self._paused(fail_at=3)
        inf = FakeInference()
        r = await _core(self.tm, inf).process("sí")
        self.assertEqual(r.provider, "autonomy")
        self.assertEqual(inf.calls, n - 2)
        self.assertEqual(self.tm.get(tid).status, TaskStatus.COMPLETED)

    async def test_si_aislado_con_varias_candidatas_pregunta_cual_sin_tocar_nada(self):
        a, _ = await self._paused(fail_at=3, goal="Analiza el mercado de cafeterias en Tijuana")
        b, _ = await self._paused(fail_at=3, goal="Analiza el mercado de panaderias en Mexicali")
        snaps = {x: _snap(self.tm, x) for x in (a, b)}
        inf = FakeInference()
        r = await _core(self.tm, inf).process("sí")
        self.assertEqual(inf.calls, 0)
        self.assertIn(a, r.text)
        self.assertIn(b, r.text)
        self.assertIn("continúa <id>", r.text)
        for x in (a, b):
            self.assertEqual(_snap(self.tm, x), snaps[x])
        # "continua" sin id tambien pregunta
        r = await _core(self.tm, inf).process("continúa")
        self.assertEqual(inf.calls, 0)
        self.assertIn(a, r.text)
        # con id retoma SOLO esa y deja la otra intacta
        r = await _core(self.tm, inf).process(f"continúa {a}")
        self.assertEqual(r.task_id, a)
        self.assertEqual(self.tm.get(a).status, TaskStatus.COMPLETED)
        self.assertEqual(_snap(self.tm, b), snaps[b])

    async def test_id_inexistente_no_ejecuta_nada(self):
        tid, _ = await self._paused(fail_at=3)
        snap = _snap(self.tm, tid)
        inf = FakeInference()
        r = await _core(self.tm, inf).process("continúa 12345678-abc")
        self.assertEqual(inf.calls, 0)
        self.assertIn("No encontré una tarea abierta", r.text)
        self.assertEqual(_snap(self.tm, tid), snap)

    async def test_si_con_pausa_vieja_no_retoma_pero_continua_explicito_si(self):
        tid, n = await self._paused(fail_at=3)
        _age(self.tm, tid, 3600)
        inf = FakeInference()
        r = await _core(self.tm, inf).process("sí")
        self.assertEqual(inf.calls, 0)                     # ambiguo: no arranca nada
        self.assertNotEqual(r.provider, "autonomy")
        self.assertEqual(self.tm.get(tid).status, TaskStatus.IN_PROGRESS)
        r = await _core(self.tm, inf).process("continúa")  # explicito: si retoma
        self.assertEqual(r.provider, "autonomy")
        self.assertEqual(inf.calls, n - 2)

    async def test_si_con_tarea_esperando_datos_no_retoma(self):
        t = _make_task(self.tm, result="¿Cuál es tu presupuesto?\n\nNota: ...")
        plan = _plan_with([("Paso uno", StepStatus.PENDING, None, None)])
        _put_checkpoint(self.tm, t.task_id, plan, run_state="paused", reason="needs_input")
        inf = FakeInference()
        before = _snap(self.tm, t.task_id)
        r = await _core(self.tm, inf).process("sí")
        self.assertEqual(inf.calls, 0)
        r = await _core(self.tm, inf).process("continúa")  # solo repite la pregunta
        self.assertEqual(inf.calls, 0)
        self.assertIn("¿Cuál es tu presupuesto?", r.text)
        self.assertEqual(_snap(self.tm, t.task_id), before)

    async def test_si_sin_checkpoint_no_inicia_una_tarea_ambigua(self):
        t = _make_task(self.tm, result="texto guardado")
        inf = FakeInference()
        r = await _core(self.tm, inf).process("sí")
        self.assertEqual(inf.calls, 0)
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.IN_PROGRESS)

    async def test_no_no_reinicia_ni_cambia_nada(self):
        tid, _ = await self._paused(fail_at=3)
        snap = _snap(self.tm, tid)
        inf = FakeInference()
        r = await _core(self.tm, inf).process("no")
        self.assertEqual(inf.calls, 0)
        self.assertIn("no cambié nada", r.text)
        self.assertIn("continúa", r.text)
        self.assertEqual(_snap(self.tm, tid), snap)

    async def test_sin_tareas_abiertas_no_arranca_ni_crea_nada(self):
        done = _make_task(self.tm, status=TaskStatus.COMPLETED, result="listo")
        failed = _make_task(self.tm, status=TaskStatus.FAILED)
        snaps = {x.task_id: _snap(self.tm, x.task_id) for x in (done, failed)}
        inf = FakeInference()
        for msg in ("sí", "continúa", "no", "ok"):
            await _core(self.tm, inf).process(msg)
        self.assertEqual(inf.calls, 0)
        self.assertEqual(self.tm.stats()["total"], 2)
        for tid, s in snaps.items():
            self.assertEqual(_snap(self.tm, tid), s)

    async def test_dame_el_resultado_sigue_devolviendo_el_texto_guardado(self):
        t = _make_task(self.tm, result="TEXTO-GUARDADO-DEL-PLAN")
        r = await _core(self.tm, FakeInference()).process("dame el resultado")
        self.assertEqual(r.text, "TEXTO-GUARDADO-DEL-PLAN")

    async def test_filas_ajenas_intactas_al_reanudar(self):
        done = _make_task(self.tm, "otra completada", status=TaskStatus.COMPLETED, result="ok")
        failed = _make_task(self.tm, "otra fallida", status=TaskStatus.FAILED)
        canc = _make_task(self.tm, "otra cancelada", status=TaskStatus.CANCELLED)
        others = {x.task_id: _snap(self.tm, x.task_id) for x in (done, failed, canc)}
        tid, _ = await self._paused(fail_at=3)
        total = self.tm.stats()["total"]
        await _core(self.tm, FakeInference()).process("continúa")
        for k, s in others.items():
            self.assertEqual(_snap(self.tm, k), s)
        self.assertEqual(self.tm.stats()["total"], total)


# =====================================================================
# Piezas de bajo nivel
# =====================================================================
class TestTaskStateNewMethods(unittest.TestCase):
    def setUp(self):
        self.tm = TaskStateManager(":memory:")
        self.t = self.tm.create("tarea", intent="task")
        self.tm.update(self.t.task_id, TaskStatus.IN_PROGRESS, result="R")

    def tearDown(self):
        self.tm.close()

    def test_transition_solo_desde_estados_permitidos(self):
        tid = self.t.task_id
        self.tm.update(tid, TaskStatus.CANCELLED)
        ok = self.tm.transition(tid, TaskStatus.IN_PROGRESS,
                                (TaskStatus.PENDING, TaskStatus.IN_PROGRESS))
        self.assertFalse(ok)
        self.assertEqual(self.tm.get(tid).status, TaskStatus.CANCELLED)
        self.assertFalse(self.tm.transition(tid, TaskStatus.COMPLETED, ()))

    def test_transition_conserva_result_y_error_por_defecto(self):
        tid = self.t.task_id
        self.assertTrue(self.tm.transition(tid, TaskStatus.COMPLETED, (TaskStatus.IN_PROGRESS,)))
        t = self.tm.get(tid)
        self.assertEqual((t.status, t.result), (TaskStatus.COMPLETED, "R"))
        self.assertTrue(self.tm.transition(tid, TaskStatus.FAILED, (TaskStatus.COMPLETED,),
                                           result=None, error="x"))
        t = self.tm.get(tid)
        self.assertEqual((t.result, t.error), (None, "x"))

    def test_merge_metadata_no_toca_estado_ni_updated_at(self):
        tid = self.t.task_id
        before = self.tm.get(tid)
        self.assertEqual(self.tm.merge_metadata(tid, {"a": 1}), {"a": 1})
        after = self.tm.get(tid)
        self.assertEqual((after.status, after.result, after.updated_at),
                         (before.status, before.result, before.updated_at))
        self.assertEqual(after.metadata, {"a": 1})
        self.tm.merge_metadata(tid, {"b": 2})
        self.assertEqual(self.tm.get_metadata(tid), {"a": 1, "b": 2})

    def test_merge_metadata_respeta_allowed_status_y_mutator_none(self):
        tid = self.t.task_id
        self.tm.update(tid, TaskStatus.CANCELLED)
        self.assertIsNone(self.tm.merge_metadata(
            tid, {"a": 1}, allowed_status=(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)))
        self.assertEqual(self.tm.get_metadata(tid), {})
        self.assertIsNone(self.tm.merge_metadata(tid, mutator=lambda d: None))
        self.assertIsNone(self.tm.merge_metadata("no-existe", {"a": 1}))

    def test_esquema_y_metodos_existentes_intactos(self):
        cols = [r[1] for r in self.tm._conn.execute("PRAGMA table_info(tasks)")]
        self.assertEqual(cols, ["task_id", "description", "intent", "domain", "status",
                                "created_at", "updated_at", "result", "error", "metadata"])
        tid = self.t.task_id
        self.tm.update(tid, TaskStatus.COMPLETED)           # update() sigue sobrescribiendo
        self.assertIsNone(self.tm.get(tid).result)


class TestCheckpointModule(unittest.TestCase):
    def _cp(self):
        plan = _plan_with([("Uno", StepStatus.DONE, None, "R1"),
                           ("Dos", StepStatus.FAILED, None, None),
                           ("Tres", StepStatus.PENDING, None, None)])
        plan.steps[1].retries = 3
        return pc.serialize_plan(plan, GOAL, "analysis", "paused", "partial", run_token="t")

    def test_ida_y_vuelta_conserva_pasos_terminados(self):
        cp = json.loads(json.dumps(self._cp()))
        plan, uncertain, reset = pc.restore_plan(cp)
        self.assertEqual([s.status for s in plan.steps],
                         [StepStatus.DONE, StepStatus.PENDING, StepStatus.PENDING])
        self.assertEqual(plan.steps[0].result, "R1")
        self.assertEqual((plan.steps[1].retries, uncertain, reset), (0, [], 1))
        self.assertFalse(plan.done)

    def test_validacion_rechaza_estructuras_invalidas(self):
        good = self._cp()
        self.assertTrue(pc.validate_checkpoint(good))
        bad = []
        for mutate in (
            lambda c: c.update(v=2), lambda c: c.update(goal=""), lambda c: c.update(steps=[]),
            lambda c: c.update(steps="x"), lambda c: c["steps"][0].update(status="raro"),
            lambda c: c["steps"][1].update(step_id=c["steps"][0]["step_id"]),
            lambda c: c["steps"][0].update(depends_on=["no-existe"]),
            lambda c: c["steps"][0].update(description=""),
            lambda c: c["steps"][0].update(result=123),
            lambda c: c.update(steps=c["steps"] * 30),
        ):
            c = json.loads(json.dumps(good))
            mutate(c)
            bad.append(c)
        for c in bad + [None, [], "x", 5]:
            self.assertFalse(pc.validate_checkpoint(c))
            self.assertEqual(pc.restore_plan(c), (None, [], 0))

    def test_resultado_muy_largo_se_acota_y_se_marca(self):
        plan = _plan_with([("Uno", StepStatus.DONE, None, "x" * (pc.MAX_STEP_RESULT_CHARS + 500))])
        cp = pc.serialize_plan(plan, GOAL, "analysis", "paused")
        self.assertEqual(len(cp["steps"][0]["result"]), pc.MAX_STEP_RESULT_CHARS)
        self.assertTrue(cp["steps"][0]["result_truncated"])

    def test_tool_is_repeatable_solo_con_riesgo_low_conocido(self):
        def ex(risk):
            reg = SimpleNamespace(get=lambda n: SimpleNamespace(
                risk_level=SimpleNamespace(value=risk)))
            return SimpleNamespace(_registry=reg)
        self.assertTrue(pc.tool_is_repeatable(ex("low"), "t"))
        self.assertTrue(pc.tool_is_repeatable(ex("LOW"), "t"))
        for r in ("medium", "high", "critical", "raro"):
            self.assertFalse(pc.tool_is_repeatable(ex(r), "t"))
        self.assertFalse(pc.tool_is_repeatable(SimpleNamespace(), "t"))     # sin registro
        self.assertFalse(pc.tool_is_repeatable(None, "t"))
        boom = SimpleNamespace(_registry=SimpleNamespace(
            get=lambda n: (_ for _ in ()).throw(RuntimeError("x"))))
        self.assertFalse(pc.tool_is_repeatable(boom, "t"))


class TestClaim(unittest.TestCase):
    def setUp(self):
        self.tm = TaskStateManager(":memory:")
        self.t = _make_task(self.tm)

    def tearDown(self):
        self.tm.close()

    def _store(self):
        return pc.CheckpointStore(self.tm, self.t.task_id, GOAL, "analysis")

    def test_solo_un_run_vivo_puede_reclamar_la_tarea(self):
        a, b = self._store(), self._store()
        self.assertTrue(a.claim())
        self.assertFalse(b.claim())
        self.assertEqual(b.claim_reason, "running")
        self.assertTrue(a.claim())                      # el mismo dueno puede renovar

    def test_run_sin_latido_se_considera_interrumpido_y_se_puede_reclamar(self):
        a, b = self._store(), self._store()
        self.assertTrue(a.claim())
        _age(self.tm, self.t.task_id, pc.RUNNING_STALE_SECONDS + 5)
        self.assertTrue(b.claim())

    def test_tarea_inactiva_no_se_reclama(self):
        self.tm.update(self.t.task_id, TaskStatus.CANCELLED)
        s = self._store()
        self.assertFalse(s.claim())
        self.assertEqual(s.claim_reason, "inactive")
        self.assertEqual(self.tm.get_metadata(self.t.task_id), {})

    def test_otro_run_vivo_no_pisa_el_checkpoint(self):
        a, b = self._store(), self._store()
        plan = _plan_with([("Uno", StepStatus.PENDING, None, None)])
        self.assertTrue(a.save(plan, "running"))
        self.assertFalse(b.save(plan, "running"))


class TestLoopWithPlan(unittest.IsolatedAsyncioTestCase):
    async def test_run_con_plan_recibido_no_llama_al_planner(self):
        class NoPlanner:
            async def plan(self, **kw):
                raise AssertionError("no debe replanificar")
        plan = _plan_with([("Uno", StepStatus.DONE, None, "R1 resultado previo suficiente."),
                           ("Dos", StepStatus.PENDING, None, None)])
        inf = FakeInference()
        res = await AutonomyLoop(planner=NoPlanner(), inference_layer=inf).run(
            goal=GOAL, intent_type="analysis", context=GOAL, plan=plan)
        self.assertEqual(inf.calls, 1)
        self.assertEqual(res.trace.status, LoopStatus.COMPLETED)
        self.assertIn("R1 resultado previo", res.text)
        self.assertIn("R1 resultado previo", inf.prompts[0])   # el paso ve lo anterior

    async def test_fallo_al_guardar_checkpoint_no_interrumpe_el_plan(self):
        class BadStore:
            def save(self, *a, **k):
                raise RuntimeError("disco lleno")

            def should_stop(self):
                raise RuntimeError("x")
        inf = FakeInference()
        res = await AutonomyLoop(planner=Planner(), inference_layer=inf).run(
            goal=GOAL, intent_type="analysis", context=GOAL, checkpoint=BadStore())
        self.assertEqual(res.trace.status, LoopStatus.COMPLETED)

    async def test_sin_checkpoint_el_comportamiento_4a_se_conserva(self):
        inf = FakeInference(fail_at=2)
        res = await AutonomyLoop(planner=Planner(), inference_layer=inf).run(
            goal=GOAL, intent_type="analysis", context=GOAL)
        self.assertEqual(res.trace.status, LoopStatus.PROVIDER_LIMIT)
        self.assertIn(RESTART_NOTE, res.text)               # sin almacen: "reinicia" es verdad


if __name__ == "__main__":
    unittest.main()
