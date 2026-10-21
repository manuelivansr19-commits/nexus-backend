# -*- coding: utf-8 -*-
"""
Cancelacion de tareas: verifica que "cancela la tarea" CAMBIA el estado de la
tarea en SQLite (CANCELLED) y no se limita a responder un texto.

Usa NexusCore.process() real con IntentRouter y TaskStateManager reales.
Solo lectura/escritura sobre bases temporales; no toca datos de produccion.
No modifica codigo de produccion.
"""
import tempfile
import unittest
from pathlib import Path

from backend.core.intent import IntentRouter
from backend.core.nexus import Behavior, NexusCore
from backend.core.task_state import TaskStateManager, TaskStatus


def _core(tm, **kw):
    return NexusCore(intent_router=IntentRouter(registry=None), task_state=tm, **kw)


def _make(tm, desc, status=TaskStatus.IN_PROGRESS, result=None, created_at=None):
    t = tm.create(desc, intent="task")
    tm.update(t.task_id, status, result=result)
    if created_at is not None:   # orden determinista (relojes con poca resolucion)
        tm._conn.execute("UPDATE tasks SET created_at=? WHERE task_id=?",
                         (created_at, t.task_id))
        tm._conn.commit()
    return t


def _snap(tm, task_id):
    t = tm.get(task_id)
    return (t.status, t.result, t.error, t.description, t.updated_at)


class _PMSinComando:
    """Memoria persistente activa que NO interpreta el mensaje como comando."""
    def handle_message(self, message, project="", task_id=None):
        return None


class TestCancelCambiaElEstado(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = TaskStateManager(":memory:")

    def tearDown(self):
        self.tm.close()

    async def test_cancela_la_tarea_pasa_a_cancelled(self):
        t = _make(self.tm, "Planifica el lanzamiento de una tienda", result="Resultado parcial")
        r = await _core(self.tm).process("cancela la tarea")
        self.assertEqual(r.provider, "task_state")
        self.assertEqual(r.model, "cancelled")
        self.assertEqual(r.behavior, Behavior.DETENER.value)
        self.assertEqual(r.task_id, t.task_id)
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.CANCELLED)
        self.assertEqual(self.tm.list_pending(), [])

    async def test_todas_las_frases_reconocidas_cancelan(self):
        for phrase in ("cancela la tarea", "Cancela la tarea!", "detén la tarea",
                       "detente la tarea", "para la tarea", "detener tarea",
                       "cancelar tarea"):
            with self.subTest(phrase=phrase):
                t = _make(self.tm, f"tarea para {phrase}")
                r = await _core(self.tm).process(phrase)
                self.assertEqual(r.model, "cancelled", phrase)
                self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.CANCELLED, phrase)

    async def test_pending_tambien_se_cancela(self):
        t = _make(self.tm, "tarea pendiente", status=TaskStatus.PENDING)
        await _core(self.tm).process("cancela la tarea")
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.CANCELLED)

    async def test_funciona_con_memoria_persistente_activa(self):
        t = _make(self.tm, "tarea con memoria activa")
        core = _core(self.tm, persistent_memory=_PMSinComando())
        r = await core.process("cancela la tarea")
        self.assertEqual(r.model, "cancelled")
        self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.CANCELLED)

    async def test_solo_cancela_la_mas_reciente_una_por_mensaje(self):
        old = _make(self.tm, "tarea vieja", created_at=1000.0)
        new = _make(self.tm, "tarea nueva", created_at=2000.0)
        core = _core(self.tm)
        await core.process("cancela la tarea")
        self.assertEqual(self.tm.get(new.task_id).status, TaskStatus.CANCELLED)
        self.assertEqual(self.tm.get(old.task_id).status, TaskStatus.IN_PROGRESS)
        await core.process("cancela la tarea")
        self.assertEqual(self.tm.get(old.task_id).status, TaskStatus.CANCELLED)

    async def test_no_modifica_ninguna_otra_fila(self):
        done = _make(self.tm, "ya terminada", status=TaskStatus.COMPLETED, result="listo",
                     created_at=500.0)
        failed = _make(self.tm, "fallida", status=TaskStatus.FAILED, result="x",
                       created_at=600.0)
        cancelled = _make(self.tm, "ya cancelada", status=TaskStatus.CANCELLED,
                          created_at=700.0)
        other = _make(self.tm, "otra en curso", created_at=1000.0)
        target = _make(self.tm, "la activa", created_at=2000.0)
        before = {t.task_id: _snap(self.tm, t.task_id)
                  for t in (done, failed, cancelled, other)}
        total_before = self.tm.stats()["total"]
        await _core(self.tm).process("cancela la tarea")
        self.assertEqual(self.tm.get(target.task_id).status, TaskStatus.CANCELLED)
        for tid, snap in before.items():
            self.assertEqual(_snap(self.tm, tid), snap)      # filas intactas
        self.assertEqual(self.tm.stats()["total"], total_before)   # no crea ni borra

    async def test_sin_tarea_pendiente_no_afirma_haber_detenido(self):
        done = _make(self.tm, "solo completada", status=TaskStatus.COMPLETED, result="listo")
        before = _snap(self.tm, done.task_id)
        r = await _core(self.tm).process("cancela la tarea")
        self.assertNotEqual(r.provider, "task_state")
        self.assertNotIn("detenida", r.text)
        self.assertEqual(_snap(self.tm, done.task_id), before)
        self.assertEqual(self.tm.stats()["total"], 1)

    async def test_tras_cancelar_ya_no_responde_con_el_resultado_guardado(self):
        _make(self.tm, "plan parcial", result="TEXTO-PARCIAL-GUARDADO")
        core = _core(self.tm)
        await core.process("cancela la tarea")
        r = await core.process("ok")
        self.assertNotIn("TEXTO-PARCIAL-GUARDADO", r.text)

    async def test_alcance_actual_frases_no_reconocidas_no_cancelan(self):
        # Documenta el alcance ACTUAL (frases exactas). Si se amplian los
        # disparadores en otro bloque, actualizar esta prueba.
        t = _make(self.tm, "tarea en curso")
        for phrase in ("para nexus", "cancela eso", "detén todo"):
            with self.subTest(phrase=phrase):
                r = await _core(self.tm).process(phrase)
                self.assertNotEqual(r.model, "cancelled")
                self.assertEqual(self.tm.get(t.task_id).status, TaskStatus.IN_PROGRESS)


class TestCancelPersisteEnDisco(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_sobrevive_cerrar_y_reabrir_la_base(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            db = str(Path(d) / "tasks_test.db")
            tm = TaskStateManager(db)
            t = _make(tm, "tarea persistente", result="parcial")
            other = _make(tm, "otra tarea", status=TaskStatus.COMPLETED, result="ok",
                          created_at=1.0)
            await _core(tm).process("cancela la tarea")
            tm.close()
            tm2 = TaskStateManager(db)
            try:
                self.assertEqual(tm2.get(t.task_id).status, TaskStatus.CANCELLED)
                self.assertEqual(tm2.get(other.task_id).status, TaskStatus.COMPLETED)
                self.assertEqual(tm2.list_pending(), [])
            finally:
                tm2.close()


if __name__ == "__main__":
    unittest.main()
