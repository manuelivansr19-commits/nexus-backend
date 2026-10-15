# -*- coding: utf-8 -*-
"""
Pruebas de INTEGRACION de la Memoria Persistente con NexusCore, ContextManager,
TaskStateManager y el endpoint /api/nexus/memory/clear.

Colocar en tests/ y correr:
    python -m pytest tests/test_persistent_memory_integration.py -v

Demuestra, con componentes reales (Memory + SQLiteMemoryStore, ContextManager,
TaskStateManager, NexusCore), que los comandos de memoria CONSERVAN task_id, la
tarea activa y el contexto; que la memoria persiste al reiniciar; que respeta el
aislamiento por proyecto; que es opcional y se degrada de forma segura; y que
/memory/clear nunca borra por una entrada invalida.
"""
import os
import shutil
import tempfile
import unittest

from backend.core.context import ContextManager
from backend.core.intent import Domain, IntentResult, IntentStrategy, IntentType
from backend.core.memory import Memory, MemoryType, RAMMemoryStore, SQLiteMemoryStore
from backend.core.nexus import NexusCore
from backend.core.persistent_memory import PersistentMemory
from backend.core.task_state import TaskStateManager, TaskStatus


class StaticIntentRouter:
    def route(self, message):
        return IntentResult(intent=IntentType.CHAT, domain=Domain.GENERAL,
                            confidence=0.5, strategy=IntentStrategy.LLM)


class CapturingInference:
    def __init__(self):
        self.prompts = []

    @property
    def calls(self):
        return len(self.prompts)

    async def generate(self, request):
        self.prompts.append(request.prompt)

        class R:
            text = "respuesta normal del LLM"
            runtime_name = "fake"
            model = "fake-model"
            fallback_used = False
            is_local = True
        return R()


class RotoStore:
    def __getattr__(self, name):
        def boom(*a, **k):
            raise RuntimeError("almacen caido")
        return boom


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "mem.db")
        self._closers = []
        self.addCleanup(self._close_all)

    def _close_all(self):
        for c in self._closers:
            try:
                c.close()
            except Exception:
                pass

    def build(self, persistent=True, pm_store=None, with_tasks=True):
        store = SQLiteMemoryStore(self.db)
        self._closers.append(store)
        memory = Memory(store)
        pm = PersistentMemory(pm_store or store) if persistent else None
        ts = None
        if with_tasks:
            ts = TaskStateManager(os.path.join(self.tmp, "tasks.db"))
            self._closers.append(ts)
        inference = CapturingInference()
        core = NexusCore(inference=inference, memory=memory,
                         intent_router=StaticIntentRouter(),
                         context_manager=ContextManager(memory=memory, persistent_memory=pm),
                         task_state=ts, persistent_memory=pm)
        return core, inference, store, pm, ts

    def active_task(self, ts, description="Analiza el proyecto Aura"):
        task = ts.create(description=description, intent="analysis", domain="general")
        ts.update(task.task_id, TaskStatus.IN_PROGRESS)
        return task


class TestComandosConservanTarea(Base):

    async def test_guardar_conserva_task_id_tarea_y_no_toca_el_llm(self):
        core, inf, store, pm, ts = self.build()
        task = self.active_task(ts)
        antes = ts.stats()
        resp = await core.process("recuerda que el presupuesto de Aura es de 5000 dólares", project="aura")
        self.assertEqual(resp.provider, "memory")
        self.assertEqual(resp.intent, "memory_command")
        self.assertEqual(resp.task_id, task.task_id)
        self.assertEqual(inf.calls, 0)
        self.assertEqual(ts.get(task.task_id).status, TaskStatus.IN_PROGRESS)
        self.assertEqual(ts.stats(), antes)                       # no se creo ni cambio ninguna tarea
        self.assertEqual([t.task_id for t in ts.list_pending()], [task.task_id])
        self.assertNotIn("conversation", store.stats()["by_type"])  # el comando no se guarda como conversacion
        # el flujo de tareas sigue intacto despues del comando
        sig = await core.process("ok")
        self.assertEqual((sig.provider, sig.task_id), ("task_state", task.task_id))

    async def test_todos_los_comandos_conservan_task_id(self):
        core, inf, store, pm, ts = self.build()
        task = self.active_task(ts)
        r = await core.process("recuerda que prefiero respuestas breves", project="aura")
        rec_id = pm.list_records("aura")[0].entry_id
        mensajes = [
            "¿qué recuerdas?",
            f"corrige el recuerdo {rec_id}: prefiero respuestas detalladas",
            "recuerda mi cumpleaños es el 5 de mayo",       # pregunta
            "no",                                            # cancela
            "olvida de tu memoria lo de respuestas",         # pregunta
            "sí",                                            # borra
            "recuerda que el servidor está en Tijuana",      # guarda otro (para poder pedir 'olvida todo')
            "olvida todo",
            "no",
        ]
        for m in mensajes:
            resp = await core.process(m, project="aura")
            self.assertEqual(resp.provider, "memory", m)
            self.assertEqual(resp.task_id, task.task_id, m)
        self.assertEqual(ts.get(task.task_id).status, TaskStatus.IN_PROGRESS)
        self.assertEqual(ts.stats()["total"], 1)
        self.assertEqual(inf.calls, 0)

    async def test_el_si_de_una_propuesta_no_se_confunde_con_el_de_la_tarea(self):
        core, inf, store, pm, ts = self.build()
        task = self.active_task(ts)
        pregunta = await core.process("recuerda mi cumpleaños es el 5 de mayo")
        self.assertEqual(pregunta.provider, "memory")
        self.assertEqual(pm.list_records(), [])
        confirm = await core.process("sí")
        self.assertEqual(confirm.provider, "memory")           # NO es FOLLOW_UP de la tarea
        self.assertIn("Guardado", confirm.text)
        self.assertEqual(len(pm.list_records()), 1)
        self.assertEqual(confirm.task_id, task.task_id)
        otro = await core.process("sí")                         # ya no hay propuesta: vuelve al flujo de tarea
        self.assertEqual((otro.provider, otro.task_id), ("task_state", task.task_id))

    async def test_cambio_de_intencion_conserva_task_id_y_usa_la_memoria(self):
        core, inf, store, pm, ts = self.build()
        task = self.active_task(ts)
        await core.process("recuerda que el presupuesto de Aura es de 5000 dólares", project="aura")
        resp = await core.process("ahora dime cuál es el presupuesto de Aura", project="aura")
        self.assertEqual(resp.provider, "fake")
        self.assertEqual(resp.task_id, task.task_id)             # continuidad de tarea intacta
        prompt = inf.prompts[-1]
        self.assertIn("[Contexto de tarea activa: Analiza el proyecto Aura]", prompt)   # contexto existente
        self.assertIn("[MEMORIA PERSISTENTE", prompt)
        self.assertIn("5000", prompt)
        self.assertIn("[MENSAJE]", prompt)

    async def test_mensaje_sin_relacion_no_inyecta_memoria(self):
        core, inf, store, pm, ts = self.build()
        await core.process("recuerda que el presupuesto de Aura es de 5000 dólares", project="aura")
        await core.process("cuéntame un chiste de gatos", project="aura")
        self.assertNotIn("[MEMORIA PERSISTENTE", inf.prompts[-1])


class TestPersistenciaYAislamiento(Base):

    async def test_persiste_tras_reiniciar(self):
        core1, inf1, store1, pm1, ts1 = self.build(with_tasks=False)
        await core1.process("recuerda que el presupuesto de Aura es de 5000 dólares", project="aura")
        store1.close()
        core2, inf2, store2, pm2, ts2 = self.build(with_tasks=False)    # "reinicio": todo nuevo, mismo archivo
        await core2.process("¿cuál es el presupuesto de Aura?", project="aura")
        self.assertIn("5000", inf2.prompts[-1])

    async def test_aislamiento_por_proyecto_de_extremo_a_extremo(self):
        core, inf, store, pm, ts = self.build(with_tasks=False)
        await core.process("recuerda que el presupuesto de Aura es de 5000 dólares", project="aura")
        await core.process("¿cuál es el presupuesto?", project="otro")
        self.assertNotIn("5000", inf.prompts[-1])
        await core.process("¿cuál es el presupuesto?", project="aura")
        self.assertIn("5000", inf.prompts[-1])
        rec_id = pm.list_records("aura")[0].entry_id
        r = await core.process(f"olvida el recuerdo {rec_id}", project="otro")
        self.assertIn("No existe", r.text)
        self.assertEqual(len(pm.list_records("aura")), 1)

    async def test_corregir_y_olvidar_de_extremo_a_extremo(self):
        core, inf, store, pm, ts = self.build(with_tasks=False)
        await core.process("recuerda que el codigo antiguo es zafiro", project="x")
        rec_id = pm.list_records("x")[0].entry_id
        await core.process(f"corrige el recuerdo {rec_id}: el codigo nuevo es esmeralda", project="x")
        store.close()
        core2, inf2, store2, pm2, _ = self.build(with_tasks=False)
        await core2.process("¿cuál es el codigo?", project="x")
        self.assertIn("esmeralda", inf2.prompts[-1])
        self.assertNotIn("zafiro", inf2.prompts[-1])
        self.assertEqual(store2.search("zafiro"), [])
        new_id = pm2.list_records("x")[0].entry_id
        await core2.process(f"olvida el recuerdo {new_id}", project="x")
        await core2.process("¿cuál es el codigo?", project="x")
        self.assertNotIn("esmeralda", inf2.prompts[-1])
        self.assertEqual(store2.search("esmeralda"), [])

    async def test_el_comando_olvidado_no_queda_en_la_memoria_de_conversacion(self):
        core, inf, store, pm, ts = self.build(with_tasks=False)
        await core.process("recuerda que mi clave es zafiro", project="x")
        await core.process("olvida de tu memoria lo de la clave", project="x")
        await core.process("sí", project="x")
        self.assertEqual(store.search("zafiro"), [])
        self.assertEqual(pm.list_records("x"), [])


class TestOpcionalYDegradacion(Base):

    async def test_sin_memoria_persistente_todo_sigue_igual(self):
        core, inf, store, pm, ts = self.build(persistent=False, with_tasks=False)
        resp = await core.process("recuerda que el presupuesto de Aura es de 5000 dólares")
        self.assertEqual(resp.provider, "fake")                  # lo atiende el LLM como antes
        self.assertEqual(inf.calls, 1)
        self.assertEqual(store.stats()["by_type"].get("semantic", 0), 0)

    async def test_sin_memoria_persistente_assemble_no_recibe_task_id(self):
        visto = {}

        class CM:
            def assemble(self, **kwargs):
                visto.update(kwargs)
                raise RuntimeError("forzar fallback")      # el fallback existente sigue funcionando

        inf = CapturingInference()
        core = NexusCore(inference=inf, intent_router=StaticIntentRouter(), context_manager=CM())
        await core.process("hola")
        self.assertNotIn("task_id", visto)
        self.assertEqual(inf.calls, 1)

    async def test_almacen_roto_responde_honesto_y_no_tumba_el_chat(self):
        core, inf, store, pm, ts = self.build(pm_store=RotoStore(), with_tasks=False)
        resp = await core.process("recuerda que el servidor está en Tijuana")
        self.assertEqual(resp.provider, "memory")
        self.assertIn("No pude", resp.text)
        self.assertIn("No se realizó ningún cambio", resp.text)
        self.assertEqual(inf.calls, 0)
        normal = await core.process("cuéntame un chiste")
        self.assertEqual(normal.provider, "fake")                # el chat normal sigue funcionando
        self.assertNotIn("[MEMORIA PERSISTENTE", inf.prompts[-1])

    async def test_si_la_memoria_persistente_lanza_el_mensaje_sigue_su_curso(self):
        class Rota:
            def handle_message(self, *a, **k):
                raise RuntimeError("bug inesperado")

            def context_for(self, *a, **k):
                raise RuntimeError("bug inesperado")

        inf = CapturingInference()
        core = NexusCore(inference=inf, intent_router=StaticIntentRouter(),
                         context_manager=ContextManager(persistent_memory=Rota()),
                         persistent_memory=Rota())
        resp = await core.process("hola, ¿cómo estás?")
        self.assertEqual(resp.provider, "fake")
        self.assertEqual(inf.calls, 1)

    async def test_firmas_antiguas_siguen_funcionando(self):
        NexusCore()
        cm = ContextManager(memory=None)
        bundle = cm.assemble(message="hola", system_prompt="sys")
        self.assertEqual(bundle.sources_used, [])


class TestContextManager(unittest.TestCase):

    class PM:
        def __init__(self, text="[MEMORIA PERSISTENTE — x]\n• (hecho) dato"):
            self.text, self.args = text, None

        def context_for(self, message, project="", task_id=None, **kw):
            self.args = (message, project, task_id)
            return self.text

    def test_agrega_seccion_y_fuente(self):
        pm = self.PM()
        b = ContextManager(persistent_memory=pm).assemble(
            message="hola", system_prompt="sys", project="aura", task_id="T1")
        self.assertIn("persistent_memory", b.sources_used)
        self.assertIn("• (hecho) dato", b.assembled_prompt)
        self.assertEqual(pm.args, ("hola", "aura", "T1"))

    def test_sin_memoria_persistente_el_resultado_es_identico(self):
        a = ContextManager().assemble(message="hola", system_prompt="sys")
        b = ContextManager(persistent_memory=self.PM(text="")).assemble(message="hola", system_prompt="sys")
        self.assertEqual((a.assembled_prompt, a.sources_used), (b.assembled_prompt, b.sources_used))

    def test_presupuesto_bajo_la_omite(self):
        b = ContextManager(persistent_memory=self.PM(), token_limit=300).assemble(
            message="hola", system_prompt="sys")
        self.assertNotIn("persistent_memory", b.sources_used)

    def test_falla_del_almacen_no_rompe_assemble(self):
        class Rota:
            def context_for(self, *a, **k):
                raise RuntimeError("x")
        b = ContextManager(persistent_memory=Rota()).assemble(message="hola", system_prompt="sys")
        self.assertEqual(b.assembled_prompt, "hola")


class TestMemoryClearEndpoint(Base):
    """/api/nexus/memory/clear: un MemoryType invalido jamas puede borrar datos."""

    def setUp(self):
        super().setUp()
        try:
            import backend.main as main
        except Exception as exc:                                  # pragma: no cover
            self.skipTest(f"no se pudo importar backend.main: {exc!r}")
        self.m = main
        self.addCleanup(setattr, main, "nexus_core", main.nexus_core)
        self.store = SQLiteMemoryStore(self.db)
        self._closers.append(self.store)
        self.memory = Memory(self.store)
        self.pm = PersistentMemory(self.store)
        self.core = NexusCore(memory=self.memory, persistent_memory=self.pm)
        main.nexus_core = self.core
        for t in MemoryType:
            if t != MemoryType.SEMANTIC:
                self.memory.remember(f"dato de tipo {t.value}", t)
        self.pm.remember("dato persistente autorizado", scope="aura")
        self.total = self.store.stats()["total"]

    def _fts_ok(self):
        a = {r[0] for r in self.store._conn.execute("SELECT entry_id FROM memories")}
        b = [r[0] for r in self.store._conn.execute("SELECT entry_id FROM memories_fts")]
        return a == set(b)

    async def test_tipos_invalidos_responden_400_y_no_borran_nada(self):
        invalidos = ["foo", "", None, 123, "WORKING", " working", "all", "*", ["x"], {"a": 1}, True]
        for bad in invalidos:
            with self.subTest(tipo=bad):
                with self.assertRaises(self.m.HTTPException) as cm:
                    await self.m.nexus_memory_clear({"type": bad})
                self.assertEqual(cm.exception.status_code, 400)
                self.assertEqual(self.store.stats()["total"], self.total)
                self.assertTrue(self._fts_ok())

    async def test_cuerpo_que_no_es_un_dict_tampoco_borra(self):
        for bad in (None, [], "working", 5):
            with self.subTest(body=bad):
                with self.assertRaises(self.m.HTTPException):
                    await self.m.nexus_memory_clear(bad)
                self.assertEqual(self.store.stats()["total"], self.total)

    async def test_sin_tipo_sigue_limpiando_solo_working(self):
        r = await self.m.nexus_memory_clear({})
        self.assertEqual((r["success"], r["cleared"]), (True, 1))
        by = self.store.stats()["by_type"]
        self.assertNotIn("working", by)
        self.assertEqual(self.store.stats()["total"], self.total - 1)

    async def test_tipo_valido_borra_solo_ese_tipo(self):
        r = await self.m.nexus_memory_clear({"type": "conversation"})
        self.assertEqual(r["cleared"], 1)
        self.assertEqual(self.store.stats()["total"], self.total - 1)
        self.assertTrue(self._fts_ok())

    async def test_semantic_exige_confirmacion_si_hay_memoria_persistente(self):
        for body in ({"type": "semantic"}, {"type": "semantic", "confirm": "true"},
                     {"type": "semantic", "confirm": 1}, {"type": "semantic", "confirm": False}):
            with self.subTest(body=body):
                with self.assertRaises(self.m.HTTPException) as cm:
                    await self.m.nexus_memory_clear(body)
                self.assertEqual(cm.exception.status_code, 409)
                self.assertEqual(self.store.stats()["total"], self.total)
        r = await self.m.nexus_memory_clear({"type": "semantic", "confirm": True})
        self.assertEqual(r["cleared"], 1)
        self.assertEqual(self.pm.list_records("aura"), [])
        self.assertTrue(self._fts_ok())

    async def test_semantic_sin_memoria_persistente_conserva_el_comportamiento_anterior(self):
        self.core._persistent_memory = None
        r = await self.m.nexus_memory_clear({"type": "semantic"})
        self.assertTrue(r["success"])

    async def test_endpoints_de_memoria_persistente(self):
        lst = await self.m.nexus_persistent_memory_list(project="aura")
        self.assertTrue(lst["success"])
        self.assertEqual([x["text"] for x in lst["memories"]], ["dato persistente autorizado"])
        otro = await self.m.nexus_persistent_memory_list(project="otro")
        self.assertEqual(otro["memories"], [])
        entry_id = lst["memories"][0]["entry_id"]
        r = await self.m.nexus_persistent_memory_delete(entry_id, project="otro")
        self.assertFalse(r["success"])                              # otro proyecto: no borra
        self.assertEqual(len(self.pm.list_records("aura")), 1)
        r = await self.m.nexus_persistent_memory_delete(entry_id, project="aura")
        self.assertTrue(r["success"])
        self.assertEqual(self.pm.list_records("aura"), [])
        self.assertTrue(self._fts_ok())

    async def test_endpoints_con_memoria_persistente_desactivada(self):
        self.core._persistent_memory = None
        self.assertFalse((await self.m.nexus_persistent_memory_list(project="aura"))["success"])
        self.assertFalse((await self.m.nexus_persistent_memory_delete("pm-12345678"))["success"])

    async def test_endpoints_con_almacen_roto_no_lanzan(self):
        self.core._persistent_memory = PersistentMemory(RotoStore())
        self.assertFalse((await self.m.nexus_persistent_memory_list(project="aura"))["success"])
        self.assertFalse((await self.m.nexus_persistent_memory_delete("pm-12345678"))["success"])


if __name__ == "__main__":
    unittest.main()
