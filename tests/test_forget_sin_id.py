# -*- coding: utf-8 -*-
"""
Memoria persistente: "olvida el recuerdo" SIN id ni tema.

Reglas verificadas: pide aclaracion cuando hay varios candidatos, confirma cuando hay uno,
avisa cuando no hay ninguno, NUNCA borra por aproximacion y NUNCA hace una limpieza global.
Usa PersistentMemory y NexusCore reales sobre almacenes en memoria / SQLite temporal.
"""
import os
import shutil
import tempfile
import unittest

from backend.core.intent import IntentRouter
from backend.core.memory import RAMMemoryStore, SQLiteMemoryStore
from backend.core.nexus import NexusCore
from backend.core.persistent_memory import PersistentMemory
from backend.core.task_state import TaskStateManager


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        self.t += 1.0          # cada llamada avanza: orden determinista por updated_at
        return self.t

    def jump(self, s):
        self.t += s


def _pm(store=None, clock=None, **kw):
    return PersistentMemory(store or RAMMemoryStore(), clock=clock or Clock(), **kw)


def _snap(pm, scope=""):
    return sorted((r.entry_id, r.text, r.kind, r.scope) for r in pm.list_records(scope, limit=1000))


def _ids(text):
    import re
    return re.findall(r"pm-[0-9a-f]{8}", text)


NOREF_PHRASES = ("olvida el recuerdo", "Olvida el recuerdo.", "borra ese recuerdo",
                 "elimina el recuerdo!", "quita este recuerdo", "olvida un recuerdo",
                 "por favor olvida el recuerdo", "nexus, borra el recuerdo",
                 "olvida de tu memoria el recuerdo")


class TestSinRecuerdos(unittest.TestCase):
    def test_sin_recuerdos_avisa_y_no_hay_nada_que_borrar(self):
        pm = _pm()
        r = pm.handle_message("olvida el recuerdo")
        self.assertIsNotNone(r)                           # ya no cae al flujo normal
        self.assertIn("No tengo recuerdos guardados", r)
        self.assertEqual(_snap(pm), [])
        self.assertEqual(pm._pending, {})


class TestUnSoloRecuerdo(unittest.TestCase):
    def setUp(self):
        self.pm = _pm()
        self.rec = self.pm.remember("mi perro se llama Max")

    def test_pide_confirmacion_y_no_borra_solo(self):
        r = self.pm.handle_message("olvida el recuerdo")
        self.assertIn(self.rec.entry_id, r)
        self.assertIn("Todavía no he borrado nada", r)
        self.assertEqual(len(_snap(self.pm)), 1)

    def test_no_conserva_el_recuerdo(self):
        self.pm.handle_message("olvida el recuerdo")
        r = self.pm.handle_message("no")
        self.assertIn("no hice ningún cambio", r)
        self.assertEqual(len(_snap(self.pm)), 1)

    def test_si_lo_borra_exactamente_ese(self):
        self.pm.handle_message("olvida el recuerdo")
        r = self.pm.handle_message("sí")
        self.assertIn("lo olvidé", r)
        self.assertEqual(_snap(self.pm), [])

    def test_confirmacion_vence_y_un_si_tardio_no_borra(self):
        clock = Clock()
        pm = _pm(clock=clock)
        pm.remember("mi color favorito es el azul")
        pm.handle_message("olvida el recuerdo")
        clock.jump(500)                                   # > 120 s
        self.assertIsNone(pm.handle_message("sí"))        # no es comando: sigue el flujo normal
        self.assertEqual(len(_snap(pm)), 1)

    def test_otro_mensaje_descarta_la_propuesta(self):
        self.pm.handle_message("olvida el recuerdo")
        self.assertIsNone(self.pm.handle_message("cuéntame un chiste"))
        self.assertIsNone(self.pm.handle_message("sí"))
        self.assertEqual(len(_snap(self.pm)), 1)


class TestVariosRecuerdos(unittest.TestCase):
    def setUp(self):
        self.pm = _pm()
        self.recs = [self.pm.remember(t) for t in (
            "mi perro se llama Max", "mi color favorito es el azul",
            "decidimos usar Python para el backend")]

    def test_lista_todos_con_id_pide_aclaracion_y_no_borra(self):
        before = _snap(self.pm)
        r = self.pm.handle_message("olvida el recuerdo")
        self.assertIn("no sé cuál", r)
        self.assertIn("No he borrado nada", r)
        for rec in self.recs:
            self.assertIn(rec.entry_id, r)
        self.assertEqual(_snap(self.pm), before)

    def test_no_deja_confirmacion_pendiente_un_si_posterior_no_borra(self):
        before = _snap(self.pm)
        self.pm.handle_message("olvida el recuerdo")
        self.assertEqual(self.pm._pending, {})
        for msg in ("sí", "ok", "dale", "claro", "hazlo", "confirmo", "sí, olvida todo",
                    "confirmo olvidar todo"):
            with self.subTest(msg=msg):
                self.assertIsNone(self.pm.handle_message(msg))
                self.assertEqual(_snap(self.pm), before)

    def test_todas_las_frases_pasan_por_la_aclaracion(self):
        before = _snap(self.pm)
        for phrase in NOREF_PHRASES:
            with self.subTest(phrase=phrase):
                r = self.pm.handle_message(phrase)
                self.assertIsNotNone(r)
                self.assertIn("no sé cuál", r)
                self.assertEqual(_snap(self.pm), before)

    def test_mas_de_cinco_muestra_cinco_y_cuenta_el_resto(self):
        pm = _pm()
        for i in range(8):
            pm.remember(f"dato numero {i} sobre el proyecto alfa")
        before = _snap(pm)
        r = pm.handle_message("olvida el recuerdo")
        self.assertEqual(len(_ids(r)), 5)
        self.assertIn("y 3 más", r)
        self.assertEqual(_snap(pm), before)

    def test_la_palabra_recuerdo_dentro_de_un_texto_no_lo_convierte_en_candidato_unico(self):
        pm = _pm()
        pm.remember("mi recuerdo favorito es la playa")
        pm.remember("mi perro se llama Max")
        before = _snap(pm)
        r = pm.handle_message("olvida el recuerdo")
        self.assertIn("no sé cuál", r)                    # no adivina por coincidencia de palabra
        self.assertEqual(_snap(pm), before)
        self.assertEqual(pm._pending, {})

    def test_con_una_propuesta_global_pendiente_se_descarta_y_no_se_ejecuta(self):
        before = _snap(self.pm)
        self.pm.handle_message("olvida todo")             # deja pendiente forget_all
        self.assertIn("forget_all", [p.action for p in self.pm._pending.values()])
        self.pm.handle_message("olvida el recuerdo")      # descarta la propuesta anterior
        self.assertEqual(self.pm._pending, {})
        self.assertIsNone(self.pm.handle_message("sí, olvida todo"))
        self.assertEqual(_snap(self.pm), before)

    def test_ningun_camino_llama_a_forget_all(self):
        calls = []
        orig = self.pm.forget_all
        self.pm.forget_all = lambda *a, **k: calls.append(1) or orig(*a, **k)
        for msg in ("olvida el recuerdo", "sí", "olvida el recuerdo", "ok"):
            self.pm.handle_message(msg)
        self.assertEqual(calls, [])


class TestAislamientoPorProyecto(unittest.TestCase):
    def test_solo_cuenta_y_lista_el_proyecto_actual(self):
        pm = _pm()
        pm.remember("dato de alfa uno", scope="alfa")
        pm.remember("dato de alfa dos", scope="alfa")
        pm.remember("dato de beta", scope="beta")
        before_beta = _snap(pm, "beta")
        r = pm.handle_message("olvida el recuerdo", project="alfa")
        self.assertIn("2 recuerdos", r)
        self.assertNotIn("dato de beta", r)
        self.assertEqual(_snap(pm, "beta"), before_beta)
        self.assertEqual(len(_snap(pm, "alfa")), 2)

    def test_proyecto_con_un_recuerdo_confirma_aunque_otro_proyecto_tenga_mas(self):
        pm = _pm()
        pm.remember("dato de alfa", scope="alfa")
        pm.remember("dato de beta uno", scope="beta")
        pm.remember("dato de beta dos", scope="beta")
        r = pm.handle_message("olvida el recuerdo", project="alfa")
        self.assertIn("Todavía no he borrado nada", r)
        pm.handle_message("sí", project="alfa")
        self.assertEqual(_snap(pm, "alfa"), [])
        self.assertEqual(len(_snap(pm, "beta")), 2)


class TestNoCambiaLoExistente(unittest.TestCase):
    def setUp(self):
        self.pm = _pm()
        self.a = self.pm.remember("mi perro se llama Max")
        self.b = self.pm.remember("mi gato se llama Luna")

    def test_borrar_por_id_sigue_inmediato(self):
        r = self.pm.handle_message(f"olvida el recuerdo {self.a.entry_id}")
        self.assertIn("olvidé", r)
        self.assertEqual([x[0] for x in _snap(self.pm)], [self.b.entry_id])

    def test_borrar_por_tema_con_un_candidato_pide_confirmacion(self):
        r = self.pm.handle_message("olvida el recuerdo de mi perro")
        self.assertIn(self.a.entry_id, r)
        self.assertIn("Todavía no he borrado nada", r)
        self.assertEqual(len(_snap(self.pm)), 2)

    def test_borrar_por_tema_con_varios_lista_y_no_borra(self):
        r = self.pm.handle_message("olvida el recuerdo sobre mi")
        self.assertEqual(len(_snap(self.pm)), 2)

    def test_olvida_todo_sigue_exigiendo_la_frase_exacta(self):
        self.pm.handle_message("olvida todo")
        r = self.pm.handle_message("sí")
        self.assertIn("«sí, olvida todo»", r)
        self.assertEqual(len(_snap(self.pm)), 2)
        r = self.pm.handle_message("sí, olvida todo")
        self.assertIn("borré 2", r)

    def test_preguntas_nunca_borran_ni_se_interpretan_como_comando(self):
        for msg in ("¿olvida el recuerdo?", "¿puedes olvidar el recuerdo?", "olvida el recuerdo?"):
            with self.subTest(msg=msg):
                self.assertIsNone(self.pm.handle_message(msg))
        self.assertEqual(len(_snap(self.pm)), 2)

    def test_referencias_vagas_con_texto_siguen_sin_borrar_nada(self):
        for msg in ("olvida el recuerdo ese", "olvida el recuerdo anterior",
                    "olvida el recuerdo pm-1234", "olvida el recuerdo pm-zzzzzzzz"):
            with self.subTest(msg=msg):
                r = self.pm.handle_message(msg)
                self.assertEqual(len(_snap(self.pm)), 2)
        self.assertEqual(self.pm._pending, {})

    def test_mensajes_normales_no_se_ven_afectados(self):
        for msg in ("hola", "recuerdo bien aquel día", "el recuerdo de mi abuela es bonito",
                    "olvida"):
            with self.subTest(msg=msg):
                self.assertIsNone(self.pm.handle_message(msg))
        self.assertEqual(len(_snap(self.pm)), 2)


class TestAlmacenSQLite(unittest.TestCase):
    def test_aclaracion_con_sqlite_y_fts_consistente(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        store = SQLiteMemoryStore(os.path.join(tmp, "m.db"))
        self.addCleanup(store.close)
        pm = _pm(store)
        for t in ("mi perro se llama Max", "mi color favorito es el azul"):
            pm.remember(t)
        before = _snap(pm)
        r = pm.handle_message("olvida el recuerdo")
        self.assertIn("no sé cuál", r)
        self.assertEqual(_snap(pm), before)
        n_mem = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        n_fts = store._conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        self.assertEqual(n_mem, n_fts)


class TestIntegracionConNexusCore(unittest.IsolatedAsyncioTestCase):
    def _core(self, pm, tm=None):
        return NexusCore(intent_router=IntentRouter(registry=None), task_state=tm,
                         persistent_memory=pm)

    async def test_process_responde_desde_la_memoria_y_no_cae_al_flujo_normal(self):
        pm = _pm()
        for t in ("mi perro se llama Max", "mi color favorito es el azul"):
            pm.remember(t)
        before = _snap(pm)
        tm = TaskStateManager(":memory:")
        self.addCleanup(tm.close)
        r = await self._core(pm, tm).process("olvida el recuerdo")
        self.assertEqual((r.provider, r.model), ("memory", "persistent"))
        self.assertIn("no sé cuál", r.text)
        self.assertNotIn("No encontré información relevante", r.text)
        self.assertEqual(_snap(pm), before)
        self.assertEqual(tm.stats()["total"], 0)           # no crea tareas

    async def test_process_un_recuerdo_pide_confirmacion_y_el_si_lo_borra(self):
        pm = _pm()
        pm.remember("mi perro se llama Max")
        core = self._core(pm)
        r = await core.process("olvida el recuerdo")
        self.assertIn("Todavía no he borrado nada", r.text)
        self.assertEqual(len(_snap(pm)), 1)
        r = await core.process("sí")
        self.assertEqual((r.provider, r.model), ("memory", "persistent"))
        self.assertEqual(_snap(pm), [])

    async def test_process_varios_y_luego_si_no_borra_nada(self):
        pm = _pm()
        for t in ("mi perro se llama Max", "mi color favorito es el azul"):
            pm.remember(t)
        before = _snap(pm)
        core = self._core(pm)
        await core.process("olvida el recuerdo")
        r = await core.process("sí")                       # no es comando de memoria
        self.assertNotEqual(r.provider, "memory")
        self.assertEqual(_snap(pm), before)


if __name__ == "__main__":
    unittest.main()
