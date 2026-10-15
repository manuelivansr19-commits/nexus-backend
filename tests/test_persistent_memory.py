# -*- coding: utf-8 -*-
"""
Pruebas unitarias de la Memoria Persistente (backend/core/persistent_memory.py).

Colocar en tests/ y correr:
    python -m pytest tests/test_persistent_memory.py -v

Cubre: persistencia entre reaperturas, recuperacion, aislamiento por proyecto,
corregir/olvidar con indice FTS sincronizado, comandos ambiguos (preguntan antes
de actuar) y degradacion segura ante fallos del almacen. Se ejecuta con ambos
almacenes reales: SQLiteMemoryStore (archivo temporal) y RAMMemoryStore.
"""
import os
import shutil
import tempfile
import unittest

from backend.core.memory import Memory, MemoryEntry, MemoryType, RAMMemoryStore, SQLiteMemoryStore
from backend.core.persistent_memory import (
    PersistentMemory, PersistentMemoryError, infer_kind, normalize_scope, tokenize,
)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "mem.db")
        self._stores = []
        self.addCleanup(self._close_all)

    def _close_all(self):
        for s in self._stores:
            try:
                s.close()
            except Exception:
                pass

    def sqlite(self):
        s = SQLiteMemoryStore(self.db)
        self._stores.append(s)
        return s

    def pm(self, store=None, **kw):
        return PersistentMemory(store or self.sqlite(), **kw)


def fts_consistent(store):
    """memories y memories_fts deben tener las mismas filas (mismos entry_id)."""
    a = {r[0] for r in store._conn.execute("SELECT entry_id FROM memories")}
    rows = [r[0] for r in store._conn.execute("SELECT entry_id FROM memories_fts")]
    return a == set(rows) and len(rows) == len(a)


class TestPersistence(TmpDirCase):

    def test_sobrevive_al_cierre_y_reapertura(self):
        s1 = self.sqlite()
        pm1 = PersistentMemory(s1)
        a = pm1.remember("prefiero respuestas breves", scope="aura")
        b = pm1.remember("el presupuesto de Aura es de 5000 dólares", scope="aura")
        s1.close()
        pm2 = PersistentMemory(self.sqlite())
        textos = {r.text for r in pm2.list_records("aura")}
        self.assertEqual(textos, {a.text, b.text})
        self.assertEqual({r.kind for r in pm2.list_records("aura")}, {"preference", "fact"})
        self.assertIn("5000", pm2.context_for("cuál es el presupuesto de Aura", "aura"))

    def test_los_cinco_tipos(self):
        pm = self.pm()
        casos = {
            "prefiero respuestas breves": "preference",
            "decidimos usar SQLite para la memoria": "decision",
            "mi objetivo es lanzar en diciembre": "goal",
            "la tarea de análisis está bloqueada": "task_state",
            "el servidor está en Tijuana": "fact",
        }
        for texto, kind in casos.items():
            self.assertEqual(infer_kind(texto), kind, texto)
            self.assertEqual(pm.remember(texto).kind, kind, texto)
        self.assertEqual({r.kind for r in pm.list_records()}, set(casos.values()))

    def test_solo_task_state_guarda_task_id_y_estado(self):
        pm = self.pm()
        t = pm.remember("la tarea de análisis está bloqueada", task_id="T1")
        f = pm.remember("el servidor está en Tijuana", task_id="T1")
        self.assertEqual((t.kind, t.task_id, t.status), ("task_state", "T1", "blocked"))
        self.assertEqual((f.kind, f.task_id, f.status), ("fact", None, "active"))

    def test_metadata_de_autorizacion_en_el_almacen(self):
        s = self.sqlite()
        rec = PersistentMemory(s).remember("el servidor está en Tijuana", scope="x")
        e = s.get(rec.entry_id)
        self.assertEqual(e.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(e.metadata["namespace"], "persistent_memory")
        self.assertEqual(e.metadata["authorized_by"], "user_explicit")
        self.assertEqual(e.metadata["scope"], "x")

    def test_duplicado_no_se_guarda_dos_veces(self):
        pm = self.pm()
        a = pm.remember("el servidor está en Tijuana")
        b = pm.remember("El servidor esta en Tijuana.")
        self.assertEqual(a.entry_id, b.entry_id)
        self.assertEqual(len(pm.list_records()), 1)

    def test_limite_por_proyecto(self):
        pm = self.pm(max_per_scope=2)
        pm.remember("dato uno aquí")
        pm.remember("dato dos aquí")
        with self.assertRaises(PersistentMemoryError):
            pm.remember("dato tres aquí")
        pm.remember("dato otro proyecto", scope="otro")   # otro proyecto: cupo propio

    def test_texto_vacio_o_demasiado_largo(self):
        pm = self.pm(max_chars=50)
        with self.assertRaises(PersistentMemoryError):
            pm.remember("   ")
        with self.assertRaises(PersistentMemoryError):
            pm.remember("x" * 51)
        self.assertEqual(pm.list_records(), [])


class TestSoloConAutorizacion(TmpDirCase):

    NO_COMANDOS = [
        "hola, ¿cómo estás?",
        "mi presupuesto para Aura es de 5000 dólares",
        "ahora crea un plan de marketing",
        "¿recuerdas que dije algo sobre el presupuesto?",
        "no recuerdes que fumo",
        "me gusta el café",
        "prefiero respuestas breves",
        "guarda el archivo en el escritorio",
        "corrige el texto de arriba",
        "qué hora es",
        "¿qué sabes de Python?",
        "olvida lo anterior",
    ]

    def test_mensajes_normales_no_guardan_ni_responden(self):
        s = self.sqlite()
        pm = PersistentMemory(s)
        for m in self.NO_COMANDOS:
            with self.subTest(m=m):
                self.assertIsNone(pm.handle_message(m))
        self.assertEqual(s.stats()["total"], 0)

    def test_conversacion_y_tool_no_entran_en_la_memoria_persistente(self):
        s = self.sqlite()
        mem = Memory(s)
        mem.conversation.add_user("quiero ver el presupuesto del servidor")
        mem.conversation.add_assistant("[TOOL:clock] {'time': '10:00:00'}")
        mem.remember("[TOOL:clock] {'time': '10:00:00'}", MemoryType.WORKING, importance=0.4)
        pm = PersistentMemory(s)
        self.assertEqual(pm.list_records(), [])
        self.assertEqual(pm.context_for("presupuesto del servidor clock"), "")
        self.assertEqual(pm.stats()["total"], 0)

    def test_comando_explicito_guarda(self):
        s = self.sqlite()
        pm = PersistentMemory(s)
        r = pm.handle_message("Recuerda que mi presupuesto para Aura es de 5000 dólares")
        self.assertIn("Guardado", r)
        self.assertEqual(len(pm.list_records()), 1)

    def test_comando_con_tipo_explicito(self):
        pm = self.pm()
        pm.handle_message("guarda como decisión: usaremos SQLite")
        pm.handle_message("guarda esta preferencia: respuestas en español")
        kinds = {r.text: r.kind for r in pm.list_records()}
        self.assertEqual(kinds["usaremos SQLite"], "decision")
        self.assertEqual(kinds["respuestas en español"], "preference")


class TestAmbiguedad(TmpDirCase):
    """Si el comando es ambiguo: pregunta ANTES de guardar, corregir o borrar."""

    def test_guardar_ambiguo_pregunta_y_solo_guarda_con_si(self):
        pm = self.pm()
        r = pm.handle_message("recuerda mi cumpleaños es el 5 de mayo")
        self.assertIn("Todavía no he guardado nada", r)
        self.assertEqual(pm.list_records(), [])
        self.assertIn("Guardado", pm.handle_message("sí"))
        self.assertEqual([x.text for x in pm.list_records()], ["mi cumpleaños es el 5 de mayo"])

    def test_guardar_ambiguo_con_no_no_guarda(self):
        pm = self.pm()
        pm.handle_message("no olvides que la junta es el lunes")
        self.assertIn("no hice ningún cambio", pm.handle_message("no"))
        self.assertEqual(pm.list_records(), [])

    def test_propuesta_vence_y_otro_mensaje_la_descarta(self):
        clock = Clock()
        pm = self.pm(clock=clock)
        pm.handle_message("recuerda mi cumpleaños es el 5 de mayo")
        clock.advance(121)
        self.assertIsNone(pm.handle_message("sí"))          # vencida: no guarda
        self.assertEqual(pm.list_records(), [])
        pm.handle_message("recuerda mi cumpleaños es el 5 de mayo")
        self.assertIsNone(pm.handle_message("mejor hablemos de marketing"))
        self.assertIsNone(pm.handle_message("sí"))          # descartada: no guarda
        self.assertEqual(pm.list_records(), [])

    def test_recuerda_esto_pregunta_que(self):
        pm = self.pm()
        self.assertIn("¿Qué quieres que recuerde", pm.handle_message("recuerda esto"))
        self.assertIn("¿Qué quieres que recuerde", pm.handle_message("recuerda lo anterior"))
        self.assertEqual(pm.list_records(), [])

    def test_olvidar_por_texto_pregunta_antes_de_borrar(self):
        pm = self.pm()
        pm.remember("el presupuesto de Aura es de 5000 dólares")
        r = pm.handle_message("olvida de tu memoria lo del presupuesto")
        self.assertIn("Todavía no he borrado nada", r)
        self.assertEqual(len(pm.list_records()), 1)
        pm.handle_message("no")
        self.assertEqual(len(pm.list_records()), 1)
        pm.handle_message("olvida de tu memoria lo del presupuesto")
        self.assertIn("olvidé", pm.handle_message("sí"))
        self.assertEqual(pm.list_records(), [])

    def test_olvidar_con_varias_coincidencias_no_borra_nada(self):
        pm = self.pm()
        pm.remember("el presupuesto de Aura es de 5000 dólares")
        pm.remember("el presupuesto de marketing es de 900 dólares")
        r = pm.handle_message("borra de tu memoria el presupuesto")
        self.assertIn("2 recuerdos", r)
        self.assertEqual(len(pm.list_records()), 2)
        self.assertIsNone(pm.handle_message("sí"))     # no hay nada pendiente
        self.assertEqual(len(pm.list_records()), 2)

    def test_olvidar_sin_coincidencias(self):
        pm = self.pm()
        pm.remember("el servidor está en Tijuana")
        self.assertIn("No encontré", pm.handle_message("olvida de tu memoria lo de Aura"))
        self.assertEqual(len(pm.list_records()), 1)

    def test_olvidar_sin_decir_memoria_solo_si_coincide_algo_guardado(self):
        pm = self.pm()
        self.assertIsNone(pm.handle_message("olvida eso"))           # nada guardado
        pm.remember("mi color favorito es el azul")
        self.assertIsNone(pm.handle_message("olvida lo de la reunión"))   # sin coincidencia
        r = pm.handle_message("olvida que mi color favorito es el azul")
        self.assertIn("¿Borro este recuerdo", r)                     # coincide: pregunta
        self.assertEqual(len(pm.list_records()), 1)

    def test_olvidar_por_id_es_inequivoco(self):
        pm = self.pm()
        rec = pm.remember("el servidor está en Tijuana")
        self.assertIn("olvidé", pm.handle_message(f"olvida el recuerdo {rec.entry_id}"))
        self.assertEqual(pm.list_records(), [])
        self.assertIn("No existe", pm.handle_message("olvida el recuerdo pm-00000000"))

    def test_olvidar_todo_exige_la_frase_exacta(self):
        pm = self.pm()
        pm.remember("dato uno aquí")
        pm.remember("dato dos aquí")
        self.assertIn("«sí, olvida todo»", pm.handle_message("olvida todo"))
        self.assertIn("frase exacta", pm.handle_message("sí"))
        self.assertEqual(len(pm.list_records()), 2)
        self.assertIn("borré 2", pm.handle_message("sí, olvida todo"))
        self.assertEqual(pm.list_records(), [])

    def test_olvidar_todo_sin_nada_guardado(self):
        self.assertIn("no hay nada que borrar", self.pm().handle_message("olvida todo"))

    def test_olvidar_todo_se_cancela(self):
        pm = self.pm()
        pm.remember("dato uno aquí")
        pm.handle_message("olvida todo")
        pm.handle_message("no")
        self.assertEqual(len(pm.list_records()), 1)

    def test_corregir_por_texto_pregunta_y_separador_ambiguo_pide_id(self):
        pm = self.pm()
        rec = pm.remember("el presupuesto de Aura es de 5000 dólares")
        r = pm.handle_message("corrige de tu memoria el presupuesto de Aura por el presupuesto de Aura es de 8000 dólares")
        self.assertIn("Todavía no he cambiado nada", r)
        self.assertEqual(pm.list_records()[0].text, rec.text)
        pm.handle_message("no")
        r = pm.handle_message("corrige de tu memoria presupuesto por 1 por 2")
        self.assertIn("identificador", r)
        self.assertEqual(pm.list_records()[0].text, rec.text)

    def test_corregir_por_texto_con_si_aplica(self):
        pm = self.pm()
        pm.remember("el presupuesto de Aura es de 5000 dólares")
        pm.handle_message("corrige de tu memoria el presupuesto de Aura por el presupuesto de Aura es de 8000 dólares")
        self.assertIn("Corregido", pm.handle_message("sí"))
        self.assertEqual([r.text for r in pm.list_records()], ["el presupuesto de Aura es de 8000 dólares"])

    def test_corregir_por_id(self):
        pm = self.pm()
        rec = pm.remember("el presupuesto de Aura es de 5000 dólares")
        r = pm.handle_message(f"corrige el recuerdo {rec.entry_id}: el presupuesto de Aura es de 9000 dólares")
        self.assertIn("Corregido", r)
        self.assertEqual([x.text for x in pm.list_records()], ["el presupuesto de Aura es de 9000 dólares"])

    def test_preguntas_nunca_actuan(self):
        pm = self.pm()
        pm.remember("el servidor está en Tijuana")
        for m in ("¿puedes olvidar el recuerdo pm-12345678?", "¿recuerda que mañana hay junta?",
                  "¿borra de tu memoria lo del servidor?"):
            with self.subTest(m=m):
                self.assertIsNone(pm.handle_message(m))
        self.assertEqual(len(pm.list_records()), 1)

    def test_texto_demasiado_largo_se_rechaza_sin_guardar(self):
        pm = self.pm(max_chars=40)
        r = pm.handle_message("recuerda que " + "palabra " * 20)
        self.assertIn("No pude", r)
        self.assertIn("demasiado largo", r)
        self.assertEqual(pm.list_records(), [])

    def test_recall(self):
        pm = self.pm()
        pm.remember("el presupuesto de Aura es de 5000 dólares")
        pm.remember("prefiero respuestas breves")
        self.assertIn("5000", pm.handle_message("¿Qué recuerdas?"))
        self.assertIn("breves", pm.handle_message("¿Qué recuerdas?"))
        r = pm.handle_message("qué recuerdas de presupuesto")
        self.assertIn("5000", r)
        self.assertNotIn("breves", r)
        self.assertIn("No tengo recuerdos", pm.handle_message("qué recuerdas de ballenas"))
        self.assertIsNone(pm.handle_message("¿qué sabes de Python?"))


class TestRecuperacion(TmpDirCase):

    def setUp(self):
        super().setUp()
        self.pm_ = self.pm()
        self.pm_.remember("el presupuesto del proyecto Aura es de 5000 dólares", scope="aura")

    def test_lenguaje_natural_con_signos_y_acentos(self):
        # Con SQLiteMemoryStore.search estas consultas devolvian 0 resultados.
        for q in ("cuál es el presupuesto de Aura", "¿cuál es el presupuesto de Aura?",
                  "presupuesto: Aura", "presupuesto Aura monto total",
                  "cuanto es el PRESUPUESTO del proyecto aura???"):
            with self.subTest(q=q):
                self.assertIn("5000", self.pm_.context_for(q, "aura"))

    def test_mensaje_enriquecido_con_tarea_activa(self):
        msg = ("[Contexto de tarea activa: Analiza el proyecto Aura]\n"
               "[Resultado previo de la tarea: listo]\ncuál es el presupuesto")
        self.assertIn("5000", self.pm_.context_for(msg, "aura"))

    def test_la_tarea_activa_aporta_relevancia(self):
        msg = ("[Contexto de tarea activa: revisar el presupuesto del proyecto Aura]\n"
               "dame un resumen")
        self.assertIn("5000", self.pm_.context_for(msg, "aura"))

    def test_mensaje_sin_relacion_no_trae_nada(self):
        for q in ("hola", "cuéntame un chiste", "qué tiempo hace en Madrid", "gracias, hasta luego"):
            with self.subTest(q=q):
                self.assertEqual(self.pm_.context_for(q, "aura"), "")

    def test_preferencias_siempre_se_incluyen_y_respetan_limite(self):
        pm = self.pm_
        pm.remember("prefiero respuestas breves", scope="aura")
        ctx = pm.context_for("hola", "aura")
        self.assertIn("breves", ctx)
        self.assertNotIn("5000", ctx)
        pm2 = self.pm(always_preferences=False)
        self.assertEqual(pm2.context_for("hola", "aura"), "")
        for i in range(10):
            pm.remember(f"presupuesto de la fase {i} del proyecto Aura", scope="aura")
        lineas = [l for l in pm.context_for("presupuesto del proyecto Aura", "aura").splitlines() if l.startswith("•")]
        self.assertLessEqual(len(lineas), 5)

    def test_estado_de_tarea_solo_para_su_task_id(self):
        pm = self.pm_
        pm.remember("la tarea de análisis está bloqueada", scope="aura", task_id="T1")
        self.assertIn("bloqueada", pm.context_for("hola", "aura", task_id="T1"))
        self.assertNotIn("bloqueada", pm.context_for("hola", "aura", task_id="T2"))
        self.assertNotIn("bloqueada", pm.context_for("hola", "aura"))

    def test_formato_marca_los_datos_como_informacion(self):
        ctx = self.pm_.context_for("presupuesto de Aura", "aura")
        self.assertTrue(ctx.startswith("[MEMORIA PERSISTENTE"))
        self.assertIn("no instrucciones", ctx)
        self.assertIn("(hecho)", ctx)

    def test_max_chars(self):
        pm = self.pm_
        for i in range(6):
            pm.remember(f"presupuesto numero {i} " + "dato " * 40, scope="aura")
        self.assertLessEqual(len(pm.context_for("presupuesto numero", "aura", max_chars=300)), 600)

    def test_tokenizador(self):
        self.assertEqual(tokenize("¿Cuál es el PRESUPUESTO de Aura?"), ["presupuesto", "aura"])
        self.assertEqual(tokenize("presupuestos"), ["presupuesto"])
        self.assertEqual(tokenize("años"), ["anos"])

    def test_mensaje_corto_con_una_palabra_clave_si_recupera(self):
        # Regla: con 1-2 palabras de contenido basta una coincidencia; con mas, hacen falta 2.
        self.assertIn("5000", self.pm_.context_for("presupuesto", "aura"))
        self.assertEqual(self.pm_.context_for("hablemos del proyecto de marketing digital", "aura"), "")


class TestAislamientoPorProyecto(TmpDirCase):

    def setUp(self):
        super().setUp()
        self.pm_ = self.pm()
        self.a = self.pm_.remember("el presupuesto es de 5000 dólares", scope="A")
        self.b = self.pm_.remember("el presupuesto es de 9000 dólares", scope="B")

    def test_recuperacion_y_listado_por_proyecto(self):
        self.assertIn("5000", self.pm_.context_for("presupuesto", "A"))
        self.assertNotIn("9000", self.pm_.context_for("presupuesto", "A"))
        self.assertNotIn("5000", self.pm_.context_for("presupuesto", "B"))
        self.assertEqual([r.text for r in self.pm_.list_records("A")], [self.a.text])
        self.assertEqual(self.pm_.context_for("presupuesto", "C"), "")

    def test_no_se_puede_olvidar_ni_corregir_desde_otro_proyecto(self):
        self.assertFalse(self.pm_.forget(self.a.entry_id, "B"))
        with self.assertRaises(PersistentMemoryError):
            self.pm_.correct(self.a.entry_id, "texto nuevo cualquiera", "B")
        self.assertEqual(len(self.pm_.list_records("A")), 1)
        self.assertIn("No existe", self.pm_.handle_message(f"olvida el recuerdo {self.a.entry_id}", "B"))
        self.assertEqual(self.pm_.list_records("A")[0].text, self.a.text)

    def test_olvidar_todo_solo_afecta_al_proyecto(self):
        self.assertEqual(self.pm_.forget_all("A"), 1)
        self.assertEqual(self.pm_.list_records("A"), [])
        self.assertEqual(len(self.pm_.list_records("B")), 1)

    def test_confirmacion_pendiente_no_cruza_proyectos(self):
        self.pm_.handle_message("recuerda mi cumpleaños es el 5 de mayo", "A")
        self.assertIsNone(self.pm_.handle_message("sí", "B"))
        self.assertEqual(len(self.pm_.list_records("B")), 1)
        self.assertIn("Guardado", self.pm_.handle_message("sí", "A"))

    def test_normalizacion_del_proyecto(self):
        self.assertEqual(normalize_scope(""), "default")
        self.assertEqual(normalize_scope(None), "default")
        self.assertEqual(normalize_scope("  Aura  "), "aura")
        self.pm_.remember("dato de prueba global")
        self.assertEqual(len(self.pm_.list_records("")), 1)
        self.assertEqual(len(self.pm_.list_records("default")), 1)
        self.assertEqual(len(self.pm_.list_records("AURA")), 0)

    def test_sin_aislamiento_por_usuario_documentado(self):
        # Limite conocido de esta version: dos "usuarios" del mismo proyecto comparten memoria.
        self.pm_.handle_message("recuerda que la clave del wifi es azul123", "A")
        otro = PersistentMemory(self.pm_._store)       # otra instancia, mismo almacen y proyecto
        self.assertIn("azul123", otro.context_for("clave del wifi", "A"))


class TestCorregirYOlvidarConIndiceSincronizado(TmpDirCase):

    def test_texto_antiguo_no_reaparece_tras_corregir_cerrar_y_reabrir(self):
        s1 = self.sqlite()
        pm1 = PersistentMemory(s1)
        rec = pm1.remember("el codigo antiguo del proyecto es zafiro", scope="aura")
        nueva = pm1.correct(rec.entry_id, "el codigo nuevo del proyecto es esmeralda", scope="aura")
        self.assertNotEqual(nueva.entry_id, rec.entry_id)
        self.assertTrue(fts_consistent(s1))
        s1.close()
        s2 = self.sqlite()
        pm2 = PersistentMemory(s2)
        self.assertTrue(fts_consistent(s2))
        # en ninguna via aparece "zafiro": recuperacion propia, find, store.search (FTS) y SQL directo
        self.assertEqual(pm2.context_for("cual es el codigo zafiro del proyecto", "aura").count("zafiro"), 0)
        self.assertEqual(pm2.find("zafiro", "aura"), [])
        self.assertEqual(s2.search("zafiro"), [])
        self.assertEqual(s2.search("antiguo"), [])
        self.assertEqual(s2._conn.execute("SELECT COUNT(*) FROM memories WHERE content LIKE '%zafiro%'").fetchone()[0], 0)
        self.assertEqual(s2._conn.execute("SELECT COUNT(*) FROM memories_fts WHERE content LIKE '%zafiro%'").fetchone()[0], 0)
        self.assertEqual([e.content for e in s2.search("esmeralda")], ["el codigo nuevo del proyecto es esmeralda"])
        self.assertEqual([r.text for r in pm2.list_records("aura")], ["el codigo nuevo del proyecto es esmeralda"])

    def test_corregir_conserva_tipo_y_deja_traza(self):
        s = self.sqlite()
        pm = PersistentMemory(s)
        rec = pm.remember("prefiero respuestas largas", scope="x")
        nueva = pm.correct(rec.entry_id, "prefiero respuestas breves", scope="x")
        self.assertEqual(nueva.kind, "preference")
        self.assertEqual(nueva.created_at, rec.created_at)
        self.assertEqual(s.get(nueva.entry_id).metadata["supersedes"], rec.entry_id)
        self.assertIsNone(s.get(rec.entry_id))

    def test_olvidar_limpia_ambas_tablas_y_no_reaparece(self):
        s1 = self.sqlite()
        pm1 = PersistentMemory(s1)
        a = pm1.remember("mi clave secreta es zafiro", scope="x")
        pm1.remember("otro dato que se queda", scope="x")
        self.assertTrue(pm1.forget(a.entry_id, "x"))
        self.assertTrue(fts_consistent(s1))
        s1.close()
        s2 = self.sqlite()
        pm2 = PersistentMemory(s2)
        self.assertTrue(fts_consistent(s2))
        self.assertEqual(s2.search("zafiro"), [])
        self.assertEqual(pm2.find("zafiro", "x"), [])
        self.assertEqual(pm2.context_for("cual es mi clave secreta zafiro", "x"), "")
        self.assertEqual(len(pm2.list_records("x")), 1)

    def test_olvidar_todo_deja_indice_consistente(self):
        s = self.sqlite()
        pm = PersistentMemory(s)
        for i in range(5):
            pm.remember(f"dato numero {i} del proyecto", scope="x")
        pm.forget_all("x")
        self.assertTrue(fts_consistent(s))
        self.assertEqual(s._conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0], 0)

    def test_corregir_dos_veces_seguidas(self):
        s = self.sqlite()
        pm = PersistentMemory(s)
        r = pm.remember("version uno del plan")
        r = pm.correct(r.entry_id, "version dos del plan")
        r = pm.correct(r.entry_id, "version tres del plan")
        self.assertTrue(fts_consistent(s))
        self.assertEqual([x.text for x in pm.list_records()], ["version tres del plan"])
        self.assertEqual(s.search("uno"), [])
        self.assertEqual(s.search("dos"), [])

    def test_si_falla_el_borrado_la_correccion_se_deshace(self):
        s = self.sqlite()

        class FallaAlBorrarLoViejo:
            def __init__(self, inner, old_id):
                self._i, self._old = inner, old_id

            def __getattr__(self, n):
                return getattr(self._i, n)

            def delete(self, entry_id):
                if entry_id == self._old:
                    raise RuntimeError("disco lleno")
                return self._i.delete(entry_id)

        rec = PersistentMemory(s).remember("dato original del proyecto")
        pm = PersistentMemory(FallaAlBorrarLoViejo(s, rec.entry_id))
        with self.assertRaises(PersistentMemoryError):
            pm.correct(rec.entry_id, "dato modificado del proyecto")
        self.assertEqual([r.text for r in PersistentMemory(s).list_records()], ["dato original del proyecto"])
        self.assertTrue(fts_consistent(s))

    def test_olvidar_id_inexistente(self):
        self.assertFalse(self.pm().forget("pm-deadbeef"))

    def test_corregir_a_texto_duplicado_se_rechaza(self):
        pm = self.pm()
        pm.remember("dato uno distinto")
        b = pm.remember("dato dos distinto")
        with self.assertRaises(PersistentMemoryError):
            pm.correct(b.entry_id, "dato uno distinto")


class RotoStore:
    """Almacen que falla en todo."""
    def __getattr__(self, name):
        def boom(*a, **k):
            raise RuntimeError("almacen caido")
        return boom


class TestDegradacionSegura(TmpDirCase):

    def test_contexto_vacio_si_el_almacen_falla(self):
        pm = PersistentMemory(RotoStore())
        self.assertEqual(pm.context_for("presupuesto de Aura", "aura"), "")

    def test_comandos_con_almacen_caido_responden_honestamente(self):
        pm = PersistentMemory(RotoStore())
        for m in ("recuerda que el servidor está en Tijuana", "olvida el recuerdo pm-12345678",
                  "qué recuerdas", "olvida de tu memoria lo del servidor"):
            with self.subTest(m=m):
                r = pm.handle_message(m)
                self.assertIsNotNone(r)
                self.assertIn("No pude", r)
                self.assertIn("No se realizó ningún cambio", r)

    def test_mensajes_normales_con_almacen_caido_no_se_interceptan(self):
        pm = PersistentMemory(RotoStore())
        for m in ("hola", "olvida lo anterior", "cuéntame un chiste", "olvida eso"):
            with self.subTest(m=m):
                self.assertIsNone(pm.handle_message(m))

    def test_api_directa_lanza_error_tipado(self):
        pm = PersistentMemory(RotoStore())
        with self.assertRaises(PersistentMemoryError):
            pm.remember("algo que guardar")
        with self.assertRaises(PersistentMemoryError):
            pm.list_records()

    def test_sin_store_es_error_de_construccion(self):
        with self.assertRaises(ValueError):
            PersistentMemory(None)

    def test_store_ram_tambien_funciona(self):
        pm = PersistentMemory(RAMMemoryStore())
        rec = pm.remember("el servidor está en Tijuana", scope="x")
        self.assertIn("Tijuana", pm.context_for("dónde está el servidor", "x"))
        new = pm.correct(rec.entry_id, "el servidor está en Ensenada", scope="x")
        self.assertEqual([r.text for r in pm.list_records("x")], [new.text])
        self.assertTrue(pm.forget(new.entry_id, "x"))
        self.assertEqual(pm.list_records("x"), [])


if __name__ == "__main__":
    unittest.main()
