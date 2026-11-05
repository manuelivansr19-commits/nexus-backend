"""
NEXUS — conftest de la suite (Fase 1: aislamiento de datos y determinismo).

Este archivo se carga ANTES de que pytest importe cualquier módulo de prueba,
y por tanto antes de que alguien importe ``backend.config`` (que lee el entorno
al importarse). Garantías:

  1. Las tres bases SQLite (memoria, conocimiento, tareas) apuntan a un
     directorio temporal exclusivo de esta sesión, nunca al directorio actual.
  2. Las variables que cambian la política de inferencia y las claves de
     proveedores se limpian, para que la suite no dependa del equipo donde corre.
  3. Si ``backend.config`` ya estaba importado, la sesión se detiene: ya no se
     podría garantizar el aislamiento.

No se importa nada de ``backend`` aquí.
"""
import atexit
import os
import shutil
import sys
import tempfile

if "backend.config" in sys.modules:
    raise RuntimeError(
        "backend.config ya fue importado antes de tests/conftest.py: "
        "no se puede garantizar el aislamiento de las bases de datos."
    )

_TMP_ROOT = os.path.realpath(tempfile.mkdtemp(prefix="nexus_pytest_"))
atexit.register(shutil.rmtree, _TMP_ROOT, True)

# Se SOBRESCRIBE siempre (no setdefault): un valor heredado del entorno del
# usuario no debe poder apuntar la suite a datos reales.
os.environ["NEXUS_TEST_TMP"] = _TMP_ROOT
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP_ROOT, "nexus_memory.db")
os.environ["KNOWLEDGE_DB_PATH"] = os.path.join(_TMP_ROOT, "nexus_knowledge.db")
os.environ["TASKS_DB_PATH"] = os.path.join(_TMP_ROOT, "nexus_tasks.db")

# Determinismo: ni claves reales ni política heredada del equipo.
for _name in (
    "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY",
    "INFERENCE_MODE", "ALLOW_CLOUD_INFERENCE", "NEXUS_LOCAL_ONLY",
    "CLOUD_RUNTIMES_ENABLED", "USE_OLLAMA_ONLY",
):
    os.environ.pop(_name, None)


def pytest_report_header(config):
    return [
        f"nexus aislamiento: raíz temporal = {_TMP_ROOT}",
        f"nexus aislamiento: MEMORY_DB_PATH = {os.environ['MEMORY_DB_PATH']}",
        f"nexus aislamiento: KNOWLEDGE_DB_PATH = {os.environ['KNOWLEDGE_DB_PATH']}",
        f"nexus aislamiento: TASKS_DB_PATH = {os.environ['TASKS_DB_PATH']}",
    ]
