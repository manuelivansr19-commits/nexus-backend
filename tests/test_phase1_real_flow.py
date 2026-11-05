"""
NEXUS — Fase 1: pruebas de flujo real (subprocesos con la app completa).

QUÉ DEMUESTRAN
  Evidencia N: con la configuración indicada, la aplicación real NO intenta
  conectarse a Gemini, Groq ni OpenRouter (guarda de red que registra y bloquea
  destinos no loopback, claves FALSAS, Ollama apuntando a un puerto cerrado).
  Evidencia de política: los cuatro endpoints informan la misma política efectiva.
  Evidencia de rutas: las tres bases quedan donde se configuran.

QUÉ NO DEMUESTRAN
  Que un modelo local responda correctamente. Eso es una validación distinta,
  con un modelo real, y sigue POR VALIDAR. Aquí ningún modelo local responde.

Cada escenario arranca un proceso nuevo porque backend.config lee el entorno
al importarse. Los escenarios con "control positivo" (se espera ver el host)
garantizan que la guarda realmente detecta tráfico cloud: sin ellos, un
"cero intentos" podría ser un falso negativo.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
from functools import lru_cache

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "_probe_inference.py")

HOSTS = {
    "gemini": "generativelanguage.googleapis.com",
    "groq": "api.groq.com",
    "openrouter": "openrouter.ai",
}
FAKE_KEYS = {
    "GEMINI_API_KEY": "FAKEKEY-GEMINI-0001",
    "GROQ_API_KEY": "FAKEKEY-GROQ-0001",
    "OPENROUTER_API_KEY": "FAKEKEY-OPENROUTER-0001",
}
# Variables que el subproceso NO debe heredar del equipo/sesión.
SCRUB = set(FAKE_KEYS) | {
    "OLLAMA_API_KEY", "OLLAMA_URL", "INFERENCE_MODE", "ALLOW_CLOUD_INFERENCE",
    "NEXUS_LOCAL_ONLY", "CLOUD_RUNTIMES_ENABLED", "USE_OLLAMA_ONLY",
    "MEMORY_DB_PATH", "KNOWLEDGE_DB_PATH", "TASKS_DB_PATH", "LOCAL_MODEL_PATH",
}

# nombre -> (entorno extra, ¿rutas de BD por entorno?)
SCENARIOS = {
    "default_all_keys":      ({}, True),
    "all_clouds_enabled":    ({"CLOUD_RUNTIMES_ENABLED": "gemini,groq,openrouter"}, True),
    "only_groq_enabled":     ({"CLOUD_RUNTIMES_ENABLED": "groq"}, True),
    "empty_allowlist":       ({"CLOUD_RUNTIMES_ENABLED": ""}, True),
    "local_only_flag":       ({"NEXUS_LOCAL_ONLY": "true"}, True),
    "local_only_flag_vs_cloud_only":
                             ({"NEXUS_LOCAL_ONLY": "true", "INFERENCE_MODE": "cloud_only",
                               "ALLOW_CLOUD_INFERENCE": "true"}, True),
    "local_only_flag_typo":  ({"NEXUS_LOCAL_ONLY": "ture"}, True),
    "invalid_mode":          ({"INFERENCE_MODE": "localfirst"}, True),
    "db_paths_by_env":       ({}, True),
    "db_defaults":           ({}, False),
}


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@lru_cache(maxsize=None)
def run_scenario(name):
    extra, db_by_env = SCENARIOS[name]
    work = os.path.realpath(tempfile.mkdtemp(prefix="nexus_probe_"))
    data = os.path.join(work, "data")
    os.makedirs(data)
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(FAKE_KEYS)
    env.update({
        "LOCAL_OLLAMA_URL": f"http://127.0.0.1:{_closed_port()}",
        "GEMINI_MAX_RETRIES": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if db_by_env:
        env["MEMORY_DB_PATH"] = os.path.join(data, "nexus_memory.db")
        env["KNOWLEDGE_DB_PATH"] = os.path.join(data, "nexus_knowledge.db")
        env["TASKS_DB_PATH"] = os.path.join(data, "nexus_tasks.db")
    env.update(extra)
    proc = subprocess.run(
        [sys.executable, PROBE], cwd=work, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=240,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("PROBE_JSON:")), None)
    assert line, f"[{name}] el sondeo no produjo resultado.\nSTDERR:\n{proc.stderr[-1500:]}"
    result = json.loads(line[len("PROBE_JSON:"):])
    assert "fatal" not in result, f"[{name}] fallo del sondeo:\n{result['fatal']}"
    result["_raw_output"] = proc.stdout + proc.stderr
    result["_data_dir"] = data
    result["_work"] = work
    return result


def cloud_hits(result):
    return sorted(k for k, host in HOSTS.items() if host in result["attempts"])


def effective(result):
    ep = result["endpoints"]
    return {k: (v["inference_mode"], v["cloud_allowed"]) for k, v in ep.items()}


# ── Evidencia N: ausencia de tráfico externo ───────────────────────────

class TestNoExternalTraffic:
    """Solo prueba que NO sale tráfico cloud. No prueba que un modelo local responda."""

    @pytest.mark.parametrize("name", [
        "local_only_flag", "local_only_flag_vs_cloud_only",
        "local_only_flag_typo", "invalid_mode", "empty_allowlist",
    ])
    def test_no_cloud_attempts(self, name):
        r = run_scenario(name)
        assert cloud_hits(r) == [], f"[{name}] intentos cloud inesperados: {r['attempts']}"

    @pytest.mark.parametrize("name", [
        "local_only_flag", "local_only_flag_vs_cloud_only",
        "local_only_flag_typo", "invalid_mode",
    ])
    def test_effective_mode_is_local_only_and_cloud_blocked(self, name):
        for ep, (mode, cloud) in effective(run_scenario(name)).items():
            assert (mode, cloud) == ("local_only", False), f"[{name}] endpoint {ep}"

    @pytest.mark.parametrize("name", ["local_only_flag_typo", "invalid_mode"])
    def test_invalid_values_produce_visible_warning(self, name):
        warnings = run_scenario(name)["endpoints"]["inference"]["config_warnings"]
        assert isinstance(warnings, list) and warnings, f"[{name}] sin advertencia visible"

    def test_empty_allowlist_reported(self):
        r = run_scenario("empty_allowlist")
        assert r["endpoints"]["inference"]["cloud_enabled"] == []


# ── A-03: sin fallback automático; control positivo de la guarda ───────

class TestNoAutomaticProviderFallback:
    def test_default_only_gemini_is_attempted(self):
        r = run_scenario("default_all_keys")
        # control positivo: la guarda SÍ ve a Gemini; luego no hay salto a otros.
        assert "gemini" in cloud_hits(r), f"control positivo fallido: {r['attempts']}"
        assert cloud_hits(r) == ["gemini"], f"fallback inesperado: {r['attempts']}"

    def test_explicit_optin_enables_all_three_hosts(self):
        # control positivo de los TRES hosts vigilados
        r = run_scenario("all_clouds_enabled")
        assert cloud_hits(r) == ["gemini", "groq", "openrouter"], r["attempts"]

    def test_only_groq_when_only_groq_enabled(self):
        assert cloud_hits(run_scenario("only_groq_enabled")) == ["groq"]


# ── Los cuatro endpoints informan la misma política efectiva ───────────

class TestEffectivePolicyConsistency:
    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_four_endpoints_agree(self, name):
        r = run_scenario(name)
        for ep, view in r["endpoints"].items():
            assert view["status_code"] == 200, f"[{name}] {ep}"
        assert len(set(effective(r).values())) == 1, f"[{name}] contradicción: {effective(r)}"


# ── D1 / A-04 mínimo: rutas de bases de datos ──────────────────────────

class TestDatabasePaths:
    def test_all_three_databases_go_to_configured_paths(self):
        r = run_scenario("db_paths_by_env")
        files = sorted(os.listdir(r["_data_dir"]))
        assert files == ["nexus_knowledge.db", "nexus_memory.db", "nexus_tasks.db"], files
        assert r["db_files_in_cwd"] == [], "alguna base se creó en el directorio actual"

    def test_defaults_unchanged_when_env_absent(self):
        r = run_scenario("db_defaults")
        assert r["db_files_in_cwd"] == [
            "nexus_knowledge.db", "nexus_memory.db", "nexus_tasks.db"], r["db_files_in_cwd"]


# ── Secretos ───────────────────────────────────────────────────────────

class TestNoSecretsInOutput:
    @pytest.mark.parametrize("name", ["invalid_mode", "local_only_flag_typo", "default_all_keys"])
    def test_fake_keys_not_in_logs_or_responses(self, name):
        raw = run_scenario(name)["_raw_output"]
        assert "FAKEKEY" not in raw
