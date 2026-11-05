"""
NEXUS — Fase 1: soberanía de inferencia (pruebas unitarias de política/config)
y aislamiento de bases de datos de la suite.

Cubre A-02 (NEXUS_LOCAL_ONLY sin efecto), A-03 (fallback automático a Groq /
OpenRouter), D2 (modo inválido -> local_only), D3 (NEXUS_LOCAL_ONLY no
reconocido -> restrictivo) y D1 (TASKS_DB_PATH).

Ninguna de estas pruebas toca la red.
"""
import asyncio
import os

import pytest

from backend.inference.errors import AllRuntimesFailed, RuntimeUnavailable
from backend.inference.interface import InferenceRuntime
from backend.inference.layer import InferenceLayer
from backend.inference.models import (
    InferenceMode, InferenceRequest, InferenceResult,
    RuntimeHealth, RuntimeStatus, RuntimeType,
)
from backend.inference.policy import InferencePolicy
from backend.inference.registry import InferenceRegistry

SECRET_LOOKALIKE = "FAKEKEY-NO-DEBE-APARECER-0001"


# ── Runtimes simulados (solo cuentan llamadas; no hay red) ─────────────

class _Local(InferenceRuntime):
    def __init__(self, name="ollama", fail=True):
        self._name, self._fail, self.calls = name, fail, 0
    @property
    def name(self): return self._name
    @property
    def runtime_type(self): return RuntimeType.LOCAL
    async def generate(self, req):
        self.calls += 1
        if self._fail:
            raise RuntimeUnavailable("local caído", runtime=self._name)
        return InferenceResult(text="ok", runtime_name=self._name,
            runtime_type=RuntimeType.LOCAL, model="m", mode=None, is_local=True)
    async def health(self):
        return RuntimeHealth(status=RuntimeStatus.UNAVAILABLE, runtime_name=self._name,
            runtime_type=RuntimeType.LOCAL, model="m")


class _Cloud(InferenceRuntime):
    def __init__(self, name, fail=False):
        self._name, self._fail, self.calls = name, fail, 0
    @property
    def name(self): return self._name
    @property
    def runtime_type(self): return RuntimeType.CLOUD
    async def generate(self, req):
        self.calls += 1
        if self._fail:
            raise RuntimeUnavailable("cloud caído", runtime=self._name)
        return InferenceResult(text="ok", runtime_name=self._name,
            runtime_type=RuntimeType.CLOUD, model="m", mode=None, is_local=False)
    async def health(self):
        return RuntimeHealth(status=RuntimeStatus.AVAILABLE, runtime_name=self._name,
            runtime_type=RuntimeType.CLOUD, model="m")


def _layer(runtimes, **policy_kwargs):
    registry = InferenceRegistry()
    for r in runtimes:
        registry.register(r)
    return InferenceLayer(registry=registry, policy=InferencePolicy(**policy_kwargs))


def _generate(layer):
    return asyncio.run(layer.generate(InferenceRequest(prompt="x")))


# ── Lectura estricta de NEXUS_LOCAL_ONLY (D3) ──────────────────────────

class TestStrictBool:
    def _fn(self):
        from backend.config import parse_strict_bool
        return parse_strict_bool

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
    def test_true_values(self, raw):
        value, warnings = self._fn()("NEXUS_LOCAL_ONLY", raw, False)
        assert value is True and warnings == []

    @pytest.mark.parametrize("raw", ["0", "false", "No", "off", ""])
    def test_false_values(self, raw):
        value, warnings = self._fn()("NEXUS_LOCAL_ONLY", raw, False)
        assert value is False and warnings == []

    def test_unset_uses_default(self):
        assert self._fn()("NEXUS_LOCAL_ONLY", None, False) == (False, [])

    @pytest.mark.parametrize("raw", ["ture", "verdadero", "2", SECRET_LOOKALIKE])
    def test_unrecognized_is_restrictive_with_warning(self, raw):
        value, warnings = self._fn()("NEXUS_LOCAL_ONLY", raw, False)
        assert value is True
        assert len(warnings) == 1 and "NEXUS_LOCAL_ONLY" in warnings[0]
        assert raw not in warnings[0], "la advertencia no debe repetir el valor recibido"


# ── Lista de runtimes cloud permitidos (A-03) ──────────────────────────

class TestParseCloudRuntimes:
    def _fn(self):
        from backend.config import parse_cloud_runtimes
        return parse_cloud_runtimes

    def test_default_is_gemini_only(self):
        assert self._fn()(None) == (("gemini",), [])

    def test_empty_means_no_cloud(self):
        assert self._fn()("") == ((), [])
        assert self._fn()("  ,  ") == ((), [])

    def test_explicit_list_normalized_and_deduplicated(self):
        value, warnings = self._fn()(" Gemini , GROQ,groq ,openrouter")
        assert value == ("gemini", "groq", "openrouter") and warnings == []

    def test_unknown_names_ignored_with_warning_without_echo(self):
        value, warnings = self._fn()("gemini," + SECRET_LOOKALIKE)
        assert value == ("gemini",)
        assert len(warnings) == 1 and "CLOUD_RUNTIMES_ENABLED" in warnings[0]
        assert SECRET_LOOKALIKE not in warnings[0]


# ── Modo efectivo: la política más restrictiva prevalece (D2, A-02) ────

class TestResolveInference:
    def _fn(self):
        from backend.config import resolve_inference_settings
        return resolve_inference_settings

    def test_defaults_preserved(self):
        assert self._fn()("local_first", True, False) == ("local_first", True, [])

    @pytest.mark.parametrize("mode", ["local_first", "local_only", "cloud_first", "cloud_only"])
    def test_valid_modes_untouched(self, mode):
        assert self._fn()(mode, True, False) == (mode, True, [])

    @pytest.mark.parametrize("raw", ["localfirst", "", "auto", SECRET_LOOKALIKE])
    def test_invalid_mode_falls_to_local_only_with_warning(self, raw):
        mode, allow, warnings = self._fn()(raw, True, False)
        assert (mode, allow) == ("local_only", False)
        assert len(warnings) == 1 and "INFERENCE_MODE" in warnings[0]
        assert raw == "" or raw not in warnings[0]

    @pytest.mark.parametrize("mode", ["local_first", "cloud_first", "cloud_only", "local_only"])
    def test_local_only_flag_forces_local_only(self, mode):
        assert self._fn()(mode, True, True)[:2] == ("local_only", False)

    def test_allow_cloud_false_is_never_relaxed(self):
        assert self._fn()("local_first", False, False)[1] is False
        assert self._fn()("cloud_only", False, False)[:2] == ("cloud_only", False)


# ── Política: lista blanca de runtimes cloud ───────────────────────────

class TestPolicyAllowlist:
    def test_none_means_no_filter_for_backward_compatibility(self):
        local, g, q = _Local(), _Cloud("gemini"), _Cloud("groq")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        assert [r.name for r in policy.select_runtimes([local, g, q])] == ["ollama", "gemini", "groq"]

    def test_allowlist_filters_cloud(self):
        local, g, q = _Local(), _Cloud("gemini"), _Cloud("groq")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_FIRST, allow_cloud=True,
                                 cloud_enabled=("gemini",))
        assert [r.name for r in policy.select_runtimes([local, g, q])] == ["ollama", "gemini"]
        assert policy.is_cloud_blocked(q) is True
        assert policy.is_cloud_blocked(g) is False

    def test_empty_allowlist_means_local_only_in_effect(self):
        local, g = _Local(), _Cloud("gemini")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_FIRST, allow_cloud=True,
                                 cloud_enabled=())
        assert [r.name for r in policy.select_runtimes([local, g])] == ["ollama"]

    def test_allowlist_applies_in_cloud_first_too(self):
        local, g, q = _Local(), _Cloud("gemini"), _Cloud("groq")
        policy = InferencePolicy(mode=InferenceMode.CLOUD_FIRST, allow_cloud=True,
                                 cloud_enabled=("gemini",))
        assert [r.name for r in policy.select_runtimes([local, g, q])] == ["gemini", "ollama"]


class TestNoAutomaticFallback:
    def test_gemini_failure_does_not_call_groq_or_openrouter(self):
        local = _Local()
        g, q, o = _Cloud("gemini", fail=True), _Cloud("groq"), _Cloud("openrouter")
        layer = _layer([local, g, q, o], mode=InferenceMode.LOCAL_FIRST,
                       allow_cloud=True, cloud_enabled=("gemini",))
        with pytest.raises(AllRuntimesFailed):
            _generate(layer)
        assert g.calls == 1
        assert q.calls == 0 and o.calls == 0

    def test_explicit_optin_still_allows_groq(self):
        local = _Local()
        g, q = _Cloud("gemini", fail=True), _Cloud("groq")
        layer = _layer([local, g, q], mode=InferenceMode.LOCAL_FIRST,
                       allow_cloud=True, cloud_enabled=("gemini", "groq"))
        assert _generate(layer).runtime_name == "groq"
        assert q.calls == 1

    def test_unfiltered_policy_keeps_legacy_fallback(self):
        local = _Local()
        g, q = _Cloud("gemini", fail=True), _Cloud("groq")
        layer = _layer([local, g, q], mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        assert _generate(layer).runtime_name == "groq"


class TestStatusExposesEffectivePolicy:
    def test_status_includes_allowlist_and_warnings(self):
        layer = _layer([_Local(), _Cloud("gemini")], mode=InferenceMode.LOCAL_ONLY,
                       allow_cloud=False, cloud_enabled=("gemini",))
        status = asyncio.run(layer.status())
        assert status["mode"] == "local_only"
        assert status["cloud_allowed"] is False
        assert status["cloud_enabled"] == ["gemini"]
        assert isinstance(status["config_warnings"], list)

    def test_status_cloud_enabled_none_when_unfiltered(self):
        layer = _layer([_Local()], mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        assert asyncio.run(layer.status())["cloud_enabled"] is None


# ── Aislamiento de bases de datos de la suite (D1 + A-04 mínimo) ───────

class TestSuiteDatabaseIsolation:
    def _paths(self):
        import backend.config as cfg
        return {
            "memory": cfg.MEMORY_DB_PATH,
            "knowledge": cfg.KNOWLEDGE_DB_PATH,
            "tasks": cfg.TASKS_DB_PATH,
        }

    def test_three_paths_absolute_distinct_inside_temp_root(self):
        root = os.path.realpath(os.environ["NEXUS_TEST_TMP"])
        paths = self._paths()
        assert len(set(paths.values())) == 3
        for name, p in paths.items():
            assert os.path.isabs(p), f"{name}: la ruta no es absoluta: {p}"
            assert os.path.realpath(p).startswith(root + os.sep), f"{name} fuera del temporal: {p}"

    def test_default_tasks_path_is_unchanged_when_env_absent(self):
        # La constante se calcula al importar; aquí se verifica el valor por defecto
        # documentado sin recargar el módulo: la función pública lo expone.
        from backend.config import DEFAULT_TASKS_DB_PATH
        assert DEFAULT_TASKS_DB_PATH == "nexus_tasks.db"
