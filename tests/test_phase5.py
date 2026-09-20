"""
NEXUS Ω — Phase 5 Tests: Local-First / Provider Independence

Pruebas A-F (fallback), G (local_only_no_cloud), H (desconexión simulada),
I (regresión), J (telemetría), K (errores tipados), L (health), M (policy).
"""

import asyncio
import pytest
from fastapi.testclient import TestClient


# ── Mock Runtimes ─────────────────────────────────────────────

from backend.inference.interface import InferenceRuntime
from backend.inference.models import (
    InferenceRequest, InferenceResult, InferenceMode,
    RuntimeCapabilities, RuntimeHealth, RuntimeStatus, RuntimeType, Message,
)
from backend.inference.errors import (
    AllRuntimesFailed, CloudInferenceBlocked, RuntimeUnavailable,
    InferenceRateLimited, InferenceTimeout,
)
from backend.inference.policy import InferencePolicy
from tools.registry import InferenceRegistry
from backend.inference.layer import InferenceLayer


class MockLocalRuntime(InferenceRuntime):
    def __init__(self, name="mock_local", text="respuesta local", fail=False):
        self._name = name; self._text = text; self._fail = fail; self.calls = 0
    @property
    def name(self): return self._name
    @property
    def runtime_type(self): return RuntimeType.LOCAL
    async def generate(self, req: InferenceRequest) -> InferenceResult:
        self.calls += 1
        if self._fail:
            raise RuntimeUnavailable("Runtime local caído", runtime=self._name)
        return InferenceResult(text=self._text, runtime_name=self._name,
            runtime_type=RuntimeType.LOCAL, model="mock-local", mode=None, is_local=True)
    async def health(self) -> RuntimeHealth:
        status = RuntimeStatus.UNAVAILABLE if self._fail else RuntimeStatus.AVAILABLE
        return RuntimeHealth(status=status, runtime_name=self._name,
            runtime_type=RuntimeType.LOCAL, model="mock-local")


class MockCloudRuntime(InferenceRuntime):
    def __init__(self, name="mock_cloud", text="respuesta cloud", fail=False, rate_limit=False):
        self._name = name; self._text = text; self._fail = fail
        self._rate_limit = rate_limit; self.calls = 0
    @property
    def name(self): return self._name
    @property
    def runtime_type(self): return RuntimeType.CLOUD
    async def generate(self, req: InferenceRequest) -> InferenceResult:
        self.calls += 1
        if self._rate_limit:
            raise InferenceRateLimited("429 rate limited", runtime=self._name)
        if self._fail:
            raise RuntimeUnavailable("Cloud caído", runtime=self._name)
        return InferenceResult(text=self._text, runtime_name=self._name,
            runtime_type=RuntimeType.CLOUD, model="mock-cloud", mode=None, is_local=False)
    async def health(self) -> RuntimeHealth:
        status = RuntimeStatus.UNAVAILABLE if self._fail else RuntimeStatus.AVAILABLE
        return RuntimeHealth(status=status, runtime_name=self._name,
            runtime_type=RuntimeType.CLOUD, model="mock-cloud")


def make_layer(local_runtimes=None, cloud_runtimes=None,
               mode=InferenceMode.LOCAL_FIRST, allow_cloud=True) -> InferenceLayer:
    registry = InferenceRegistry()
    for r in (local_runtimes or []):
        registry.register(r)
    for r in (cloud_runtimes or []):
        registry.register(r)
    policy = InferencePolicy(mode=mode, allow_cloud=allow_cloud)
    return InferenceLayer(registry=registry, policy=policy)


def make_req(prompt="hola") -> InferenceRequest:
    return InferenceRequest(prompt=prompt, system="Test")


# ============================================================
# TEST A — LOCAL_FIRST + local OK → LOCAL
# ============================================================

class TestA_LocalFirstLocalOK:

    @pytest.mark.asyncio
    async def test_local_responds_first(self):
        local = MockLocalRuntime("ollama", "respuesta ollama")
        cloud = MockCloudRuntime("gemini", "respuesta gemini")
        layer = make_layer([local], [cloud], mode=InferenceMode.LOCAL_FIRST)

        result = await layer.generate(make_req())
        assert result.runtime_name == "ollama"
        assert result.is_local is True
        assert result.fallback_used is False
        assert cloud.calls == 0  # cloud nunca se intentó

    @pytest.mark.asyncio
    async def test_is_local_flag_set(self):
        local = MockLocalRuntime()
        layer = make_layer([local], mode=InferenceMode.LOCAL_FIRST)
        result = await layer.generate(make_req())
        assert result.is_local is True
        assert result.runtime_type == RuntimeType.LOCAL


# ============================================================
# TEST B — LOCAL_FIRST + local FAIL + cloud permitido → CLOUD FALLBACK
# ============================================================

class TestB_LocalFailCloudFallback:

    @pytest.mark.asyncio
    async def test_fallback_to_cloud_when_local_fails(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini", "respuesta gemini")
        layer = make_layer([local], [cloud],
                           mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)

        result = await layer.generate(make_req())
        assert result.runtime_name == "gemini"
        assert result.is_local is False
        assert result.fallback_used is True
        assert local.calls == 1
        assert cloud.calls == 1

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_fallback(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini", rate_limit=True)
        cloud2 = MockCloudRuntime("groq", "groq responde")
        layer = make_layer([local], [cloud, cloud2],
                           mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        result = await layer.generate(make_req())
        assert result.runtime_name == "groq"


# ============================================================
# TEST C — LOCAL_FIRST + local FAIL + cloud prohibido → CONTROLLED FAILURE
# ============================================================

class TestC_LocalFailCloudBlocked:

    @pytest.mark.asyncio
    async def test_controlled_failure_when_cloud_blocked(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini")
        layer = make_layer([local], [cloud],
                           mode=InferenceMode.LOCAL_FIRST, allow_cloud=False)

        with pytest.raises(AllRuntimesFailed):
            await layer.generate(make_req())
        assert cloud.calls == 0  # cloud nunca intentado

    @pytest.mark.asyncio
    async def test_cloud_never_called_when_blocked(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini")
        layer = make_layer([local], [cloud],
                           mode=InferenceMode.LOCAL_FIRST, allow_cloud=False)
        try:
            await layer.generate(make_req())
        except AllRuntimesFailed:
            pass
        assert cloud.calls == 0


# ============================================================
# TEST D — LOCAL_ONLY + local FAIL → CONTROLLED FAILURE sin cloud
# ============================================================

class TestD_LocalOnlyFail:

    @pytest.mark.asyncio
    async def test_local_only_no_cloud_fallback(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini")
        layer = make_layer([local], [cloud], mode=InferenceMode.LOCAL_ONLY)

        with pytest.raises(AllRuntimesFailed):
            await layer.generate(make_req())
        assert cloud.calls == 0

    @pytest.mark.asyncio
    async def test_local_only_blocks_cloud_entirely(self):
        """LOCAL_ONLY: cloud no aparece en la lista de runtimes elegibles."""
        cloud = MockCloudRuntime("gemini")
        local = MockLocalRuntime("ollama")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_ONLY, allow_cloud=True)
        registry = InferenceRegistry()
        registry.register(local)
        registry.register(cloud)
        selected = policy.select_runtimes(registry.all_runtimes())
        names = [r.name for r in selected]
        assert "gemini" not in names
        assert "ollama" in names


# ============================================================
# TEST E — CLOUD_ONLY → CLOUD
# ============================================================

class TestE_CloudOnly:

    @pytest.mark.asyncio
    async def test_cloud_only_uses_cloud(self):
        local = MockLocalRuntime("ollama")
        cloud = MockCloudRuntime("gemini", "cloud responde")
        layer = make_layer([local], [cloud], mode=InferenceMode.CLOUD_ONLY, allow_cloud=True)
        result = await layer.generate(make_req())
        assert result.runtime_name == "gemini"
        assert local.calls == 0

    @pytest.mark.asyncio
    async def test_cloud_only_blocked_fails(self):
        cloud = MockCloudRuntime("gemini")
        layer = make_layer([], [cloud], mode=InferenceMode.CLOUD_ONLY, allow_cloud=False)
        with pytest.raises(CloudInferenceBlocked):
            await layer.generate(make_req())


# ============================================================
# TEST F — CLOUD_FIRST (DEPRECATED) → CLOUD
# ============================================================

class TestF_CloudFirstDeprecated:

    def test_cloud_first_is_deprecated(self):
        from backend.inference.models import DEPRECATED_MODES
        assert InferenceMode.CLOUD_FIRST in DEPRECATED_MODES

    @pytest.mark.asyncio
    async def test_cloud_first_uses_cloud_first(self):
        local = MockLocalRuntime("ollama")
        cloud = MockCloudRuntime("gemini", "cloud primero")
        layer = make_layer([local], [cloud], mode=InferenceMode.CLOUD_FIRST, allow_cloud=True)
        result = await layer.generate(make_req())
        assert result.runtime_name == "gemini"
        assert local.calls == 0


# ============================================================
# TEST G — LOCAL_ONLY: ninguna llamada cloud posible
# ============================================================

class TestG_LocalOnlyNoCloudDependency:

    @pytest.mark.asyncio
    async def test_local_only_no_cloud_dependency(self):
        """
        INFERENCE_MODE=LOCAL_ONLY + ALLOW_CLOUD_INFERENCE=false
        Todos los cloud simulados como caídos.
        NEXUS debe funcionar si local está disponible.
        """
        local  = MockLocalRuntime("ollama", "NEXUS local funcionando")
        clouds = [
            MockCloudRuntime("gemini", fail=True),
            MockCloudRuntime("groq",   fail=True),
            MockCloudRuntime("openrouter", fail=True),
        ]
        layer  = make_layer([local], clouds, mode=InferenceMode.LOCAL_ONLY, allow_cloud=False)
        result = await layer.generate(make_req("hola"))
        assert result.text == "NEXUS local funcionando"
        assert result.is_local is True
        for c in clouds:
            assert c.calls == 0

    @pytest.mark.asyncio
    async def test_nexus_core_local_only_no_cloud(self):
        """NexusCore con LOCAL_ONLY: directs van sin inferencia, otros van a local."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore

        local = MockLocalRuntime("ollama", "respuesta local completa")
        layer = make_layer([local], mode=InferenceMode.LOCAL_ONLY)
        mem   = Memory(RAMMemoryStore())
        core  = NexusCore(inference=layer, memory=mem, intent_router=IntentRouter())

        # Direct intent — sin inferencia
        result = await core.process("hola")
        assert result.provider == "system"
        assert local.calls == 0   # direct no usa inferencia

        # General query — usa local
        result = await core.process("que es machine learning")
        assert result.provider == "ollama"
        assert local.calls >= 1


# ============================================================
# TEST H — Prueba de desconexión simulada
# ============================================================

class TestH_SimulatedDisconnection:
    """
    SIMULATED — No se desconectó Internet físicamente.
    Se simulan todos los runtimes cloud como no disponibles.
    """

    @pytest.mark.asyncio
    async def test_simulated_offline_local_responds(self):
        """Simular: Internet desconectado, solo local disponible."""
        local = MockLocalRuntime("ollama", "respuesta sin internet")
        layer = make_layer([local], mode=InferenceMode.LOCAL_FIRST, allow_cloud=False)

        for prompt in ["Hola NEXUS", "Que es una red neuronal", "que hora es"]:
            req    = make_req(prompt)
            result = await layer.generate(req)
            assert result.text != ""
            assert result.is_local is True

    @pytest.mark.asyncio
    async def test_clock_tool_no_inference_needed(self):
        """Clock tool funciona sin ningún runtime de inferencia."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.executor import Executor
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.tools.builtin import create_default_registry

        mem       = Memory(RAMMemoryStore())
        reg       = create_default_registry(mem)
        executor = Executor(registry=reg, memory=mem)
        core      = NexusCore(
            inference=None,   # sin inferencia
            memory=mem,
            intent_router=IntentRouter(registry=reg),
            executor=executor,
        )
        result = await core.process("que hora es")
        assert result.provider == "tool"
        assert result.text != ""


# ============================================================
# TEST I — Regresión: endpoints existentes no rotos
# ============================================================

class TestI_Regression:

    def setup_method(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_health_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["version"] == "3.8.0"
        assert "inference_mode" in data
        assert "cloud_allowed" in data

    def test_chat_empty_static(self):
        r = self.client.post("/api/nexus/chat", json={"message": ""})
        assert r.status_code == 200
        assert r.json()["provider"] == "system"
        assert "request_id" in r.json()

    def test_chat_contract_preserved(self):
        r = self.client.post("/api/nexus/chat", json={"message": ""})
        data = r.json()
        for field in ["success", "response", "provider", "fallback", "request_id"]:
            assert field in data

    def test_intent_endpoint(self):
        r = self.client.post("/api/nexus/intent", json={"message": "hola"})
        assert r.status_code == 200
        assert r.json()["intent"] == "chat"

    def test_tools_endpoint(self):
        r = self.client.get("/api/nexus/tools")
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_autonomy_endpoint(self):
        r = self.client.get("/api/nexus/autonomy")
        assert r.status_code == 200
        assert "max_plan_steps" in r.json()

    def test_knowledge_domains(self):
        r = self.client.get("/api/knowledge/domains")
        assert r.status_code == 200
        assert "domains" in r.json()

    def test_inference_health_endpoint(self):
        r = self.client.get("/api/inference/health")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "mode" in data
        assert "local_runtimes" in data
        assert "cloud_runtimes" in data

    def test_aura_status(self):
        r = self.client.get("/api/aura/status")
        assert r.status_code == 200
        assert r.json()["system"] == "AURA"

    def test_sw_js(self):
        r = self.client.get("/sw.js")
        assert r.status_code == 200

    def test_head_root(self):
        r = self.client.head("/")
        assert r.status_code == 200


# ============================================================
# TEST J — Telemetría sin secretos
# ============================================================

class TestJ_Telemetry:

    @pytest.mark.asyncio
    async def test_result_has_telemetry_fields(self):
        local  = MockLocalRuntime("ollama", "ok")
        layer  = make_layer([local])
        result = await layer.generate(make_req())
        telem  = result.to_telemetry()
        assert "runtime" in telem
        assert "mode" in telem
        assert "is_local" in telem
        assert "success" in telem
        assert "duration_ms" in telem
        # Sin secretos
        telem_str = str(telem)
        for secret_marker in ["api_key", "password", "token", "bearer"]:
            assert secret_marker not in telem_str.lower()

    def test_safe_error_redacts_secrets(self):
        from backend.main import safe_error_message
        from backend.config import ALL_SECRETS
        fake = "sk-fake-secret-key-xyz"
        ALL_SECRETS.append(fake)
        try:
            safe = safe_error_message(Exception(f"Error con clave={fake}"))
            assert fake not in safe
            assert "[REDACTED]" in safe
        finally:
            ALL_SECRETS.remove(fake)


# ============================================================
# TEST K — Errores tipados
# ============================================================

class TestK_TypedErrors:

    def test_classify_rate_limit(self):
        from backend.inference.errors import classify_error, InferenceRateLimited
        e = classify_error(Exception("HTTP 429 RESOURCE_EXHAUSTED"), runtime="gemini")
        assert isinstance(e, InferenceRateLimited)
        assert e.code == "RATE_LIMITED"

    def test_classify_timeout(self):
        from backend.inference.errors import classify_error, InferenceTimeout
        e = classify_error(Exception("Request timed out"), runtime="ollama")
        assert isinstance(e, InferenceTimeout)

    def test_classify_connection(self):
        from backend.inference.errors import classify_error, RuntimeUnavailable
        e = classify_error(Exception("Connection refused"), runtime="ollama")
        assert isinstance(e, RuntimeUnavailable)

    def test_classify_auth(self):
        from backend.inference.errors import classify_error, InferenceAuthenticationError
        e = classify_error(Exception("401 Unauthorized API KEY invalid"), runtime="gemini")
        assert isinstance(e, InferenceAuthenticationError)

    @pytest.mark.asyncio
    async def test_all_runtimes_failed_has_details(self):
        local = MockLocalRuntime("ollama", fail=True)
        cloud = MockCloudRuntime("gemini", fail=True)
        layer = make_layer([local], [cloud],
                           mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        try:
            await layer.generate(make_req())
        except AllRuntimesFailed as e:
            assert len(e.runtime_errors) >= 1
            assert e.code == "ALL_RUNTIMES_FAILED"


# ============================================================
# TEST L — Health checks
# ============================================================

class TestL_HealthChecks:

    @pytest.mark.asyncio
    async def test_health_returns_all_runtimes(self):
        local = MockLocalRuntime("ollama")
        cloud = MockCloudRuntime("gemini")
        layer = make_layer([local], [cloud])
        health = await layer.health()
        assert "ollama" in health
        assert "gemini" in health

    @pytest.mark.asyncio
    async def test_status_has_required_fields(self):
        local = MockLocalRuntime("ollama")
        layer = make_layer([local])
        status = await layer.status()
        assert "inference_layer" in status
        assert "mode" in status
        assert "cloud_allowed" in status
        assert "local_runtimes" in status
        assert "cloud_runtimes" in status

    @pytest.mark.asyncio
    async def test_unavailable_runtime_health(self):
        local = MockLocalRuntime("ollama", fail=True)
        layer = make_layer([local])
        health = await layer.health()
        assert health["ollama"].status == RuntimeStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_llamacpp_scaffolded_status(self):
        from backend.inference.local.llamacpp import LlamaCppRuntime
        rt     = LlamaCppRuntime(model_path="")
        health = await rt.health()
        assert health.status in (RuntimeStatus.SCAFFOLDED, RuntimeStatus.NOT_CONFIGURED)


# ============================================================
# TEST M — InferencePolicy
# ============================================================

class TestM_Policy:

    def test_local_only_excludes_cloud(self):
        local  = MockLocalRuntime("ollama")
        cloud  = MockCloudRuntime("gemini")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_ONLY, allow_cloud=True)
        sel    = policy.select_runtimes([local, cloud])
        assert all(r.is_local for r in sel)

    def test_cloud_blocked_raises(self):
        cloud  = MockCloudRuntime("gemini")
        policy = InferencePolicy(mode=InferenceMode.CLOUD_ONLY, allow_cloud=False)
        with pytest.raises(CloudInferenceBlocked):
            policy.select_runtimes([cloud])

    def test_local_first_order(self):
        local  = MockLocalRuntime("ollama")
        cloud  = MockCloudRuntime("gemini")
        policy = InferencePolicy(mode=InferenceMode.LOCAL_FIRST, allow_cloud=True)
        sel    = policy.select_runtimes([local, cloud])
        assert sel[0].is_local is True
        assert sel[-1].is_local is False

    def test_cloud_first_deprecated_order(self):
        local  = MockLocalRuntime("ollama")
        cloud  = MockCloudRuntime("gemini")
        policy = InferencePolicy(mode=InferenceMode.CLOUD_FIRST, allow_cloud=True)
        sel    = policy.select_runtimes([local, cloud])
        assert sel[0].is_local is False

    def test_nexus_core_has_no_provider_imports(self):
        """NexusCore no importa directamente Gemini, Groq, Ollama, etc."""
        import inspect
        from backend.core import nexus as nexus_module
        source = inspect.getsource(nexus_module)
        forbidden = ["from backend.providers.gemini", "from backend.providers.groq",
                     "from backend.providers.openrouter", "from backend.providers.ollama",
                     "import google.genai", "import httpx\nfrom backend"]
        for f in forbidden:
            assert f not in source, f"NexusCore contiene referencia prohibida: {f}"