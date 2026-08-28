"""
NEXUS Ω — Tests de modo offline y AURA Brain.

Prueba que NEXUS funciona sin Gemini, OpenRouter ni Groq.
"""

import pytest
from fastapi.testclient import TestClient


# ── Mock LocalProvider que siempre responde ──────────────────

from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse

class MockLocalProvider(BaseModelProvider):
    is_local: bool = True

    @property
    def name(self) -> str: return "local"
    @property
    def model(self) -> str: return "mock-local-v1"
    @property
    def is_configured(self) -> bool: return True

    async def generate(self, req: GenerateRequest) -> ProviderResponse:
        return ProviderResponse(
            text=f"Respuesta local a: {req.prompt[:50]}",
            provider="local", model="mock-local-v1", duration_ms=5,
        )


# ── Router offline tests ─────────────────────────────────────

from backend.router import ModelRouter, RouterResult
from backend.providers.base import Message

def make_req(prompt="Hola"):
    return GenerateRequest(prompt=prompt, system="Test")


class TestOfflineMode:

    @pytest.mark.asyncio
    async def test_local_provider_responds_without_internet(self):
        """El LocalProvider responde cuando todos los externos están down."""
        local = MockLocalProvider()
        router = ModelRouter([local])
        result = await router.generate(make_req())

        assert result.response.provider == "local"
        assert result.response.text != ""
        assert result.fallback is False

    @pytest.mark.asyncio
    async def test_local_only_mode_skips_external(self):
        """Con NEXUS_LOCAL_ONLY, los providers externos se ignoran aunque estén configurados."""
        import backend.config as cfg
        original = cfg.NEXUS_LOCAL_ONLY
        cfg.NEXUS_LOCAL_ONLY = True

        try:
            from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse

            class FakeExternalProvider(BaseModelProvider):
                is_local = False
                @property
                def name(self): return "gemini"
                @property
                def model(self): return "gemini-test"
                @property
                def is_configured(self): return True
                async def generate(self, req): raise AssertionError("No debería llamarse en LOCAL_ONLY")

            local    = MockLocalProvider()
            external = FakeExternalProvider()
            router   = ModelRouter([local, external])

            result = await router.generate(make_req())
            assert result.response.provider == "local"

        finally:
            cfg.NEXUS_LOCAL_ONLY = original

    @pytest.mark.asyncio
    async def test_local_mode_raises_if_no_local_provider(self):
        """LOCAL_ONLY sin provider local configurado → RuntimeError claro."""
        import backend.config as cfg
        original = cfg.NEXUS_LOCAL_ONLY
        cfg.NEXUS_LOCAL_ONLY = True

        try:
            from backend.providers.base import BaseModelProvider

            class UnconfiguredLocal(BaseModelProvider):
                is_local = True
                @property
                def name(self): return "local"
                @property
                def model(self): return "none"
                @property
                def is_configured(self): return False
                async def generate(self, req): raise RuntimeError("never")

            router = ModelRouter([UnconfiguredLocal()])
            with pytest.raises(RuntimeError, match="NEXUS_LOCAL_ONLY"):
                await router.generate(make_req())
        finally:
            cfg.NEXUS_LOCAL_ONLY = original

    @pytest.mark.asyncio
    async def test_fallback_flag_true_when_local_is_primary_but_external_responds(self):
        """Si local falla y responde external, fallback debe ser True."""
        from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse

        class FailingLocal(BaseModelProvider):
            is_local = True
            @property
            def name(self): return "local"
            @property
            def model(self): return "none"
            @property
            def is_configured(self): return True
            async def generate(self, req): raise RuntimeError("local no disponible")

        class WorkingExternal(BaseModelProvider):
            is_local = False
            @property
            def name(self): return "gemini"
            @property
            def model(self): return "gemini-test"
            @property
            def is_configured(self): return True
            async def generate(self, req):
                return ProviderResponse(text="OK externo", provider="gemini",
                                        model="gemini-test", duration_ms=10)

        router = ModelRouter([FailingLocal(), WorkingExternal()])
        result = await router.generate(make_req())

        assert result.response.provider == "gemini"
        assert result.fallback is True


# ── AURA Brain tests ─────────────────────────────────────────

from backend.core.perception import Perception, PerceptionEvent, Modality
from backend.core.memory import Memory, MemoryType
from backend.core.reasoning import Reasoning, ReasoningStrategy
from backend.simulation.engine import SimulationEngine, SimulationScenario


class TestPerception:

    def test_receive_text_event(self):
        p = Perception()
        event = PerceptionEvent(modality=Modality.TEXT, data="Hola NEXUS", source="user")
        p.receive(event)
        assert p.latest(Modality.TEXT) == event

    def test_queue_drains_on_pending(self):
        p = Perception()
        p.receive(PerceptionEvent(modality=Modality.TEXT, data="A", source="test"))
        p.receive(PerceptionEvent(modality=Modality.IMU, data={}, source="imu_0"))
        events = p.pending()
        assert len(events) == 2
        assert len(p.pending()) == 0  # vacío tras drenar

    def test_to_context_text(self):
        p = Perception()
        p.receive(PerceptionEvent(modality=Modality.TEXT, data="test", source="user"))
        ctx = p.to_context()
        assert "TEXT" in ctx
        assert "user" in ctx


class TestMemory:

    def test_remember_and_recall(self):
        m = Memory()
        m.remember("NEXUS usa Gemini como provider primario", tags=["nexus", "gemini"])
        results = m.recall("Gemini")
        assert any("Gemini" in r.content for r in results)

    def test_working_context(self):
        m = Memory()
        m.remember("Mensaje 1", MemoryType.WORKING)
        m.remember("Mensaje 2", MemoryType.WORKING)
        ctx = m.working_context()
        assert "Mensaje 1" in ctx or "Mensaje 2" in ctx

    def test_stats(self):
        m = Memory()
        m.remember("A", MemoryType.WORKING)
        m.remember("B", MemoryType.EPISODIC)
        stats = m.stats()
        assert stats["total"] == 2


class TestReasoning:

    def test_direct_response_for_hora(self):
        r = Reasoning()
        result = r.analyze("hora")
        assert result.strategy == ReasoningStrategy.DIRECT
        assert result.direct_response is not None

    def test_llm_strategy_for_complex_prompt(self):
        r = Reasoning()
        result = r.analyze("Explícame la teoría de juegos")
        assert result.strategy == ReasoningStrategy.LLM

    def test_tool_strategy_for_search(self):
        r = Reasoning()
        result = r.analyze("busca noticias de IA hoy")
        assert result.strategy == ReasoningStrategy.TOOL
        assert result.tool_name == "web_search"


class TestSimulation:

    def test_idle_scenario_generates_imu(self):
        sim = SimulationEngine()
        sim.set_scenario(SimulationScenario.IDLE)
        events = sim.tick()
        assert any(e.modality == Modality.IMU for e in events)

    def test_exploring_generates_lidar(self):
        sim = SimulationEngine()
        sim.set_scenario(SimulationScenario.EXPLORING)
        events = sim.tick()
        assert any(e.modality == Modality.LIDAR for e in events)

    def test_obstacle_scenario_lidar_has_close_point(self):
        sim = SimulationEngine()
        sim.set_scenario(SimulationScenario.OBSTACLE)
        events = sim.tick()
        lidar_events = [e for e in events if e.modality == Modality.LIDAR]
        assert len(lidar_events) > 0
        assert lidar_events[0].data.get("obstacle_detected") is True

    def test_inject_text(self):
        sim = SimulationEngine()
        event = sim.inject_text("Test de usuario", source="user")
        assert event.modality == Modality.TEXT
        assert event.data == "Test de usuario"

    def test_multiple_ticks(self):
        sim = SimulationEngine()
        sim.set_scenario(SimulationScenario.EXPLORING)
        for _ in range(5):
            sim.tick()
        events = sim.pop_events()
        assert len(events) > 0


# ── API Brain endpoints ───────────────────────────────────────

class TestAURARest:

    def setup_method(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_aura_status_returns_200(self):
        r = self.client.get("/api/aura/status")
        assert r.status_code == 200
        data = r.json()
        assert data["system"] == "AURA"
        assert "brain" in data

    def test_perceive_text_event(self):
        r = self.client.post("/api/aura/perceive", json={
            "modality": "text", "data": "Hola NEXUS", "source": "test"
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_perceive_invalid_modality(self):
        r = self.client.post("/api/aura/perceive", json={
            "modality": "telepathy", "data": "test", "source": "test"
        })
        assert r.status_code == 422

    def test_simulate_idle(self):
        r = self.client.post("/api/aura/simulate", json={
            "scenario": "idle", "ticks": 2
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["events"] > 0

    def test_simulate_obstacle(self):
        r = self.client.post("/api/aura/simulate", json={
            "scenario": "obstacle", "ticks": 1
        })
        assert r.status_code == 200

    def test_simulate_invalid_scenario(self):
        r = self.client.post("/api/aura/simulate", json={
            "scenario": "flying", "ticks": 1
        })
        assert r.status_code == 422
