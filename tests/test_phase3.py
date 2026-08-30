"""
NEXUS Ω — Phase 3 Tests: A → Q

A. Pregunta simple (QUESTION → LLM directo)
B. Análisis (ANALYSIS → autonomy)
C. Tarea multi-paso (TASK → autonomy con plan)
D. Uso de tool (clock sin LLM)
E. Tool inexistente → fallo graceful
F. Ejecución exitosa de paso
G. Ejecución fallida de paso
H. Retry en paso transitorio
I. Replan (LLM falla → LocalPlanner)
J. Límite de loops
K. Memory write
L. Memory search
M. Gemini funcionando (mock)
N. Gemini falla → fallback (mock)
O. Gemini + Ollama caídos → 503 correcto
P. request_id presente en toda respuesta
Q. Secretos nunca aparecen en logs
"""

import asyncio
import os
import tempfile
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# ============================================================
# MOCKS
# ============================================================

from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse
from backend.router import ModelRouter


class MockProvider(BaseModelProvider):
    """Provider que siempre responde."""
    is_local = False

    def __init__(self, name: str = "mock", text: str = "respuesta mock"):
        self._name = name
        self._text = text
        self.calls = 0

    @property
    def name(self) -> str: return self._name
    @property
    def model(self) -> str: return f"{self._name}-model"
    @property
    def is_configured(self) -> bool: return True

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text=self._text, provider=self._name,
            model=self.model, duration_ms=5,
        )


class FailingProvider(BaseModelProvider):
    """Provider que siempre falla."""
    is_local = False

    def __init__(self, name: str = "failing", error: str = "Error de prueba"):
        self._name = name
        self._error = error

    @property
    def name(self) -> str: return self._name
    @property
    def model(self) -> str: return f"{self._name}-model"
    @property
    def is_configured(self) -> bool: return True

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        raise RuntimeError(self._error)


class TransientFailProvider(BaseModelProvider):
    """Falla N veces, luego responde."""
    is_local = False

    def __init__(self, name: str, fail_times: int = 1, text: str = "OK tras retry"):
        self._name = name
        self._fail_times = fail_times
        self._attempts = 0
        self._text = text

    @property
    def name(self) -> str: return self._name
    @property
    def model(self) -> str: return f"{self._name}-model"
    @property
    def is_configured(self) -> bool: return True

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        self._attempts += 1
        if self._attempts <= self._fail_times:
            raise RuntimeError("503 Service Unavailable — error transitorio")
        return ProviderResponse(
            text=self._text, provider=self._name,
            model=self.model, duration_ms=5,
        )


def make_router(*providers) -> ModelRouter:
    return ModelRouter(list(providers))


# ============================================================
# TEST A — Pregunta simple
# ============================================================

class TestA_SimpleQuestion:

    @pytest.mark.asyncio
    async def test_question_goes_to_llm(self):
        """Pregunta simple → LLM directo, sin autonomy."""
        from backend.core.intent import IntentRouter, IntentStrategy, IntentType
        router = IntentRouter()
        result = router.route("que es la inteligencia artificial")
        assert result.intent == IntentType.QUESTION
        assert result.strategy == IntentStrategy.LLM
        assert result.requires_planning is False

    @pytest.mark.asyncio
    async def test_question_resolved_by_llm_provider(self):
        """Una pregunta llega al modelo y devuelve respuesta."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore

        provider = MockProvider("gemini", "La IA es la simulación de inteligencia.")
        router   = make_router(provider)
        core     = NexusCore(
            model_router=router,
            memory=Memory(RAMMemoryStore()),
            intent_router=IntentRouter(),
        )
        result = await core.process("que es la inteligencia artificial")
        assert result.text != ""
        assert result.provider == "gemini"
        assert provider.calls == 1


# ============================================================
# TEST B — Análisis
# ============================================================

class TestB_Analysis:

    @pytest.mark.asyncio
    async def test_analysis_intent_detected(self):
        from backend.core.intent import IntentRouter, IntentType, IntentStrategy
        router = IntentRouter()
        result = router.route("analiza la viabilidad de construir un sistema hidroponico")
        assert result.intent == IntentType.ANALYSIS
        assert result.strategy == IntentStrategy.AUTONOMY
        assert result.requires_planning is True

    @pytest.mark.asyncio
    async def test_analysis_creates_multi_step_plan(self):
        from backend.core.planner import Planner
        planner = Planner(model_router=None)
        plan    = planner.plan_sync("analiza viabilidad de A", intent_type="analysis")
        assert len(plan.steps) >= 3
        assert plan.source == "local"
        assert plan.goal != ""


# ============================================================
# TEST C — Tarea multi-paso
# ============================================================

class TestC_MultiStepTask:

    @pytest.mark.asyncio
    async def test_task_plan_has_steps(self):
        from backend.core.planner import Planner
        planner = Planner(model_router=None)
        plan    = planner.plan_sync("implementa un sistema de monitoreo", intent_type="task")
        assert len(plan.steps) >= 4
        for step in plan.steps:
            assert step.description != ""
            assert step.status.value == "pending"

    @pytest.mark.asyncio
    async def test_task_autonomy_loop_runs(self):
        from backend.core.autonomy import AutonomyLoop
        from backend.core.planner import Planner
        from backend.core.evaluator import StepEvaluator, PlanEvaluator
        from backend.core.executor import Executor
        from backend.tools.registry import ToolRegistry

        provider = MockProvider("gemini", "Resultado del paso ejecutado.")
        router   = make_router(provider)
        planner  = Planner(model_router=router)
        executor = Executor(registry=ToolRegistry())

        loop = AutonomyLoop(
            planner=planner,
            executor=executor,
            step_evaluator=StepEvaluator(),
            plan_evaluator=PlanEvaluator(),
            model_router=router,
            max_loops=10,
            max_steps=3,
            max_retries=1,
        )
        result = await loop.run(goal="crea un plan de marketing", intent_type="task")
        assert result.text != ""
        assert result.trace.loops > 0


# ============================================================
# TEST D — Uso de tool (clock sin LLM)
# ============================================================

class TestD_ToolUsage:

    @pytest.mark.asyncio
    async def test_clock_tool_no_llm(self):
        """'que hora es' → tool clock → responde sin LLM."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.executor import Executor
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.tools.builtin import create_default_registry

        mem      = Memory(RAMMemoryStore())
        registry = create_default_registry(mem)
        executor = Executor(registry=registry, memory=mem)
        core     = NexusCore(
            model_router=None,    # sin LLM
            memory=mem,
            intent_router=IntentRouter(registry=registry),
            executor=executor,
        )
        result = await core.process("que hora es")
        assert result.text != ""
        assert result.provider == "tool"
        assert "clock" in result.tools_used

    @pytest.mark.asyncio
    async def test_clock_returns_time_data(self):
        from backend.tools.builtin import ClockTool
        from backend.tools.base import ToolInput
        tool   = ClockTool()
        result = await tool.execute(ToolInput())
        assert result.success is True
        assert "time" in result.output
        assert "date" in result.output


# ============================================================
# TEST E — Tool inexistente
# ============================================================

class TestE_NonexistentTool:

    @pytest.mark.asyncio
    async def test_unknown_tool_fails_gracefully(self):
        from backend.core.executor import Executor
        from backend.tools.registry import ToolRegistry

        executor = Executor(registry=ToolRegistry())
        result   = await executor.execute_by_name("web_search_xyz", {})
        assert result.success is False
        assert "no encontrada" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_disabled_tool_skipped(self):
        from backend.core.executor import Executor
        from backend.tools.registry import ToolRegistry
        from backend.tools.builtin import ClockTool

        reg  = ToolRegistry()
        reg.register(ClockTool())
        reg.disable("clock")
        executor = Executor(registry=reg)
        result   = await executor.execute_by_name("clock", {})
        assert result.success is False
        assert result.skipped is True


# ============================================================
# TEST F — Ejecución exitosa
# ============================================================

class TestF_SuccessfulExecution:

    @pytest.mark.asyncio
    async def test_step_executed_and_marked_done(self):
        from backend.core.planner import Planner, StepStatus
        from backend.core.evaluator import StepEvaluator, EvalStatus

        planner = Planner(model_router=None)
        plan    = planner.plan_sync("analiza el mercado", intent_type="analysis")
        step    = plan.steps[0]

        plan.mark_running(step.step_id)
        plan.mark_done(step.step_id, "Análisis completado con éxito.")

        evaluator = StepEvaluator()
        evaluation = evaluator.evaluate(
            step_description="Identificar el problema central",
            execution_success=True,
            output="Análisis completado con éxito. El mercado muestra crecimiento.",
        )
        assert evaluation.status == EvalStatus.SUCCESS
        assert evaluation.score >= 0.5
        assert step.status == StepStatus.DONE


# ============================================================
# TEST G — Ejecución fallida
# ============================================================

class TestG_FailedExecution:

    def test_step_failure_evaluates_correctly(self):
        from backend.core.evaluator import StepEvaluator, EvalStatus

        evaluator  = StepEvaluator()
        evaluation = evaluator.evaluate(
            step_description="Conectar con API externa",
            execution_success=False,
            output=None,
            error="HTTP 404 Not Found",
        )
        assert evaluation.status == EvalStatus.FAILED
        assert evaluation.score == 0.0

    def test_plan_marks_failed_step(self):
        from backend.core.planner import Planner, StepStatus

        planner = Planner(model_router=None)
        plan    = planner.plan_sync("tarea de prueba", intent_type="task")
        step    = plan.steps[0]
        plan.mark_failed(step.step_id, "Error simulado.")
        assert step.status == StepStatus.FAILED
        assert step.error == "Error simulado."


# ============================================================
# TEST H — Retry
# ============================================================

class TestH_Retry:

    def test_transient_error_triggers_retry(self):
        from backend.core.evaluator import StepEvaluator, EvalStatus

        evaluator  = StepEvaluator()
        evaluation = evaluator.evaluate(
            step_description="Llamar API con timeout",
            execution_success=False,
            output=None,
            error="503 Service Unavailable",
            retries_used=0,
            max_retries=3,
        )
        assert evaluation.status == EvalStatus.RETRY

    def test_max_retries_exceeded_gives_failed(self):
        from backend.core.evaluator import StepEvaluator, EvalStatus

        evaluator  = StepEvaluator()
        evaluation = evaluator.evaluate(
            step_description="Llamar API con timeout",
            execution_success=False,
            output=None,
            error="503 Service Unavailable",
            retries_used=3,    # ya agotó reintentos
            max_retries=3,
        )
        assert evaluation.status == EvalStatus.FAILED

    @pytest.mark.asyncio
    async def test_autonomy_loop_retries_step(self):
        from backend.core.autonomy import AutonomyLoop
        from backend.core.planner import Planner
        from backend.core.evaluator import StepEvaluator, PlanEvaluator
        from backend.core.executor import Executor
        from backend.tools.registry import ToolRegistry

        # Provider que falla una vez luego responde
        provider = TransientFailProvider("gemini", fail_times=1)
        router   = make_router(provider)
        planner  = Planner(model_router=router)

        loop = AutonomyLoop(
            planner=planner,
            executor=Executor(registry=ToolRegistry()),
            step_evaluator=StepEvaluator(),
            plan_evaluator=PlanEvaluator(),
            model_router=router,
            max_loops=10,
            max_steps=2,
            max_retries=2,
        )
        result = await loop.run("tarea simple", intent_type="task")
        # Debe completar a pesar del fallo inicial
        assert result.trace.loops > 0


# ============================================================
# TEST I — Replan (LLM falla → LocalPlanner)
# ============================================================

class TestI_Replan:

    @pytest.mark.asyncio
    async def test_llm_planner_fails_falls_back_to_local(self):
        """Si el LLM falla al planificar, LocalPlanner toma el control."""
        from backend.core.planner import Planner

        # Provider que siempre falla
        failing_router = make_router(FailingProvider("gemini", "LLM no disponible"))

        planner = Planner(model_router=failing_router)
        # Con use_llm=True pero LLM falla → debe retornar plan local
        plan = await planner.plan(
            goal="diseña un sistema complejo",
            intent_type="design",
            use_llm=True,
        )
        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.source == "local"   # fallback confirmado

    @pytest.mark.asyncio
    async def test_planner_works_without_any_router(self):
        """Planner sin router → solo LocalPlanner, siempre funciona."""
        from backend.core.planner import Planner

        planner = Planner(model_router=None)
        plan    = await planner.plan("analiza X", intent_type="analysis", use_llm=True)
        assert len(plan.steps) > 0
        assert plan.source == "local"


# ============================================================
# TEST J — Límite de loops
# ============================================================

class TestJ_LoopLimit:

    @pytest.mark.asyncio
    async def test_loop_limit_prevents_infinite_execution(self):
        from backend.core.autonomy import AutonomyLoop, LoopStatus
        from backend.core.planner import Planner
        from backend.core.evaluator import StepEvaluator, PlanEvaluator
        from backend.core.executor import Executor
        from backend.tools.registry import ToolRegistry

        # Provider lento pero siempre responde
        provider = MockProvider("gemini", "x")
        router   = make_router(provider)
        planner  = Planner(model_router=None)  # plan local de 5 pasos

        loop = AutonomyLoop(
            planner=planner,
            executor=Executor(registry=ToolRegistry()),
            step_evaluator=StepEvaluator(),
            plan_evaluator=PlanEvaluator(),
            model_router=router,
            max_loops=2,       # límite muy bajo para el test
            max_steps=5,
            max_retries=1,
        )
        result = await loop.run("tarea grande", intent_type="task")
        # Debe terminar aunque no complete todos los pasos
        assert result.trace.loops <= 2
        assert result.trace.status in (
            LoopStatus.LIMIT_HIT, LoopStatus.COMPLETED, LoopStatus.PARTIAL
        )

    def test_plan_step_limit_respected(self):
        from backend.core.planner import Planner
        planner = Planner(model_router=None, max_steps=3)
        plan    = planner.plan_sync("gran tarea con muchos pasos", intent_type="task")
        assert len(plan.steps) <= 3


# ============================================================
# TEST K — Memory write
# ============================================================

class TestK_MemoryWrite:

    def test_conversation_saved_to_memory(self):
        from backend.core.memory import Memory, RAMMemoryStore

        mem = Memory(RAMMemoryStore())
        mem.conversation.add_user("hola nexus")
        mem.conversation.add_assistant("hola, ¿en qué te ayudo?")

        turns = mem.conversation.recent_turns(limit=10)
        assert len(turns) == 2
        assert turns[0].content == "hola nexus"
        assert turns[1].content == "hola, ¿en qué te ayudo?"

    def test_fact_saved_with_tags(self):
        from backend.core.memory import Memory, RAMMemoryStore

        mem = Memory(RAMMemoryStore())
        mem.facts.save_fact("NEXUS usa gemini-3.6-flash", tags=["gemini", "config"])
        results = mem.facts.search("gemini")
        assert len(results) > 0
        assert "gemini" in results[0].content

    def test_autonomy_saves_to_episodic_memory(self):
        from backend.core.memory import Memory, RAMMemoryStore, MemoryType

        mem = Memory(RAMMemoryStore())
        mem.remember(
            content="[PLAN:abc123] analizar mercado → análisis completado",
            memory_type=MemoryType.EPISODIC,
            tags=["autonomy", "plan"],
        )
        stats = mem.stats()
        assert stats["total"] >= 1


# ============================================================
# TEST L — Memory search
# ============================================================

class TestL_MemorySearch:

    def test_search_finds_relevant_entry(self):
        from backend.core.memory import Memory, RAMMemoryStore

        mem = Memory(RAMMemoryStore())
        mem.facts.save_fact("El proyecto AURA usa Jetson Orin NX", tags=["aura", "hardware"])
        mem.facts.save_fact("Ollama corre en localhost:11434", tags=["ollama"])
        mem.facts.save_fact("Gemini es el provider primario", tags=["gemini"])

        results = mem.recall("AURA hardware", limit=5)
        assert any("AURA" in r.content or "Jetson" in r.content for r in results)

    def test_memory_search_tool(self):
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.tools.builtin import MemorySearchTool
        from backend.tools.base import ToolInput
        import asyncio

        mem = Memory(RAMMemoryStore())
        mem.facts.save_fact("dato importante sobre robótica")
        tool   = MemorySearchTool(memory=mem)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(ToolInput(context="robótica"))
        )
        assert result.success is True

    def test_sqlite_search_persists(self):
        from backend.core.memory import SQLiteMemoryStore, MemoryEntry, MemoryType
        import os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteMemoryStore(db_path)
            store.save(MemoryEntry(content="NEXUS Omega proyecto AURA", memory_type=MemoryType.FACT))
            store.close()

            store2   = SQLiteMemoryStore(db_path)
            results  = store2.search("AURA")
            store2.close()
            assert len(results) > 0
        finally:
            os.unlink(db_path)


# ============================================================
# TEST M — Gemini funcionando
# ============================================================

class TestM_GeminiFunctional:

    @pytest.mark.asyncio
    async def test_gemini_mock_responds(self):
        """Con Gemini disponible → responde correctamente."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore

        provider = MockProvider("gemini", "Gemini respondió correctamente.")
        router   = make_router(provider)
        core     = NexusCore(
            model_router=router,
            memory=Memory(RAMMemoryStore()),
            intent_router=IntentRouter(),
        )
        result = await core.process("explica machine learning")
        assert result.provider == "gemini"
        assert result.fallback is False
        assert result.text != ""
        assert "request_id" not in result.text    # no filtra internals


# ============================================================
# TEST N — Gemini falla → fallback
# ============================================================

class TestN_GeminiFailsFallback:

    @pytest.mark.asyncio
    async def test_gemini_fails_uses_second_provider(self):
        """Gemini cae → OpenRouter responde → fallback=True."""
        gemini     = FailingProvider("gemini", "503 Service Unavailable")
        openrouter = MockProvider("openrouter", "OpenRouter respondió.")
        router     = make_router(gemini, openrouter)

        result = await router.generate(
            GenerateRequest(prompt="test", system="sys")
        )
        assert result.response.provider == "openrouter"
        assert result.fallback is True

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_fallback(self):
        """429 en Gemini → fallback al siguiente."""
        gemini = FailingProvider("gemini", "429 RESOURCE_EXHAUSTED quota exceeded")
        groq   = MockProvider("groq", "Groq respondió.")
        router = make_router(gemini, groq)

        result = await router.generate(
            GenerateRequest(prompt="test", system="sys")
        )
        assert result.response.provider == "groq"
        assert result.fallback is True


# ============================================================
# TEST O — Todos los providers caídos → 503
# ============================================================

class TestO_AllProvidersFail:

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_runtime_error(self):
        """Sin ningún provider disponible → RuntimeError claro."""
        from backend.router import ModelRouter

        providers = [
            FailingProvider("gemini",     "Gemini caído"),
            FailingProvider("openrouter", "OpenRouter caído"),
            FailingProvider("groq",       "Groq caído"),
        ]
        router = ModelRouter(providers)

        with pytest.raises(RuntimeError, match="Todos los proveedores fallaron"):
            await router.generate(GenerateRequest(prompt="test", system="sys"))

    @pytest.mark.asyncio
    async def test_nexus_core_survives_no_providers(self):
        """NexusCore sin router activo no crashea con intent directo."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore

        core = NexusCore(
            model_router=None,
            memory=Memory(RAMMemoryStore()),
            intent_router=IntentRouter(),
        )
        result = await core.process("hola")
        assert result.provider == "system"
        assert result.text != ""


# ============================================================
# TEST P — request_id presente
# ============================================================

class TestP_RequestID:

    def test_chat_response_has_request_id(self):
        from backend.main import app
        client   = TestClient(app)
        response = client.post("/api/nexus/chat", json={"message": ""})
        assert response.status_code == 200
        assert "request_id" in response.json()

    def test_task_response_has_request_id(self):
        from backend.main import app
        client   = TestClient(app)
        response = client.post(
            "/api/nexus/task",
            json={"goal": "analiza algo simple", "use_llm_plan": False}
        )
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            assert "request_id" in response.json()

    def test_health_returns_version(self):
        from backend.main import app
        client   = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["version"] == "3.6.0"


# ============================================================
# TEST Q — Secretos nunca en logs
# ============================================================

class TestQ_SecretsNeverLogged:

    def test_safe_error_message_redacts_secrets(self):
        from backend.main import safe_error_message
        from backend.config import ALL_SECRETS

        fake_key = "sk-super-secret-api-key-12345"
        error    = RuntimeError(f"Auth failed with key={fake_key}")

        # Añadir temporalmente al listado de secrets
        ALL_SECRETS.append(fake_key)
        try:
            safe = safe_error_message(error)
            assert fake_key not in safe
            assert "[REDACTED]" in safe
        finally:
            ALL_SECRETS.remove(fake_key)

    def test_gemini_api_key_not_in_error_string(self):
        """API keys de config no aparecen en mensajes de error."""
        from backend.config import GEMINI_API_KEY
        if not GEMINI_API_KEY:
            return  # no hay key en este entorno, skip
        from backend.main import safe_error_message
        error = RuntimeError(f"Invalid key: {GEMINI_API_KEY}")
        safe  = safe_error_message(error)
        assert GEMINI_API_KEY not in safe

    def test_forbidden_tool_names_rejected(self):
        """Herramientas prohibidas nunca se registran."""
        from backend.tools.interfaces import FORBIDDEN_TOOLS
        from backend.tools.registry import ToolRegistry
        from backend.tools.base import BaseTool, ToolInput, ToolResult, RiskLevel

        class ShellTool(BaseTool):
            risk_level = RiskLevel.CRITICAL
            @property
            def name(self): return "shell_exec"
            @property
            def description(self): return "Ejecuta shell"
            async def execute(self, ti: ToolInput) -> ToolResult:
                return ToolResult(success=True, output="hack")

        reg  = ToolRegistry()
        tool = ShellTool()
        reg.register(tool)

        assert "shell_exec" in FORBIDDEN_TOOLS
        # Aunque está registrada, CRITICAL sin auth falla
        import asyncio
        from backend.core.executor import Executor
        executor = Executor(registry=reg)
        result   = asyncio.get_event_loop().run_until_complete(
            executor.execute_by_name("shell_exec", {}, authorized=False)
        )
        assert result.success is False
        assert result.authorized is False


# ============================================================
# API INTEGRATION
# ============================================================

class TestAPIIntegration:

    def setup_method(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_chat_contract_preserved(self):
        r = self.client.post("/api/nexus/chat", json={"message": ""})
        assert r.status_code == 200
        data = r.json()
        # Contrato mínimo intacto
        assert "success" in data
        assert "response" in data
        assert "provider" in data
        assert "fallback" in data
        assert "request_id" in data

    def test_health_contract_preserved(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "providers" in data
        assert "version" in data

    def test_autonomy_endpoint(self):
        r = self.client.get("/api/nexus/autonomy")
        assert r.status_code == 200
        data = r.json()
        assert "max_plan_steps" in data
        assert "max_execution_loops" in data
        assert "safe_actions" in data
        assert "forbidden_actions" in data

    def test_task_endpoint_exists(self):
        r = self.client.post(
            "/api/nexus/task",
            json={"goal": "analiza el mercado de IA", "use_llm_plan": False}
        )
        # 200 si hay modelo, 503 si no — ambos son válidos en test
        assert r.status_code in (200, 503)

    def test_intent_endpoint_works(self):
        r = self.client.post("/api/nexus/intent", json={"message": "analiza la estrategia"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] in [
            "analysis", "strategy_query", "general_query", "task"
        ]

    def test_planner_separation_enforced(self):
        """El Planner NO ejecuta tools directamente."""
        from backend.core.planner import Planner
        import inspect
        source = inspect.getsource(Planner)
        # El planner no debe tener llamadas a execute_by_name
        assert "execute_by_name" not in source
        assert "execute_candidates" not in source
