"""
NEXUS Ω — Phase 2 Tests.

Cubre:
1.  Tool registration
2.  Tool discovery
3.  Intent routing (determinístico)
4.  Intent routing (keyword scoring)
5.  Memory save (RAM)
6.  Memory retrieval
7.  Memory persistence (SQLite)
8.  Context assembly
9.  Provider abstraction
10. Executor validation (risk level)
11. API compatibility (endpoints existentes)
12. Test offline (sin Gemini)
13. Memory restart persistence
14. NexusCore direct response
15. NexusCore tool response
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


# ============================================================
# TOOL TESTS
# ============================================================

from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult
from backend.tools.registry import ToolRegistry
from backend.tools.builtin import ClockTool, StatusTool, MemorySearchTool


class MockTool(BaseTool):
    enabled    = True
    risk_level = RiskLevel.LOW

    def __init__(self, name: str, keywords: list[str] = []):
        self._name     = name
        self._keywords = keywords

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool: {self._name}"

    @property
    def intent_keywords(self) -> list[str]:
        return self._keywords

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        return ToolResult(success=True, output=f"Result from {self._name}", tool_name=self._name)


class TestToolRegistration:

    def test_register_tool(self):
        reg = ToolRegistry()
        tool = MockTool("test_tool")
        reg.register(tool)
        assert reg.get("test_tool") is tool

    def test_unregister_tool(self):
        reg = ToolRegistry()
        reg.register(MockTool("tool_a"))
        assert reg.unregister("tool_a") is True
        assert reg.get("tool_a") is None

    def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        assert reg.unregister("nonexistent") is False

    def test_list_enabled_only(self):
        reg = ToolRegistry()
        t1 = MockTool("enabled_tool")
        t2 = MockTool("disabled_tool")
        t2.enabled = False
        reg.register(t1)
        reg.register(t2)
        enabled = reg.list(enabled_only=True)
        names = [t.name for t in enabled]
        assert "enabled_tool" in names
        assert "disabled_tool" not in names

    def test_stats(self):
        reg = ToolRegistry()
        reg.register(MockTool("tool_1"))
        reg.register(MockTool("tool_2"))
        stats = reg.stats()
        assert stats["total"] == 2
        assert stats["enabled"] == 2


class TestToolDiscovery:

    def test_find_by_intent_with_keyword(self):
        reg = ToolRegistry()
        reg.register(MockTool("clock_tool", ["hora", "tiempo"]))
        reg.register(MockTool("search_tool", ["buscar", "web"]))
        found = reg.find_by_intent("qué hora es")
        assert len(found) == 1
        assert found[0].name == "clock_tool"

    def test_find_by_intent_no_match(self):
        reg = ToolRegistry()
        reg.register(MockTool("clock_tool", ["hora"]))
        found = reg.find_by_intent("explícame física cuántica")
        assert len(found) == 0

    def test_builtin_clock_tool(self):
        reg = ToolRegistry()
        reg.register(ClockTool())
        found = reg.find_by_intent("qué hora es")
        assert any(t.name == "clock" for t in found)


# ============================================================
# INTENT TESTS
# ============================================================

from backend.core.intent import IntentRouter, IntentStrategy, Domain


class TestIntentRouting:

    def test_greeting_is_direct(self):
        router = IntentRouter()
        result = router.route("hola")
        assert result.strategy == IntentStrategy.DIRECT
        assert result.confidence == 1.0
        assert result.direct_response is not None

    def test_ping_is_direct(self):
        router = IntentRouter()
        result = router.route("ping")
        assert result.strategy == IntentStrategy.DIRECT
        assert result.direct_response == "pong"

    def test_ayuda_is_direct(self):
        router = IntentRouter()
        result = router.route("ayuda")
        assert result.strategy == IntentStrategy.DIRECT

    def test_strategy_domain(self):
        router = IntentRouter()
        result = router.route("necesito una estrategia de negocio para mi empresa")
        assert result.domain == Domain.STRATEGY
        assert result.strategy == IntentStrategy.LLM

    def test_technology_domain(self):
        router = IntentRouter()
        result = router.route("implementa un algoritmo de búsqueda binaria en Python")
        assert result.domain == Domain.TECHNOLOGY

    def test_memory_required(self):
        router = IntentRouter()
        result = router.route("recuerda lo que dijiste antes sobre estrategia")
        assert result.requires_memory is True

    def test_time_query_uses_tool(self):
        reg = ToolRegistry()
        reg.register(ClockTool())
        router = IntentRouter(registry=reg)
        result = router.route("qué hora es")
        assert result.requires_tool is True
        assert "clock" in result.candidate_tools

    def test_unknown_domain_falls_to_general(self):
        router = IntentRouter()
        result = router.route("xyzzy frobnicator")
        assert result.domain == Domain.GENERAL


# ============================================================
# MEMORY TESTS
# ============================================================

from backend.core.memory import (
    Memory, MemoryEntry, MemoryType,
    RAMMemoryStore, SQLiteMemoryStore,
    ConversationMemory, FactMemory,
)


class TestMemoryRAM:

    def test_save_and_get(self):
        store = RAMMemoryStore()
        entry = MemoryEntry(content="NEXUS usa Gemini", memory_type=MemoryType.FACT)
        entry_id = store.save(entry)
        fetched  = store.get(entry_id)
        assert fetched is not None
        assert fetched.content == "NEXUS usa Gemini"

    def test_search(self):
        store = RAMMemoryStore()
        store.save(MemoryEntry(content="El modelo principal es Gemini", memory_type=MemoryType.FACT))
        store.save(MemoryEntry(content="Ollama es el fallback local", memory_type=MemoryType.FACT))
        results = store.search("Gemini")
        assert any("Gemini" in r.content for r in results)

    def test_recent_by_type(self):
        store = RAMMemoryStore()
        store.save(MemoryEntry(content="conv msg", memory_type=MemoryType.CONVERSATION))
        store.save(MemoryEntry(content="fact data", memory_type=MemoryType.FACT))
        conv = store.recent(MemoryType.CONVERSATION)
        assert all(e.memory_type == MemoryType.CONVERSATION for e in conv)

    def test_delete(self):
        store = RAMMemoryStore()
        entry = MemoryEntry(content="to delete")
        eid   = store.save(entry)
        assert store.delete(eid) is True
        assert store.get(eid) is None

    def test_clear_by_type(self):
        store = RAMMemoryStore()
        store.save(MemoryEntry(content="w1", memory_type=MemoryType.WORKING))
        store.save(MemoryEntry(content="w2", memory_type=MemoryType.WORKING))
        store.save(MemoryEntry(content="f1", memory_type=MemoryType.FACT))
        count = store.clear(MemoryType.WORKING)
        assert count == 2
        assert store.stats()["total"] == 1

    def test_stats(self):
        store = RAMMemoryStore()
        store.save(MemoryEntry(content="a", memory_type=MemoryType.FACT))
        store.save(MemoryEntry(content="b", memory_type=MemoryType.FACT))
        store.save(MemoryEntry(content="c", memory_type=MemoryType.CONVERSATION))
        stats = store.stats()
        assert stats["total"] == 3
        assert stats["by_type"].get("fact") == 2


class TestMemorySQLite:

    def test_persist_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store1 = SQLiteMemoryStore(db_path)
            entry  = MemoryEntry(
                content="Dato persistido en SQLite",
                memory_type=MemoryType.FACT,
                importance=0.9,
            )
            eid = store1.save(entry)
            store1.close()

            # Nueva conexión → mismo archivo
            store2   = SQLiteMemoryStore(db_path)
            fetched  = store2.get(eid)
            store2.close()

            assert fetched is not None
            assert fetched.content == "Dato persistido en SQLite"
        finally:
            os.unlink(db_path)

    def test_search_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteMemoryStore(db_path)
            store.save(MemoryEntry(content="NEXUS proyecto AURA", memory_type=MemoryType.FACT))
            store.save(MemoryEntry(content="Robotica avanzada", memory_type=MemoryType.FACT))
            results = store.search("AURA")
            assert len(results) > 0
            assert any("AURA" in r.content for r in results)
            store.close()
        finally:
            os.unlink(db_path)

    def test_stats_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteMemoryStore(db_path)
            store.save(MemoryEntry(content="x", memory_type=MemoryType.FACT))
            store.save(MemoryEntry(content="y", memory_type=MemoryType.CONVERSATION))
            stats = store.stats()
            assert stats["total"] == 2
            assert stats["backend"] == "sqlite"
            store.close()
        finally:
            os.unlink(db_path)


class TestConversationMemory:

    def test_add_and_retrieve(self):
        mem = Memory(RAMMemoryStore())
        mem.conversation.add_user("Hola NEXUS")
        mem.conversation.add_assistant("¿Qué necesitas?")
        turns = mem.conversation.recent_turns(limit=10)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_to_history_list(self):
        mem = Memory(RAMMemoryStore())
        mem.conversation.add_user("mensaje 1")
        mem.conversation.add_assistant("respuesta 1")
        history = mem.conversation.to_history_list()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


# ============================================================
# CONTEXT TESTS
# ============================================================

from backend.core.context import ContextManager


class TestContextAssembly:

    def test_basic_assembly(self):
        cm     = ContextManager(memory=None, token_limit=4000)
        bundle = cm.assemble(
            message="Analiza el mercado de IA",
            system_prompt="Eres NEXUS.",
        )
        assert bundle.assembled_prompt != ""
        assert bundle.system_prompt == "Eres NEXUS."

    def test_history_trimming(self):
        cm = ContextManager(memory=None, token_limit=500)
        long_history = [
            {"role": "user",      "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
            {"role": "user",      "content": "z" * 100},
        ]
        bundle = cm.assemble(
            message="pregunta corta",
            system_prompt="sistema",
            history=long_history,
        )
        # El historial recortado debe ser <= original
        assert len(bundle.history) <= len(long_history)

    def test_tool_results_included(self):
        cm     = ContextManager(memory=None)
        bundle = cm.assemble(
            message="prueba",
            system_prompt="sys",
            tool_results=["[TOOL:clock] 14:30:00"],
        )
        assert "TOOL" in bundle.assembled_prompt or "clock" in bundle.assembled_prompt


# ============================================================
# EXECUTOR TESTS
# ============================================================

from backend.core.executor import Executor


class HighRiskTool(BaseTool):
    enabled    = True
    risk_level = RiskLevel.HIGH

    @property
    def name(self) -> str: return "high_risk_tool"
    @property
    def description(self) -> str: return "Risky"
    async def execute(self, ti: ToolInput) -> ToolResult:
        return ToolResult(success=True, output="danger", tool_name=self.name)


class TestExecutorValidation:

    @pytest.mark.asyncio
    async def test_low_risk_executes(self):
        reg = ToolRegistry()
        reg.register(ClockTool())
        executor = Executor(registry=reg)
        result = await executor.execute_by_name("clock", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_high_risk_blocked_without_auth(self):
        reg = ToolRegistry()
        reg.register(HighRiskTool())
        executor = Executor(registry=reg)
        result = await executor.execute_by_name("high_risk_tool", {})
        assert result.success is False
        assert result.authorized is False

    @pytest.mark.asyncio
    async def test_high_risk_allowed_with_auth(self):
        reg = ToolRegistry()
        reg.register(HighRiskTool())
        executor = Executor(registry=reg)
        result = await executor.execute_by_name(
            "high_risk_tool", {}, authorized=True
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_tool_fails_gracefully(self):
        executor = Executor(registry=ToolRegistry())
        result = await executor.execute_by_name("nonexistent", {})
        assert result.success is False
        assert "no encontrada" in (result.error or "")

    @pytest.mark.asyncio
    async def test_disabled_tool_skipped(self):
        reg  = ToolRegistry()
        tool = ClockTool()
        reg.register(tool)
        reg.disable("clock")
        executor = Executor(registry=reg)
        result = await executor.execute_by_name("clock", {})
        assert result.success is False
        assert result.skipped is True


# ============================================================
# API COMPATIBILITY TESTS
# ============================================================

class TestAPICompatibility:

    def setup_method(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["version"] == APP_VERSION
        assert "providers" in data
        assert "local_mode" in data

    def test_status_returns_200(self):
        r = self.client.get("/api/nexus/status")
        assert r.status_code == 200
        assert r.json()["status"] == "online"

    def test_config_returns_200(self):
        r = self.client.get("/api/nexus/config")
        assert r.status_code == 200
        assert r.json()["multi_provider"] is True

    def test_chat_empty_message(self):
        r = self.client.post("/api/nexus/chat", json={"message": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["provider"] == "system"
        assert "request_id" in data

    def test_chat_validates_message_length(self):
        r = self.client.post("/api/nexus/chat", json={"message": "x" * 30001})
        assert r.status_code == 422

    def test_tools_endpoint(self):
        r = self.client.get("/api/nexus/tools")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "tools" in data

    def test_intent_endpoint(self):
        r = self.client.post("/api/nexus/intent", json={"message": "hola"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "greeting"
        assert data["strategy"] == "direct"

    def test_intent_strategy_domain(self):
        r = self.client.post(
            "/api/nexus/intent",
            json={"message": "analiza la estrategia de mi empresa"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["strategy"] in ("llm", "tool")

    def test_memory_endpoint(self):
        r = self.client.get("/api/nexus/memory")
        assert r.status_code == 200
        assert r.json()["success"] is True

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
# OFFLINE TEST
# ============================================================

class TestOfflineMode:

    @pytest.mark.asyncio
    async def test_nexus_starts_without_gemini(self):
        """El sistema inicia aunque Gemini no esté disponible."""
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.core.intent import IntentRouter
        from backend.core.context import ContextManager
        from backend.core.executor import Executor
        from backend.core.evaluation import Evaluator
        from backend.core.nexus import NexusCore
        from backend.tools.registry import ToolRegistry

        mem     = Memory(RAMMemoryStore())
        reg     = ToolRegistry()
        reg.register(ClockTool())
        executor = Executor(registry=reg, memory=mem)
        core     = NexusCore(
            model_router=None,   # sin router
            memory=mem,
            intent_router=IntentRouter(registry=reg),
            context_manager=ContextManager(memory=mem),
            executor=executor,
            evaluator=Evaluator(),
        )
        assert core is not None
        status = core.status()
        assert status["memory"] is True
        assert status["intent_router"] is True

    @pytest.mark.asyncio
    async def test_direct_intent_works_without_model(self):
        """Respuestas directas (sin LLM) funcionan sin modelo."""
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.core.intent import IntentRouter
        from backend.core.context import ContextManager
        from backend.core.executor import Executor
        from backend.core.evaluation import Evaluator
        from backend.core.nexus import NexusCore

        mem  = Memory(RAMMemoryStore())
        core = NexusCore(
            model_router=None,
            memory=mem,
            intent_router=IntentRouter(),
            context_manager=ContextManager(memory=mem),
            executor=Executor(registry=ToolRegistry(), memory=mem),
            evaluator=Evaluator(),
        )
        result = await core.process("hola")
        assert result.text != ""
        assert result.provider == "system"

    @pytest.mark.asyncio
    async def test_clock_tool_works_without_model(self):
        """La tool clock responde sin LLM."""
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.core.intent import IntentRouter
        from backend.core.context import ContextManager
        from backend.core.executor import Executor
        from backend.core.evaluation import Evaluator
        from backend.core.nexus import NexusCore

        mem      = Memory(RAMMemoryStore())
        reg      = ToolRegistry()
        reg.register(ClockTool())
        executor = Executor(registry=reg, memory=mem)
        core     = NexusCore(
            model_router=None,
            memory=mem,
            intent_router=IntentRouter(registry=reg),
            context_manager=ContextManager(memory=mem),
            executor=executor,
            evaluator=Evaluator(),
        )
        result = await core.process("qué hora es")
        assert result.text != ""
        assert result.provider == "tool"
        assert "clock" in result.tools_used


# Import necesario para APP_VERSION en tests
from backend.config import APP_VERSION
