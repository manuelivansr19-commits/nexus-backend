from backend.core.intent import IntentRouter, IntentType, IntentStrategy, Domain
from backend.tools.builtin import create_default_registry


def _router():
    return IntentRouter(registry=create_default_registry())


def test_help_question_is_direct_response():
    result = _router().route("¿Qué puedes hacer?")
    assert result.strategy == IntentStrategy.DIRECT
    assert result.direct_response
    assert not result.requires_planning


def test_how_is_nexus_is_normal_conversation():
    result = _router().route("¿Cómo está NEXUS?")
    assert result.intent == IntentType.CHAT
    assert result.strategy == IntentStrategy.LLM
    assert "system_status" not in result.candidate_tools


def test_explicit_system_status_still_selects_tool():
    result = _router().route("¿Cuál es el estado del sistema?")
    assert result.strategy == IntentStrategy.TOOL
    assert "system_status" in result.candidate_tools
