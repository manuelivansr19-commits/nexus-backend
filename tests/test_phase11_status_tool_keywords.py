"""
NEXUS Ω — Phase 11 Tests: acotar StatusTool.intent_keywords para
evitar falsos positivos con palabras sueltas comunes (corrección
aislada tras el Bloque 5).

Causa: ToolRegistry.find_by_intent() hace match por substring simple
(sin límites de palabra), y StatusTool declaraba keywords de una sola
palabra muy comunes en conversación de negocios ("estado", "sistema",
"operativo", "funcionando", "activo") — el mismo dominio para el que
NEXUS está especializado (gestión empresarial, economía, estrategia).
Cualquier mensaje que mencionara esas palabras sueltas, sin relación
con el estado del propio NEXUS, terminaba siendo candidato a
StatusTool.

Corrección: las keywords sueltas se reemplazan por frases compuestas
que exigen co-ocurrencia de palabras específicas ("estado" + "sistema"/
"nexus", "sistema"/"nexus" + "operativo", etc.). No se modifica
ToolRegistry.find_by_intent() ni ninguna otra herramienta.
"""

import pytest

from backend.tools.registry import ToolRegistry
from backend.tools.builtin import ClockTool, StatusTool, MemorySearchTool


def _make_registry():
    reg = ToolRegistry()
    reg.register(ClockTool())
    reg.register(StatusTool())
    reg.register(MemorySearchTool())
    return reg


class TestStatusToolExplicitQueriesPreserved:
    """Las consultas explícitas de estado deben seguir funcionando."""

    def test_sistema_esta_operativo(self):
        reg = _make_registry()
        found = reg.find_by_intent("¿el sistema está operativo?")
        assert any(t.name == "system_status" for t in found)

    def test_estado_del_sistema(self):
        reg = _make_registry()
        found = reg.find_by_intent("¿cuál es el estado del sistema?")
        assert any(t.name == "system_status" for t in found)

    def test_estado_de_nexus(self):
        reg = _make_registry()
        found = reg.find_by_intent("muéstrame el estado de NEXUS")
        assert any(t.name == "system_status" for t in found)


class TestStatusToolNormalConversationNotCaptured:
    """Mensajes normales que antes disparaban falsos positivos por
    palabras sueltas ya no deben capturar StatusTool."""

    @pytest.mark.parametrize("message", [
        "¿qué hora es?",
        "¿cuáles fueron las noticias más relevantes de hoy?",
        "¿qué es la fotosíntesis?",
        "¿cómo funciona esto?",
    ])
    def test_plan_required_negative_cases(self, message):
        reg = _make_registry()
        found = reg.find_by_intent(message)
        assert not any(t.name == "system_status" for t in found), message

    @pytest.mark.parametrize("message", [
        "quiero automatizar mi sistema de ventas",
        "el estado financiero de mi empresa es preocupante",
        "necesito un plan de vida más activo",
        "quiero que el negocio esté funcionando en 3 meses",
        "analiza el sistema económico actual",
        "cuál es el estado de la economía",
    ])
    def test_business_domain_phrases_not_captured(self, message):
        """Frases de negocio plausibles en el dominio de NEXUS
        (economía, estrategia, gestión empresarial) que contienen
        palabras antes-conflictivas sueltas, pero no son una consulta
        de estado del propio sistema."""
        reg = _make_registry()
        found = reg.find_by_intent(message)
        assert not any(t.name == "system_status" for t in found), message


class TestOtherToolsUnaffected:
    """El fix es aislado a StatusTool: clock y memory_search deben
    seguir funcionando exactamente igual."""

    def test_clock_still_matches_time_queries(self):
        reg = _make_registry()
        found = reg.find_by_intent("¿qué hora es?")
        assert any(t.name == "clock" for t in found)

    def test_memory_search_still_matches(self):
        reg = _make_registry()
        found = reg.find_by_intent("¿recuerdas lo que te dije antes?")
        assert any(t.name == "memory_search" for t in found)

    def test_no_match_case_still_empty(self):
        reg = _make_registry()
        found = reg.find_by_intent("explícame física cuántica")
        assert found == []
