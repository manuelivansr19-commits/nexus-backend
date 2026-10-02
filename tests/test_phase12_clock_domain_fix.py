"""
NEXUS Ω — Phase 12 Tests: corrección del bug "clock → noticias"
(Domain.TIME sobrescribía la estrategia para cualquier mensaje con
"hoy"/"mañana"/"fecha"/"día", no solo consultas explícitas de hora).

Causa raíz confirmada por diagnóstico de sesión: en
IntentRouter.route(), la condición
    domain == Domain.TIME or any(w in lower for w in ["hora","fecha"])
se disparaba con CUALQUIER mensaje clasificado como Domain.TIME
(incluye "hoy", "mañana", "dia" — temporalidad contextual, no reloj),
forzando strategy=TOOL/clock incluso cuando el mensaje no tenía
relación con la hora.

No es un bug de contaminación entre turnos, contexto, TaskStateManager
ni ToolRegistry: reproduce de forma aislada, en una sola llamada
stateless a IntentRouter.route().

Corrección: solo "hora" dispara clock. Domain.TIME sigue existiendo y
clasificando exactamente igual (no se tocó _DOMAIN_KEYWORDS ni
_score_domain).
"""

import pytest

from backend.core.intent import IntentRouter, IntentStrategy, Domain


@pytest.fixture
def router():
    return IntentRouter(registry=None)


def _is_clock(result):
    return result.strategy == IntentStrategy.TOOL and "clock" in result.candidate_tools


class TestExplicitClockQueriesStillWork:
    """Consultas explícitas de hora deben seguir activando clock."""

    @pytest.mark.parametrize("message", [
        "¿Qué hora es?",
        "¿Qué hora tenemos ahora?",
        "¿Qué hora es actualmente?",
    ])
    def test_explicit_time_query_triggers_clock(self, router, message):
        result = router.route(message)
        assert _is_clock(result), f"{message!r} debería activar clock"


class TestContextualTimeWordsDoNotTriggerClock(object):
    """Mensajes con palabras de temporalidad contextual ('hoy',
    'mañana', 'ayer', 'fecha', 'día') sin ser consultas de hora NO
    deben activar clock."""

    @pytest.mark.parametrize("message", [
        "¿Cuáles fueron las noticias más relevantes del día de hoy?",
        "¿Qué pasó hoy en el mercado?",
        "Mañana tengo una reunión importante",
        "¿Qué ocurrió ayer?",
    ])
    def test_contextual_time_word_does_not_trigger_clock(self, router, message):
        result = router.route(message)
        assert not _is_clock(result), f"{message!r} NO debería activar clock"


class TestDomainTimeStillValidClassification:
    """Domain.TIME sigue existiendo como clasificación temporal válida;
    solo se dejó de usar como disparador automático de clock."""

    def test_domain_time_still_scored_for_explicit_hour_query(self, router):
        result = router.route("¿Qué hora es?")
        assert result.domain == Domain.TIME

    def test_domain_time_still_scored_for_contextual_today_mention(self, router):
        result = router.route("¿Qué pasó hoy en el mercado?")
        assert result.domain == Domain.TIME
        # Pero ya NO fuerza clock, aunque el dominio siga siendo TIME:
        assert not _is_clock(result)


class TestConversationalRegression:
    """Secuencia de regresión: un turno sobre la hora, seguido de un
    turno sobre noticias de hoy, no debe contaminar el segundo turno.
    IntentRouter es stateless, así que cada .route() es independiente;
    esto documenta el comportamiento esperado turno por turno."""

    def test_hour_then_news_sequence(self, router):
        turn1 = router.route("¿Qué hora es?")
        turn2 = router.route("¿Y cuáles fueron las noticias más relevantes de hoy?")

        assert _is_clock(turn1), "El primer turno sí debe activar clock"
        assert not _is_clock(turn2), "El segundo turno NO debe activar clock"
