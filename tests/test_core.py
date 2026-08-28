"""
NEXUS Ω — Tests mínimos de Fase 1.5.

Ejecutar: python -m pytest tests/ -v
"""

import asyncio

import pytest

from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    Message,
    ProviderResponse,
)
from backend.router import ModelRouter, RouterResult, is_rate_limit_error


# ============================================================
# MOCK PROVIDERS
# ============================================================

class MockProvider(BaseModelProvider):
    """Provider de prueba que responde texto fijo."""

    def __init__(self, provider_name: str, text: str = "OK", configured: bool = True):
        self._name = provider_name
        self._text = text
        self._configured = configured
        self._called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return f"{self._name}-test-model"

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        self._called = True
        return ProviderResponse(
            text=self._text,
            provider=self.name,
            model=self.model,
            duration_ms=10,
        )


class FailingProvider(BaseModelProvider):
    """Provider que siempre falla."""

    def __init__(self, provider_name: str, error_msg: str = "Error de prueba"):
        self._name = provider_name
        self._error_msg = error_msg

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return f"{self._name}-test-model"

    @property
    def is_configured(self) -> bool:
        return True

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        raise RuntimeError(self._error_msg)


class RateLimitProvider(BaseModelProvider):
    """Provider que devuelve error 429."""

    def __init__(self, provider_name: str):
        self._name = provider_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return f"{self._name}-test-model"

    @property
    def is_configured(self) -> bool:
        return True

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded")


# ============================================================
# HELPERS
# ============================================================

def make_request(prompt: str = "Hola") -> GenerateRequest:
    return GenerateRequest(
        prompt=prompt,
        system="Eres un asistente de prueba.",
    )


# ============================================================
# TESTS: RATE LIMIT DETECTION
# ============================================================

class TestRateLimitDetection:

    def test_detects_429_code(self):
        class E(Exception):
            code = 429
        assert is_rate_limit_error(E()) is True

    def test_detects_429_in_text(self):
        assert is_rate_limit_error(RuntimeError("HTTP 429")) is True

    def test_detects_resource_exhausted(self):
        assert is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED")) is True

    def test_detects_quota(self):
        assert is_rate_limit_error(RuntimeError("Quota exceeded")) is True

    def test_normal_error_not_rate_limit(self):
        assert is_rate_limit_error(RuntimeError("Connection refused")) is False


# ============================================================
# TESTS: ROUTER
# ============================================================

class TestRouter:

    @pytest.mark.asyncio
    async def test_primary_provider_success(self):
        """Provider primario responde → no es fallback."""
        providers = [
            MockProvider("gemini", "Respuesta Gemini"),
            MockProvider("openrouter", "Respuesta OR"),
        ]
        router = ModelRouter(providers)
        result = await router.generate(make_request())

        assert result.response.text == "Respuesta Gemini"
        assert result.response.provider == "gemini"
        assert result.fallback is False
        assert result.primary_provider == "gemini"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Provider primario falla → fallback al segundo."""
        providers = [
            FailingProvider("gemini"),
            MockProvider("openrouter", "Respuesta OR"),
        ]
        router = ModelRouter(providers)
        result = await router.generate(make_request())

        assert result.response.text == "Respuesta OR"
        assert result.response.provider == "openrouter"
        assert result.fallback is True
        assert result.attempted_providers == ["gemini", "openrouter"]

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self):
        """Provider primario da 429 → fallback al segundo."""
        providers = [
            RateLimitProvider("gemini"),
            MockProvider("groq", "Respuesta Groq"),
        ]
        router = ModelRouter(providers)
        result = await router.generate(make_request())

        assert result.response.text == "Respuesta Groq"
        assert result.fallback is True

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Todos fallan → RuntimeError."""
        providers = [
            FailingProvider("gemini", "Gemini muerto"),
            FailingProvider("openrouter", "OR muerto"),
        ]
        router = ModelRouter(providers)

        with pytest.raises(RuntimeError, match="Todos los proveedores fallaron"):
            await router.generate(make_request())

    @pytest.mark.asyncio
    async def test_no_providers_configured(self):
        """Sin providers → RuntimeError."""
        providers = [
            MockProvider("gemini", configured=False),
        ]
        router = ModelRouter(providers)

        with pytest.raises(RuntimeError, match="Ningún proveedor"):
            await router.generate(make_request())

    @pytest.mark.asyncio
    async def test_skips_unconfigured_providers(self):
        """Provider no configurado no se intenta."""
        unconfigured = MockProvider("gemini", configured=False)
        configured = MockProvider("openrouter", "Respuesta OR")
        providers = [unconfigured, configured]
        router = ModelRouter(providers)
        result = await router.generate(make_request())

        assert result.response.provider == "openrouter"
        assert result.fallback is False  # Es el primero CONFIGURADO
        assert unconfigured._called is False

    @pytest.mark.asyncio
    async def test_history_passed_through(self):
        """El historial se pasa al provider."""
        class HistoryCapture(MockProvider):
            captured_request = None
            async def generate(self, request):
                HistoryCapture.captured_request = request
                return await super().generate(request)

        provider = HistoryCapture("gemini")
        router = ModelRouter([provider])

        req = GenerateRequest(
            prompt="¿Qué dije antes?",
            system="Test",
            history=[
                Message(role="user", content="Hola"),
                Message(role="assistant", content="Hola, ¿qué tal?"),
            ],
        )
        await router.generate(req)

        assert HistoryCapture.captured_request is not None
        assert len(HistoryCapture.captured_request.history) == 2
        assert HistoryCapture.captured_request.history[0].content == "Hola"


# ============================================================
# TESTS: PROVIDER STATUS
# ============================================================

class TestProviderStatus:

    def test_status_reports_all(self):
        providers = [
            MockProvider("gemini", configured=True),
            MockProvider("openrouter", configured=False),
            MockProvider("groq", configured=True),
        ]
        router = ModelRouter(providers)
        status = router.provider_status()

        assert status == {
            "gemini": True,
            "openrouter": False,
            "groq": True,
        }

    def test_model_names(self):
        providers = [MockProvider("gemini"), MockProvider("groq")]
        router = ModelRouter(providers)
        models = router.model_names()

        assert "gemini" in models
        assert "groq" in models
