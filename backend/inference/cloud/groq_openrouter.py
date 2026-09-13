"""NEXUS Ω — Groq + OpenRouter Cloud Runtimes v3.8.0"""
from __future__ import annotations
import time
from backend.config import GROQ_API_KEY, GROQ_MODEL, OPENROUTER_API_KEY, OPENROUTER_MODEL, logger
from backend.inference.cloud.base import CloudRuntime
from backend.inference.errors import InferenceAuthenticationError, InferenceInvalidResponse, classify_error
from backend.inference.models import InferenceRequest, InferenceResult, RuntimeCapabilities, RuntimeHealth, RuntimeStatus, RuntimeType

class GroqRuntime(CloudRuntime):
    STATUS = "IMPLEMENTADO"
    def __init__(self, http_client=None):
        self._provider = None
        if GROQ_API_KEY and http_client:
            try:
                from backend.providers.groq import GroqProvider
                self._provider = GroqProvider(http_client)
            except Exception:
                pass

    @property
    def name(self): return "groq"
    @property
    def model(self): return GROQ_MODEL

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self._provider is None:
            raise InferenceAuthenticationError("GroqRuntime no disponible.", runtime=self.name)
        started = time.perf_counter()
        try:
            from backend.providers.base import GenerateRequest, Message as PMessage
            gen_req = GenerateRequest(prompt=request.prompt, system=request.system,
                history=[PMessage(role=m.role if hasattr(m,"role") else m["role"],
                                  content=m.content if hasattr(m,"content") else m["content"])
                         for m in request.history],
                max_tokens=request.max_tokens, temperature=request.temperature)
            resp = await self._provider.generate(gen_req)
        except Exception as e:
            raise classify_error(e, runtime=self.name) from e
        if not resp.text:
            raise InferenceInvalidResponse("Groq respuesta vacia.", runtime=self.name)
        elapsed = int((time.perf_counter()-started)*1000)
        return InferenceResult(text=resp.text, runtime_name=self.name, runtime_type=RuntimeType.CLOUD,
            model=resp.model, mode=None, is_local=False, duration_ms=elapsed)

    async def health(self):
        status = RuntimeStatus.AVAILABLE if GROQ_API_KEY else RuntimeStatus.NOT_CONFIGURED
        return RuntimeHealth(status=status, runtime_name=self.name, runtime_type=RuntimeType.CLOUD,
            model=self.model, message="OK" if GROQ_API_KEY else "GROQ_API_KEY no configurada.")

    def capabilities(self):
        return RuntimeCapabilities(tool_calling=True, max_context=32768)


class OpenRouterRuntime(CloudRuntime):
    STATUS = "IMPLEMENTADO"
    def __init__(self, http_client=None):
        self._provider = None
        if OPENROUTER_API_KEY and http_client:
            try:
                from backend.providers.openrouter import OpenRouterProvider
                self._provider = OpenRouterProvider(http_client)
            except Exception:
                pass

    @property
    def name(self): return "openrouter"
    @property
    def model(self): return OPENROUTER_MODEL

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self._provider is None:
            raise InferenceAuthenticationError("OpenRouterRuntime no disponible.", runtime=self.name)
        started = time.perf_counter()
        try:
            from backend.providers.base import GenerateRequest, Message as PMessage
            gen_req = GenerateRequest(prompt=request.prompt, system=request.system,
                history=[PMessage(role=m.role if hasattr(m,"role") else m["role"],
                                  content=m.content if hasattr(m,"content") else m["content"])
                         for m in request.history],
                max_tokens=request.max_tokens, temperature=request.temperature)
            resp = await self._provider.generate(gen_req)
        except Exception as e:
            raise classify_error(e, runtime=self.name) from e
        if not resp.text:
            raise InferenceInvalidResponse("OpenRouter respuesta vacia.", runtime=self.name)
        elapsed = int((time.perf_counter()-started)*1000)
        return InferenceResult(text=resp.text, runtime_name=self.name, runtime_type=RuntimeType.CLOUD,
            model=resp.model, mode=None, is_local=False, duration_ms=elapsed)

    async def health(self):
        status = RuntimeStatus.AVAILABLE if OPENROUTER_API_KEY else RuntimeStatus.NOT_CONFIGURED
        return RuntimeHealth(status=status, runtime_name=self.name, runtime_type=RuntimeType.CLOUD,
            model=self.model, message="OK" if OPENROUTER_API_KEY else "OPENROUTER_API_KEY no configurada.")

    def capabilities(self):
        return RuntimeCapabilities(tool_calling=True, max_context=200000)
