"""NEXUS Ω — Gemini Cloud Runtime v3.8.0"""
from __future__ import annotations
import time
from backend.config import GEMINI_API_KEY, GEMINI_MODEL, logger
from backend.inference.cloud.base import CloudRuntime
from backend.inference.errors import InferenceAuthenticationError, InferenceInvalidResponse, classify_error
from backend.inference.models import InferenceRequest, InferenceResult, RuntimeCapabilities, RuntimeHealth, RuntimeStatus, RuntimeType

class GeminiRuntime(CloudRuntime):
    STATUS = "IMPLEMENTADO"
    def __init__(self):
        self._provider = None
        if GEMINI_API_KEY:
            try:
                from backend.providers.gemini import GeminiProvider
                self._provider = GeminiProvider()
            except Exception:
                logger.exception("GeminiRuntime: no pudo inicializar.")

    @property
    def name(self): return "gemini"
    @property
    def model(self): return GEMINI_MODEL

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self._provider is None:
            raise InferenceAuthenticationError("GeminiRuntime no disponible.", runtime=self.name)
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
            raise InferenceInvalidResponse("Gemini respuesta vacia.", runtime=self.name)
        elapsed = int((time.perf_counter()-started)*1000)
        return InferenceResult(text=resp.text, runtime_name=self.name, runtime_type=RuntimeType.CLOUD,
            model=resp.model, mode=None, is_local=False, duration_ms=elapsed)

    async def health(self):
        if not GEMINI_API_KEY:
            return RuntimeHealth(status=RuntimeStatus.NOT_CONFIGURED, runtime_name=self.name,
                runtime_type=RuntimeType.CLOUD, model=self.model, message="GEMINI_API_KEY no configurada.")
        if self._provider is None:
            return RuntimeHealth(status=RuntimeStatus.UNAVAILABLE, runtime_name=self.name,
                runtime_type=RuntimeType.CLOUD, model=self.model, message="Provider no disponible.")
        return RuntimeHealth(status=RuntimeStatus.AVAILABLE, runtime_name=self.name,
            runtime_type=RuntimeType.CLOUD, model=self.model, message="Configurado.")

    def capabilities(self):
        return RuntimeCapabilities(streaming=False, tool_calling=True, vision=True, max_context=128000)

    async def shutdown(self):
        if self._provider:
            await self._provider.shutdown()
