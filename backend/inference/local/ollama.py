"""NEXUS Ω — Ollama Runtime v3.8.0"""
from __future__ import annotations
import asyncio, time
from typing import Optional
import httpx
from backend.config import LOCAL_OLLAMA_MODEL, LOCAL_OLLAMA_URL, REQUEST_TIMEOUT_SECONDS, logger
from backend.inference.errors import InferenceInvalidResponse, InferenceTimeout, ModelUnavailable, RuntimeUnavailable
from backend.inference.local.base import LocalRuntime
from backend.inference.models import InferenceRequest, InferenceResult, RuntimeCapabilities, RuntimeHealth, RuntimeStatus, RuntimeType

class OllamaRuntime(LocalRuntime):
    STATUS = "IMPLEMENTADO"
    def __init__(self, http_client=None, base_url=LOCAL_OLLAMA_URL, model=LOCAL_OLLAMA_MODEL, timeout=REQUEST_TIMEOUT_SECONDS):
        self._http       = http_client
        self._base_url   = base_url.rstrip("/")
        self._model      = model
        self._timeout    = timeout
        self._owns_client = http_client is None

    @property
    def name(self): return "ollama"
    @property
    def model(self): return self._model

    async def _client(self):
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=15.0, pool=15.0))
        return self._http

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        started  = time.perf_counter()
        messages = [{"role": "system", "content": request.system}] if request.system else []
        for msg in request.history:
            messages.append({"role": msg.role if hasattr(msg,"role") else msg["role"],
                              "content": msg.content if hasattr(msg,"content") else msg["content"]})
        messages.append({"role": "user", "content": request.prompt})
        payload = {"model": self._model, "messages": messages, "stream": False,
                   "options": {"num_predict": request.max_tokens, "temperature": request.temperature}}
        client = await self._client()
        try:
            response = await asyncio.wait_for(client.post(f"{self._base_url}/api/chat", json=payload), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise InferenceTimeout(f"Ollama timeout", runtime=self.name, model=self._model)
        except Exception as e:
            raise RuntimeUnavailable(f"Ollama no responde en {self._base_url}: {e}", runtime=self.name)
        if response.status_code == 404:
            raise ModelUnavailable(f"Modelo '{self._model}' no encontrado.", runtime=self.name, model=self._model)
        if response.status_code >= 400:
            raise RuntimeUnavailable(f"Ollama HTTP {response.status_code}", runtime=self.name)
        text = response.json().get("message", {}).get("content", "").strip()
        if not text:
            raise InferenceInvalidResponse("Ollama respuesta vacia.", runtime=self.name, model=self._model)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("OllamaRuntime OK | model=%s | %dms", self._model, elapsed)
        return InferenceResult(text=text, runtime_name=self.name, runtime_type=RuntimeType.LOCAL,
            model=self._model, mode=None, is_local=True, duration_ms=elapsed)

    async def health(self):
        started = time.perf_counter()
        client  = await self._client()
        try:
            response = await asyncio.wait_for(client.get(f"{self._base_url}/api/tags"), timeout=5.0)
        except Exception as e:
            return RuntimeHealth(status=RuntimeStatus.UNAVAILABLE, runtime_name=self.name,
                runtime_type=RuntimeType.LOCAL, model=self._model, message=str(e)[:80],
                latency_ms=int((time.perf_counter()-started)*1000))
        if response.status_code != 200:
            return RuntimeHealth(status=RuntimeStatus.UNAVAILABLE, runtime_name=self.name,
                runtime_type=RuntimeType.LOCAL, model=self._model, message=f"HTTP {response.status_code}",
                latency_ms=int((time.perf_counter()-started)*1000))
        try:
            models = [m.get("name","") for m in response.json().get("models",[])]
            found  = any(self._model in m or m.startswith(self._model.split(":")[0]) for m in models)
        except Exception:
            found = False
        elapsed = int((time.perf_counter()-started)*1000)
        if not found:
            return RuntimeHealth(status=RuntimeStatus.DEGRADED, runtime_name=self.name,
                runtime_type=RuntimeType.LOCAL, model=self._model,
                message=f"Runtime OK pero modelo no encontrado.", latency_ms=elapsed)
        return RuntimeHealth(status=RuntimeStatus.AVAILABLE, runtime_name=self.name,
            runtime_type=RuntimeType.LOCAL, model=self._model, message="OK", latency_ms=elapsed)

    def capabilities(self):
        return RuntimeCapabilities(streaming=False, tool_calling=False, max_context=4096)

    async def shutdown(self):
        if self._owns_client and self._http:
            await self._http.aclose()
