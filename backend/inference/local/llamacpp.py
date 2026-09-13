"""NEXUS Ω — LlamaCpp Runtime v3.8.0 — SCAFFOLDED"""
from __future__ import annotations
from backend.config import LOCAL_MODEL_PATH, logger
from backend.inference.errors import InferenceConfigurationError, RuntimeUnavailable
from backend.inference.local.base import LocalRuntime
from backend.inference.models import InferenceRequest, InferenceResult, RuntimeCapabilities, RuntimeHealth, RuntimeStatus, RuntimeType

try:
    from llama_cpp import Llama as _LlamaCpp
    _AVAILABLE = True
except ImportError:
    _LlamaCpp  = None
    _AVAILABLE = False

class LlamaCppRuntime(LocalRuntime):
    STATUS = "SCAFFOLDED"
    def __init__(self, model_path=LOCAL_MODEL_PATH, n_ctx=4096, n_threads=4, timeout=120.0):
        self._model_path = model_path
        self._timeout    = timeout
        self._n_ctx      = n_ctx
        self._llm        = None
        self._load_error = None
        if not _AVAILABLE:
            self._load_error = "llama-cpp-python no instalado."
            return
        if not model_path:
            self._load_error = "LOCAL_MODEL_PATH no configurado."
            return
        import os
        if not os.path.exists(model_path):
            self._load_error = f"Modelo no encontrado: {model_path}"
            return
        try:
            self._llm = _LlamaCpp(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
            logger.info("LlamaCppRuntime: modelo listo.")
        except Exception as e:
            self._load_error = str(e)[:200]

    @property
    def name(self): return "llama_cpp"
    @property
    def model(self):
        if self._model_path:
            import os
            return f"gguf:{os.path.basename(self._model_path)}"
        return "gguf:not_configured"

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if not _AVAILABLE:
            raise InferenceConfigurationError("SCAFFOLDED: llama-cpp-python no instalado.", runtime=self.name)
        if self._llm is None:
            raise RuntimeUnavailable(f"LlamaCpp no disponible: {self._load_error}", runtime=self.name)
        import asyncio, time
        started = time.perf_counter()
        parts   = []
        if request.system:
            parts.append(f"<|im_start|>system\n{request.system}<|im_end|>")
        for msg in request.history:
            role = msg.role if hasattr(msg,"role") else msg["role"]
            cont = msg.content if hasattr(msg,"content") else msg["content"]
            parts.append(f"<|im_start|>{role}\n{cont}<|im_end|>")
        parts.append(f"<|im_start|>user\n{request.prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)
        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self._llm(prompt, max_tokens=request.max_tokens,
                temperature=request.temperature, stop=["<|im_end|>"], echo=False)),
            timeout=self._timeout)
        text = result["choices"][0]["text"].strip()
        elapsed = int((time.perf_counter()-started)*1000)
        return InferenceResult(text=text, runtime_name=self.name, runtime_type=RuntimeType.LOCAL,
            model=self.model, mode=None, is_local=True, duration_ms=elapsed)

    async def health(self):
        if not _AVAILABLE:
            return RuntimeHealth(status=RuntimeStatus.SCAFFOLDED, runtime_name=self.name,
                runtime_type=RuntimeType.LOCAL, model=self.model, message="SCAFFOLDED")
        if self._llm is None:
            return RuntimeHealth(status=RuntimeStatus.NOT_CONFIGURED, runtime_name=self.name,
                runtime_type=RuntimeType.LOCAL, model=self.model, message=self._load_error or "No cargado.")
        return RuntimeHealth(status=RuntimeStatus.AVAILABLE, runtime_name=self.name,
            runtime_type=RuntimeType.LOCAL, model=self.model, message="OK")

    def capabilities(self):
        return RuntimeCapabilities(max_context=self._n_ctx)
