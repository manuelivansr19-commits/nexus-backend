"""
NEXUS Ω — Local Engine Provider.

Motor de inferencia LOCAL para AURA.
No depende de Internet ni de ningún proveedor externo.

Soporta dos runtimes (en orden de prioridad):
1. llama-cpp-python  — ejecuta modelos GGUF directamente
2. Ollama local      — Ollama corriendo en localhost

En producción (Render), este provider estará no-configurado
porque no hay modelo local. En el hardware físico de AURA,
será el provider primario.

Activar con NEXUS_LOCAL_ONLY=true para modo offline completo.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from backend.config import (
    LOCAL_MODEL_PATH,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_URL,
    LOCAL_RUNTIME,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    logger,
)
from backend.providers.base import (
    BaseModelProvider,
    GenerateRequest,
    ProviderResponse,
)

# ── Importación opcional de llama-cpp-python ──────────────────
# No está en requirements.txt porque Render no puede compilarlo.
# En el hardware físico de AURA: pip install llama-cpp-python
try:
    from llama_cpp import Llama as _LlamaCpp  # type: ignore
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LlamaCpp = None
    LLAMA_CPP_AVAILABLE = False


class LocalProvider(BaseModelProvider):
    """
    Provider local. Orden de intento:
      1. llama-cpp-python (si LOCAL_MODEL_PATH existe y lib disponible)
      2. Ollama en LOCAL_OLLAMA_URL (localhost)

    is_local = True → el router lo prioriza en NEXUS_LOCAL_ONLY.
    """

    is_local: bool = True

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http   = http_client
        self._llm: Optional[object] = None
        self._runtime: Optional[str] = None

        if LLAMA_CPP_AVAILABLE and LOCAL_MODEL_PATH:
            try:
                logger.info(
                    "LocalProvider: cargando modelo llama.cpp desde %s",
                    LOCAL_MODEL_PATH,
                )
                self._llm = _LlamaCpp(
                    model_path=LOCAL_MODEL_PATH,
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False,
                )
                self._runtime = "llama_cpp"
                logger.info("LocalProvider: modelo llama.cpp listo.")
            except Exception:
                logger.exception(
                    "LocalProvider: no se pudo cargar el modelo llama.cpp. "
                    "Usando Ollama local como fallback."
                )

        if self._runtime is None:
            # Usará Ollama local; disponibilidad verificada en generate()
            self._runtime = "ollama_local"
            logger.info(
                "LocalProvider: runtime = ollama_local (%s / %s)",
                LOCAL_OLLAMA_URL, LOCAL_OLLAMA_MODEL,
            )

    @property
    def name(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        if self._runtime == "llama_cpp" and LOCAL_MODEL_PATH:
            import os
            return f"llama_cpp:{os.path.basename(LOCAL_MODEL_PATH)}"
        return f"ollama_local:{LOCAL_OLLAMA_MODEL}"

    @property
    def is_configured(self) -> bool:
        """
        Configurado si:
        - hay modelo llama.cpp cargado, O
        - hay una URL de Ollama local definida
        """
        if self._runtime == "llama_cpp" and self._llm is not None:
            return True
        return bool(LOCAL_OLLAMA_URL)

    async def is_available(self) -> bool:
        """Verifica disponibilidad real (no solo config)."""
        if self._runtime == "llama_cpp" and self._llm is not None:
            return True
        # Para Ollama local, verificar que responde
        try:
            r = await self._http.get(
                f"{LOCAL_OLLAMA_URL.rstrip('/')}/api/tags",
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    async def generate(self, request: GenerateRequest) -> ProviderResponse:
        if self._runtime == "llama_cpp" and self._llm is not None:
            return await self._generate_llama_cpp(request)
        return await self._generate_ollama_local(request)

    # ── llama.cpp ────────────────────────────────────────────

    async def _generate_llama_cpp(self, request: GenerateRequest) -> ProviderResponse:
        """
        Genera texto con llama-cpp-python.
        La llamada es síncrona; la ejecutamos en un executor para
        no bloquear el event loop.
        """
        import asyncio

        started = time.perf_counter()

        # Construir prompt en formato ChatML
        prompt = self._build_chatml(request)

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._llm(  # type: ignore
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=["<|im_end|>", "<|endoftext|>", "USER:", "NEXUS:"],
                    echo=False,
                ),
            )
        except Exception as e:
            raise RuntimeError(f"llama.cpp error: {e}") from e

        text = result["choices"][0]["text"].strip()
        if not text:
            raise RuntimeError("llama.cpp devolvió respuesta vacía.")

        elapsed = time.perf_counter() - started
        logger.info("Local llama.cpp OK | %.2fs", elapsed)

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            duration_ms=int(elapsed * 1000),
            raw_metadata={"runtime": "llama_cpp"},
        )

    def _build_chatml(self, request: GenerateRequest) -> str:
        """Formato ChatML estándar para modelos GGUF."""
        parts = [f"<|im_start|>system\n{request.system}<|im_end|>"]
        for msg in request.history:
            role = "user" if msg.role == "user" else "assistant"
            parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{request.prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ── Ollama local ─────────────────────────────────────────

    async def _generate_ollama_local(self, request: GenerateRequest) -> ProviderResponse:
        started = time.perf_counter()

        messages = [{"role": "system", "content": request.system}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": LOCAL_OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        url = f"{LOCAL_OLLAMA_URL.rstrip('/')}/api/chat"

        try:
            response = await self._http.post(url, json=payload)
        except Exception as e:
            raise RuntimeError(
                f"Ollama local no responde en {LOCAL_OLLAMA_URL}: {e}"
            ) from e

        if response.status_code >= 400:
            raise RuntimeError(f"Ollama local HTTP {response.status_code}")

        data = response.json()
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama local devolvió respuesta vacía.")

        elapsed = time.perf_counter() - started
        logger.info("Local Ollama OK | modelo=%s | %.2fs", LOCAL_OLLAMA_MODEL, elapsed)

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            duration_ms=int(elapsed * 1000),
            raw_metadata={"runtime": "ollama_local"},
        )
