"""NEXUS Ω — Inference Layer v3.8.0"""
from __future__ import annotations
import time
from backend.config import logger
from backend.inference.errors import AllRuntimesFailed, CloudInferenceBlocked, InferenceError, classify_error
from backend.inference.models import InferenceMode, InferenceRequest, InferenceResult, RuntimeHealth, RuntimeType
from backend.inference.policy import InferencePolicy
from backend.inference.registry import InferenceRegistry

class InferenceLayer:
    def __init__(self, registry: InferenceRegistry, policy: InferencePolicy):
        self._registry = registry
        self._policy   = policy

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        started      = time.perf_counter()
        all_runtimes = self._registry.all_runtimes()
        try:
            runtimes = self._policy.select_runtimes(all_runtimes)
        except CloudInferenceBlocked:
            raise
        if not runtimes:
            raise AllRuntimesFailed([("none", "No hay runtimes elegibles.")])
        blocked = [r for r in all_runtimes if r not in runtimes]
        self._policy.log_decision(runtimes, blocked)
        primary = runtimes[0].name
        errors  = []
        first   = True
        for runtime in runtimes:
            if self._policy.is_cloud_blocked(runtime):
                continue
            logger.info("[%s] Intentando runtime=%s [%s]", request.request_id, runtime.name, runtime.runtime_type.value)
            try:
                result = await runtime.generate(request)
                elapsed = int((time.perf_counter() - started) * 1000)
                result.duration_ms   = elapsed
                result.fallback_used = not first
                result.mode          = self._policy.mode
                result.request_id    = request.request_id
                result.is_local      = runtime.is_local
                logger.info("[%s] InferenceLayer OK | runtime=%s | fallback=%s | %dms",
                    request.request_id, runtime.name, result.fallback_used, elapsed)
                return result
            except InferenceError as e:
                errors.append((runtime.name, f"{e.code}: {str(e)[:100]}"))
                logger.warning("[%s] Runtime %s fallo: %s", request.request_id, runtime.name, e.code)
                first = False
            except Exception as e:
                typed = classify_error(e, runtime=runtime.name)
                errors.append((runtime.name, f"{typed.code}: {str(typed)[:100]}"))
                first = False
        raise AllRuntimesFailed(errors)

    async def health(self):
        return await self._registry.health()

    async def status(self):
        health_map   = await self._registry.health()
        local_status = {}
        cloud_status = {}
        for name, h in health_map.items():
            entry = {"status": h.status.value, "model": h.model,
                     "latency_ms": h.latency_ms, "message": h.message}
            if h.runtime_type == RuntimeType.LOCAL:
                local_status[name] = entry
            else:
                cloud_status[name] = entry
        return {
            "inference_layer": "OK",
            "mode":            self._policy.mode.value,
            "deprecated":      self._policy.mode.value == "cloud_first",
            "cloud_allowed":   self._policy.cloud_allowed,
            "local_runtimes":  local_status,
            "cloud_runtimes":  cloud_status,
            "registry":        self._registry.stats(),
        }

    async def shutdown(self):
        await self._registry.shutdown()
