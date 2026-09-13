"""NEXUS Ω — Inference Policy v3.8.0"""
from __future__ import annotations
from backend.config import ALLOW_CLOUD_INFERENCE, INFERENCE_MODE, logger
from backend.inference.errors import AllRuntimesFailed, CloudInferenceBlocked, InferenceRateLimited, RuntimeUnavailable
from backend.inference.models import InferenceMode, DEPRECATED_MODES, RuntimeType

class InferencePolicy:
    def __init__(self, mode=None, allow_cloud=None):
        self._mode        = mode or InferenceMode(INFERENCE_MODE)
        self._allow_cloud = allow_cloud if allow_cloud is not None else ALLOW_CLOUD_INFERENCE
        if self._mode in DEPRECATED_MODES:
            logger.warning("InferencePolicy: modo '%s' DEPRECATED.", self._mode.value)

    @property
    def mode(self): return self._mode

    @property
    def cloud_allowed(self): return self._allow_cloud

    def select_runtimes(self, all_runtimes):
        local  = [r for r in all_runtimes if r.is_local]
        cloud  = [r for r in all_runtimes if not r.is_local]
        if self._mode == InferenceMode.LOCAL_ONLY:
            if not local:
                raise CloudInferenceBlocked("LOCAL_ONLY: no hay runtimes locales.", runtime="none")
            return local
        if self._mode == InferenceMode.CLOUD_ONLY:
            if not self._allow_cloud:
                raise CloudInferenceBlocked("CLOUD_ONLY pero ALLOW_CLOUD_INFERENCE=false.", runtime="none")
            return cloud
        if self._mode == InferenceMode.LOCAL_FIRST:
            return local + cloud if self._allow_cloud else local
        if self._mode == InferenceMode.CLOUD_FIRST:
            return cloud + local if self._allow_cloud else local
        return local + cloud

    def should_retry(self, error): return True

    def is_cloud_blocked(self, runtime):
        return not runtime.is_local and not self._allow_cloud

    def log_decision(self, selected, blocked):
        logger.info("InferencePolicy: mode=%s cloud=%s | runtimes=[%s] blocked=[%s]",
            self._mode.value, "allowed" if self._allow_cloud else "BLOCKED",
            ",".join(r.name for r in selected), ",".join(r.name for r in blocked))
