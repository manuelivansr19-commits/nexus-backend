"""NEXUS Ω — Inference Runtime Interface v3.8.0"""
from __future__ import annotations
import abc
from backend.inference.models import InferenceRequest, InferenceResult, RuntimeCapabilities, RuntimeHealth, RuntimeType

class InferenceRuntime(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def runtime_type(self) -> RuntimeType: pass

    @property
    def is_local(self) -> bool:
        return self.runtime_type == RuntimeType.LOCAL

    @abc.abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResult: pass

    @abc.abstractmethod
    async def health(self) -> RuntimeHealth: pass

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def shutdown(self) -> None: pass
