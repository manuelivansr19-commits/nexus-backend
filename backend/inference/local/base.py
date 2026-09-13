"""NEXUS Ω — Local Runtime Base v3.8.0"""
from backend.inference.interface import InferenceRuntime
from backend.inference.models import RuntimeType

class LocalRuntime(InferenceRuntime):
    @property
    def runtime_type(self): return RuntimeType.LOCAL
    @property
    def is_local(self): return True
