"""NEXUS Ω — Cloud Runtime Base v3.8.0"""
from backend.inference.interface import InferenceRuntime
from backend.inference.models import RuntimeType

class CloudRuntime(InferenceRuntime):
    @property
    def runtime_type(self): return RuntimeType.CLOUD
    @property
    def is_local(self): return False
