"""NEXUS Ω — Inference Package v3.8.0"""
from backend.inference.models import (
    InferenceMode, InferenceRequest, InferenceResult,
    RuntimeType, RuntimeStatus, RuntimeHealth, RuntimeCapabilities, Message,
    DEPRECATED_MODES,
)
from backend.inference.interface import InferenceRuntime
from backend.inference.errors import (
    InferenceError, RuntimeUnavailable, ModelUnavailable,
    InferenceTimeout, InferenceAuthenticationError, InferenceRateLimited,
    InferenceInvalidResponse, InferenceConfigurationError,
    CloudInferenceBlocked, AllRuntimesFailed, classify_error,
)
from backend.inference.policy import InferencePolicy
from backend.inference.registry import InferenceRegistry
from backend.inference.layer import InferenceLayer
from backend.inference.local.ollama import OllamaRuntime
from backend.inference.local.llamacpp import LlamaCppRuntime
from backend.inference.cloud.gemini import GeminiRuntime
from backend.inference.cloud.groq_openrouter import GroqRuntime, OpenRouterRuntime

__all__ = [
    "InferenceMode", "InferenceRequest", "InferenceResult", "Message",
    "RuntimeType", "RuntimeStatus", "RuntimeHealth", "DEPRECATED_MODES",
    "InferenceRuntime", "InferencePolicy", "InferenceRegistry", "InferenceLayer",
    "InferenceError", "RuntimeUnavailable", "ModelUnavailable",
    "InferenceTimeout", "InferenceAuthenticationError", "InferenceRateLimited",
    "InferenceInvalidResponse", "CloudInferenceBlocked", "AllRuntimesFailed",
    "classify_error", "OllamaRuntime", "LlamaCppRuntime",
    "GeminiRuntime", "GroqRuntime", "OpenRouterRuntime",
]
