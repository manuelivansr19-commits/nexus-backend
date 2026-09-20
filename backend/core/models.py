"""
NEXUS Ω — Inference Models v3.8.0

Tipos de datos del sistema de inferencia.
Sin dependencias de SDKs específicos.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class InferenceMode(str, Enum):
    LOCAL_FIRST  = "local_first"   # local preferred, cloud fallback si permitido
    LOCAL_ONLY   = "local_only"    # solo local, nunca cloud
    CLOUD_FIRST  = "cloud_first"   # DEPRECATED — cloud preferred, local fallback
    CLOUD_ONLY   = "cloud_only"    # solo cloud, nunca local


DEPRECATED_MODES = {InferenceMode.CLOUD_FIRST}


class RuntimeType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class RuntimeStatus(str, Enum):
    AVAILABLE     = "available"
    UNAVAILABLE   = "unavailable"
    DEGRADED      = "degraded"
    NOT_CONFIGURED = "not_configured"
    SCAFFOLDED    = "scaffolded"    # interfaz lista, implementación pendiente


@dataclass
class Message:
    role:    str    # "user" | "assistant" | "system"
    content: str


@dataclass
class InferenceRequest:
    """
    Request unificado para cualquier runtime.
    NexusCore solo conoce este tipo.
    """
    prompt:      str
    system:      str              = ""
    history:     list[Message]    = field(default_factory=list)
    max_tokens:  int              = 8192
    temperature: float            = 0.5
    request_id:  str              = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata:    dict             = field(default_factory=dict)


@dataclass
class RuntimeCapabilities:
    """Capacidades declaradas por un runtime."""
    streaming:        bool = False
    tool_calling:     bool = False
    structured_output: bool = False
    vision:           bool = False
    embeddings:       bool = False
    max_context:      int  = 4096
    languages:        list[str] = field(default_factory=lambda: ["es", "en"])


@dataclass
class RuntimeHealth:
    """Estado de salud de un runtime."""
    status:       RuntimeStatus
    runtime_name: str
    runtime_type: RuntimeType
    model:        str        = ""
    latency_ms:   int        = 0
    message:      str        = ""
    checked_at:   float      = field(default_factory=time.time)

    def is_available(self) -> bool:
        return self.status == RuntimeStatus.AVAILABLE


@dataclass
class InferenceResult:
    """
    Resultado de una inferencia.
    Agnóstico al runtime concreto que respondió.
    """
    text:         str
    runtime_name: str           # "ollama", "gemini", "groq", etc.
    runtime_type: RuntimeType   # LOCAL o CLOUD
    model:        str
    mode:         InferenceMode
    fallback_used: bool         = False
    is_local:     bool          = False
    duration_ms:  int           = 0
    request_id:   str           = ""
    telemetry:    dict          = field(default_factory=dict)

    def to_telemetry(self) -> dict:
        """Telemetría sin secretos."""
        return {
            "request_id":   self.request_id,
            "mode":         self.mode.value,
            "runtime":      self.runtime_name,
            "runtime_type": self.runtime_type.value,
            "model":        self.model,
            "is_local":     self.is_local,
            "fallback_used": self.fallback_used,
            "duration_ms":  self.duration_ms,
            "success":      bool(self.text),
        }
