"""NEXUS Ω — Inference Models v3.8.0"""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class InferenceMode(str, Enum):
    LOCAL_FIRST = "local_first"
    LOCAL_ONLY  = "local_only"
    CLOUD_FIRST = "cloud_first"
    CLOUD_ONLY  = "cloud_only"

DEPRECATED_MODES = {InferenceMode.CLOUD_FIRST}

class RuntimeType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"

class RuntimeStatus(str, Enum):
    AVAILABLE      = "available"
    UNAVAILABLE    = "unavailable"
    DEGRADED       = "degraded"
    NOT_CONFIGURED = "not_configured"
    SCAFFOLDED     = "scaffolded"

@dataclass
class Message:
    role: str
    content: str

@dataclass
class InferenceRequest:
    prompt:      str
    system:      str           = ""
    history:     list          = field(default_factory=list)
    max_tokens:  int           = 8192
    temperature: float         = 0.5
    request_id:  str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata:    dict          = field(default_factory=dict)

@dataclass
class RuntimeCapabilities:
    streaming:         bool      = False
    tool_calling:      bool      = False
    structured_output: bool      = False
    vision:            bool      = False
    embeddings:        bool      = False
    max_context:       int       = 4096
    languages:         list      = field(default_factory=lambda: ["es","en"])

@dataclass
class RuntimeHealth:
    status:       RuntimeStatus
    runtime_name: str
    runtime_type: RuntimeType
    model:        str   = ""
    latency_ms:   int   = 0
    message:      str   = ""
    checked_at:   float = field(default_factory=time.time)
    def is_available(self) -> bool:
        return self.status == RuntimeStatus.AVAILABLE

@dataclass
class InferenceResult:
    text:          str
    runtime_name:  str
    runtime_type:  RuntimeType
    model:         str
    mode:          object
    fallback_used: bool  = False
    is_local:      bool  = False
    duration_ms:   int   = 0
    request_id:    str   = ""
    telemetry:     dict  = field(default_factory=dict)
    def to_telemetry(self) -> dict:
        return {
            "request_id":    self.request_id,
            "mode":          self.mode.value if self.mode else "unknown",
            "runtime":       self.runtime_name,
            "runtime_type":  self.runtime_type.value,
            "model":         self.model,
            "is_local":      self.is_local,
            "fallback_used": self.fallback_used,
            "duration_ms":   self.duration_ms,
            "success":       bool(self.text),
        }
