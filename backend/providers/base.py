from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Message:
    role: str
    content: str

@dataclass
class GenerateRequest:
    prompt: str
    system: str
    history: list = field(default_factory=list)
    temperature: float = 0.5
    max_tokens: int = 8192

@dataclass
class ProviderResponse:
    text: str
    provider: str
    model: str
    duration_ms: int
    raw_metadata: dict = field(default_factory=dict)

class BaseModelProvider(abc.ABC):
    is_local: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def model(self) -> str: pass

    @property
    @abc.abstractmethod
    def is_configured(self) -> bool: pass

    async def is_available(self) -> bool:
        return self.is_configured

    @abc.abstractmethod
    async def generate(self, request: GenerateRequest) -> ProviderResponse: pass

    async def shutdown(self) -> None: pass
