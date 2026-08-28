"""NEXUS Ω — Microphone Interface."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class AudioChunk:
    data: Any = None           # bytes | numpy array
    sample_rate: int = 16000
    duration_ms: int = 0
    transcript: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "mic_0"

class Microphone(HardwareInterface):
    """
    Interfaz de micrófono.
    Para hardware real: PyAudio / sounddevice.
    """
    def __init__(self, device_id: str = "mic_0",
                 sample_rate: int = 16000, simulated: bool = True) -> None:
        super().__init__(device_id, simulated)
        self.sample_rate = sample_rate

    async def connect(self) -> bool:
        if self.simulated:
            self._status = HardwareStatus.SIMULATED
            return True
        self._status = HardwareStatus.ERROR
        return False

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED

    async def record(self, duration_ms: int = 1000) -> Optional[AudioChunk]:
        if not self.is_ready: return None
        if self.simulated:
            return AudioChunk(sample_rate=self.sample_rate,
                              duration_ms=duration_ms, source=self.device_id)
        return None

    def describe(self) -> dict:
        return {"device": self.device_id, "sample_rate": self.sample_rate,
                "simulated": self.simulated, "status": self._status.value}
