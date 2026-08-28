"""NEXUS Ω — Camera Interface."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class CameraFrame:
    width: int; height: int; channels: int = 3
    data: Any = None          # bytes | numpy array | base64 str
    timestamp: float = field(default_factory=time.time)
    source: str = "camera_0"

class Camera(HardwareInterface):
    """
    Interfaz de cámara.
    En modo simulado devuelve frames vacíos.
    Para hardware real: implementar con OpenCV / libcamera.
    """
    def __init__(self, device_id: str = "camera_0",
                 width: int = 640, height: int = 480,
                 simulated: bool = True) -> None:
        super().__init__(device_id, simulated)
        self.width = width; self.height = height

    async def connect(self) -> bool:
        if self.simulated:
            self._status = HardwareStatus.SIMULATED
            return True
        # TODO: inicializar captura real (OpenCV, libcamera)
        self._status = HardwareStatus.ERROR
        return False

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED

    async def capture(self) -> Optional[CameraFrame]:
        if not self.is_ready:
            return None
        if self.simulated:
            return CameraFrame(
                width=self.width, height=self.height,
                data=None, source=self.device_id,
            )
        # TODO: captura real
        return None

    def describe(self) -> dict:
        return {"device": self.device_id, "resolution": f"{self.width}x{self.height}",
                "simulated": self.simulated, "status": self._status.value}
