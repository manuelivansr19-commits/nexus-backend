"""NEXUS Ω — LiDAR Interface."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class LidarScan:
    points: list[dict] = field(default_factory=list)  # [{"x","y","z","intensity"}]
    max_range: float = 10.0
    timestamp: float = field(default_factory=time.time)
    source: str = "lidar_front"

class Lidar(HardwareInterface):
    """
    Interfaz LiDAR.
    Para hardware real: RPLIDAR, Livox, Velodyne vía SDK.
    """
    def __init__(self, device_id: str = "lidar_front",
                 max_range: float = 10.0, simulated: bool = True) -> None:
        super().__init__(device_id, simulated)
        self.max_range = max_range

    async def connect(self) -> bool:
        if self.simulated:
            self._status = HardwareStatus.SIMULATED
            return True
        self._status = HardwareStatus.ERROR
        return False

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED

    async def scan(self) -> Optional[LidarScan]:
        if not self.is_ready: return None
        if self.simulated:
            return LidarScan(points=[], max_range=self.max_range, source=self.device_id)
        return None

    def describe(self) -> dict:
        return {"device": self.device_id, "max_range": self.max_range,
                "simulated": self.simulated, "status": self._status.value}
