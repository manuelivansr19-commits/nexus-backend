"""NEXUS Ω — IMU Interface."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class IMUReading:
    acceleration: dict = field(default_factory=lambda: {"x":0.0,"y":0.0,"z":9.81})
    gyroscope:    dict = field(default_factory=lambda: {"x":0.0,"y":0.0,"z":0.0})
    magnetometer: dict = field(default_factory=lambda: {"x":0.0,"y":0.0,"z":0.0})
    orientation:  dict = field(default_factory=lambda: {"roll":0.0,"pitch":0.0,"yaw":0.0})
    temperature:  float = 25.0
    timestamp:    float = field(default_factory=time.time)
    source:       str = "imu_0"

class IMU(HardwareInterface):
    """
    Interfaz IMU (MPU-6050, ICM-42688, BNO085).
    Para hardware real: smbus2 / I2C.
    """
    def __init__(self, device_id: str = "imu_0", simulated: bool = True) -> None:
        super().__init__(device_id, simulated)

    async def connect(self) -> bool:
        self._status = HardwareStatus.SIMULATED if self.simulated else HardwareStatus.ERROR
        return self.simulated

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED

    async def read(self) -> Optional[IMUReading]:
        if not self.is_ready: return None
        return IMUReading(source=self.device_id) if self.simulated else None

    def describe(self) -> dict:
        return {"device": self.device_id, "simulated": self.simulated,
                "status": self._status.value}
