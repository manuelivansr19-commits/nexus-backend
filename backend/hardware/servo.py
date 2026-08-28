"""NEXUS Ω — Servo Interface."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class ServoCommand:
    angle: float          # grados, típico 0-180 o -90 a +90
    speed: float = 1.0    # 0.0-1.0
    servo_id: str = "servo_0"

class Servo(HardwareInterface):
    """
    Interfaz de servo motor.
    Para hardware real: pigpio / RPi.GPIO / PCA9685.
    NO ejecuta movimientos sin sandbox activado.
    """
    def __init__(self, device_id: str = "servo_0",
                 min_angle: float = 0.0, max_angle: float = 180.0,
                 simulated: bool = True) -> None:
        super().__init__(device_id, simulated)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._current_angle: float = 90.0

    async def connect(self) -> bool:
        self._status = HardwareStatus.SIMULATED if self.simulated else HardwareStatus.ERROR
        return self.simulated

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED

    async def move(self, cmd: ServoCommand) -> bool:
        """
        SIMULADO: registra el comando pero no mueve hardware.
        Para hardware real: implementar vía GPIO.
        """
        if not self.is_ready: return False
        angle = max(self.min_angle, min(self.max_angle, cmd.angle))
        self._current_angle = angle
        return True

    @property
    def current_angle(self) -> float:
        return self._current_angle

    def describe(self) -> dict:
        return {"device": self.device_id, "range": f"{self.min_angle}-{self.max_angle}°",
                "current": self._current_angle, "simulated": self.simulated,
                "status": self._status.value}
