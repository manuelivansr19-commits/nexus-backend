"""NEXUS Ω — Motor Interface."""
from __future__ import annotations
from dataclasses import dataclass
from backend.hardware.base import HardwareInterface, HardwareStatus

@dataclass
class MotorCommand:
    speed: float      # -1.0 (reversa máx) a +1.0 (adelante máx)
    duration_ms: int = 0   # 0 = continuo hasta nuevo comando
    motor_id: str = "motor_left"

class Motor(HardwareInterface):
    """
    Interfaz de motor DC / paso a paso.
    Para hardware real: L298N, DRV8833, ODrive.
    NO ejecuta movimientos sin sandbox activado.
    """
    def __init__(self, device_id: str = "motor_left", simulated: bool = True) -> None:
        super().__init__(device_id, simulated)
        self._speed: float = 0.0

    async def connect(self) -> bool:
        self._status = HardwareStatus.SIMULATED if self.simulated else HardwareStatus.ERROR
        return self.simulated

    async def disconnect(self) -> None:
        self._status = HardwareStatus.DISCONNECTED
        self._speed = 0.0

    async def set_speed(self, cmd: MotorCommand) -> bool:
        if not self.is_ready: return False
        self._speed = max(-1.0, min(1.0, cmd.speed))
        return True

    async def stop(self) -> None:
        self._speed = 0.0

    def describe(self) -> dict:
        return {"device": self.device_id, "current_speed": self._speed,
                "simulated": self.simulated, "status": self._status.value}
