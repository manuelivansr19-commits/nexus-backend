"""
NEXUS Ω — Simulation Engine.

Genera PerceptionEvents sintéticos para probar el
pipeline completo sin hardware.

Escenarios disponibles:
  IDLE          → sin movimiento, todo estático
  EXPLORING     → movimiento lento, scan LiDAR activo
  CONVERSATION  → input de texto/audio, sin movimiento
  OBSTACLE      → objeto detectado por LiDAR
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.core.perception import Modality, PerceptionEvent


class SimulationScenario(str, Enum):
    IDLE         = "idle"
    EXPLORING    = "exploring"
    CONVERSATION = "conversation"
    OBSTACLE     = "obstacle"


@dataclass
class SimulationEngine:
    """
    Motor de simulación de sensores.

    Produce eventos de percepción realistas sin hardware.
    Los eventos se pasan directamente a Perception.receive().
    """

    scenario:  SimulationScenario = SimulationScenario.IDLE
    tick_rate: float = 0.5          # segundos entre ticks
    _running:  bool = False
    _events:   list[PerceptionEvent] = field(default_factory=list)
    _t:        float = field(default_factory=time.time)

    def tick(self) -> list[PerceptionEvent]:
        """
        Genera un lote de eventos para el tick actual.
        Llamar periódicamente (e.g. cada 0.5s).
        """
        events = []
        self._t = time.time()

        if self.scenario == SimulationScenario.IDLE:
            events += self._imu_static()

        elif self.scenario == SimulationScenario.EXPLORING:
            events += self._imu_moving()
            events += self._lidar_clear()
            events += self._camera_frame()

        elif self.scenario == SimulationScenario.CONVERSATION:
            events += self._imu_static()

        elif self.scenario == SimulationScenario.OBSTACLE:
            events += self._imu_static()
            events += self._lidar_obstacle()

        self._events.extend(events)
        return events

    def pop_events(self) -> list[PerceptionEvent]:
        """Retorna y limpia todos los eventos acumulados."""
        events, self._events = self._events, []
        return events

    def inject_text(self, text: str, source: str = "user") -> PerceptionEvent:
        """Inyectar un evento de texto manualmente."""
        event = PerceptionEvent(
            modality=Modality.TEXT,
            data=text,
            source=source,
            confidence=1.0,
        )
        self._events.append(event)
        return event

    def inject_audio_transcript(self, transcript: str) -> PerceptionEvent:
        event = PerceptionEvent(
            modality=Modality.AUDIO,
            data={"transcript": transcript, "duration_ms": len(transcript) * 50},
            source="mic_0",
            confidence=0.92,
        )
        self._events.append(event)
        return event

    def set_scenario(self, scenario: SimulationScenario) -> None:
        self.scenario = scenario

    # ── Private generators ───────────────────────────────────

    def _imu_static(self) -> list[PerceptionEvent]:
        noise = lambda: random.uniform(-0.01, 0.01)
        return [PerceptionEvent(
            modality=Modality.IMU,
            data={
                "acceleration": {"x": noise(), "y": noise(), "z": 9.81 + noise()},
                "gyroscope":    {"x": noise(), "y": noise(), "z": noise()},
                "orientation":  {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "temperature":  25.0 + noise(),
            },
            source="imu_0",
            confidence=0.99,
        )]

    def _imu_moving(self) -> list[PerceptionEvent]:
        t = self._t
        return [PerceptionEvent(
            modality=Modality.IMU,
            data={
                "acceleration": {
                    "x": 0.3 * math.sin(t),
                    "y": 0.1 * math.cos(t),
                    "z": 9.81,
                },
                "gyroscope": {
                    "x": 0.05 * math.sin(t * 2),
                    "y": 0.02,
                    "z": 0.1 * math.cos(t),
                },
                "orientation": {
                    "roll":  2.0 * math.sin(t * 0.5),
                    "pitch": 1.5 * math.cos(t * 0.3),
                    "yaw":   math.degrees(t * 0.1) % 360,
                },
                "temperature": 26.5,
            },
            source="imu_0",
            confidence=0.98,
        )]

    def _lidar_clear(self, num_points: int = 36) -> list[PerceptionEvent]:
        """LiDAR sin obstáculos — puntos a distancia máxima."""
        points = []
        for i in range(num_points):
            angle = math.radians(i * (360 / num_points))
            dist  = 8.0 + random.uniform(-0.2, 0.2)
            points.append({
                "x": dist * math.cos(angle),
                "y": dist * math.sin(angle),
                "z": 0.0,
                "intensity": random.uniform(0.8, 1.0),
            })
        return [PerceptionEvent(
            modality=Modality.LIDAR,
            data={"points": points, "max_range": 10.0},
            source="lidar_front",
            confidence=0.99,
        )]

    def _lidar_obstacle(self) -> list[PerceptionEvent]:
        """LiDAR con obstáculo a 1.5m al frente."""
        points = []
        for i in range(36):
            angle = math.radians(i * 10)
            dist  = 8.0
            # obstáculo entre -15° y +15°
            if -0.26 < angle < 0.26 or angle > (2 * math.pi - 0.26):
                dist = 1.5 + random.uniform(-0.05, 0.05)
            points.append({
                "x": dist * math.cos(angle),
                "y": dist * math.sin(angle),
                "z": 0.0,
                "intensity": 1.0 if dist < 3.0 else 0.8,
            })
        return [PerceptionEvent(
            modality=Modality.LIDAR,
            data={"points": points, "max_range": 10.0, "obstacle_detected": True},
            source="lidar_front",
            confidence=0.97,
        )]

    def _camera_frame(self) -> list[PerceptionEvent]:
        return [PerceptionEvent(
            modality=Modality.VISION,
            data={"width": 640, "height": 480, "format": "RGB", "data": None},
            source="camera_0",
            confidence=1.0,
        )]
