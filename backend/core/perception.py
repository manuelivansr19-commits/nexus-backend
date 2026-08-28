"""
NEXUS Ω — Perception.

Recibe señales del mundo (sensores reales o simulados)
y las convierte en eventos estructurados que el cerebro
puede procesar.

Modalities soportadas:
  VISION    → frames de cámara (base64 o numpy array)
  LIDAR     → nube de puntos 3D
  AUDIO     → fragmentos de audio / transcripciones
  IMU       → aceleración, giroscopio, orientación
  TEXT      → entrada de texto (teclado, STT)
  SYSTEM    → señales internas (timers, alertas)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Modality(str, Enum):
    VISION  = "vision"
    LIDAR   = "lidar"
    AUDIO   = "audio"
    IMU     = "imu"
    TEXT    = "text"
    SYSTEM  = "system"


@dataclass
class PerceptionEvent:
    """Unidad atómica de información percibida."""
    modality:   Modality
    data:       Any                     # payload según modality
    timestamp:  float = field(default_factory=time.time)
    source:     str   = "unknown"       # "camera_0", "lidar_front", "user", …
    confidence: float = 1.0             # 0.0 – 1.0
    event_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata:   dict  = field(default_factory=dict)

    def to_text(self) -> str:
        """Representación textual para el LLM."""
        return (
            f"[{self.modality.value.upper()}:{self.source}] "
            f"confidence={self.confidence:.2f} | {self._data_summary()}"
        )

    def _data_summary(self) -> str:
        if self.modality == Modality.TEXT:
            return str(self.data)[:500]
        if self.modality == Modality.IMU:
            return (
                f"acc={self.data.get('acceleration',{})} "
                f"gyro={self.data.get('gyroscope',{})}"
            )
        if self.modality == Modality.LIDAR:
            pts = self.data.get("points", [])
            return f"{len(pts)} puntos, rango={self.data.get('max_range','?')}m"
        if self.modality == Modality.VISION:
            return f"frame {self.data.get('width','?')}x{self.data.get('height','?')}"
        if self.modality == Modality.AUDIO:
            return self.data.get("transcript", f"audio {self.data.get('duration_ms','?')}ms")
        return str(self.data)[:200]


class Perception:
    """
    Procesador de percepción.

    Recibe eventos de sensores, los valida y los encola
    para que el sistema de razonamiento los consuma.
    """

    MAX_QUEUE = 100

    def __init__(self) -> None:
        self._queue: list[PerceptionEvent] = []
        self._handlers: dict[Modality, list] = {m: [] for m in Modality}

    def receive(self, event: PerceptionEvent) -> None:
        """Registrar un evento de percepción."""
        if len(self._queue) >= self.MAX_QUEUE:
            self._queue.pop(0)   # drop oldest
        self._queue.append(event)
        for handler in self._handlers.get(event.modality, []):
            try:
                handler(event)
            except Exception:
                pass

    def on(self, modality: Modality, handler) -> None:
        """Registrar un callback para una modalidad."""
        self._handlers[modality].append(handler)

    def pending(self) -> list[PerceptionEvent]:
        """Retorna y vacía la cola."""
        events, self._queue = self._queue, []
        return events

    def latest(self, modality: Optional[Modality] = None) -> Optional[PerceptionEvent]:
        """Último evento de una modalidad (o el más reciente en general)."""
        filtered = [e for e in self._queue if modality is None or e.modality == modality]
        return filtered[-1] if filtered else None

    def to_context(self, max_events: int = 5) -> str:
        """Genera texto de contexto perceptual para el LLM."""
        recent = self._queue[-max_events:]
        if not recent:
            return "Sin percepción activa."
        return "\n".join(e.to_text() for e in recent)
