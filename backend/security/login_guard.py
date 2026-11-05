"""Intentos de login: ventana fija, retardo progresivo y bloqueo. Un contador global por proceso."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Decision:
    kind: str            # "ok" | "backoff" | "locked"
    retry_after: int = 0


class LoginGuard:
    def __init__(self, max_attempts: int = 5, window: int = 900, lockout: int = 900,
                 backoff_base: int = 1, clock: Callable[[], float] = time.monotonic):
        self._max, self._window, self._lockout, self._base = max_attempts, window, lockout, backoff_base
        self._clock = clock
        self.failures = 0
        self._window_start = 0.0
        self._next_allowed = 0.0
        self._locked_until = 0.0
        self._inflight = False

    def _reset(self) -> None:
        self.failures = 0
        self._window_start = self._next_allowed = self._locked_until = 0.0

    def _roll(self) -> None:
        now = self._clock()
        if self._locked_until and now >= self._locked_until:
            self._reset()
        elif self.failures and not self._locked_until and now - self._window_start >= self._window:
            self._reset()

    @staticmethod
    def _ceil(x: float) -> int:
        return max(1, math.ceil(x))

    def begin(self) -> Decision:
        """Reserva el turno de evaluación. 'ok' => hay que llamar a fail() o succeed()."""
        self._roll()
        now = self._clock()
        if self._locked_until:
            return Decision("locked", min(self._lockout, self._ceil(self._locked_until - now)))
        if self._inflight:                    # una sola evaluación a la vez
            return Decision("backoff", 1)
        if now < self._next_allowed:
            return Decision("backoff", self._ceil(self._next_allowed - now))
        self._inflight = True
        return Decision("ok")

    def fail(self) -> Decision:
        """Contraseña incorrecta evaluada. 'locked' si fue el que activa el bloqueo."""
        self._inflight = False
        now = self._clock()
        if self.failures == 0:
            self._window_start = now
        self.failures += 1
        if self.failures >= self._max:
            self._locked_until = now + self._lockout
            return Decision("locked", self._lockout)
        self._next_allowed = now + self._base * (2 ** (self.failures - 1))
        return Decision("ok")

    def succeed(self) -> None:
        self._inflight = False
        self._reset()

    def abort(self) -> None:
        """Se reservó turno pero no se llegó a evaluar (error interno): no cuenta."""
        self._inflight = False

    def clear(self) -> int:
        """unlock: borra contador, retardo y bloqueo. Devuelve cuántos fallos había."""
        n = self.failures
        was_locked = bool(self._locked_until)
        self._reset()
        return n if n else (1 if was_locked else 0)
