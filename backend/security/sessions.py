"""Sesiones opacas en memoria del proceso. Solo se guarda el hash del id.

Con varios procesos cada uno tiene SU almacén: la Fase 2 soporta un solo proceso.
"""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Callable, Dict, List


class SessionStore:
    def __init__(self, max_seconds: int, idle_seconds: int, max_active: int,
                 clock: Callable[[], float] = time.monotonic):
        self._max, self._idle, self._active = max_seconds, idle_seconds, max_active
        self._clock = clock
        self._data: Dict[str, List[float]] = {}      # hash -> [creada, ultimo_uso]

    @staticmethod
    def _h(sid: str) -> str:
        return hashlib.sha256(sid.encode("utf-8")).hexdigest()

    def _reap(self) -> None:
        now = self._clock()
        for k in [k for k, (c, l) in self._data.items()
                  if now - c >= self._max or now - l >= self._idle]:
            del self._data[k]

    def create(self) -> str:
        self._reap()
        sid = secrets.token_urlsafe(32)
        now = self._clock()
        self._data[self._h(sid)] = [now, now]
        while len(self._data) > self._active:        # expulsa la más antigua
            oldest = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]
        return sid

    def validate(self, sid: str) -> bool:
        if not isinstance(sid, str) or not sid:
            return False
        self._reap()
        rec = self._data.get(self._h(sid))
        if rec is None:
            return False
        rec[1] = self._clock()
        return True

    def destroy(self, sid: str) -> None:
        self._data.pop(self._h(sid), None)

    def destroy_all(self) -> int:
        n = len(self._data)
        self._data.clear()
        return n

    def count(self) -> int:
        self._reap()
        return len(self._data)
