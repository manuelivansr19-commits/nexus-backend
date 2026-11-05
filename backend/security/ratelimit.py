"""Límites de uso por principal y por ventana de 60 s (por proceso)."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Dict, Tuple


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic, window: float = 60.0):
        self._clock, self._window = clock, window
        self._hits: Dict[Tuple[str, str], deque] = {}

    def check(self, bucket: str, principal_id: str, limit: int):
        """-> (permitido, retry_after_segundos). Registra el uso solo si se permite."""
        now = self._clock()
        q = self._hits.setdefault((bucket, principal_id), deque())
        while q and now - q[0] >= self._window:
            q.popleft()
        if len(q) >= limit:
            return False, max(1, math.ceil(self._window - (now - q[0])))
        q.append(now)
        return True, 0
