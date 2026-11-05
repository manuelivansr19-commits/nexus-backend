"""Redacción de lo que sale hacia un runtime de inferencia (local o de nube).

Cubre prompt, system, cada content del historial y los valores de texto de metadata.
Solo se registran recuentos y nombres de familia, nunca el texto detectado.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

from backend.security.secrets_filter import redact, redact_obj

log = logging.getLogger("NEXUS.security")


def _redact_item(item: Any, counts: dict) -> Any:
    if isinstance(item, str):
        new, c = redact(item)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        return new
    if isinstance(item, dict):
        return {k: _redact_item(v, counts) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(_redact_item(v, counts) for v in item)
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        changes = {f.name: _redact_item(getattr(item, f.name), counts)
                   for f in dataclasses.fields(item) if isinstance(getattr(item, f.name), (str, dict, list, tuple))}
        return dataclasses.replace(item, **changes)
    return item


def redact_request(request: Any, where: str = "runtime"):
    """Devuelve una COPIA de la petición con prompt, system, history y metadata redactados."""
    counts: dict = {}
    names = ("prompt", "system", "history", "metadata")
    changes = {n: _redact_item(getattr(request, n), counts) for n in names if hasattr(request, n)}
    if counts:
        log.warning("Filtro de secretos: se redactó texto antes de enviarlo a %s (%s).",
                    where, ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if dataclasses.is_dataclass(request) and not isinstance(request, type):
        return dataclasses.replace(request, **changes)
    try:                                   # objetos que no son dataclass (p. ej. dobles de prueba)
        clone = copy.copy(request)
    except Exception:
        clone = request
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone
