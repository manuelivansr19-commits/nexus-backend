"""Tabla ÚNICA de rutas: cada una declara aquí su permiso y su clase de alcance.

Lo no declarado se niega. Una ruta pasa de GLOBAL a PROYECTO solo cuando su almacén
filtre por proyecto y exista su prueba (Fase 3).
"""
from __future__ import annotations

import re
from typing import NamedTuple, Optional

PUBLICA, SESION, UNLOCK, SISTEMA, PROYECTO, GLOBAL = (
    "PUBLICA", "SESION", "UNLOCK", "SISTEMA", "PROYECTO", "GLOBAL")


class Route(NamedTuple):
    n: int
    method: str
    template: str
    permission: Optional[str]
    cls: str
    bucket: Optional[str] = None       # cubo de límite de uso


_RAW = [
    (1, "GET", "/", None, PUBLICA, None),
    (2, "HEAD", "/", None, PUBLICA, None),
    (3, "GET", "/health", None, PUBLICA, None),
    (4, "GET", "/sw.js", None, PUBLICA, None),
    (5, "GET", "/api/auth/me", None, PUBLICA, None),
    (6, "POST", "/api/auth/login", None, PUBLICA, None),
    (7, "POST", "/api/auth/logout", None, SESION, None),
    (8, "POST", "/api/auth/logout-all", None, SESION, None),
    (9, "POST", "/api/auth/unlock", "admin", UNLOCK, "unlock"),
    (10, "GET", "/api/nexus/status", "read", SISTEMA, None),
    (11, "GET", "/api/nexus/config", "read", SISTEMA, None),
    (12, "POST", "/api/nexus/chat", "chat", GLOBAL, "chat"),
    (13, "GET", "/api/nexus/tasks", "read", GLOBAL, None),
    (14, "GET", "/api/nexus/tasks/pending", "read", GLOBAL, None),
    (15, "GET", "/api/nexus/tasks/{task_id}", "read", GLOBAL, None),
    (16, "GET", "/api/inference/health", "read", SISTEMA, None),
    (17, "GET", "/api/nexus/memory", "read", GLOBAL, None),
    (18, "POST", "/api/nexus/memory/clear", "admin", GLOBAL, None),
    (19, "GET", "/api/nexus/persistent-memory", "read", PROYECTO, None),
    (20, "DELETE", "/api/nexus/persistent-memory/{entry_id}", "admin", PROYECTO, None),
    (21, "GET", "/api/nexus/tools", "read", SISTEMA, None),
    (22, "POST", "/api/nexus/intent", "read", SISTEMA, None),
    (23, "POST", "/api/nexus/task", "run", GLOBAL, "task"),
    (24, "GET", "/api/nexus/autonomy", "read", SISTEMA, None),
    (25, "POST", "/api/knowledge/add", "write", GLOBAL, None),
    (26, "POST", "/api/knowledge/search", "read", GLOBAL, None),
    (27, "GET", "/api/knowledge/domains", "read", GLOBAL, None),
    (28, "GET", "/api/knowledge/stats", "read", GLOBAL, None),
    (29, "GET", "/api/knowledge/{entry_id}", "read", GLOBAL, None),
    (30, "DELETE", "/api/knowledge/{entry_id}", "admin", GLOBAL, None),
    (31, "POST", "/api/knowledge/refresh", "admin", GLOBAL, None),
    (32, "GET", "/api/aura/status", "read", GLOBAL, None),
    (33, "POST", "/api/aura/perceive", "write", GLOBAL, None),
    (34, "POST", "/api/aura/simulate", "write", GLOBAL, None),
]

ROUTES = tuple(Route(*r) for r in _RAW)
AUTH_PATHS = frozenset(r.template for r in ROUTES if r.template.startswith("/api/auth/"))
# Rutas que NUNCA se abren a tokens de proyecto en la Fase 2 (solo SISTEMA y PROYECTO se abren).
PROJECT_TOKEN_OPEN = tuple(r.n for r in ROUTES if r.cls in (SISTEMA, PROYECTO))


def _compile(template: str):
    return re.compile("^" + re.sub(r"\{[a-z_]+\}", r"[^/]+", re.escape(template).replace(r"\{", "{").replace(r"\}", "}")) + "$")


_COMPILED = tuple((r, _compile(r.template)) for r in ROUTES)


def match(method: str, path: str) -> Optional[Route]:
    """Ruta declarada que corresponde, o None (=> se niega). Las literales ganan a las {param}."""
    method = method.upper()
    best = None
    for route, rx in _COMPILED:
        if route.method == method and rx.match(path):
            if "{" not in route.template:
                return route
            best = best or route
    return best


def usable_routes(permissions, is_global: bool):
    """Números de las rutas protegidas que un token con esos permisos puede usar."""
    out = []
    for r in ROUTES:
        if r.cls in (PUBLICA, SESION, UNLOCK):
            if r.cls == UNLOCK and is_global and "admin" in permissions:
                out.append(r.n)
            continue
        if r.permission not in permissions:
            continue
        if r.cls == GLOBAL and not is_global:
            continue
        out.append(r.n)
    return out
