"""Construcción de configuraciones y middleware para las pruebas de la Fase 2.

Todo es SINTÉTICO: contraseñas, tokens y valores sensibles son centinelas falsos
generados al vuelo (nunca credenciales reales). Cada prueba construye su propia
configuración a partir de un diccionario: no depende del entorno del equipo.
"""
import json
import logging

from backend.security import routes as R
from backend.security.access import SecurityMiddleware, build_state
from backend.security.passwords import hash_password
from backend.security.settings import SecuritySettings
from backend.security.tokens import new_token
from tests.asgi_helper import RecordingApp, call

logging.getLogger("NEXUS.security").setLevel(logging.CRITICAL)

ORIGIN = "http://127.0.0.1:8000"
PASSWORD = "contrasena-sintetica-de-prueba-1"
_HASH = hash_password(PASSWORD, n=1024)            # N bajo: solo para pruebas (no production)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def token_entry(tid, perms, projects, env="test", **extra):
    raw, digest = new_token()
    entry = {"id": tid, "env": env, "hash": digest, "permissions": perms, "projects": projects}
    entry.update(extra)
    return raw, entry


def make_env(extra_tokens=(), **over):
    """-> (env, tokens{id: valor_en_claro})"""
    specs = [("adm", ["read", "admin"], ["*"]),
             ("rd", ["read"], ["*"]),
             ("all", ["read", "chat", "run", "write", "admin"], ["*"]),
             ("pa", ["read", "chat", "run", "write", "admin"], ["aura"]),
             ("pb", ["read", "admin"], ["beta"]),
             ("pro", ["read"], ["aura"])]
    tokens, entries = {}, []
    for tid, perms, projects in list(specs) + list(extra_tokens):
        raw, e = token_entry(tid, perms, projects)
        tokens[tid] = raw
        entries.append(e)
    env = {"NEXUS_ENVIRONMENT": "test", "NEXUS_PUBLIC_ORIGIN": ORIGIN,
           "NEXUS_OWNER_PASSWORD_HASH": _HASH, "NEXUS_API_TOKENS": json.dumps(entries),
           "NEXUS_DEFAULT_PROJECT": "default"}
    env.update(over)
    return env, tokens


def build(clock=None, extra_tokens=(), app=None, **over):
    """-> (mw, app, state, tokens, settings)"""
    env, tokens = make_env(extra_tokens, **over)
    settings = SecuritySettings.from_env(env)
    state = build_state(settings, clock=clock) if clock else build_state(settings)
    app = app or RecordingApp()
    mw = SecurityMiddleware(app, settings, state)
    return mw, app, state, tokens, settings


def bearer(tokens, tid, **extra):
    h = {"Authorization": "Bearer " + tokens[tid]}
    h.update(extra)
    return h


JSON_H = {"content-type": "application/json"}
LOGIN_H = {"Origin": ORIGIN, "content-type": "application/json"}


def login(mw, password=PASSWORD, headers=None):
    return call(mw, "POST", "/api/auth/login", headers or LOGIN_H,
                json.dumps({"password": password}).encode())


def cookie_of(resp):
    sc = resp.header("set-cookie") or ""
    return sc.split(";")[0]


def concrete_path(template):
    return (template.replace("{task_id}", "abc123").replace("{entry_id}", "abc123"))


def valid_request(route, extra_headers=None, with_project=False):
    """Petición VÁLIDA para esa ruta (precondición P): método y ruta existentes, JSON válido."""
    path = concrete_path(route.template)
    if with_project:
        path += "?project=aura"
    headers = dict(JSON_H) if route.method in ("POST", "PUT", "PATCH", "DELETE") else {}
    headers.update(extra_headers or {})
    body = b"{}" if route.method in ("POST", "PUT", "PATCH") else b""
    return route.method, path, headers, body
