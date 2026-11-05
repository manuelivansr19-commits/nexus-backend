"""Middleware ASGI de acceso (Fase 2).

Orden de capas (decisión de diseño, ver v4 §4 y §8):
  1. encuadre HTTP, límite y plazo de cuerpo          (400 / 408 / 413)
  2. autenticación, CSRF, clase de ruta, permisos      (401 / 403 / 503)
  3. límites de uso y resolución de proyecto           (429 / 400)
  4. la aplicación (validación del modelo y manejador)

Por ir como middleware ASGI, un JSON malformado sin credencial recibe 401 y no 422: el control
de acceso se ejecuta antes de que FastAPI analice el cuerpo. El cuerpo (máx. 1 MB) se lee
COMPLETO aquí antes de entregárselo a la aplicación: el manejador nunca ve un cuerpo excesivo.

Sin dependencias fuera de la biblioteca estándar.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode

from backend.security import routes as R
from backend.security.login_guard import LoginGuard
from backend.security.passwords import verify_password
from backend.security.principal import Principal, owner_principal
from backend.security.ratelimit import RateLimiter
from backend.security.sessions import SessionStore
from backend.security.settings import SecuritySettings, normalize_project
from backend.security.tokens import TokenRegistry

log = logging.getLogger("NEXUS.security")

BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MUTATING = BODY_METHODS
_DIGITS = re.compile(r"^[0-9]+$")
_BEARER = re.compile(r"^Bearer ([^\s]+)$", re.I)


class SecurityState:
    """Estado en memoria del proceso: sesiones, contador de login, límites, boot_id."""

    def __init__(self, settings: SecuritySettings, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.settings = settings
        self.wall = wall
        self.boot_id = secrets.token_hex(8)
        self.tokens = TokenRegistry(settings.tokens, settings.environment)
        self.sessions = SessionStore(settings.session_max_seconds, settings.session_idle_seconds,
                                     settings.session_max_active, clock)
        self.guard = LoginGuard(settings.login_max_attempts, settings.login_window_seconds,
                                settings.login_lockout_seconds, settings.login_backoff_seconds, clock)
        self.limiter = RateLimiter(clock)


def build_state(settings: SecuritySettings, **kw) -> SecurityState:
    return SecurityState(settings, **kw)


class _Timeout(Exception): ...
class _TooLarge(Exception): ...
class _Disconnected(Exception): ...


class SecurityMiddleware:
    def __init__(self, app, settings: SecuritySettings, state: Optional[SecurityState] = None):
        self.app = app
        self.s = settings
        self.state = state or SecurityState(settings)

    # ------------------------------------------------------------------ ASGI
    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await self._http(scope, receive, send)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _hdrs(scope) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, v in scope.get("headers", []):
            out.setdefault(k.decode("latin-1").lower(), []).append(v.decode("latin-1"))
        return out

    @staticmethod
    async def _reply(send, status: int, body: dict, *, close: bool, extra: Optional[list] = None):
        raw = json.dumps(body, separators=(",", ":")).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode()),
                   (b"cache-control", b"no-store")]
        if close:
            headers.append((b"connection", b"close"))
        for k, v in (extra or []):
            headers.append((k.encode(), v.encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": raw})

    async def _read_body(self, receive) -> bytes:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.s.body_timeout_seconds
        chunks, total = [], 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise _Timeout
            try:
                msg = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                raise _Timeout
            if msg["type"] == "http.disconnect":
                raise _Disconnected
            if msg["type"] != "http.request":
                continue
            chunk = msg.get("body", b"") or b""
            total += len(chunk)
            if total > self.s.max_body_bytes:
                raise _TooLarge
            chunks.append(chunk)
            if not msg.get("more_body"):
                return b"".join(chunks)

    # ------------------------------------------------------------------ main
    async def _http(self, scope, receive, send):
        s = self.s
        method = scope["method"].upper()
        path = scope.get("path", "/")
        h = self._hdrs(scope)
        has_body = method in BODY_METHODS or "content-length" in h or "transfer-encoding" in h
        is_api = path.startswith("/api/")

        async def send_w(message):
            if message["type"] == "http.response.start":
                hs = list(message.get("headers", []))
                names = {k.lower() for k, _ in hs}
                if has_body and b"connection" not in names:
                    hs.append((b"connection", b"close"))
                if is_api:
                    hs = [(k, v) for k, v in hs if k.lower() != b"cache-control"]
                    hs.append((b"cache-control", b"no-store"))
                message = {**message, "headers": hs}
            await send(message)

        # ---- 1. encuadre HTTP -------------------------------------------------
        cl, te = h.get("content-length", []), h.get("transfer-encoding", [])
        framing_bad = (len(cl) > 1 or (cl and not _DIGITS.match(cl[0].strip()))
                       or (cl and te) or len(te) > 1
                       or (te and te[0].strip().lower() != "chunked"))
        if framing_bad:
            await self._reply(send, 400, {"error": "bad_request_framing"}, close=True)
            return
        if cl and int(cl[0]) > s.max_body_bytes:
            await self._reply(send, 413, {"error": "payload_too_large"}, close=True)
            return
        body_receive = receive
        if has_body:
            try:
                body = await self._read_body(receive)
            except _TooLarge:
                await self._reply(send, 413, {"error": "payload_too_large"}, close=True)
                return
            except _Timeout:
                await self._reply(send, 408, {"error": "request_timeout"}, close=True)
                return
            except _Disconnected:
                return
            sent = False

            async def body_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()
        else:
            body = b""

        # ---- 2. rutas de autenticación propias ---------------------------------
        route = R.match(method, path)
        if route is not None and path in R.AUTH_PATHS:
            await self._auth_endpoint(route, scope, h, body, send, has_body)
            return
        if s.auth_mode == "off":
            await self.app(scope, body_receive, send_w)
            return

        # ---- 3. control de acceso ----------------------------------------------
        principal, cred = self._authenticate(h)
        verdict = self._decide(route, principal, cred, h, method)
        if verdict is not None:
            status, body_json, reason = verdict
            log.warning("acceso denegado: ruta=%s estado=%s motivo=%s principal=%s",
                        route.n if route else "-", status, reason, principal.id if principal else "-")
            if s.auth_mode == "enforce":
                await self._reply(send, status, body_json, close=has_body)
                return
        elif route is not None and route.cls != R.PUBLICA:
            # ---- 4. límites de uso y proyecto -----------------------------------
            if route.bucket in ("chat", "task", "unlock"):
                limit = {"chat": s.rate_chat_per_minute, "task": s.rate_task_per_minute,
                         "unlock": s.unlock_rate_per_minute}[route.bucket]
                ok, wait = self.state.limiter.check(route.bucket, principal.id, limit)
                if not ok:
                    log.warning("límite de uso: ruta=%s principal=%s", route.n, principal.id)
                    if s.auth_mode == "enforce":
                        await self._reply(send, 429, {"error": "rate_limited"}, close=has_body,
                                          extra=[("retry-after", str(wait))])
                        return
            if route.cls == R.PROYECTO and s.auth_mode == "enforce":
                err = self._resolve_project(scope, principal)
                if err is not None:
                    await self._reply(send, err[0], err[1], close=has_body)
                    return
        if principal is not None:
            scope["nexus.principal"] = principal.id
        await self.app(scope, body_receive, send_w)

    # -------------------------------------------------------- autenticación
    def _cookie_sid(self, h) -> Optional[str]:
        found = []
        for line in h.get("cookie", []):
            for part in line.split(";"):
                k, _, v = part.strip().partition("=")
                if k == self.s.cookie_name:
                    found.append(v)
        return found[0] if len(found) == 1 else None

    def _authenticate(self, h):
        """-> (Principal|None, 'none'|'invalid'|'ok'). Un Authorization presente NUNCA cae a la cookie."""
        auth = h.get("authorization")
        if auth is not None:
            m = _BEARER.match(auth[0]) if len(auth) == 1 else None
            if not m:
                return None, "invalid"
            principal, reason, tid = self.state.tokens.authenticate(m.group(1), self.state.wall())
            if principal is None:
                log.warning("credencial rechazada: motivo=%s token=%s", reason, tid or "-")
                return None, "invalid"
            return principal, "ok"
        sid = self._cookie_sid(h)
        if sid is not None and self.state.sessions.validate(sid):
            return owner_principal(), "ok"
        return None, "none"

    def _origin_ok(self, h) -> bool:
        po = self.s.public_origin
        if po is None:
            return False
        origin = h.get("origin")
        if origin is not None:
            return len(origin) == 1 and origin[0] == po
        sfs = h.get("sec-fetch-site", [])
        return len(sfs) == 1 and sfs[0] == "same-origin"

    def _decide(self, route, principal, cred, h, method):
        """None si se permite; si no, (estado, cuerpo, motivo)."""
        if route is not None and route.cls == R.PUBLICA:
            return None
        if not self.s.has_auth_method:
            return 503, {"error": "auth_not_configured"}, "sin método de credencial"
        if cred != "ok":
            return 401, {"error": "unauthorized"}, "credencial " + cred
        if route is None:
            return 403, {"error": "forbidden"}, "ruta no declarada"
        if principal.via == "cookie" and method in MUTATING and not self._origin_ok(h):
            return 403, {"error": "origin_not_allowed"}, "csrf"
        cls = route.cls
        if cls == R.SESION:
            return None if principal.via == "cookie" else (403, {"error": "forbidden"}, "requiere sesión")
        if cls == R.UNLOCK:
            if principal.via == "bearer" and principal.has("admin") and principal.is_global:
                return None
            return 403, {"error": "unlock_requires_bearer_admin"}, "unlock"
        if cls == R.GLOBAL and not principal.is_global:
            return 403, {"error": "global_route_denied"}, "ruta global"
        if not principal.has(route.permission):
            return 403, {"error": "forbidden"}, "permiso insuficiente"
        return None

    def _resolve_project(self, scope, principal: Principal):
        """Resuelve el proyecto de las rutas PROYECTO (sin inferir) y reescribe la consulta."""
        pairs = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        values = [v for k, v in pairs if k == "project"]
        if len(values) > 1:
            return 400, {"error": "project_invalid"}
        raw = values[0].strip() if values else ""
        if raw == "":
            if self.s.default_project is None:
                return 400, {"error": "project_required"}
            project = self.s.default_project
        elif raw == "*":
            return 400, {"error": "project_invalid"}
        else:
            project = normalize_project(raw)
            if project is None:
                return 400, {"error": "project_invalid"}
        if not principal.allows_project(project):
            return 403, {"error": "forbidden"}
        pairs = [(k, v) for k, v in pairs if k != "project"] + [("project", project)]
        scope["query_string"] = urlencode(pairs).encode("latin-1")
        return None

    # ----------------------------------------------------- rutas /api/auth/*
    def _set_cookie(self, sid: str, max_age: int) -> list:
        attrs = f"{self.s.cookie_name}={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}"
        if self.s.cookie_secure:
            attrs += "; Secure"
        return [("set-cookie", attrs)]

    async def _auth_endpoint(self, route, scope, h, body, send, close):
        s, st = self.s, self.state
        n = route.n
        if n == 5:                                           # GET /api/auth/me
            _, cred = self._authenticate(h)
            await self._reply(send, 200, {"authenticated": cred == "ok"}, close=close)
            return
        if n == 6:
            await self._login(h, body, send, close)
            return
        principal, cred = self._authenticate(h)
        verdict = self._decide(route, principal, cred, h, scope["method"].upper())
        if verdict is not None and s.auth_mode != "off":
            status, bj, reason = verdict
            log.warning("acceso denegado: ruta=%s estado=%s motivo=%s", n, status, reason)
            await self._reply(send, status, bj, close=close)
            return
        if n == 7:                                           # logout
            sid = self._cookie_sid(h)
            if sid:
                st.sessions.destroy(sid)
            await self._reply(send, 200, {"authenticated": False}, close=close,
                              extra=self._set_cookie("", 0))
        elif n == 8:                                         # logout-all
            count = st.sessions.destroy_all()
            await self._reply(send, 200, {"authenticated": False, "sessions_closed": count},
                              close=close, extra=self._set_cookie("", 0))
        elif n == 9:                                         # unlock
            ok, wait = st.limiter.check("unlock", principal.id if principal else "-", s.unlock_rate_per_minute)
            if not ok:
                await self._reply(send, 429, {"error": "rate_limited"}, close=close,
                                  extra=[("retry-after", str(wait))])
                return
            cleared = st.guard.clear()
            log.warning("unlock: token=%s hora=%s bloqueos_limpiados=%d", principal.id if principal else "-",
                        st.wall().isoformat(timespec="seconds"), cleared)
            await self._reply(send, 200, {"unlocked": True, "cleared": cleared}, close=close)

    async def _login(self, h, body: bytes, send, close):
        s, st = self.s, self.state
        if not s.browser_login_enabled:
            await self._reply(send, 503, {"error": "login_unavailable"}, close=close)
            return
        if not self._origin_ok(h):                           # 403 ANTES de mirar la contraseña
            await self._reply(send, 403, {"error": "origin_not_allowed"}, close=close)
            return
        ctype = (h.get("content-type", [""])[0].split(";")[0].strip().lower())
        if len(h.get("content-type", [])) != 1 or ctype != "application/json":
            await self._reply(send, 415, {"error": "unsupported_media_type"}, close=close)
            return
        try:
            data = json.loads(body.decode("utf-8"))
            password = data.get("password") if isinstance(data, dict) else None
            if not isinstance(password, str) or not password or len(password) > 1024:
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            await self._reply(send, 400, {"error": "bad_request"}, close=close)
            return
        d = st.guard.begin()
        if d.kind != "ok":
            code = "login_locked" if d.kind == "locked" else "login_backoff"
            await self._reply(send, 429, {"error": code}, close=close,
                              extra=[("retry-after", str(d.retry_after))])
            return
        try:
            ok = await asyncio.get_running_loop().run_in_executor(
                None, verify_password, password, s.owner_hash)
        except BaseException:
            st.guard.abort()
            raise
        if ok:
            st.guard.succeed()
            sid = st.sessions.create()
            log.info("login correcto")
            await self._reply(send, 200, {"authenticated": True}, close=close,
                              extra=self._set_cookie(sid, s.session_max_seconds))
            return
        after = st.guard.fail()
        log.warning("login fallido")
        if after.kind == "locked":
            await self._reply(send, 429, {"error": "login_locked"}, close=close,
                              extra=[("retry-after", str(after.retry_after))])
        else:
            await self._reply(send, 401, {"error": "invalid_credentials"}, close=close)
