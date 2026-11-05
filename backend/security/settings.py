"""Configuración de seguridad: se lee y valida UNA vez. Nunca se imprime un valor secreto.

Todas las advertencias nombran la variable (o el id de un token), jamás su contenido.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional
from urllib.parse import urlsplit

from backend.security.passwords import ParsedHash, parse_hash

ENVIRONMENTS = ("production", "development", "test")
AUTH_MODES = ("enforce", "audit", "off")
PERMISSIONS = frozenset({"read", "chat", "run", "write", "admin"})
PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,79}$")
TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HASH_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


def normalize_project(value) -> Optional[str]:
    """Nombre de proyecto normalizado, o None si no es válido ('*' y vacío no lo son)."""
    if not isinstance(value, str):
        return None
    s = re.sub(r"\s+", " ", value).strip().lower()
    return s if PROJECT_RE.match(s) else None


def validate_origin(value: str, production: bool) -> Optional[str]:
    """Origen exacto (esquema://host[:puerto]) sin barra final. None si no es válido."""
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    try:
        parts = urlsplit(value)
        host, _ = parts.hostname, parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host:
        return None
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        return None
    if parts.scheme == "http" and (production or host not in LOOPBACK_HOSTS):
        return None
    canonical = f"{parts.scheme}://{parts.netloc}"
    return value if value == canonical.lower() and value == canonical else None


@dataclass(frozen=True)
class TokenEntry:
    id: str
    env: str
    hash_hex: str
    permissions: frozenset
    projects: tuple            # ("*",) o nombres
    expires: Optional[datetime]
    revoked: bool


@dataclass
class SecuritySettings:
    environment: str = "production"
    auth_mode: str = "enforce"
    owner_hash: Optional[ParsedHash] = None
    tokens: tuple = ()
    token_ids_discarded: tuple = ()
    public_origin: Optional[str] = None
    default_project: Optional[str] = None
    single_process_ack: bool = False
    allowed_origins: tuple = ()
    cookie_secure: bool = True
    session_max_seconds: int = 28800
    session_idle_seconds: int = 1800
    session_max_active: int = 5
    login_max_attempts: int = 5
    login_window_seconds: int = 900
    login_lockout_seconds: int = 900
    login_backoff_seconds: int = 1
    rate_chat_per_minute: int = 30
    rate_task_per_minute: int = 5
    unlock_rate_per_minute: int = 5
    max_body_bytes: int = 1048576
    body_timeout_seconds: int = 30
    browser_login_enabled: bool = False
    owner_configured: bool = False
    warnings: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def production(self) -> bool:
        return self.environment == "production"

    @property
    def has_auth_method(self) -> bool:
        """¿Hay alguna forma válida de autenticarse? (si no, enforce responde 503)"""
        return self.browser_login_enabled or bool(self.tokens)

    @property
    def cookie_name(self) -> str:
        return "__Host-nexus_session" if self.cookie_secure else "nexus_session"

    def startup_summary(self) -> str:
        ack = ("declared_not_verified" if self.single_process_ack else "not_declared")
        return (f"auth_mode={self.auth_mode} environment={self.environment} "
                f"owner={'yes' if self.owner_configured else 'no'} "
                f"browser_login={'enabled' if self.browser_login_enabled else 'disabled'} "
                f"tokens={len(self.tokens)} "
                f"default_project={'set' if self.default_project else 'unset'} "
                f"single_process_ack={ack}")

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "SecuritySettings":
        s = cls()
        w = s.warnings

        def get(name: str) -> Optional[str]:
            v = environ.get(name)
            return v if v is not None and v.strip() != "" else None

        def num(name: str, default: int) -> int:
            raw = get(name)
            if raw is None:
                return default
            try:
                v = int(raw.strip())
                if v <= 0:
                    raise ValueError
                return v
            except ValueError:
                w.append(f"{name} inválida: se usa el valor por defecto ({default}).")
                return default

        def strict_bool(name: str, default: bool) -> bool:
            raw = get(name)
            if raw is None:
                return default
            v = raw.strip().lower()
            if v in ("true", "false"):
                return v == "true"
            w.append(f"{name} inválida (solo true/false): se trata como {str(default).lower()}.")
            return default

        # --- entorno y modo ------------------------------------------------
        env = (get("NEXUS_ENVIRONMENT") or "production").strip().lower()
        if env not in ENVIRONMENTS:
            w.append("NEXUS_ENVIRONMENT inválida: se usa production.")
            env = "production"
        s.environment = env
        mode = (get("NEXUS_AUTH_MODE") or "enforce").strip().lower()
        if mode not in AUTH_MODES:
            w.append("NEXUS_AUTH_MODE inválida: se usa enforce.")
            mode = "enforce"
        if mode != "enforce" and env == "production":
            w.append("NEXUS_AUTH_MODE distinto de enforce se ignora en production: se usa enforce.")
            mode = "enforce"
        s.auth_mode = mode
        s.cookie_secure = strict_bool("NEXUS_COOKIE_SECURE", True)
        if not s.cookie_secure and env == "production":
            w.append("NEXUS_COOKIE_SECURE=false se ignora en production.")
            s.cookie_secure = True

        # --- números -------------------------------------------------------
        s.session_max_seconds = num("NEXUS_SESSION_MAX_SECONDS", 28800)
        s.session_idle_seconds = num("NEXUS_SESSION_IDLE_SECONDS", 1800)
        s.session_max_active = num("NEXUS_SESSION_MAX_ACTIVE", 5)
        s.login_max_attempts = num("NEXUS_LOGIN_MAX_ATTEMPTS", 5)
        s.login_window_seconds = num("NEXUS_LOGIN_WINDOW_SECONDS", 900)
        s.login_lockout_seconds = num("NEXUS_LOGIN_LOCKOUT_SECONDS", 900)
        s.login_backoff_seconds = num("NEXUS_LOGIN_BACKOFF_SECONDS", 1)
        s.rate_chat_per_minute = num("NEXUS_RATE_CHAT_PER_MINUTE", 30)
        s.rate_task_per_minute = num("NEXUS_RATE_TASK_PER_MINUTE", 5)
        s.unlock_rate_per_minute = num("NEXUS_UNLOCK_RATE_PER_MINUTE", 5)
        s.max_body_bytes = num("NEXUS_MAX_BODY_BYTES", 1048576)
        s.body_timeout_seconds = num("NEXUS_BODY_TIMEOUT_SECONDS", 30)

        # --- proyecto por defecto, orígenes --------------------------------
        raw = get("NEXUS_DEFAULT_PROJECT")
        if raw is not None:
            s.default_project = normalize_project(raw)
            if s.default_project is None:
                w.append("NEXUS_DEFAULT_PROJECT inválida: se trata como no definida.")
        raw = get("NEXUS_PUBLIC_ORIGIN")
        if raw is not None:
            s.public_origin = validate_origin(raw.strip(), s.production)
            if s.public_origin is None:
                w.append("NEXUS_PUBLIC_ORIGIN inválida: login por navegador desactivado.")
        else:
            w.append("NEXUS_PUBLIC_ORIGIN no definida: login por navegador desactivado.")
        allowed = []
        raw = get("NEXUS_ALLOWED_ORIGINS")
        if raw is not None:
            for item in raw.split(","):
                item = item.strip()
                ok = validate_origin(item, s.production) if item != "*" else None
                if ok:
                    allowed.append(ok)
                else:
                    w.append("NEXUS_ALLOWED_ORIGINS: se descartó una entrada inválida (nunca se admite '*').")
        s.allowed_origins = tuple(dict.fromkeys(allowed))

        # --- guardia de un solo proceso -----------------------------------
        s.single_process_ack = strict_bool("NEXUS_SINGLE_PROCESS_ACK", False)
        multi = False
        for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
            raw = get(name)
            if raw is not None:
                try:
                    if int(raw.strip()) > 1:
                        multi = True
                        w.append(f"{name} mayor que 1: el login por navegador queda desactivado.")
                except ValueError:
                    pass

        # --- propietario ----------------------------------------------------
        raw = get("NEXUS_OWNER_PASSWORD_HASH")
        if raw is not None:
            s.owner_hash = parse_hash(raw, production=s.production)
            if s.owner_hash is None:
                w.append("NEXUS_OWNER_PASSWORD_HASH inválido: no hay propietario.")
        s.owner_configured = s.owner_hash is not None
        login = s.owner_configured and s.public_origin is not None and not multi
        if login and s.production and not s.single_process_ack:
            w.append("NEXUS_SINGLE_PROCESS_ACK no es true: login por navegador desactivado en production.")
            login = False
        s.browser_login_enabled = login

        # --- tokens ----------------------------------------------------------
        s.tokens, s.token_ids_discarded = cls._parse_tokens(get("NEXUS_API_TOKENS"), w)
        for t in s.tokens:
            if t.projects != ("*",):
                dead = sorted(t.permissions & {"chat", "run", "write"})
                if dead or "read" not in t.permissions and "admin" not in t.permissions:
                    w.append(f"token '{t.id}': permisos sin rutas utilizables hasta la Fase 3 "
                             f"({', '.join(dead) or 'ninguno útil'}).")
        return s

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tokens(raw: Optional[str], w: list):
        if raw is None:
            return (), ()
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError
        except ValueError:
            w.append("NEXUS_API_TOKENS inválido (no es una lista JSON): TODOS los tokens desactivados.")
            return (), ()
        ids_seen: dict = {}
        good: dict = {}
        discarded: list = []
        for idx, item in enumerate(data):
            tid = item.get("id") if isinstance(item, dict) else None
            label = tid if isinstance(tid, str) and TOKEN_ID_RE.match(tid) else f"#{idx}"
            entry = SecuritySettings._parse_token(item, w, label)
            if isinstance(tid, str):
                ids_seen[tid] = ids_seen.get(tid, 0) + 1
            if entry is None:
                discarded.append(label)
            else:
                good[entry.id] = entry
        for tid, count in ids_seen.items():
            if count > 1 and tid in good:
                del good[tid]
                discarded.append(tid)
                w.append(f"token '{tid}': id repetido, se descartan todas sus entradas.")
        return tuple(good.values()), tuple(discarded)

    @staticmethod
    def _parse_token(item, w: list, label: str) -> Optional[TokenEntry]:
        def bad(reason: str):
            w.append(f"token '{label}' descartado: {reason}.")
            return None
        if not isinstance(item, dict):
            return bad("no es un objeto")
        tid = item.get("id")
        if not isinstance(tid, str) or not TOKEN_ID_RE.match(tid):
            return bad("id inválido")
        env = item.get("env")
        if env not in ENVIRONMENTS:
            return bad("env inválido")
        m = HASH_RE.match(item.get("hash") if isinstance(item.get("hash"), str) else "")
        if not m:
            return bad("hash inválido")
        perms = item.get("permissions")
        if (not isinstance(perms, list) or not perms
                or not all(isinstance(p, str) for p in perms) or not set(perms) <= PERMISSIONS):
            return bad("permisos inválidos o desconocidos")
        projects = item.get("projects")
        if not isinstance(projects, list) or not projects:
            return bad("projects ausente o vacío")
        if "*" in projects:
            if projects != ["*"]:
                return bad("projects mezcla '*' con nombres")
            proj = ("*",)
        else:
            names = [normalize_project(p) for p in projects]
            if any(n is None for n in names):
                return bad("projects con nombre inválido")
            proj = tuple(dict.fromkeys(names))
        expires = None
        if item.get("expires") is not None:
            try:
                raw = str(item["expires"]).replace("Z", "+00:00")
                expires = datetime.fromisoformat(raw)
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
            except ValueError:
                return bad("expires inválido")
        revoked = item.get("revoked", False)
        if not isinstance(revoked, bool):
            return bad("revoked inválido")
        return TokenEntry(id=tid, env=env, hash_hex=m.group(1), permissions=frozenset(perms),
                          projects=proj, expires=expires, revoked=revoked)
