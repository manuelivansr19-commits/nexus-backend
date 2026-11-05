"""Tokens Bearer: solo se guarda el hash SHA-256; comparación en tiempo constante."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

from backend.security.principal import Principal

TOKEN_RE = re.compile(r"^nxs_[A-Za-z0-9_-]{32,128}$")


def new_token() -> Tuple[str, str]:
    """(token en claro, hash 'sha256:...'). El token en claro se muestra una sola vez."""
    raw = "nxs_" + secrets.token_urlsafe(32)
    return raw, hash_token(raw)


def hash_token(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TokenRegistry:
    def __init__(self, entries, environment: str):
        self._entries = tuple(entries)
        self._env = environment

    def __len__(self) -> int:
        return len(self._entries)

    def authenticate(self, raw: str, now: Optional[datetime] = None):
        """-> (Principal | None, motivo, id_del_token | None). El motivo solo va al registro."""
        if not isinstance(raw, str) or not TOKEN_RE.match(raw):
            return None, "malformed", None
        digest = hashlib.sha256(raw.encode("utf-8")).digest()
        found = None
        for e in self._entries:                 # se recorren TODAS: sin salida temprana
            if hmac.compare_digest(digest, bytes.fromhex(e.hash_hex)):
                found = e
        if found is None:
            return None, "unknown", None
        now = now or datetime.now(timezone.utc)
        if found.env != self._env:
            return None, "other_env", found.id
        if found.revoked:
            return None, "revoked", found.id
        if found.expires is not None and found.expires <= now:
            return None, "expired", found.id
        return (Principal(id=found.id, kind="token", via="bearer",
                          permissions=found.permissions, projects=found.projects),
                "ok", found.id)
