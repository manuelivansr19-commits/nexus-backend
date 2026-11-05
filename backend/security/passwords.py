"""Contraseña del propietario: scrypt con sal propia. Solo biblioteca estándar.

Formato guardado (todo en hexadecimal, sin caracteres especiales de shell):
    scrypt$N$r$p$sal_hex$hash_hex
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import NamedTuple, Optional

SCRYPT_N = 2 ** 16
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
DK_BYTES = 32
MIN_N_PRODUCTION = 2 ** 14
MIN_N_ANY = 2 ** 10
MAX_N = 2 ** 20
MAX_PASSWORD_CHARS = 1024


class ParsedHash(NamedTuple):
    n: int
    r: int
    p: int
    salt: bytes
    digest: bytes


def _maxmem(n: int, r: int, p: int) -> int:
    return 128 * n * r * (p + 1) + (1 << 20)


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                          dklen=dklen, maxmem=_maxmem(n, r, p))


def hash_password(password: str, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, n, r, p, DK_BYTES)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def parse_hash(value: str, production: bool = True) -> Optional[ParsedHash]:
    """Devuelve el hash analizado o None si el formato o los parámetros no son válidos."""
    try:
        parts = str(value).strip().split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            return None
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, digest = bytes.fromhex(parts[4]), bytes.fromhex(parts[5])
    except (ValueError, TypeError):
        return None
    min_n = MIN_N_PRODUCTION if production else MIN_N_ANY
    if n & (n - 1) or not (min_n <= n <= MAX_N):
        return None
    if not (1 <= r <= 16 and 1 <= p <= 4):
        return None
    if len(salt) < 8 or not (16 <= len(digest) <= 64):
        return None
    return ParsedHash(n, r, p, salt, digest)


def verify_password(password: str, parsed: ParsedHash) -> bool:
    """Comparación en tiempo constante. Nunca lanza por contraseñas raras."""
    if not isinstance(password, str) or len(password) > MAX_PASSWORD_CHARS:
        return False
    try:
        candidate = _derive(password, parsed.salt, parsed.n, parsed.r, parsed.p, len(parsed.digest))
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, parsed.digest)
