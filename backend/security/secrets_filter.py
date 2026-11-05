"""Filtro de secretos (D6). HEURÍSTICO: no garantiza detectar todos los secretos.

Cubre una lista cerrada de familias de formato (ver FAMILIES) y los valores literales de
las claves configuradas en el entorno. No cubre formatos no listados. Los registros solo
guardan recuentos y nombres de familia, nunca el texto detectado.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

REDACTION = "[SECRETO OMITIDO]"

_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----|.*\Z)", re.S)
_ASSIGN = re.compile(
    r"(?i)\b(password|passwd|passphrase|secret|token|api[_-]?key|apikey|clave|contrase(?:n|ñ)a)"
    r"(\s*[:=]\s*)([\"']?)([^\s\"',;]{8,})")

# (familia, regex, grupo a redactar)  -- 0 = toda la coincidencia
FAMILIES: List[Tuple[str, "re.Pattern", int]] = [
    ("pem_private_key", _PEM, 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("authorization_bearer", re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{8,}"), 0),
    ("nexus_scrypt", re.compile(r"\bscrypt\$\d+\$\d+\$\d+\$[0-9a-fA-F]+\$[0-9a-fA-F]+"), 0),
    ("nexus_token", re.compile(r"\bnxs_[A-Za-z0-9_-]{20,}"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), 0),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}"), 0),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}"), 0),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"), 0),
]
FAMILY_NAMES = tuple(f[0] for f in FAMILIES) + ("assignment", "configured_secret")

_extra: List[str] = []


def set_known_secrets(values: Iterable[str]) -> None:
    """Valores literales (p. ej. las claves del entorno) que también se redactan. Mín. 8 caracteres."""
    global _extra
    _extra = sorted({v for v in values if isinstance(v, str) and len(v) >= 8}, key=len, reverse=True)


def _assignment_ok(value: str) -> bool:
    return (any(c.isdigit() for c in value) or any(not c.isalnum() for c in value)
            or (any(c.isupper() for c in value) and any(c.islower() for c in value)))


def scan(text: str) -> List[Tuple[str, int, int]]:
    """[(familia, inicio, fin)] sin solapes (gana la primera, de izquierda a derecha)."""
    if not isinstance(text, str) or not text:
        return []
    spans: List[Tuple[str, int, int]] = []
    for name, rx, g in FAMILIES:
        for m in rx.finditer(text):
            spans.append((name, m.start(g), m.end(g)))
    pos = 0
    while True:                      # un intento rechazado NO consume texto: «clave: password = X»
        m = _ASSIGN.search(text, pos)
        if m is None:
            break
        if _assignment_ok(m.group(4)):
            spans.append(("assignment", m.start(4), m.end(4)))
            pos = m.end()
        else:
            pos = m.start() + 1
    for secret in _extra:
        i = text.find(secret)
        while i != -1:
            spans.append(("configured_secret", i, i + len(secret)))
            i = text.find(secret, i + len(secret))
    spans.sort(key=lambda s: (s[1], -(s[2] - s[1])))
    out: List[Tuple[str, int, int]] = []
    end = -1
    for name, a, b in spans:
        if a >= end:
            out.append((name, a, b))
            end = b
    return out


def contains_secret(text: str) -> bool:
    return bool(scan(text))


def redact(text: str) -> Tuple[str, Dict[str, int]]:
    """-> (texto redactado, {familia: recuento})."""
    spans = scan(text)
    if not spans:
        return text, {}
    counts: Dict[str, int] = {}
    parts, pos = [], 0
    for name, a, b in spans:
        parts.append(text[pos:a])
        parts.append(REDACTION)
        pos = b
        counts[name] = counts.get(name, 0) + 1
    parts.append(text[pos:])
    return "".join(parts), counts


def redact_obj(obj: Any, counts: Dict[str, int] = None) -> Tuple[Any, Dict[str, int]]:
    """Redacta recursivamente cadenas en dict/list/tuple (las claves de dict no se tocan)."""
    counts = {} if counts is None else counts
    if isinstance(obj, str):
        new, c = redact(obj)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        return new, counts
    if isinstance(obj, dict):
        return {k: redact_obj(v, counts)[0] for k, v in obj.items()}, counts
    if isinstance(obj, list):
        return [redact_obj(v, counts)[0] for v in obj], counts
    if isinstance(obj, tuple):
        return tuple(redact_obj(v, counts)[0] for v in obj), counts
    return obj, counts


def obj_contains_secret(obj: Any) -> bool:
    return bool(redact_obj(obj)[1])


class SecretDetected(ValueError):
    """Guardado explícito o argumento de herramienta rechazado por contener un secreto."""
    def __init__(self, families):
        self.families = tuple(sorted(families))
        super().__init__("Se detectó un posible secreto (" + ", ".join(self.families) + "); no se guardó ni se ejecutó.")


def reject_if_secret(obj: Any) -> None:
    _, counts = redact_obj(obj)
    if counts:
        raise SecretDetected(counts.keys())
