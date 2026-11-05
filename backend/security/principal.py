"""El 'principal': quién hace la petición y qué puede hacer."""
from __future__ import annotations

from dataclasses import dataclass

ALL_PROJECTS = "*"


@dataclass(frozen=True)
class Principal:
    id: str                      # id del token, o "owner"
    kind: str                    # "token" | "owner"
    via: str                     # "bearer" | "cookie"
    permissions: frozenset
    projects: tuple              # ("*",) o nombres normalizados

    @property
    def is_global(self) -> bool:
        return self.projects == (ALL_PROJECTS,)

    def has(self, permission: str) -> bool:
        # Ningún permiso implica a otro: admin NO incluye read, etc.
        return permission in self.permissions

    def allows_project(self, project: str) -> bool:
        return self.is_global or project in self.projects


OWNER_PERMISSIONS = frozenset({"read", "chat", "run", "write", "admin"})


def owner_principal() -> Principal:
    return Principal(id="owner", kind="owner", via="cookie",
                     permissions=OWNER_PERMISSIONS, projects=(ALL_PROJECTS,))
