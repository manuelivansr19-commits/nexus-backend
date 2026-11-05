"""Herramienta LOCAL de credenciales:  python -m backend.security.cli <comando>

  new-token       genera un token (se muestra UNA vez) y su entrada para NEXUS_API_TOKENS
  hash-password   pide la contraseña SIN eco y muestra el hash para NEXUS_OWNER_PASSWORD_HASH
  check-env       valida la configuración del entorno actual (sin imprimir valores)
  scan-stored     localiza posibles secretos en una COPIA local de una base SQLite (solo ids)

Garantías: cero conexiones de red, cero archivos de salida y ningún secreto en argumentos.
Todo sale por la salida estándar de la terminal.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sqlite3
import sys

from backend.security import routes as R
from backend.security.passwords import hash_password
from backend.security.secrets_filter import scan
from backend.security.settings import ENVIRONMENTS, PERMISSIONS, SecuritySettings
from backend.security.tokens import new_token


def cmd_new_token(args, out) -> int:
    perms = sorted(set(p.strip() for p in args.permissions.split(",") if p.strip()))
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    if not perms or not set(perms) <= PERMISSIONS:
        print("Permisos inválidos. Válidos: " + ", ".join(sorted(PERMISSIONS)), file=out)
        return 2
    if args.env not in ENVIRONMENTS:
        print("Entorno inválido. Válidos: " + ", ".join(ENVIRONMENTS), file=out)
        return 2
    raw, digest = new_token()
    entry = {"id": args.id, "env": args.env, "hash": digest, "permissions": perms,
             "projects": projects or ["*"], "expires": args.expires, "revoked": False}
    print("TOKEN (se muestra UNA sola vez; guárdalo en tu gestor de contraseñas):", file=out)
    print(raw, file=out)
    print("\nENTRADA para NEXUS_API_TOKENS (solo contiene el hash):", file=out)
    print(json.dumps(entry, ensure_ascii=False), file=out)
    return 0


def cmd_hash_password(args, out, inp=getpass.getpass) -> int:
    first = inp("Contraseña: ")
    if len(first) < 12:
        print("La contraseña debe tener al menos 12 caracteres.", file=out)
        return 2
    if inp("Repite la contraseña: ") != first:
        print("Las contraseñas no coinciden.", file=out)
        return 2
    print(hash_password(first), file=out)
    return 0


def cmd_check_env(args, out, environ=None) -> int:
    s = SecuritySettings.from_env(os.environ if environ is None else environ)
    print("Resumen: " + s.startup_summary(), file=out)
    for w in s.warnings:
        print("AVISO: " + w, file=out)
    for t in s.tokens:
        nums = R.usable_routes(t.permissions, t.projects == ("*",))
        alcance = "global" if t.projects == ("*",) else "proyectos: " + ", ".join(t.projects)
        print(f"token '{t.id}' ({alcance}; env={t.env}): rutas utilizables {nums or 'ninguna'}", file=out)
    problems = []
    if s.auth_mode == "enforce" and not s.has_auth_method:
        problems.append("enforce sin ningún método válido de credencial: el servicio respondería 503")
    if s.public_origin is None:
        problems.append("NEXUS_PUBLIC_ORIGIN no es válida: el login por navegador estará desactivado")
    print("NEXUS_PUBLIC_ORIGIN válida: " + ("sí" if s.public_origin else "no"), file=out)
    if problems:
        for p in problems:
            print("PROBLEMA: " + p, file=out)
        return 1
    print("OK", file=out)
    return 0


def cmd_scan_stored(args, out) -> int:
    path = args.db
    if not os.path.isfile(path):
        print("No existe ese archivo.", file=out)
        return 2
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    found = 0
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")') if str(r[2]).upper() in ("TEXT", "")]
            if not cols:
                continue
            sel = ", ".join(f'"{c}"' for c in cols)
            for row in con.execute(f'SELECT rowid, {sel} FROM "{t}"'):
                fams = sorted({f for v in row[1:] if isinstance(v, str) for f, _, _ in scan(v)})
                if fams:
                    found += 1
                    print(f"tabla={t} rowid={row[0]} familias={','.join(fams)}", file=out)   # sin contenido
    finally:
        con.close()
    print(f"Filas con posibles secretos: {found}", file=out)
    return 0


def main(argv=None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="backend.security.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new-token")
    p.add_argument("--id", required=True)
    p.add_argument("--env", default="production")
    p.add_argument("--permissions", default="read")
    p.add_argument("--projects", default="*", help="'*' o nombres separados por comas")
    p.add_argument("--expires", default=None, help="ej. 2027-06-30T00:00:00Z")
    sub.add_parser("hash-password")
    sub.add_parser("check-env")
    p = sub.add_parser("scan-stored")
    p.add_argument("db", help="ruta de una COPIA local de la base SQLite")
    args = ap.parse_args(argv)
    return {"new-token": cmd_new_token, "hash-password": cmd_hash_password,
            "check-env": cmd_check_env, "scan-stored": cmd_scan_stored}[args.cmd](args, out)


if __name__ == "__main__":
    sys.exit(main())
