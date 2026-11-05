"""Servidor de juguete para las pruebas de sockets reales de la Fase 2.

Se arranca con:  python -m uvicorn tests.framing_server:app   (con middleware real de NEXUS)
                 python -m uvicorn tests.framing_server:bare  (SIN middleware: caracterización)
La aplicación de juguete solo registra qué manejadores se ejecutaron (GET /__calls).
"""
import json
import os

from backend.security.access import SecurityMiddleware
from backend.security.settings import SecuritySettings

CALLS = []


async def toy(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    body = b""
    while True:
        msg = await receive()
        if msg["type"] != "http.request":
            break
        body += msg.get("body", b"")
        if not msg.get("more_body"):
            break
    if scope["path"] == "/__calls":
        raw = json.dumps(CALLS).encode()
    elif scope["path"] == "/api/nexus/status":          # como main.py: boot_id del proceso
        raw = json.dumps({"status": "online", "boot_id": app.state.boot_id}).encode()
    else:
        CALLS.append({"method": scope["method"], "path": scope["path"], "body_len": len(body)})
        raw = b'{"ok":true}'
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
    await send({"type": "http.response.body", "body": raw})


bare = toy
app = SecurityMiddleware(toy, SecuritySettings.from_env(os.environ))
