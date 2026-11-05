"""Ayudas para probar aplicaciones ASGI sin servidor, sin httpx y sin FastAPI.

Se usan en las pruebas de la Fase 2 (middleware de acceso). No dependen de pytest.
"""
import asyncio
import json
from urllib.parse import urlsplit


class Resp:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = {}
        for k, v in headers:
            self.headers.setdefault(k.decode("latin-1").lower(), []).append(v.decode("latin-1"))
        self.body = body

    def header(self, name, default=None):
        v = self.headers.get(name.lower())
        return v[0] if v else default

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class RecordingApp:
    """Aplicación de juguete: registra cada invocación del 'manejador' y responde 200."""

    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        self.calls.append({"method": scope["method"], "path": scope["path"],
                           "query": scope.get("query_string", b"").decode("latin-1"), "body": body,
                           "principal": scope.get("nexus.principal")})
        raw = b'{"ok":true}'
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})


async def acall(app, method, url, headers=None, body=b"", chunks=None):
    parts = urlsplit(url)
    items = headers.items() if isinstance(headers, dict) else (headers or [])
    hs = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in items]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": parts.path or "/", "raw_path": (parts.path or "/").encode(),
             "query_string": parts.query.encode("latin-1"), "headers": hs,
             "client": ("127.0.0.1", 1234), "server": ("testserver", 80)}
    queue = list(chunks) if chunks is not None else [body]
    msgs = [{"type": "http.request", "body": c, "more_body": i < len(queue) - 1}
            for i, c in enumerate(queue)]
    out = []

    async def receive():
        if msgs:
            return msgs.pop(0)
        await asyncio.sleep(3600)

    async def send(message):
        out.append(message)

    await app(scope, receive, send)
    start = next(m for m in out if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body")
    return Resp(start["status"], start["headers"], data)


def call(app, method, url, headers=None, body=b"", chunks=None):
    return asyncio.run(acall(app, method, url, headers, body, chunks))
