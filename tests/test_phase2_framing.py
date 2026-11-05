"""Fase 2 — encuadre HTTP, límite y plazo de cuerpo, en dos niveles.

(A) ASGI directo con cabeceras manipuladas: S59–S62, S65(ii)–S69, S71–S72, S63e/f(i), S64 (plazo).
(B) Sockets reales con uvicorn real (parser h11 y, si está instalado, httptools): S63a–S63d, S64a–S64c,
    S70, S60/S62 en red, y la caracterización del servidor S63f(ii)/S65(i)/S66(i)/S67(i).

Los resultados de (B) describen la versión de uvicorn/h11 INSTALADA donde se ejecutan: no son los de
Render ni validan su proxy.
"""
import asyncio
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import unittest

from tests.asgi_helper import RecordingApp, acall, call
from tests.security_helper import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 1048576


def mw_off(**over):
    over.setdefault("NEXUS_AUTH_MODE", "off")
    mw, app, state, tokens, settings = build(**over)
    return mw, app


def body_of(n):
    return b"x" * n


class TestFramingAsgi:
    def test_s59_content_length_over_limit_413_without_reading(self):
        mw, app = mw_off()
        r = call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1), "content-type": "application/json"}, b"")
        assert r.status == 413 and r.header("connection") == "close" and app.calls == []

    def test_s60_chunked_over_limit_413_and_no_side_effects(self):
        mw, app = mw_off()
        chunks = [body_of(400000)] * 3
        r = call(mw, "POST", "/api/nexus/chat", {"transfer-encoding": "chunked"}, chunks=chunks)
        assert r.status == 413 and app.calls == [] and r.header("connection") == "close"

    def test_s61_exact_limit_passes_both_framings(self):
        mw, app = mw_off()
        b = body_of(LIMIT)
        assert call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT)}, b).status == 200
        assert call(mw, "POST", "/api/nexus/chat", {"transfer-encoding": "chunked"}, chunks=[b[:700000], b[700000:]]).status == 200
        assert len(app.calls) == 2 and len(app.calls[0]["body"]) == LIMIT

    def test_s62_one_byte_over_413_both_framings(self):
        mw, app = mw_off()
        b = body_of(LIMIT + 1)
        assert call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1)}, b).status == 413
        assert call(mw, "POST", "/api/nexus/chat", {"transfer-encoding": "chunked"}, chunks=[b[:700000], b[700000:]]).status == 413
        assert app.calls == []

    def test_s65_invalid_content_length_values_400(self):
        mw, app = mw_off()
        for bad in ("abc", "-5", "+10", " 1 0", "1.5", "", "0x10"):
            r = call(mw, "POST", "/api/nexus/chat", {"content-length": bad}, b"")
            assert r.status == 400, repr(bad)
        assert app.calls == []

    def test_s66_s67_duplicate_content_length_arriving_separately_400(self):
        mw, app = mw_off()
        for a, b in (("10", "10"), ("10", "20")):
            scope_headers = [("content-length", a), ("content-length", b)]
            r = call(mw, "POST", "/api/nexus/chat", scope_headers, b"0123456789")
            assert r.status == 400, (a, b)
        assert app.calls == []

    def test_s68_content_length_plus_transfer_encoding_400(self):
        mw, app = mw_off()
        r = call(mw, "POST", "/api/nexus/chat", {"content-length": "5", "transfer-encoding": "chunked"}, b"hello")
        assert r.status == 400 and app.calls == []
        r = call(mw, "POST", "/api/nexus/chat", {"transfer-encoding": "gzip"}, b"hello")
        assert r.status == 400

    def test_s69_get_with_oversized_body_413(self):
        mw, app = mw_off()
        r = call(mw, "GET", "/api/nexus/status", {"content-length": str(LIMIT + 1)}, b"")
        assert r.status == 413 and app.calls == []

    def test_s71_413_is_fixed_message_and_no_store_and_never_echoes_body(self):
        mw, app = mw_off()
        r = call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1)}, b"CANARIO-CUERPO")
        assert r.json() == {"error": "payload_too_large"} and r.header("cache-control") == "no-store"
        assert b"CANARIO" not in r.body

    def test_s72_limit_is_configurable_and_invalid_values_fall_back(self):
        mw, app = mw_off(NEXUS_MAX_BODY_BYTES="1000")
        assert call(mw, "POST", "/api/nexus/chat", {"content-length": "1000"}, body_of(1000)).status == 200
        assert call(mw, "POST", "/api/nexus/chat", {"content-length": "1001"}, body_of(1001)).status == 413
        mw, app = mw_off(NEXUS_MAX_BODY_BYTES="grande")
        assert call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1)}, b"").status == 413

    def test_s63e_get_head_keep_alive(self):
        mw, app = mw_off()
        for m, p in (("GET", "/api/nexus/status"), ("HEAD", "/"), ("GET", "/health")):
            assert call(mw, m, p).header("connection") is None, (m, p)

    def test_s63f_i_every_nexus_response_to_a_body_request_closes(self):
        mw, app, state, tokens, settings = build()
        from tests.security_helper import JSON_H, LOGIN_H, bearer
        cases = [
            call(mw, "POST", "/api/nexus/chat", JSON_H, b"{}"),                                   # 401
            call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "pa"), **JSON_H}, b"{}"),       # 403
            call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}"),      # 200 (app)
            call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1)}, b""),          # 413
            call(mw, "POST", "/api/nexus/chat", {"content-length": "x"}, b""),                     # 400
            call(mw, "POST", "/api/auth/login", {"Origin": "http://evil.example", **JSON_H}, b"{}"),  # 403
            call(mw, "POST", "/api/auth/login", {"Origin": LOGIN_H["Origin"], "content-type": "text/plain"}, b"x"),  # 415
        ]
        for _ in range(settings.rate_task_per_minute + 1):
            cases.append(call(mw, "POST", "/api/nexus/task", {**bearer(tokens, "all"), **JSON_H}, b"{}"))   # 429
        statuses = {c.status for c in cases}
        assert {200, 400, 401, 403, 413, 415, 429} <= statuses
        for c in cases:
            assert c.header("connection") == "close", c.status

    def test_s64_body_timeout_408_and_close_for_each_slow_pattern(self):
        mw, app = mw_off(NEXUS_BODY_TIMEOUT_SECONDS="1")

        async def slow(headers, first, gap, n_more):
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                     "scheme": "http", "path": "/api/nexus/chat", "raw_path": b"/api/nexus/chat",
                     "query_string": b"", "headers": [(k.encode(), v.encode()) for k, v in headers],
                     "client": ("127.0.0.1", 1), "server": ("t", 80)}
            sent_first = False
            out = []

            async def receive():
                nonlocal sent_first, n_more
                if not sent_first:
                    sent_first = True
                    return {"type": "http.request", "body": first, "more_body": True}
                await asyncio.sleep(gap)
                if n_more > 0:
                    n_more -= 1
                    return {"type": "http.request", "body": b"x", "more_body": True}
                await asyncio.sleep(3600)

            async def send(m):
                out.append(m)
            t0 = time.time()
            await mw(scope, receive, send)
            return out, time.time() - t0
        for headers, first, gap, more in (
                ([("content-length", "500")], b"0123456789", 3600, 0),                # a) CL mayor que lo enviado
                ([("transfer-encoding", "chunked")], b"abc", 3600, 0),                # b) chunked que no termina
                ([("transfer-encoding", "chunked")], b"a", 0.3, 100)):               # c) goteo
            out, elapsed = asyncio.run(slow(headers, first, gap, more))
            start = next(m for m in out if m["type"] == "http.response.start")
            hs = {k.decode(): v.decode() for k, v in start["headers"]}
            assert start["status"] == 408 and hs.get("connection") == "close" and elapsed < 2.5, headers
        assert app.calls == []

    def test_negative_malformed_json_without_credential_is_401_not_422(self):
        """Riesgo 18: el control de acceso va ANTES de que se analice el cuerpo JSON."""
        mw, app, state, tokens, _ = build()
        r = call(mw, "POST", "/api/nexus/chat", {"content-type": "application/json"}, b"{esto no es json")
        assert r.status == 401 and app.calls == []

    def test_s2ii_layer_order_is_characterized(self):
        """Sin credencial: encuadre inválido/413/415 ocurren antes del 401 (deliberado)."""
        mw, app, state, tokens, _ = build()
        seen = {
            "framing_400": call(mw, "POST", "/api/nexus/chat", {"content-length": "x"}, b"").status,
            "too_big_413": call(mw, "POST", "/api/nexus/chat", {"content-length": str(LIMIT + 1)}, b"").status,
            "valid_401": call(mw, "POST", "/api/nexus/chat", {"content-type": "application/json"}, b"{}").status,
            "malformed_json": call(mw, "POST", "/api/nexus/chat", {"content-type": "application/json"}, b"{x").status,
        }
        assert seen == {"framing_400": 400, "too_big_413": 413, "valid_401": 401, "malformed_json": 401}


# ============================================================ (B) sockets reales
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, target="app", parser="h11", extra_env=None):
        self.port = _free_port()
        env = dict(os.environ)
        env.update({"NEXUS_ENVIRONMENT": "test", "NEXUS_AUTH_MODE": "off", "NEXUS_MAX_BODY_BYTES": "1000",
                    "NEXUS_BODY_TIMEOUT_SECONDS": "2", "PYTHONPATH": ROOT, "PYTHONDONTWRITEBYTECODE": "1"})
        env.update(extra_env or {})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"tests.framing_server:{target}", "--host", "127.0.0.1",
             "--port", str(self.port), "--http", parser, "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError("el servidor de prueba no arrancó")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def raw(self, data, wait=3.0, then_silent=False):
        """Envía bytes, lee hasta cierre o 'wait' segundos. -> (bytes recibidos, ¿cerró el servidor?)"""
        s = socket.create_connection(("127.0.0.1", self.port), 3)
        s.sendall(data)
        s.settimeout(wait)
        buf, closed = b"", False
        end = time.time() + wait
        while time.time() < end:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            except OSError:
                closed = True
                break
            if not chunk:
                closed = True
                break
            buf += chunk
        s.close()
        return buf, closed

    def calls(self):
        buf, _ = self.raw(b"GET /__calls HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", 2)
        return json.loads(buf.split(b"\r\n\r\n", 1)[1].decode() or "[]")


def parsers():
    out = ["h11"]
    if importlib.util.find_spec("httptools") is not None:
        out.append("httptools")
    else:
        print("OMITIDO (registrado): httptools no está instalado; solo se prueba el parser h11")
    return out


def status_lines(buf):
    """Líneas de estado de TODAS las respuestas del búfer (un cuerpo puede ir pegado a la siguiente)."""
    return re.findall(rb"HTTP/1\.1 \d{3}[^\r\n]*", buf)


def header_of(buf, name):
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    for line in head.split("\r\n")[1:]:
        k, _, v = line.partition(":")
        if k.strip().lower() == name:
            return v.strip()          # el VALOR conserva mayúsculas (p. ej. el id de sesión)
    return None


class TestRealSockets:
    def each_parser(self, fn, target="app"):
        for parser in parsers():
            srv = Server(target, parser)
            try:
                fn(srv, parser)
            finally:
                srv.stop()

    def test_s63a_s63b_s63c_extra_bytes_not_processed_with_close_rule(self):
        def run(srv, parser):
            buf, closed = srv.raw(b"POST /a HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n0123456789"
                                  b"GET /extra HTTP/1.1\r\nHost: x\r\n\r\n")
            assert len(status_lines(buf)) == 1 and header_of(buf, "connection") == "close" and closed, parser
            assert "/extra" not in [c["path"] for c in srv.calls()], parser
            buf, closed = srv.raw(b"POST /b HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\nabcd\x00\x01\x02garbage\r\n")
            assert len(status_lines(buf)) == 1 and closed, parser
            buf, closed = srv.raw(b"POST /c HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\n" + b"y" * 5000)
            assert len(status_lines(buf)) == 1 and closed, parser
            c = [x for x in srv.calls() if x["path"] == "/c"]
            assert c and c[-1]["body_len"] == 5, parser
        self.each_parser(run)

    def test_s63d_characterization_without_close_rule_second_request_is_processed(self):
        def run(srv, parser):
            buf, closed = srv.raw(b"POST /a HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n0123456789"
                                  b"GET /extra HTTP/1.1\r\nHost: x\r\n\r\n", wait=1.5)
            assert len(status_lines(buf)) == 2, f"[{parser}] cambió el comportamiento observado del servidor"
            assert "/extra" in [c["path"] for c in srv.calls()]
        self.each_parser(run, target="bare")

    def test_s63e_chained_gets_get_two_responses(self):
        def run(srv, parser):
            buf, _ = srv.raw(b"GET /one HTTP/1.1\r\nHost: x\r\n\r\nGET /two HTTP/1.1\r\nHost: x\r\n\r\n", wait=1.0)
            assert len(status_lines(buf)) == 2, parser
        self.each_parser(run)

    def test_s60_s62_chunked_and_content_length_limits_over_the_wire(self):
        def run(srv, parser):
            big = b"3E9\r\n" + b"z" * 1001 + b"\r\n0\r\n\r\n"
            buf, closed = srv.raw(b"POST /big HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n" + big)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 413") and closed, parser
            buf, closed = srv.raw(b"POST /big2 HTTP/1.1\r\nHost: x\r\nContent-Length: 1001\r\n\r\n" + b"z" * 1001)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 413") and closed, parser
            buf, closed = srv.raw(b"POST /ok HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n" + b"z" * 1000)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 200"), parser
            paths = [c["path"] for c in srv.calls()]
            assert "/big" not in paths and "/big2" not in paths and "/ok" in paths, parser
        self.each_parser(run)

    def test_s64_slow_bodies_get_408_and_close(self):
        def run(srv, parser):
            t0 = time.time()
            buf, closed = srv.raw(b"POST /s HTTP/1.1\r\nHost: x\r\nContent-Length: 500\r\n\r\n0123456789", wait=6)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 408") and closed and time.time() - t0 < 5, parser
            buf, closed = srv.raw(b"POST /s HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n", wait=6)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 408") and closed, parser
            s = socket.create_connection(("127.0.0.1", srv.port), 3)
            s.sendall(b"POST /drip HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n")
            t0, got = time.time(), b""
            s.settimeout(0.6)
            while time.time() - t0 < 6:
                try:
                    s.sendall(b"1\r\nx\r\n")
                except OSError:
                    pass                           # cerrado por el servidor
                try:
                    d = s.recv(4096)
                    if d:
                        got += d
                        break                      # ya llegó la respuesta: no se sigue enviando
                    else:
                        break
                except socket.timeout:
                    continue
                except OSError:                    # el servidor ya cerró (RST): se usa lo recibido
                    break
            s.close()
            assert got.startswith(b"HTTP/1.1 408") and time.time() - t0 < 5.5, parser
            assert "/s" not in [c["path"] for c in srv.calls()] and "/drip" not in [c["path"] for c in srv.calls()]
        self.each_parser(run)

    def test_s70_oversized_body_closes_and_second_request_is_not_served(self):
        def run(srv, parser):
            canary = b"CANARIO-SECRETO-XYZ"
            req = (b"POST /huge HTTP/1.1\r\nHost: x\r\nContent-Length: 5000\r\n\r\n" + canary + b"z" * 4000 +
                   b"GET /second HTTP/1.1\r\nHost: x\r\n\r\n")
            buf, closed = srv.raw(req)
            assert status_lines(buf)[0].startswith(b"HTTP/1.1 413") and len(status_lines(buf)) == 1 and closed, parser
            assert header_of(buf, "connection") == "close" and header_of(buf, "cache-control") == "no-store"
            assert canary not in buf and "/second" not in [c["path"] for c in srv.calls()], parser
        self.each_parser(run)

    def test_characterization_server_level_framing_errors(self):
        """S63f(ii), S65(i), S66(i), S67(i): lo que hace el SERVIDOR antes de ASGI. Se REGISTRA,
        no se atribuye a NEXUS; un resultado distinto es una limitación del servidor, no un fallo."""
        report = {}

        def run(srv, parser):
            cases = {
                "nonnumeric": b"POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n",
                "plus10": b"POST /p HTTP/1.1\r\nHost: x\r\nContent-Length: +10\r\n\r\n0123456789",
                "dup_identical": b"POST /dupi HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\nContent-Length: 10\r\n\r\n0123456789",
                "dup_conflict": b"POST /dupc HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\nContent-Length: 20\r\n\r\n0123456789",
            }
            for name, data in cases.items():
                buf, closed = srv.raw(data, wait=1.5)
                sl = status_lines(buf)
                report[(parser, name)] = (sl[0][9:12].decode() if sl else "-", header_of(buf, "connection"), closed)
            # lo que SÍ exigimos con cualquier parser: nunca llegó un manejador con encuadre contradictorio
            assert "/dupc" not in [c["path"] for c in srv.calls()], parser
        self.each_parser(run)
        print("CARACTERIZACION", json.dumps({f"{k[0]}:{k[1]}": v for k, v in report.items()}))

    def test_s68_content_length_plus_transfer_encoding_over_the_wire(self):
        def run(srv, parser):
            buf, closed = srv.raw(b"POST /both HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
                                  b"Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n", wait=2)
            sl = status_lines(buf)
            assert sl and sl[0][9:12] in (b"400",), (parser, sl)         # NEXUS (o el parser) lo rechaza
            assert "/both" not in [c["path"] for c in srv.calls()], parser
        self.each_parser(run)
