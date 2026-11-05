"""Fase 2 — dos procesos reales en 127.0.0.1 (S90–S97).

Demuestran cómo se comporta NEXUS con DOS procesos: sesiones, contadores y límites son POR PROCESO.
No dicen cuántos procesos tiene Render. Cada prueba arranca sus propios procesos.
"""
import json
import time

from tests.security_helper import LOGIN_H, PASSWORD, make_env
from tests.test_phase2_framing import Server, header_of, status_lines


def req(srv, method, path, headers=None, body=b""):
    lines = [f"{method} {path} HTTP/1.1", "Host: x", "Connection: close"]
    for k, v in (headers or {}).items():
        lines.append(f"{k}: {v}")
    if body:
        lines.append(f"Content-Length: {len(body)}")
    buf, _ = srv.raw(("\r\n".join(lines) + "\r\n\r\n").encode() + body, wait=3)
    code = int(status_lines(buf)[0][9:12]) if status_lines(buf) else 0
    return code, buf, buf.split(b"\r\n\r\n", 1)[-1]


def login(srv, password=PASSWORD):
    code, buf, _ = req(srv, "POST", "/api/auth/login", LOGIN_H, json.dumps({"password": password}).encode())
    ck = (header_of(buf, "set-cookie") or "").split(";")[0]
    return code, ck


def environment(extra=None):
    env, tokens = make_env()
    env["NEXUS_AUTH_MODE"] = "enforce"
    env["NEXUS_MAX_BODY_BYTES"] = "1048576"
    env.update(extra or {})
    return env, tokens


def start(env):
    return Server("app", "h11", env)


class TestTwoProcesses:
    def test_s90_s91_s93_s97_sessions_and_boot_id(self):
        env, tokens = environment()
        a, b = start(env), start(env)
        try:
            code, ck = login(a)
            assert code == 200 and ck
            assert req(b, "GET", "/api/nexus/status", {"Cookie": ck})[0] == 401          # S90: B no la conoce
            assert req(a, "GET", "/api/nexus/status", {"Cookie": ck})[0] == 200
            assert req(a, "POST", "/api/auth/logout", {"Cookie": ck, "Origin": LOGIN_H["Origin"]})[0] == 200
            assert req(a, "GET", "/api/nexus/status", {"Cookie": ck})[0] == 401          # S91
            auth = {"Authorization": "Bearer " + tokens["rd"]}                           # S93
            ca, _, body_a = req(a, "GET", "/api/nexus/status", auth)
            cb, _, body_b = req(b, "GET", "/api/nexus/status", auth)
            assert ca == cb == 200
            ida, idb = json.loads(body_a)["boot_id"], json.loads(body_b)["boot_id"]
            assert ida != idb                                                              # S97: distintos
            assert json.loads(req(a, "GET", "/api/nexus/status", auth)[2])["boot_id"] == ida  # mismo proceso
            assert req(a, "GET", "/api/nexus/status")[0] == 401                          # solo con read
            assert "boot_id" not in req(a, "GET", "/health")[2].decode()
        finally:
            a.stop(); b.stop()

    def test_s92_restart_invalidates_old_cookie(self):
        env, tokens = environment()
        a = start(env)
        try:
            code, ck = login(a)
            assert code == 200 and req(a, "GET", "/api/nexus/status", {"Cookie": ck})[0] == 200
        finally:
            a.stop()
        a2 = start(env)                                                                   # "reinicio"
        try:
            assert req(a2, "GET", "/api/nexus/status", {"Cookie": ck})[0] == 401
        finally:
            a2.stop()

    def test_s94_login_counters_are_per_process(self):
        env, tokens = environment({"NEXUS_LOGIN_MAX_ATTEMPTS": "2"})
        a, b = start(env), start(env)
        try:
            assert login(a, "mala")[0] == 401
            time.sleep(1.2)
            assert login(a, "mala")[0] == 429                                             # A: bloqueado
            assert login(a, PASSWORD)[0] == 429
            code_b, ck_b = login(b)                                                       # B no se entera
            assert code_b == 200 and ck_b                                                # límite efectivo N×
        finally:
            a.stop(); b.stop()

    def test_s95_rate_limits_are_per_process(self):
        env, tokens = environment({"NEXUS_RATE_TASK_PER_MINUTE": "1"})
        a, b = start(env), start(env)
        try:
            h = {"Authorization": "Bearer " + tokens["all"], "content-type": "application/json"}
            assert req(a, "POST", "/api/nexus/task", h, b"{}")[0] == 200
            assert req(a, "POST", "/api/nexus/task", h, b"{}")[0] == 429
            assert req(b, "POST", "/api/nexus/task", h, b"{}")[0] == 200                 # límite efectivo N×
        finally:
            a.stop(); b.stop()
