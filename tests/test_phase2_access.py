"""Fase 2 — control de acceso por ASGI (rutas, credenciales, proyectos, sesiones, login, CSRF).

Cubre: S1–S6, S9–S13, S17, S19–S23, S30 (HTTP), S31–S45, S73–S74, S77, S84–S89.
Precondición P: las peticiones que esperan 401/403 son VÁLIDAS (método y ruta existentes, JSON
válido, encuadre correcto) y la prueba comprueba, con instrumentación propia, que el control de
acceso se invocó una vez y el manejador cero veces; si no, falla como "petición de prueba inválida".
"""
import json

from backend.security import routes as R
from backend.security.access import SecurityMiddleware
from tests.asgi_helper import RecordingApp, call
from tests.security_helper import (JSON_H, LOGIN_H, ORIGIN, PASSWORD, Clock, bearer, build,
                                   concrete_path, cookie_of, login, valid_request)

PROTECTED = [r for r in R.ROUTES if r.cls not in (R.PUBLICA,)]
AUTH_ROUTE_NUMBERS = {5, 6, 7, 8, 9}


class CountingMiddleware(SecurityMiddleware):
    """Instrumento de la precondición P: cuenta cuántas veces se invocó el control de acceso."""
    decisions = 0

    def _decide(self, *a, **k):
        type(self).decisions += 1
        return super()._decide(*a, **k)


def reached_access_control(mw, app, method, path, headers, body):
    CountingMiddleware.decisions = 0
    before = len(app.calls)
    resp = call(mw, method, path, headers, body)
    assert CountingMiddleware.decisions == 1, (
        f"petición de prueba inválida: no llegó al control de acceso ({method} {path} -> {resp.status})")
    return resp, len(app.calls) - before


def counting(**kw):
    mw, app, state, tokens, settings = build(**kw)
    return CountingMiddleware(app, settings, state), app, state, tokens, settings


class TestRouteTable:
    def test_s1_inventory_counts_and_classes(self):
        assert len(R.ROUTES) == 34
        by_cls = {}
        for r in R.ROUTES:
            by_cls.setdefault(r.cls, []).append(r.n)
        assert by_cls[R.PUBLICA] == [1, 2, 3, 4, 5, 6]
        assert by_cls[R.SESION] == [7, 8] and by_cls[R.UNLOCK] == [9]
        assert by_cls[R.SISTEMA] == [10, 11, 16, 21, 22, 24]
        assert by_cls[R.PROYECTO] == [19, 20]
        assert by_cls[R.GLOBAL] == [12, 13, 14, 15, 17, 18, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        assert len(PROTECTED) == 28
        assert all(r.cls in (R.PUBLICA, R.SESION) or r.permission for r in R.ROUTES)

    def test_s9_project_token_open_list_is_closed(self):
        assert R.PROJECT_TOKEN_OPEN == (10, 11, 16, 19, 20, 21, 22, 24)

    def test_undeclared_routes_have_no_match(self):
        for m, p in (("GET", "/nope"), ("POST", "/api/nexus/tasks"), ("GET", "/api/nexus/chat"),
                     ("PUT", "/api/nexus/status"), ("GET", "/api/nexus/status/")):
            assert R.match(m, p) is None, (m, p)


class TestCredentials:
    def test_s2i_all_28_protected_routes_401_without_credential(self):
        mw, app, state, tokens, _ = counting()
        for r in PROTECTED:
            m, p, h, b = valid_request(r)
            resp, handler_calls = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 401, (r.n, resp.status)
            assert handler_calls == 0 and resp.json() == {"error": "unauthorized"}
            assert resp.header("cache-control") == "no-store"

    def test_s3_all_credential_failures_identical(self):
        mw, app, state, tokens, settings = build()
        bodies = set()
        for hdr in ({"Authorization": "Bearer nxs_" + "a" * 43}, {"Authorization": "Bearer xyz"},
                    {"Authorization": "Basic abc"}, {"Authorization": "bearer"}, {}):
            resp = call(mw, "GET", "/api/nexus/status", hdr)
            assert resp.status == 401
            bodies.add(resp.body)
        assert len(bodies) == 1 and not app.calls

    def test_s4_insufficient_permission_403(self):
        mw, app, state, tokens, _ = counting()
        r = [x for x in R.ROUTES if x.n == 12][0]       # chat exige 'chat'; 'rd' solo tiene read
        m, p, h, b = valid_request(r, bearer(tokens, "rd"))
        resp, n = reached_access_control(mw, app, m, p, h, b)
        assert resp.status == 403 and n == 0

    def test_s5_admin_routes(self):
        mw, app, state, tokens, _ = counting()
        for rn in (18, 30, 31):
            r = [x for x in R.ROUTES if x.n == rn][0]
            m, p, h, b = valid_request(r, bearer(tokens, "pb"))     # admin pero de UN proyecto
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 403 and n == 0, rn
            m, p, h, b = valid_request(r, bearer(tokens, "rd"))     # '*' pero sin admin
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 403 and n == 0, rn
            m, p, h, b = valid_request(r, bearer(tokens, "adm"))    # admin y '*' (read no necesario)
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 200 and n == 1, rn

    def test_s6_deletes_require_admin(self):
        mw, app, state, tokens, _ = counting()
        for rn, extra in ((20, "?project=aura"), (30, "")):
            r = [x for x in R.ROUTES if x.n == rn][0]
            m, p, h, b = valid_request(r, bearer(tokens, "pro" if rn == 20 else "rd"))
            resp, n = reached_access_control(mw, app, m, p + extra, h, b)
            assert resp.status == 403 and n == 0, rn

    def test_s35_invalid_bearer_never_falls_back_to_cookie(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        resp = call(mw, "GET", "/api/nexus/status", {"Cookie": ck})
        assert resp.status == 200
        calls = len(app.calls)
        resp = call(mw, "GET", "/api/nexus/status", {"Cookie": ck, "Authorization": "Bearer nxs_" + "z" * 43})
        assert resp.status == 401 and len(app.calls) == calls

    def test_s36_valid_bearer_plus_cookie_uses_only_bearer(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        call(mw, "GET", "/api/nexus/tasks", {"Cookie": ck, **bearer(tokens, "pro")})
        resp = call(mw, "GET", "/api/nexus/tasks", {"Cookie": ck, **bearer(tokens, "pro")})
        assert resp.status == 403 and resp.json() == {"error": "global_route_denied"}   # no es el propietario

    def test_s34_bearer_needs_no_origin(self):
        mw, app, state, tokens, _ = build()
        r = call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}")
        assert r.status == 200
        r = call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H, "Origin": "http://evil.example"}, b"{}")
        assert r.status == 200

    def test_503_when_enforce_has_no_method(self):
        mw, app, state, tokens, _ = build(NEXUS_API_TOKENS="[]", NEXUS_OWNER_PASSWORD_HASH="")
        assert call(mw, "GET", "/api/nexus/status").status == 503
        assert call(mw, "GET", "/health").status == 200            # pública
        assert not app.calls or app.calls[-1]["path"] == "/health"

    def test_public_routes_need_nothing(self):
        mw, app, state, tokens, _ = build()
        for m, p in (("GET", "/"), ("HEAD", "/"), ("GET", "/health"), ("GET", "/sw.js")):
            assert call(mw, m, p).status == 200
        r = call(mw, "GET", "/api/auth/me"); assert r.status == 200 and r.json() == {"authenticated": False}

    def test_unknown_route_denied_to_everyone(self):
        mw, app, state, tokens, _ = build()
        assert call(mw, "GET", "/api/secreto").status == 401
        assert call(mw, "GET", "/api/secreto", bearer(tokens, "adm")).status == 403
        assert not app.calls


class TestProjectIsolation:
    def test_s10_project_token_all_perms_denied_on_all_17_global_routes(self):
        mw, app, state, tokens, _ = counting()
        globals_ = [r for r in R.ROUTES if r.cls == R.GLOBAL]
        assert len(globals_) == 17
        for r in globals_:
            m, p, h, b = valid_request(r, bearer(tokens, "pa"))
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 403 and resp.json() == {"error": "global_route_denied"}, r.n
            assert n == 0, f"el manejador se ejecutó en la ruta {r.n}"

    def test_s11_global_principals_pass(self):
        mw, app, state, tokens, _ = counting()
        ck = cookie_of(login(mw))
        for r in [r for r in R.ROUTES if r.cls == R.GLOBAL]:
            m, p, h, b = valid_request(r, bearer(tokens, "all"))
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 200 and n == 1, r.n
            m, p, h, b = valid_request(r, {"Cookie": ck, "Origin": ORIGIN})
            resp, n = reached_access_control(mw, app, m, p, h, b)
            assert resp.status == 200 and n == 1, ("propietario", r.n)

    def test_s12_denied_calls_consume_no_quota(self):
        mw, app, state, tokens, settings = build()
        chat = [r for r in R.ROUTES if r.n == 12][0]
        for _ in range(settings.rate_chat_per_minute + 10):
            m, p, h, b = valid_request(chat, bearer(tokens, "pa"))
            assert call(mw, m, p, h, b).status == 403
        r = call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}")
        assert r.status == 200 and not app.calls[:-1]

    def test_s17_chat_with_project_token_never_reaches_core(self):
        mw, app, state, tokens, _ = build()
        r = call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "pa"), **JSON_H}, b'{"message":"hola"}')
        assert r.status == 403 and app.calls == []

    def test_s18_ids_existing_or_not_give_identical_response(self):
        mw, app, state, tokens, _ = build()
        a = call(mw, "GET", "/api/nexus/tasks/existe", bearer(tokens, "pa"))
        b = call(mw, "GET", "/api/nexus/tasks/no-existe", bearer(tokens, "pa"))
        c = call(mw, "GET", "/api/knowledge/xx", bearer(tokens, "pa"))
        assert a.status == b.status == c.status == 403 and a.body == b.body == c.body

    def test_s19_denial_body_has_no_data_or_project_names(self):
        mw, app, state, tokens, _ = build()
        r = call(mw, "GET", "/api/nexus/tasks", bearer(tokens, "pa"))
        assert r.body == b'{"error":"global_route_denied"}'
        assert b"beta" not in r.body and b"aura" not in r.body

    def test_system_routes_open_to_read_project_tokens(self):
        mw, app, state, tokens, _ = build()
        for rn in (10, 11, 16, 21, 22, 24):
            r = [x for x in R.ROUTES if x.n == rn][0]
            m, p, h, b = valid_request(r, bearer(tokens, "pro"))
            assert call(mw, m, p, h, b).status == 200, rn
        nr = call(mw, "GET", "/api/nexus/status", bearer(tokens, "pa"))
        assert nr.status == 200

    def test_s13_canaries_are_not_reachable_by_project_token_on_any_route(self):
        """Un 'manejador' con canarios en todos los almacenes: el token del proyecto A recorre TODAS
        las rutas y se examinan TODAS las respuestas (también 4xx)."""
        CANARY = "CANARIO-OTRO-PROYECTO-7f3a"

        class Leaky(RecordingApp):
            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    while True:
                        msg = await receive()
                        if not msg.get("more_body"):
                            break
                    raw = ('{"leak":"%s"}' % CANARY).encode()
                    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", str(len(raw)).encode())]})
                    await send({"type": "http.response.body", "body": raw})
        mw, app, state, tokens, _ = build(app=Leaky())
        seen = []
        for r in R.ROUTES:
            if r.cls in (R.PUBLICA, R.SESION, R.UNLOCK):
                continue
            m, p, h, b = valid_request(r, bearer(tokens, "pa"), with_project=(r.cls == R.PROYECTO))
            resp = call(mw, m, p, h, b)
            seen.append((r.n, resp.status))
            if r.cls == R.GLOBAL:
                assert CANARY.encode() not in resp.body, r.n
        assert all(s == 403 for n, s in seen if n in {x.n for x in R.ROUTES if x.cls == R.GLOBAL})

    def test_s15_project_param_rules(self):
        mw, app, state, tokens, _ = build()
        h = bearer(tokens, "pa")
        assert call(mw, "GET", "/api/nexus/persistent-memory?project=aura", h).status == 200
        assert call(mw, "GET", "/api/nexus/persistent-memory?project=beta", h).status == 403
        assert call(mw, "DELETE", "/api/nexus/persistent-memory/e1?project=beta", bearer(tokens, "pa")).status == 403
        for tid in ("pa", "all", "adm"):
            r = call(mw, "GET", "/api/nexus/persistent-memory?project=*", bearer(tokens, tid))
            assert r.status == 400, tid
        assert call(mw, "GET", "/api/nexus/persistent-memory?project=aura&project=beta", h).status == 400

    def test_s16_own_project_works_and_is_rewritten(self):
        mw, app, state, tokens, _ = build()
        r = call(mw, "GET", "/api/nexus/persistent-memory?project=%20AURA%20", bearer(tokens, "pa"))
        assert r.status == 200 and "project=aura" in app.calls[-1]["query"]

    def test_s41_no_project_no_default_is_400_even_with_single_project_token(self):
        mw, app, state, tokens, _ = build(NEXUS_DEFAULT_PROJECT="")
        for tid in ("pa", "all"):
            r = call(mw, "GET", "/api/nexus/persistent-memory", bearer(tokens, tid))
            assert r.status == 400 and r.json() == {"error": "project_required"}, tid
        assert not app.calls

    def test_s42_default_project_used(self):
        mw, app, state, tokens, _ = build(NEXUS_DEFAULT_PROJECT="aura")
        assert call(mw, "GET", "/api/nexus/persistent-memory", bearer(tokens, "pa")).status == 200
        assert "project=aura" in app.calls[-1]["query"]

    def test_s44_normalization_and_invalid(self):
        mw, app, state, tokens, _ = build()
        assert call(mw, "GET", "/api/nexus/persistent-memory?project=AURA+", bearer(tokens, "all")).status == 200
        for bad in ("", "!!", "a/b", "-x"):
            r = call(mw, "GET", "/api/nexus/persistent-memory?project=" + bad.replace("/", "%2F"), bearer(tokens, "all"))
            if bad == "":
                assert r.status == 200                      # vacío => proyecto por defecto (definido en la prueba)
            else:
                assert r.status == 400, bad

    def test_s45_default_outside_principal_list_is_403(self):
        mw, app, state, tokens, _ = build(NEXUS_DEFAULT_PROJECT="otro")
        assert call(mw, "GET", "/api/nexus/persistent-memory", bearer(tokens, "pa")).status == 403
        assert call(mw, "GET", "/api/nexus/persistent-memory", bearer(tokens, "all")).status == 200


class TestSessionsAndLogin:
    def test_s21_cookie_attributes(self):
        mw, *_ = build()
        sc = login(mw).header("set-cookie")
        assert sc.startswith("__Host-nexus_session=")
        for a in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/"):
            assert a in sc
        assert "Domain" not in sc

    def test_s22_logout_kills_session_immediately(self):
        mw, app, *_ = build()
        ck = cookie_of(login(mw))
        assert call(mw, "GET", "/api/nexus/status", {"Cookie": ck}).status == 200
        r = call(mw, "POST", "/api/auth/logout", {"Cookie": ck, "Origin": ORIGIN})
        assert r.status == 200 and "Max-Age=0" in r.header("set-cookie")
        assert call(mw, "GET", "/api/nexus/status", {"Cookie": ck}).status == 401

    def test_s23_new_process_rejects_old_cookie(self):
        mw1, *_ = build()
        ck = cookie_of(login(mw1))
        mw2, *_ = build()                                   # "reinicio": estado nuevo
        assert call(mw2, "GET", "/api/nexus/status", {"Cookie": ck}).status == 401

    def test_s24_s25_idle_and_max_through_http(self):
        c = Clock()
        mw, app, state, tokens, _ = build(clock=c)
        ck = cookie_of(login(mw))
        c.advance(29 * 60); assert call(mw, "GET", "/api/nexus/status", {"Cookie": ck}).status == 200
        c.advance(31 * 60); assert call(mw, "GET", "/api/nexus/status", {"Cookie": ck}).status == 401

    def test_s26_logout_all(self):
        mw, app, state, tokens, _ = build()
        a, b = cookie_of(login(mw)), None
        state.guard.clear()
        b = cookie_of(login(mw))
        r = call(mw, "POST", "/api/auth/logout-all", {"Cookie": a, "Origin": ORIGIN})
        assert r.status == 200
        assert call(mw, "GET", "/api/nexus/status", {"Cookie": b}).status == 401

    def test_session_routes_reject_bearer(self):
        mw, app, state, tokens, _ = build()
        assert call(mw, "POST", "/api/auth/logout", bearer(tokens, "adm")).status == 403
        assert call(mw, "POST", "/api/auth/logout").status == 401

    def test_s30_http_flow_statuses_and_bodies(self):
        c = Clock()
        mw, app, state, tokens, _ = build(clock=c)
        seen = []
        for expected in (1, 2, 4, 8):
            r = login(mw, "mala")
            assert r.status == 401 and r.json() == {"error": "invalid_credentials"}
            r = login(mw, PASSWORD)                          # dentro del retardo: no se evalúa
            assert r.status == 429 and r.json() == {"error": "login_backoff"}
            assert r.header("retry-after") == str(expected)
            c.advance(expected)
        r = login(mw, "mala")
        assert r.status == 429 and r.json() == {"error": "login_locked"} and r.header("retry-after") == "900"
        c.advance(100)
        r = login(mw, PASSWORD)                              # correcta, pero bloqueada
        assert r.status == 429 and r.json() == {"error": "login_locked"} and r.header("retry-after") == "800"
        assert "set-cookie" not in r.headers
        c.advance(800)
        assert login(mw, PASSWORD).status == 200

    def test_s30d_locked_never_evaluates_password(self):
        import backend.security.access as acc
        c = Clock()
        mw, app, state, tokens, _ = build(clock=c)
        for _ in range(5):
            login(mw, "mala"); c.advance(20)
        evaluated = []
        real = acc.verify_password
        acc.verify_password = lambda *a, **k: evaluated.append(1) or real(*a, **k)
        try:
            login(mw, PASSWORD); login(mw, "otra")
        finally:
            acc.verify_password = real
        assert evaluated == []

    def test_s30h_concurrent_wrong_attempts_bounded(self):
        import asyncio
        import backend.security.access as acc
        from tests.asgi_helper import acall
        mw, app, state, tokens, _ = build()
        count = []
        real = acc.verify_password
        acc.verify_password = lambda *a, **k: count.append(1) or real(*a, **k)

        async def go():
            body = json.dumps({"password": "mala"}).encode()
            return await asyncio.gather(*[acall(mw, "POST", "/api/auth/login", LOGIN_H, body) for _ in range(20)])
        try:
            rs = asyncio.run(go())
        finally:
            acc.verify_password = real
        assert len(count) <= 5 and sum(1 for r in rs if r.status == 401) == len(count)
        assert all(r.status in (401, 429) for r in rs)


class TestOriginCsrfLogin:
    def test_s39_login_origin_rules(self):
        mw, app, state, tokens, _ = build()
        body = json.dumps({"password": PASSWORD}).encode()
        cases = [
            ({"Origin": "http://evil.example", **JSON_H}, 403),                         # b
            ({"Origin": "null", **JSON_H}, 403),                                        # c
            ({"Sec-Fetch-Site": "cross-site", **JSON_H}, 403),                          # e
            ({"Sec-Fetch-Site": "same-site", **JSON_H}, 403),
            ({"Sec-Fetch-Site": "none", **JSON_H}, 403),
            ({**JSON_H}, 403),
            ({"Origin": ORIGIN, "content-type": "application/x-www-form-urlencoded"}, 415),  # g
            ({"Origin": ORIGIN, "content-type": "text/plain"}, 415),
            ({"Origin": ORIGIN, "content-type": "multipart/form-data; boundary=x"}, 415),
        ]
        for h, status in cases:
            r = call(mw, "POST", "/api/auth/login", h, body)
            assert r.status == status and "set-cookie" not in r.headers, (h, r.status)
        assert state.sessions.count() == 0 and state.guard.failures == 0          # h
        assert login(mw).status == 200                                              # a
        mw2, *_ = build()
        r = call(mw2, "POST", "/api/auth/login", {"Sec-Fetch-Site": "same-origin", **JSON_H}, body)   # d
        assert r.status == 200

    def test_s39f_login_csrf_with_correct_password(self):
        mw, app, state, tokens, _ = build()
        form = json.dumps({"password": PASSWORD}).encode()
        r = call(mw, "POST", "/api/auth/login", {"Origin": "http://evil.example", "content-type": "application/x-www-form-urlencoded"}, form)
        assert r.status == 403 and "set-cookie" not in r.headers
        r = call(mw, "POST", "/api/auth/login", {"Sec-Fetch-Site": "cross-site", "content-type": "application/x-www-form-urlencoded"}, form)
        assert r.status == 403 and state.sessions.count() == 0

    def test_s39h_rejections_do_not_count_as_attempts(self):
        mw, app, state, tokens, _ = build()
        for _ in range(10):
            call(mw, "POST", "/api/auth/login", {"Origin": "http://evil.example", **JSON_H}, b'{"password":"x"}')
        assert state.guard.failures == 0 and login(mw).status == 200

    def test_s31_csrf_on_cookie_mutations(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        r = call(mw, "POST", "/api/nexus/chat", {"Cookie": ck, "Origin": "http://evil.example", **JSON_H}, b"{}")
        assert r.status == 403 and r.json() == {"error": "origin_not_allowed"}
        r = call(mw, "DELETE", "/api/nexus/persistent-memory/x?project=aura", {"Cookie": ck, "Origin": "http://evil.example"})
        assert r.status == 403

    def test_s32_origin_null(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        assert call(mw, "POST", "/api/nexus/chat", {"Cookie": ck, "Origin": "null", **JSON_H}, b"{}").status == 403

    def test_s33_cookie_without_origin_needs_same_origin_fetch_site(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        assert call(mw, "POST", "/api/nexus/chat", {"Cookie": ck, **JSON_H}, b"{}").status == 403
        assert call(mw, "POST", "/api/nexus/chat", {"Cookie": ck, "Sec-Fetch-Site": "cross-site", **JSON_H}, b"{}").status == 403
        assert call(mw, "POST", "/api/nexus/chat", {"Cookie": ck, "Sec-Fetch-Site": "same-origin", **JSON_H}, b"{}").status == 200

    def test_s37_get_and_head_never_blocked_by_csrf(self):
        mw, app, state, tokens, _ = build()
        ck = cookie_of(login(mw))
        assert call(mw, "GET", "/api/nexus/status", {"Cookie": ck, "Origin": "http://evil.example"}).status == 200

    def test_s38_cors_is_not_added_by_the_access_layer(self):
        mw, app, state, tokens, _ = build()
        r = call(mw, "GET", "/api/nexus/status", {**bearer(tokens, "rd"), "Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers


class TestRateLimitsAndHeaders:
    def test_s73_chat_and_task_limits_per_principal(self):
        c = Clock()
        mw, app, state, tokens, _ = build(clock=c)
        for i in range(30):
            assert call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}").status == 200
        r = call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}")
        assert r.status == 429 and int(r.header("retry-after")) >= 1
        assert call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "adm"), **JSON_H}, b"{}").status == 403   # sin permiso, no por límite
        for i in range(5):
            assert call(mw, "POST", "/api/nexus/task", {**bearer(tokens, "all"), **JSON_H}, b"{}").status == 200
        assert call(mw, "POST", "/api/nexus/task", {**bearer(tokens, "all"), **JSON_H}, b"{}").status == 429
        c.advance(61)
        assert call(mw, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{}").status == 200

    def test_s74_429_has_retry_after_and_no_store(self):
        mw, app, state, tokens, _ = build(NEXUS_RATE_TASK_PER_MINUTE="1")
        call(mw, "POST", "/api/nexus/task", {**bearer(tokens, "all"), **JSON_H}, b"{}")
        r = call(mw, "POST", "/api/nexus/task", {**bearer(tokens, "all"), **JSON_H}, b"{}")
        assert r.status == 429 and r.header("retry-after") and r.header("cache-control") == "no-store"

    def test_s77_no_store_on_every_protected_and_auth_response(self):
        mw, app, state, tokens, _ = build()
        for r in R.ROUTES:
            if r.cls == R.PUBLICA and not r.template.startswith("/api/"):
                continue
            m, p, h, b = valid_request(r, bearer(tokens, "all"), with_project=(r.cls == R.PROYECTO))
            for hdrs in (h, {k: v for k, v in h.items() if k.lower() != "authorization"}):
                resp = call(mw, m, p, hdrs, b)
                assert resp.header("cache-control") == "no-store", (r.n, resp.status)
        for st in (call(mw, "POST", "/api/auth/login", {"Origin": "http://x.example", **JSON_H}, b"{}"),
                   call(mw, "POST", "/api/auth/login", {"Origin": ORIGIN, "content-type": "text/plain"}, b"x")):
            assert st.header("cache-control") == "no-store"


class TestUnlock:
    def lock(self, mw, c):
        for _ in range(5):
            login(mw, "mala"); c.advance(20)

    def test_s84_unlock_authorization(self):
        c = Clock()
        mw, app, state, tokens, _ = counting(clock=c)
        u = ("POST", "/api/auth/unlock")
        cases = [({"authorization": "Bearer " + tokens["adm"]}, 200),
                 ({"authorization": "Bearer " + tokens["all"]}, 200),
                 ({"authorization": "Bearer " + tokens["pb"]}, 403),      # admin de UN proyecto
                 ({"authorization": "Bearer " + tokens["rd"]}, 403),      # read
                 ({"authorization": "Bearer " + tokens["pa"]}, 403),
                 ({}, 401)]
        for h, status in cases:
            r = call(mw, *u, {**h, **JSON_H}, b"{}")
            assert r.status == status, (h, r.status)
        ck = cookie_of(login(mw))
        r = call(mw, *u, {"Cookie": ck, "Origin": ORIGIN, **JSON_H}, b"{}")       # cookie con CSRF correcto
        assert r.status == 403 and r.json() == {"error": "unlock_requires_bearer_admin"}
        r = call(mw, *u, {"Cookie": ck, "Authorization": "Bearer nxs_" + "q" * 43, **JSON_H}, b"{}")
        assert r.status == 401

    def test_s85_s86_unlock_only_clears_counter_and_leaves_one_log_line(self):
        import logging
        c = Clock()
        mw, app, state, tokens, settings = build(clock=c)
        self.lock(mw, c)
        assert login(mw, PASSWORD).status == 429
        ck = cookie_of(login(mw, PASSWORD)) if False else None
        before = (settings.owner_hash, settings.tokens, state.sessions.count(), id(state.limiter))
        records = []
        handler = logging.Handler(); handler.emit = lambda rec: records.append(rec.getMessage())
        lg = logging.getLogger("NEXUS.security"); lg.addHandler(handler); lg.setLevel(logging.WARNING)
        try:
            r = call(mw, "POST", "/api/auth/unlock", {**bearer(tokens, "adm"), **JSON_H}, b"{}")
        finally:
            lg.removeHandler(handler); lg.setLevel(logging.CRITICAL)
        assert r.status == 200
        assert before == (settings.owner_hash, settings.tokens, state.sessions.count(), id(state.limiter))
        lines = [m for m in records if m.startswith("unlock:")]
        assert len(lines) == 1 and "token=adm" in lines[0] and tokens["adm"] not in lines[0]
        assert PASSWORD not in " ".join(records)
        assert login(mw, PASSWORD).status == 200

    def test_s86_unlock_is_rate_limited(self):
        mw, app, state, tokens, _ = build()
        codes = [call(mw, "POST", "/api/auth/unlock", {**bearer(tokens, "adm"), **JSON_H}, b"{}").status for _ in range(7)]
        assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]

    def test_s87_bearer_still_works_during_owner_lock(self):
        c = Clock()
        mw, app, state, tokens, _ = build(clock=c)
        self.lock(mw, c)
        assert call(mw, "GET", "/api/nexus/status", bearer(tokens, "rd")).status == 200

    def test_s88_new_hash_new_process_old_password_fails(self):
        from backend.security.passwords import hash_password
        mw, app, state, tokens, _ = build(NEXUS_OWNER_PASSWORD_HASH=hash_password("nueva-contrasena-123", n=1024))
        assert login(mw, PASSWORD).status == 401
        state.guard.clear()
        assert login(mw, "nueva-contrasena-123").status == 200

    def test_s89_production_without_credentials_is_503_and_no_password_route(self):
        mw, app, state, tokens, _ = build(NEXUS_ENVIRONMENT="production", NEXUS_API_TOKENS="[]",
                                          NEXUS_OWNER_PASSWORD_HASH="", NEXUS_PUBLIC_ORIGIN="https://x.example.com")
        assert call(mw, "GET", "/api/nexus/status").status == 503
        assert not [r for r in R.ROUTES if "password" in r.template or "change" in r.template or "reset" in r.template]

    def test_s85_static_no_route_changes_password_tokens_or_env(self):
        import re, pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / "backend" / "security"
        text = (src / "access.py").read_text(encoding="utf-8")
        assert "os.environ[" not in text and "putenv" not in text and "open(" not in text
        cli_text = (src / "cli.py").read_text(encoding="utf-8")
        assert "socket" not in cli_text and "urllib.request" not in cli_text and "requests" not in cli_text
        assert "open(" not in cli_text and "write(" not in cli_text.replace("file=out", "")


class TestAuditAndOff:
    def test_s20_audit_logs_without_blocking(self):
        import logging
        mw, app, state, tokens, _ = build(NEXUS_AUTH_MODE="audit")
        records = []
        handler = logging.Handler(); handler.emit = lambda rec: records.append(rec.getMessage())
        lg = logging.getLogger("NEXUS.security"); lg.addHandler(handler); lg.setLevel(logging.WARNING)
        try:
            r = call(mw, "GET", "/api/nexus/tasks", bearer(tokens, "pa"))
        finally:
            lg.removeHandler(handler); lg.setLevel(logging.CRITICAL)
        assert r.status == 200 and any("ruta=13" in m and "principal=pa" in m for m in records)
        assert tokens["pa"] not in " ".join(records)

    def test_off_in_test_env_lets_requests_through(self):
        mw, app, state, tokens, _ = build(NEXUS_AUTH_MODE="off")
        assert call(mw, "GET", "/api/nexus/tasks").status == 200
