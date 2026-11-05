"""Fase 2 — el middleware dentro de una pila real de Starlette (la base de FastAPI), con CORS por fuera.

Reproduce el orden de main.py: SecurityMiddleware añadido primero, CORSMiddleware después (más externo).
Se omite con motivo registrado si Starlette no está instalado.
"""
import json
import unittest

from tests.asgi_helper import call
from tests.security_helper import JSON_H, ORIGIN, bearer, build, login

OTHER = "https://panel.ejemplo.test"


def make_stack(**over):
    try:
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise unittest.SkipTest(f"Starlette no está instalado: {exc}")
    mw0, rec, state, tokens, settings = build(NEXUS_ALLOWED_ORIGINS=OTHER, **over)
    seen = []

    async def status(request):
        return JSONResponse({"status": "online"})

    async def chat(request):
        body = await request.body()
        seen.append(("chat", body))
        try:
            json.loads(body)
        except ValueError:
            return JSONResponse({"detail": "json"}, status_code=422)
        return JSONResponse({"ok": True})

    async def pm(request):
        seen.append(("pm", request.query_params.get("project")))
        return JSONResponse({"project": request.query_params.get("project")})

    app = Starlette(routes=[Route("/api/nexus/status", status), Route("/api/nexus/chat", chat, methods=["POST"]),
                            Route("/api/nexus/persistent-memory", pm)])
    from backend.security.access import SecurityMiddleware
    app.add_middleware(SecurityMiddleware, settings=settings, state=state)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])
    return app, tokens, state, seen


class TestRealStarletteStack:
    def test_401_and_403_carry_cors_for_allowed_origin_only(self):
        app, tokens, state, seen = make_stack()
        r = call(app, "GET", "/api/nexus/status", {"Origin": OTHER})
        assert r.status == 401 and r.header("access-control-allow-origin") == OTHER
        assert r.header("access-control-allow-credentials") is None and r.header("access-control-allow-origin") != "*"
        r = call(app, "GET", "/api/nexus/status", {"Origin": "https://evil.example"})
        assert r.status == 401 and r.header("access-control-allow-origin") is None
        r = call(app, "GET", "/api/nexus/status", {**bearer(tokens, "rd"), "Origin": OTHER})
        assert r.status == 200 and r.header("cache-control") == "no-store"

    def test_preflight_is_answered_by_cors_without_credentials(self):
        app, tokens, state, seen = make_stack()
        r = call(app, "OPTIONS", "/api/nexus/chat", {"Origin": OTHER, "Access-Control-Request-Method": "POST",
                                                      "Access-Control-Request-Headers": "authorization,content-type"})
        assert r.status == 200 and r.header("access-control-allow-origin") == OTHER and seen == []

    def test_401_comes_before_422_for_malformed_json(self):
        app, tokens, state, seen = make_stack()
        r = call(app, "POST", "/api/nexus/chat", JSON_H, b"{roto")
        assert r.status == 401 and seen == []
        r = call(app, "POST", "/api/nexus/chat", {**bearer(tokens, "all"), **JSON_H}, b"{roto")
        assert r.status == 422 and r.header("connection") == "close"          # 422 de la app también cierra

    def test_project_is_rewritten_for_the_handler_and_default_applies(self):
        app, tokens, state, seen = make_stack()
        r = call(app, "GET", "/api/nexus/persistent-memory?project=%20AURA", bearer(tokens, "pa"))
        assert r.status == 200 and r.json() == {"project": "aura"}
        r = call(app, "GET", "/api/nexus/persistent-memory", bearer(tokens, "all"))
        assert r.json() == {"project": "default"}                              # NEXUS_DEFAULT_PROJECT de la prueba

    def test_owner_cookie_flow_end_to_end(self):
        app, tokens, state, seen = make_stack()
        lr = login(app)
        ck = lr.header("set-cookie").split(";")[0]
        r = call(app, "POST", "/api/nexus/chat", {"Cookie": ck, "Origin": ORIGIN, **JSON_H}, b"{}")
        assert r.status == 200 and ("chat", b"{}") in seen
        r = call(app, "POST", "/api/nexus/chat", {"Cookie": ck, "Origin": "https://evil.example", **JSON_H}, b"{}")
        assert r.status == 403
