"""Fase 2 — comprobaciones estáticas y de integración con la aplicación real.

S1 (inventario contra app.routes), S75, S76, S78 (G1), S79, y la conexión del middleware en main.py.
Las que necesitan FastAPI se omiten (con motivo registrado) si no está instalado.
"""
import ast
import os
import re
import unittest

from backend.security import routes as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


class TestIndexHtml:
    def test_s76_no_worker_registration_no_credentials_in_page(self):
        html = read("index.html")
        for bad in ("serviceWorker.register", "getRegistrations", "unregister", "caches.", "Bearer", "nxs_"):
            assert bad not in html, bad
        assert not re.search(r"\b(AIza|gsk_|sk-or-|ghp_)[A-Za-z0-9_-]{10,}", html)

    def test_s79_main_sw_has_no_purge_code(self):
        main = read("backend/main.py")
        sw = main[main.index('"/sw.js"'):main.index('"/sw.js"') + 600]
        for bad in ("getRegistrations", "unregister", "caches.", "serviceWorker.register", "fetch"):
            assert bad not in sw.replace('"fetch"', ""), bad
        assert "addEventListener(\"fetch\"" not in main and "caches" not in main

    def test_s75_sw_is_no_store_without_fetch_handler(self):
        main = read("backend/main.py")
        assert "no-store" in main[main.index('"/sw.js"'):main.index('"/sw.js"') + 600]
        assert 'addEventListener("fetch"' not in main and "addEventListener('fetch'" not in main

    def test_s78_g1_logout_keeps_local_data_and_button_wipes_exactly_three_keys(self):
        html = read("index.html")
        logout = html[html.index("logoutBtn.addEventListener"):]
        logout = logout[:logout.index("});") + 3]
        assert "localStorage" not in logout and "removeItem" not in logout          # cerrar sesión NO borra
        assert "Borrar datos de este dispositivo" in html
        wipe = html[html.index("wipeBtn.addEventListener"):]
        wipe = wipe[:wipe.index("});") + 3]
        assert "window.confirm(" in wipe and wipe.index("window.confirm(") < wipe.index("removeItem")
        assert re.findall(r"LOCAL_KEYS = \[(.*?)\]", html)[0].replace("'", "").replace(" ", "") == \
            "nexus-history,nexus-reminders,nexus-personality"
        assert wipe.count("removeItem") == 1 and "LOCAL_KEYS.forEach" in wipe and "clear()" not in wipe

    def test_inline_scripts_parse_when_node_is_available(self):
        import shutil, subprocess, tempfile
        node = shutil.which("node")
        if not node:
            raise unittest.SkipTest("node no está instalado: no se comprobó la sintaxis de los scripts")
        for i, body in enumerate(re.findall(r"<script>(.*?)</script>", read("index.html"), re.S)):
            path = os.path.join(tempfile.mkdtemp(), f"s{i}.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            r = subprocess.run([node, "--check", path], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr


class TestMainWiring:
    def test_middleware_is_registered_inside_cors_and_cors_is_not_wildcard(self):
        main = read("backend/main.py")
        a = main.index("app.add_middleware(SecurityMiddleware")
        b = main.index("app.add_middleware(CORSMiddleware")
        assert a < b                                    # el último añadido es el más externo: CORS fuera
        assert 'allow_origins=["*"]' not in main and "allow_credentials=False" in main

    def test_status_exposes_boot_id_and_health_does_not(self):
        tree = ast.parse(read("backend/main.py"))
        funcs = {n.name: ast.get_source_segment(read("backend/main.py"), n) for n in ast.walk(tree)
                 if isinstance(n, ast.AsyncFunctionDef)}
        assert "boot_id" in funcs["nexus_status"] and "boot_id" not in funcs["health"]

    def test_every_route_function_in_main_is_classified(self):
        """Cada @app.<método>(ruta) de main.py debe estar en la tabla de rutas (o ser una de las 29 existentes)."""
        main = read("backend/main.py")
        declared = {(m.upper(), p) for m, p in re.findall(r'@app\.(get|post|delete|head|put|patch)\("([^"]+)"', main)}
        table = {(r.method, r.template) for r in R.ROUTES if not r.template.startswith("/api/auth/")}
        assert len(declared) == 29, len(declared)
        assert declared == table, (declared ^ table)

    def test_s1_against_real_app_routes_when_fastapi_is_installed(self):
        try:
            from backend.main import app
        except ImportError as exc:
            raise unittest.SkipTest(f"no se pudo importar la aplicación: {exc}")
        extra_ok = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}     # los niega el middleware
        seen = set()
        for r in app.routes:
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None) or set()
            if path in extra_ok or not path:
                continue
            for m in methods:
                if m in ("OPTIONS",):
                    continue
                seen.add((m, path))
                assert R.match(m, re.sub(r"\{[a-z_]+\}", "x1", path)) is not None, f"ruta sin clasificar: {m} {path}"
        table = {(r.method, r.template) for r in R.ROUTES if not r.template.startswith("/api/auth/")}
        assert seen == table, seen ^ table


class TestDocumentation:
    def test_s96e_runbook_states_ack_is_a_declaration_and_r6_has_no_promise(self):
        doc = read("docs/FASE2_ACCESO_Y_SECRETOS.md")
        for phrase in ("declaración del propietario", "no puede verificarla", "no detecta las instancias de Render",
                       "deshabilitado", "señales, no pruebas", "R6", "NO existe recuperación local garantizada",
                       "Una cookie NO sirve".lower()):
            assert phrase.lower() in doc.lower(), phrase
        for var in ("NEXUS_PUBLIC_ORIGIN", "NEXUS_SINGLE_PROCESS_ACK", "NEXUS_UNLOCK_RATE_PER_MINUTE",
                    "NEXUS_LOGIN_LOCKOUT_SECONDS", "NEXUS_BODY_TIMEOUT_SECONDS"):
            assert var in doc, var

    def test_no_new_dependencies_and_no_mesan(self):
        reqs = read("requirements.txt").split()
        assert [r for r in reqs if r.lower().startswith(("fastapi", "uvicorn", "httpx", "pydantic", "google-genai"))] == \
            ["fastapi", "uvicorn", "httpx", "pydantic", "google-genai>=1.36.0"]
        for dp, _, files in os.walk(ROOT):
            if "__pycache__" in dp or os.sep + ".git" in dp:
                continue
            for f in files:
                if f.endswith((".py", ".md", ".html", ".txt")) and "test_phase2_static" not in f:
                    assert "mesan" not in open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read().lower(), (dp, f)
