"""Fase 2 — configuración, tokens, contraseña, sesiones y herramienta local.

Cubre: S7, S8, S28, S40, S43, S81 (parte), S85 (parte), S98–S101, y utilidades.
Todo sintético: ningún secreto real.
"""
import io
import json
import socket
from datetime import datetime, timedelta, timezone

from backend.security import cli
from backend.security.login_guard import LoginGuard
from backend.security.passwords import hash_password, parse_hash, verify_password
from backend.security.sessions import SessionStore
from backend.security.settings import SecuritySettings
from backend.security.tokens import TokenRegistry, new_token
from tests.security_helper import Clock, make_env, token_entry


def settings_from(env):
    return SecuritySettings.from_env(env)


class TestPasswords:
    def test_hash_roundtrip_and_wrong(self):
        h = hash_password("una-contrasena-larga-1", n=1024)
        p = parse_hash(h, production=False)
        assert p is not None
        assert verify_password("una-contrasena-larga-1", p)
        assert not verify_password("otra", p)

    def test_format_is_hex_without_shell_specials(self):
        h = hash_password("una-contrasena-larga-1", n=1024)
        assert h.startswith("scrypt$1024$8$1$")
        assert all(c in "0123456789abcdef$scrypt" for c in h)

    def test_bad_hashes_rejected(self):
        for bad in ("", "x", "scrypt$1$2", "scrypt$1000$8$1$aa$bb", "scrypt$1024$8$1$zz$yy"):
            assert parse_hash(bad, production=False) is None
        assert parse_hash(hash_password("x" * 12, n=1024), production=True) is None   # N bajo en production

    def test_verify_never_raises_on_weird_input(self):
        p = parse_hash(hash_password("una-contrasena-larga-1", n=1024), production=False)
        for weird in ("", "\x00" * 10, "é" * 2000, None, 123):
            assert verify_password(weird, p) is False


class TestTokens:
    def make(self, **kw):
        raw, entry = token_entry("t1", ["read"], ["*"], **kw)
        s = settings_from({"NEXUS_ENVIRONMENT": "test", "NEXUS_API_TOKENS": json.dumps([entry])})
        return raw, s

    def test_valid_and_constant_shape(self):
        raw, s = self.make()
        reg = TokenRegistry(s.tokens, "test")
        p, reason, tid = reg.authenticate(raw)
        assert p is not None and p.id == "t1" and tid == "t1" and reason == "ok"

    def test_s3_unknown_malformed_expired_revoked_other_env(self):
        raw, s = self.make()
        reg = TokenRegistry(s.tokens, "test")
        assert reg.authenticate("nxs_" + "a" * 43)[0] is None
        assert reg.authenticate("xyz")[0] is None
        assert reg.authenticate("")[0] is None
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        raw2, s2 = self.make(expires=past)
        assert TokenRegistry(s2.tokens, "test").authenticate(raw2)[1] == "expired"
        raw3, s3 = self.make(revoked=True)
        assert TokenRegistry(s3.tokens, "test").authenticate(raw3)[1] == "revoked"
        raw4, e4 = token_entry("t4", ["read"], ["*"], env="development")
        s4 = settings_from({"NEXUS_ENVIRONMENT": "test", "NEXUS_API_TOKENS": json.dumps([e4])})
        assert TokenRegistry(s4.tokens, "test").authenticate(raw4)[1] == "other_env"

    def test_only_hash_is_stored(self):
        raw, s = self.make()
        assert raw not in repr(s.tokens) and raw not in repr(s.__dict__)


class TestSettings:
    def test_s7_projects_rules(self):
        _, a = token_entry("mix", ["read"], ["*", "aura"])
        _, b = token_entry("sin", ["read"], ["*"]); del b["projects"]
        _, c = token_entry("vac", ["read"], [])
        _, d = token_entry("ok", ["read"], ["Aura "])
        s = settings_from({"NEXUS_ENVIRONMENT": "test", "NEXUS_API_TOKENS": json.dumps([a, b, c, d])})
        assert [t.id for t in s.tokens] == ["ok"] and s.tokens[0].projects == ("aura",)
        assert set(s.token_ids_discarded) == {"mix", "sin", "vac"}
        assert any("'mix'" in w for w in s.warnings)

    def test_s8_project_token_without_usable_routes_warns_by_id(self):
        _, e = token_entry("solo-chat", ["chat", "run"], ["aura"])
        s = settings_from({"NEXUS_ENVIRONMENT": "test", "NEXUS_API_TOKENS": json.dumps([e])})
        assert any("'solo-chat'" in w and "Fase 3" in w for w in s.warnings)

    def test_s40_origin_missing_or_invalid_disables_browser_login(self):
        env, _ = make_env()
        del env["NEXUS_PUBLIC_ORIGIN"]
        assert not settings_from(env).browser_login_enabled
        for bad in ("http://example.com", "https://x.com/", "https://x.com/path", "x.com", "https://X.com", "*"):
            env["NEXUS_PUBLIC_ORIGIN"] = bad
            s = settings_from(env)
            assert not s.browser_login_enabled and s.public_origin is None, bad
        env["NEXUS_PUBLIC_ORIGIN"] = "http://localhost:8000"
        assert settings_from(env).browser_login_enabled
        env["NEXUS_ENVIRONMENT"] = "production"
        env["NEXUS_PUBLIC_ORIGIN"] = "http://127.0.0.1:8000"
        assert not settings_from(env).browser_login_enabled          # http no vale en production

    def test_s43_default_project(self):
        env, _ = make_env(NEXUS_DEFAULT_PROJECT="  AURA ")
        assert settings_from(env).default_project == "aura"
        for bad in ("*", "!!", "-x"):
            env["NEXUS_DEFAULT_PROJECT"] = bad
            s = settings_from(env)
            assert s.default_project is None and any("NEXUS_DEFAULT_PROJECT" in w for w in s.warnings)
            assert bad not in " ".join(s.warnings)

    def test_s98_audit_off_only_in_dev_test(self):
        env, _ = make_env(NEXUS_AUTH_MODE="off")
        assert settings_from(env).auth_mode == "off"
        env["NEXUS_ENVIRONMENT"] = "production"
        s = settings_from(env)
        assert s.auth_mode == "enforce" and any("NEXUS_AUTH_MODE" in w for w in s.warnings)
        env["NEXUS_ENVIRONMENT"] = "algo-raro"
        assert settings_from(env).environment == "production"

    def test_s99_tokens_json(self):
        s = settings_from({"NEXUS_ENVIRONMENT": "test", "NEXUS_API_TOKENS": "{no es json"})
        assert s.tokens == () and any("TODOS" in w for w in s.warnings)
        raw, ok = token_entry("ok", ["read"], ["*"])
        bad = {"id": "malo", "env": "test", "hash": "md5:xx", "permissions": ["read"], "projects": ["*"]}
        dup1 = token_entry("dup", ["read"], ["*"])[1]
        dup2 = token_entry("dup", ["read"], ["*"])[1]
        unknown_perm = token_entry("up", ["root"], ["*"])[1]
        s = settings_from({"NEXUS_ENVIRONMENT": "test",
                           "NEXUS_API_TOKENS": json.dumps([ok, bad, dup1, dup2, unknown_perm])})
        assert [t.id for t in s.tokens] == ["ok"]

    def test_s100_numeric_invalid_defaults_and_never_printed(self):
        env, _ = make_env(NEXUS_MAX_BODY_BYTES="mil", NEXUS_BODY_TIMEOUT_SECONDS="-3",
                          NEXUS_LOGIN_MAX_ATTEMPTS="0", NEXUS_RATE_CHAT_PER_MINUTE="x9")
        s = settings_from(env)
        assert s.max_body_bytes == 1048576 and s.body_timeout_seconds == 30
        assert s.login_max_attempts == 5 and s.rate_chat_per_minute == 30
        text = " ".join(s.warnings)
        for v in ("mil", "-3", "x9"):
            assert v not in text
        assert "NEXUS_MAX_BODY_BYTES" in text and "NEXUS_BODY_TIMEOUT_SECONDS" in text

    def test_s101_ack_strict(self):
        env, _ = make_env(NEXUS_ENVIRONMENT="production", NEXUS_PUBLIC_ORIGIN="https://nexus.example.com")
        env["NEXUS_OWNER_PASSWORD_HASH"] = hash_password("x" * 12, n=2 ** 14)
        for v in ("true", "TRUE", " true "):
            env["NEXUS_SINGLE_PROCESS_ACK"] = v
            assert settings_from(env).browser_login_enabled, v
        for v in ("yes", "1", "si", "verdadero", ""):
            env["NEXUS_SINGLE_PROCESS_ACK"] = v
            assert not settings_from(env).browser_login_enabled, v

    def test_s96_deployment_guard(self):
        env, _ = make_env(NEXUS_ENVIRONMENT="production", NEXUS_PUBLIC_ORIGIN="https://nexus.example.com")
        env["NEXUS_OWNER_PASSWORD_HASH"] = hash_password("x" * 12, n=2 ** 14)
        s = settings_from(env)                                     # (a) sin ACK
        assert not s.browser_login_enabled and s.has_auth_method          # los Bearer siguen
        env["NEXUS_SINGLE_PROCESS_ACK"] = "true"
        assert settings_from(env).browser_login_enabled
        env["WEB_CONCURRENCY"] = "2"                                       # (b)
        assert not settings_from(env).browser_login_enabled
        del env["WEB_CONCURRENCY"]
        env["UVICORN_WORKERS"] = "3"                                       # (c)
        assert not settings_from(env).browser_login_enabled
        del env["UVICORN_WORKERS"]
        summary = settings_from(env).startup_summary()                     # (d)
        assert "declared_not_verified" in summary and "single process" not in summary.lower()

    def test_enforce_without_method_is_flagged(self):
        s = settings_from({"NEXUS_ENVIRONMENT": "production"})
        assert not s.has_auth_method and s.auth_mode == "enforce"


class TestSessionsAndGuard:
    def test_s24_s25_idle_and_max(self):
        c = Clock()
        st = SessionStore(max_seconds=28800, idle_seconds=1800, max_active=5, clock=c)
        sid = st.create()
        c.advance(29 * 60); assert st.validate(sid)
        c.advance(29 * 60); assert st.validate(sid)          # la actividad renueva la inactividad
        c.advance(31 * 60); assert not st.validate(sid)
        sid = st.create()
        for _ in range(17):                                  # actividad continua, 8,5 h
            c.advance(1700); st.validate(sid)
        assert not st.validate(sid)

    def test_s26_s27_logout_all_and_max_active(self):
        st = SessionStore(1000, 1000, 3, Clock())
        ids = [st.create() for _ in range(5)]
        assert not st.validate(ids[0]) and not st.validate(ids[1]) and st.validate(ids[4])
        assert st.destroy_all() == 3 and not st.validate(ids[4])

    def test_s28_only_hashes_stored(self):
        st = SessionStore(1000, 1000, 3, Clock())
        sid = st.create()
        assert sid not in repr(st.__dict__) and sid not in "".join(st._data.keys())

    def test_s29_distinct_ids(self):
        st = SessionStore(1000, 1000, 5, Clock())
        assert len({st.create() for _ in range(5)}) == 5

    def test_s30_guard_timeline(self):
        c = Clock(); g = LoginGuard(clock=c)
        assert g.begin().kind == "ok"; assert g.fail().kind == "ok"           # t=0 fallo 1
        c.advance(0.5); d = g.begin(); assert d.kind == "backoff" and d.retry_after == 1
        c.advance(0.5); assert g.begin().kind == "ok"; g.fail()               # t=1 fallo 2
        c.advance(1.99); assert g.begin().kind == "backoff"
        c.advance(0.01); assert g.begin().kind == "ok"; g.fail()              # t=3 fallo 3
        c.advance(4); assert g.begin().kind == "ok"; g.fail()                 # t=7 fallo 4
        c.advance(8); assert g.begin().kind == "ok"
        d = g.fail(); assert d.kind == "locked" and d.retry_after == 900      # t=15 fallo 5
        c.advance(85); d = g.begin(); assert d.kind == "locked" and d.retry_after == 815
        c.advance(815); assert g.begin().kind == "ok"                          # t=915

    def test_s30e_window_reset_and_success_clears(self):
        c = Clock(); g = LoginGuard(clock=c)
        g.begin(); g.fail(); c.advance(2); g.begin(); g.fail()
        c.advance(900); assert g.begin().kind == "ok" and g.failures == 0
        g.succeed(); assert g.failures == 0

    def test_s30h_only_one_evaluation_at_a_time(self):
        g = LoginGuard(clock=Clock())
        assert g.begin().kind == "ok"
        assert all(g.begin().kind == "backoff" for _ in range(19))
        g.fail()

    def test_s30i_clear(self):
        c = Clock(); g = LoginGuard(clock=c)
        for _ in range(5):
            g.begin(); g.fail(); c.advance(20)
            if g.failures == 5:
                break
        g.clear()
        assert g.failures == 0 and g.begin().kind == "ok"


class TestCliLocalTool:
    def test_s81_no_network_no_files(self):
        calls = []
        real = socket.socket.connect
        socket.socket.connect = lambda self, *a, **k: calls.append(a)
        try:
            out = io.StringIO()
            assert cli.main(["new-token", "--id", "x", "--permissions", "read"], out) == 0
            assert cli.cmd_hash_password(None, out, inp=lambda _: "una-contrasena-larga-1") == 0
            assert cli.cmd_check_env(None, out, environ={"NEXUS_ENVIRONMENT": "test"}) in (0, 1)
        finally:
            socket.socket.connect = real
        assert calls == []

    def test_new_token_prints_only_hash_in_entry(self):
        out = io.StringIO()
        cli.main(["new-token", "--id", "x", "--permissions", "read,admin", "--projects", "aura"], out)
        text = out.getvalue()
        raw = [l for l in text.splitlines() if l.startswith("nxs_")][0]
        entry = json.loads([l for l in text.splitlines() if l.startswith("{")][0])
        assert raw not in json.dumps(entry) and entry["hash"].startswith("sha256:")
        assert entry["projects"] == ["aura"]

    def test_hash_password_rejects_short_and_mismatch(self):
        out = io.StringIO()
        assert cli.cmd_hash_password(None, out, inp=lambda _: "corta") == 2
        answers = iter(["una-contrasena-larga-1", "otra-contrasena-larga-2"])
        assert cli.cmd_hash_password(None, out, inp=lambda _: next(answers)) == 2

    def test_check_env_lists_routes_by_id_and_never_values(self):
        env, tokens = make_env()
        out = io.StringIO()
        rc = cli.cmd_check_env(None, out, environ=env)
        text = out.getvalue()
        assert rc == 0 and "OK" in text and "token 'pa'" in text and "token 'all'" in text
        for raw in tokens.values():
            assert raw not in text
        assert env["NEXUS_OWNER_PASSWORD_HASH"] not in text

    def test_scan_stored_lists_ids_only(self):
        import os, sqlite3, tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, "copia.db")
        con = sqlite3.connect(path)
        con.execute("create table m (id text, content text)")
        con.execute("insert into m values ('a1','hola normal')")
        con.execute("insert into m values ('a2','mi clave es AIza" + "B" * 35 + "')")
        con.commit(); con.close()
        out = io.StringIO()
        assert cli.main(["scan-stored", path], out) == 0
        text = out.getvalue()
        assert "rowid=2" in text and "google_api_key" in text and "AIza" not in text and "rowid=1" not in text
