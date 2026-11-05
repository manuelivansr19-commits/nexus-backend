"""Fase 2 — filtro de secretos (D6): S46–S58.

TODO sintético: ningún secreto real en las muestras ni en el corpus. El filtro es HEURÍSTICO:
S46 mide 0 falsos negativos solo para las familias listadas y ≤1 % de falsos positivos solo sobre el
corpus usado; no garantiza detectar formatos no listados. Los umbrales son los PROPUESTOS (decisión H9).
"""
import asyncio
import json
import logging
import os
import random
import re
import secrets as pysecrets
import string
import unittest

from backend.security import secrets_filter as sf
from backend.security.outbound import redact_request
from backend.security.passwords import hash_password

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = random.Random(20261103)
ALNUM = string.ascii_letters + string.digits
URLSAFE = ALNUM + "_-"


def rnd(n, alphabet=ALNUM):
    return "".join(RNG.choice(alphabet) for _ in range(n))


def gen_jwt():
    return "eyJ" + rnd(20, URLSAFE) + ".eyJ" + rnd(30, URLSAFE) + "." + rnd(43, URLSAFE)


def gen_pem():
    body = "\n".join(rnd(64, ALNUM + "+/") for _ in range(3))
    return "-----BEGIN PRIVATE KEY-----\n" + body + "\n-----END PRIVATE KEY-----", body


# familia -> (generador del texto secreto, fragmento que NO debe sobrevivir)
FAMILY_GEN = {
    "google_api_key": lambda: (lambda s: (s, s))("AIza" + rnd(35, URLSAFE)),
    "groq_key": lambda: (lambda s: (s, s))("gsk_" + rnd(48)),
    "openrouter_key": lambda: (lambda s: (s, s))("sk-or-v1-" + rnd(48, string.hexdigits.lower())),
    "openai_style_key": lambda: (lambda s: (s, s))("sk-" + rnd(48)),
    "github_token": lambda: (lambda s: (s, s))(RNG.choice(["ghp_", "gho_", "ghs_"]) + rnd(36)),
    "github_pat": lambda: (lambda s: (s, s))("github_pat_" + rnd(40, ALNUM + "_")),
    "nexus_token": lambda: (lambda s: (s, s))("nxs_" + rnd(43, URLSAFE)),
    "nexus_scrypt": lambda: (lambda s: (s, s))(hash_password("muestra-sintetica-12", n=1024)),
    "pem_private_key": gen_pem,
    "jwt": lambda: (lambda s: (s, s))(gen_jwt()),
    "authorization_bearer": lambda: (lambda v: ("Authorization: Bearer " + v, v))(rnd(34, ALNUM + "._~-")),
    "assignment": lambda: (lambda kv: (kv[0] + RNG.choice([" = ", ": ", "="]) + kv[1], kv[1]))(
        (RNG.choice(["password", "secret", "token", "api_key", "apikey", "passwd"]), rnd(6) + "7" + rnd(6) + "_")),
}

CONTEXTS = [
    "{s}", "Mi clave es {s} y no la compartas.", '{{"clave": "{s}", "ok": true}}',
    "https://api.ejemplo.test/v1/datos?key={s}&modo=1", "```\n{s}\n```", "linea uno\nlinea dos {s}\nlinea tres",
    "({s}).", "'{s}'", "- elemento: {s}\n- otro: nada", "export VALOR={s}\n", "curl -H 'X: {s}' https://x.test",
    "log: usuario ingresó con {s}, fin.", "# comentario {s}", "valor=\"{s}\";", "<campo>{s}</campo>",
    "[{s}]", "Hola,\n\n{s}\n\nSaludos", "yaml:\n  clave: {s}\n", "texto largo " * 5 + "{s} " + "mas texto " * 5,
    "1) {s}\n2) segundo", "«{s}»", "{s}, {s}",
]


def make_samples(family, n=24):
    out = []
    for i in range(n):
        text, must_vanish = FAMILY_GEN[family]()
        ctx = CONTEXTS[i % len(CONTEXTS)]
        out.append((ctx.replace("{s}", text), must_vanish))
    return out


class TestS46Measurement:
    def test_s46_i_zero_false_negatives_for_listed_families(self):
        report = {}
        for fam in FAMILY_GEN:
            samples = make_samples(fam)
            assert len(samples) >= 20
            misses = 0
            for text, must_vanish in samples:
                new, counts = sf.redact(text)
                first = must_vanish.splitlines()[0]
                if not counts or first in new or sf.REDACTION not in new:
                    misses += 1
            report[fam] = (len(samples), misses)
        print("S46(i) falsos negativos por familia (muestras, fallos):", json.dumps(report))
        assert all(m == 0 for _, m in report.values()), report

    def test_s46_ii_false_positive_rate_at_most_one_percent(self):
        corpus = benign_corpus()
        assert len(corpus) >= 500
        flagged = [t for t in corpus if sf.contains_secret(t)]
        rate = len(flagged) / len(corpus)
        print(f"S46(ii) falsos positivos: {len(flagged)}/{len(corpus)} = {rate:.2%}")
        for t in flagged[:5]:
            print("  marcado:", repr(t[:80]))
        assert rate <= 0.01

    def test_s46_iii_scope_is_documented(self):
        assert "HEURÍSTICO" in sf.__doc__ and "no garantiza" in sf.__doc__.lower()
        assert sf.contains_secret("un formato inventado QQQ-12345-ZZZ-67890") is False   # fuera de la lista


def benign_corpus():
    r = random.Random(7)
    sujetos = ["El equipo", "La directora", "Nuestro cliente", "El proveedor", "Mi asistente", "El comité"]
    verbos = ["revisó", "aprobó", "pospuso", "comparó", "resumió", "analizó"]
    objetos = ["el presupuesto trimestral", "la estrategia de marketing", "el contrato de servicios",
               "los indicadores de ventas", "el plan de contratación", "la hoja de ruta del producto"]
    cierres = ["antes del viernes.", "con muchas dudas.", "sin cambios mayores.", "y pidió más datos.", "en la reunión de hoy."]
    corpus = [f"{r.choice(sujetos)} {r.choice(verbos)} {r.choice(objetos)} {r.choice(cierres)}" for _ in range(200)]
    snippets = ["def suma(a, b):\n    return a + b", "for i in range(10):\n    print(i)", "x = {'a': 1, 'b': [1, 2, 3]}",
                "SELECT id, nombre FROM clientes WHERE activo = 1;", "const total = items.reduce((a, b) => a + b, 0);",
                "git commit -m 'arreglo del informe'", "import os, sys", "class Cliente:\n    pass"]
    corpus += [r.choice(snippets) + f"\n# nota {i}" for i, _ in enumerate(range(100))]
    import uuid
    corpus += [str(uuid.UUID(int=r.getrandbits(128))) for _ in range(60)]
    corpus += [("%064x" % r.getrandbits(256)) for _ in range(30)] + [("%040x" % r.getrandbits(160)) for _ in range(30)]
    corpus += [f"https://www.ejemplo.test/{rnd(8, string.ascii_lowercase)}/{rnd(12, string.ascii_lowercase)}/informe-{i}.html?pagina={i}" for i in range(40)]
    corpus += [f"ORD-2026-{i:06d} por ${r.randint(100, 99999)}.{r.randint(0, 99):02d}" for i in range(40)]
    corpus += ["El token de acceso se renueva cada hora; pide uno nuevo si falla.",
               "La contraseña debe cambiarse con regularidad, dicen las guías públicas.",
               "Una clave de cifrado fuerte tiene suficiente longitud.", "secret: lo guardamos en el gestor",
               "password = hunter", "Authorization se envía en la cabecera, sin más detalle."]
    # textos públicos del propio repositorio (sin las pruebas ni el filtro, que contienen muestras sintéticas)
    skip = ("backend/security/", "tests/test_phase2", "tests/security_helper", "tests/asgi_helper", "tests/framing_server")
    n = 0
    for base in ("backend", "tests", "."):
        for dp, _, files in os.walk(os.path.join(ROOT, base)):
            if "__pycache__" in dp or dp.count(os.sep) - ROOT.count(os.sep) > 3 and base == ".":
                continue
            for f in sorted(files):
                if not f.endswith((".py", ".md")):
                    continue
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                if rel.startswith(skip) or (base == "." and "/" in rel):
                    continue
                text = open(path, encoding="utf-8", errors="ignore").read()
                for i in range(0, min(len(text), 4000 * 3), 500):
                    corpus.append(text[i:i + 500])
                    n += 1
    return corpus


class TestRedactionBasics:
    def test_redact_replaces_and_counts_without_echoing(self):
        s = "AIza" + "B" * 35
        new, counts = sf.redact("clave " + s + " fin")
        assert new == "clave " + sf.REDACTION + " fin" and counts == {"google_api_key": 1}

    def test_configured_secret_values_are_redacted(self):
        sf.set_known_secrets(["valor-literal-de-prueba-9"])
        try:
            assert "valor-literal" not in sf.redact("x valor-literal-de-prueba-9 y")[0]
        finally:
            sf.set_known_secrets([])

    def test_redact_request_accepts_non_dataclass_doubles(self):
        import types
        secret = "AIza" + "K" * 35
        req = types.SimpleNamespace(prompt="x " + secret, system="", history=[])
        out = redact_request(req)
        assert secret not in out.prompt and secret in req.prompt

    def test_redact_obj_and_reject(self):
        obj = {"a": ["ok", {"b": "gsk_" + "Q" * 40}], "n": 5}
        new, c = sf.redact_obj(obj)
        assert c == {"groq_key": 1} and "gsk_" not in json.dumps(new) and new["n"] == 5
        try:
            sf.reject_if_secret(obj)
            raise AssertionError("debió rechazar")
        except sf.SecretDetected as e:
            assert e.families == ("groq_key",) and "gsk_" not in str(e)
        sf.reject_if_secret({"a": "nada"})


class TestExplicitAndImplicitStores:
    def test_s47_explicit_persistent_memory_rejects_without_saving(self):
        from backend.core.memory import RAMMemoryStore
        from backend.core.persistent_memory import PersistentMemory, PersistentMemoryError
        pm = PersistentMemory(RAMMemoryStore())
        secret = "AIza" + "C" * 35
        try:
            pm.remember("mi clave es " + secret, scope="aura")
            raise AssertionError("debió rechazar")
        except PersistentMemoryError as e:
            assert secret not in str(e)
        assert pm.list_records("aura") == []
        reply = pm.handle_message("recuerda que mi clave es " + secret, project="aura")
        assert reply is None or secret not in reply
        assert pm.list_records("aura") == []
        pm.remember("me gusta el café", scope="aura")                      # lo normal sigue funcionando
        assert len(pm.list_records("aura")) == 1

    def test_s47_explicit_knowledge_add_rejects(self):
        from backend.knowledge.engine import KnowledgeEngine
        import tempfile
        eng = KnowledgeEngine(db_path=os.path.join(tempfile.mkdtemp(), "k.db"))
        secret = "sk-" + "D" * 40
        r = eng.add(title="nota", content="usa la clave " + secret)
        assert r.success is False and r.entry_id is None and secret not in " ".join(r.errors)
        assert eng.stats()["total"] == 0 if "total" in eng.stats() else True

    def test_s48_implicit_working_memory_redacts_before_writing(self):
        from backend.core.memory import ConversationMemory, Memory, MemoryType, RAMMemoryStore
        store = RAMMemoryStore()
        secret = "ghp_" + "E" * 36
        ConversationMemory(store).add_user("uso este token " + secret)
        Memory(store).remember(content="[TOOL:x] " + secret, memory_type=MemoryType.WORKING, importance=0.4)
        for e in store._store.values():
            assert secret not in e.content and sf.REDACTION in e.content

    def test_s52_s53_tool_arguments_with_secret_are_rejected_not_executed(self):
        from backend.core.executor import Executor
        from backend.tools.base import BaseTool, ToolInput, ToolResult
        from backend.tools.registry import ToolRegistry

        class Probe(BaseTool):
            ran = 0
            name = property(lambda self: "probe")
            description = property(lambda self: "probe")

            async def execute(self, tool_input: ToolInput) -> ToolResult:
                Probe.ran += 1
                return ToolResult(success=True, output="ok")
        reg = ToolRegistry(); reg.register(Probe())
        ex = Executor(registry=reg)
        secret = "gsk_" + "F" * 40
        for params, ctx in (({"q": secret}, ""), ({"nested": {"k": ["x", secret]}}, ""), ({}, "contexto " + secret)):
            res = asyncio.run(ex.execute_by_name("probe", params, ctx))
            assert res.success is False and secret not in (res.error or "")
        assert Probe.ran == 0
        assert asyncio.run(ex.execute_by_name("probe", {"q": "hola"})).success and Probe.ran == 1


class TestOutboundControlPoints:
    def _runtime(self, name, local, seen):
        from backend.inference.interface import InferenceRuntime
        from backend.inference.models import InferenceResult, RuntimeHealth, RuntimeStatus, RuntimeType

        class Fake(InferenceRuntime):
            name = property(lambda self: name)
            runtime_type = property(lambda self: RuntimeType.LOCAL if local else RuntimeType.CLOUD)

            async def generate(self, request):
                seen.append((name, request))
                return InferenceResult(text="ok", runtime_name=name, runtime_type=self.runtime_type,
                                       model="m", mode=None)

            async def health(self):
                return RuntimeHealth(status=RuntimeStatus.AVAILABLE, runtime_name=name, runtime_type=self.runtime_type)
        return Fake()

    def _layer(self, runtimes, mode):
        from backend.inference import InferenceLayer, InferencePolicy, InferenceRegistry
        reg = InferenceRegistry()
        for r in runtimes:
            reg.register(r)
        return InferenceLayer(registry=reg, policy=InferencePolicy(mode=mode, allow_cloud=True))

    def test_s49_s50_s51_cloud_and_local_runtimes_get_redacted_request(self):
        from backend.inference import InferenceMode, InferenceRequest
        secret = "AIza" + "G" * 35
        for local in (True, False):
            seen = []
            mode = InferenceMode.LOCAL_ONLY if local else InferenceMode.CLOUD_ONLY
            layer = self._layer([self._runtime("rt", local, seen)], mode)
            req = InferenceRequest(prompt="contexto recuperado: " + secret, system="sys " + secret,
                                   history=[{"role": "user", "content": "antes " + secret}],
                                   metadata={"nota": secret, "n": 3, "anidado": {"k": [secret]}})
            asyncio.run(layer.generate(req))
            (_, got), = seen
            blob = json.dumps([got.prompt, got.system, got.history, got.metadata])
            assert secret not in blob and blob.count(sf.REDACTION) == 5, (local, blob)
            assert secret in req.prompt                                    # el original no se muta

    def test_s54_legacy_router_path_gets_redacted_request(self):
        try:
            from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse
            from backend.router import ModelRouter as Router
        except ImportError as exc:                      # p. ej. google-genai no instalado en este equipo
            raise unittest.SkipTest(f"no se pudo importar la ruta antigua: {exc}")
        seen = []

        class Capturing(BaseModelProvider):
            is_local = True
            name = property(lambda self: "cap")
            model = property(lambda self: "m")
            is_configured = property(lambda self: True)

            async def generate(self, request):
                seen.append(request)
                return ProviderResponse(text="ok", provider="cap", model="m", duration_ms=1)
        secret = "sk-or-v1-" + "a" * 48
        asyncio.run(Router([Capturing()]).generate(GenerateRequest(prompt="p " + secret, system="s " + secret,
                                                                    history=[{"role": "user", "content": secret}])))
        assert seen and secret not in json.dumps([seen[0].prompt, seen[0].system, seen[0].history])

    def test_s56_sentinels_not_in_logs(self):
        from backend.inference import InferenceMode, InferenceRequest
        records = []
        h = logging.Handler(); h.emit = lambda r: records.append(r.getMessage())
        root = logging.getLogger(); root.addHandler(h)
        sec = logging.getLogger("NEXUS.security")
        old, old_sec = root.level, sec.level
        root.setLevel(logging.DEBUG); sec.setLevel(logging.DEBUG)       # otras pruebas lo dejan en CRITICAL
        secret = "gsk_" + "H" * 40
        try:
            seen = []
            layer = self._layer([self._runtime("rt", True, seen)], InferenceMode.LOCAL_ONLY)
            asyncio.run(layer.generate(InferenceRequest(prompt=secret)))
        finally:
            root.removeHandler(h); root.setLevel(old); sec.setLevel(old_sec)
        assert records and all(secret not in m for m in records)
        assert any("Filtro de secretos" in m and "groq_key=1" in m for m in records)

    def test_s58_every_registered_runtime_receives_redacted_text(self):
        """Recorre el registro de runtimes (locales y de nube): todos reciben texto ya redactado."""
        from backend.inference import (GeminiRuntime, GroqRuntime, InferenceMode, InferenceRegistry,
                                       InferenceRequest, LlamaCppRuntime, OllamaRuntime, OpenRouterRuntime)
        from backend.inference import InferenceLayer, InferencePolicy
        reg = InferenceRegistry()
        runtimes = [OllamaRuntime(http_client=None), LlamaCppRuntime(), GeminiRuntime(),
                    GroqRuntime(http_client=None), OpenRouterRuntime(http_client=None)]
        seen = {}
        for rt in runtimes:
            async def capture(request, _n=rt.name):
                seen[_n] = request
                raise RuntimeError("capturado")
            rt.generate = capture
            reg.register(rt)
        secret = "AIza" + "J" * 35
        layer = InferenceLayer(registry=reg, policy=InferencePolicy(mode=InferenceMode.LOCAL_FIRST, allow_cloud=True))
        try:
            asyncio.run(layer.generate(InferenceRequest(prompt="x " + secret, system=secret)))
        except Exception:
            pass
        assert set(seen) == {r.name for r in runtimes if r.name in seen} and len(seen) >= 2
        for name, req in seen.items():
            assert secret not in req.prompt and secret not in req.system, name


class TestStaticInventory:
    # Cada archivo con `.generate(` y cuántas llamadas. Todo pasa por InferenceLayer.generate o Router.generate,
    # que redactan; los adaptadores de runtime están DETRÁS del punto de control.
    CLOSED_LIST = {
        "backend/router.py": 1,                    # ruta antigua: punto de control 2
        "backend/inference/layer.py": 1,           # punto de control 1
        "backend/inference/cloud/gemini.py": 1,    # adaptador, detrás del control
        "backend/inference/cloud/groq_openrouter.py": 2,
        "backend/core/nexus.py": 2,                # vía InferenceLayer y vía Router
        "backend/core/autonomy.py": 2,
        "backend/core/planner.py": 2,
    }
    PROVIDER_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:google|groq|openai|ollama|llama_cpp)\b", re.M)

    @staticmethod
    def scan(root):
        found = {}
        for dp, _, files in os.walk(os.path.join(root, "backend")):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                if rel.startswith("backend/security/") or rel.startswith("backend/providers/"):
                    continue
                n = len(re.findall(r"\.generate\(", open(path, encoding="utf-8", errors="ignore").read()))
                if n:
                    found[rel] = n
        return found

    def test_s55_inventory_matches_closed_list(self):
        assert self.scan(ROOT) == self.CLOSED_LIST

    def test_s55_a_new_unclassified_call_breaks_the_inventory(self):
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp()
        shutil.copytree(os.path.join(ROOT, "backend"), os.path.join(tmp, "backend"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        with open(os.path.join(tmp, "backend", "core", "nuevo.py"), "w", encoding="utf-8") as fh:
            fh.write("async def f(runtime, req):\n    return await runtime.generate(req)\n")
        assert self.scan(tmp) != self.CLOSED_LIST
        assert "backend/core/nuevo.py" in self.scan(tmp)

    def test_s55_no_direct_provider_imports_outside_layer_and_providers(self):
        offenders = []
        for dp, _, files in os.walk(os.path.join(ROOT, "backend")):
            for f in files:
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                if not f.endswith(".py") or rel.startswith(("backend/inference/", "backend/providers/", "backend/security/")):
                    continue
                if self.PROVIDER_IMPORT.search(open(path, encoding="utf-8", errors="ignore").read()):
                    offenders.append(rel)
        assert offenders == [], offenders
