# -*- coding: utf-8 -*-
"""
Proveedor Gemini: reintentos y clasificacion de errores.

No usa red ni clave real: se sustituye GeminiProvider._call por un doble y asyncio.sleep por
un registro (sin esperas). Cubre cuota agotada, 503, timeout y errores no transitorios.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import backend.providers.gemini as gem
from backend.providers.base import GenerateRequest, ProviderResponse


class ApiError(Exception):
    """Imita google.genai.errors.APIError: .code, .status, .message y str '<code> <status>. ...'."""
    def __init__(self, code, status, message=""):
        super().__init__(f"{code} {status}. {message}")
        self.code, self.status, self.message = code, status, message


QUOTA_TEXT = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
              "current quota, please check your plan and billing details.', "
              "'status': 'RESOURCE_EXHAUSTED'}}")


def _ok(text="respuesta"):
    return ProviderResponse(text=text, provider="gemini", model="m", duration_ms=1)


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sleeps = []

        async def fake_sleep(d):
            self.sleeps.append(d)
        # Proxy de asyncio SOLO dentro del modulo gemini: sleep sin espera real, el resto igual.
        proxy = SimpleNamespace(sleep=fake_sleep, wait_for=asyncio.wait_for,
                                TimeoutError=asyncio.TimeoutError)
        for p in (mock.patch.object(gem, "GEMINI_MAX_RETRIES", 3),
                  mock.patch.object(gem, "asyncio", proxy)):
            p.start()
            self.addCleanup(p.stop)
        self.provider = gem.GeminiProvider()           # sin clave: _client None
        self.provider._client = object()               # "disponible"
        self.calls = 0

    def script(self, *outcomes):
        """_call devuelve/lanza cada resultado en orden (el ultimo se repite)."""
        seq = list(outcomes)

        async def fake_call(request):
            self.calls += 1
            out = seq[min(self.calls, len(seq)) - 1]
            if isinstance(out, BaseException):
                raise out
            return out
        self.provider._call = fake_call

    async def gen(self):
        return await self.provider.generate(GenerateRequest(prompt="hola", system="s", max_tokens=100))


# =====================================================================
# Cuota agotada: NO se reintenta
# =====================================================================
class TestQuotaNoRetry(_Base):
    async def test_429_resource_exhausted_una_sola_llamada(self):
        err = Exception(QUOTA_TEXT)
        self.script(err)
        with self.assertRaises(Exception) as cm:
            await self.gen()
        self.assertIs(cm.exception, err)               # relanza el error ORIGINAL
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.sleeps, [])              # ni una espera de backoff

    async def test_distintas_formas_de_cuota(self):
        casos = [
            Exception("429 Too Many Requests"),
            Exception("RESOURCE_EXHAUSTED: quota exceeded for metric generate_content"),
            Exception("You exceeded your current quota"),
            Exception("Rate limit reached for gemini"),
            Exception("RATE_LIMITED"),
            ApiError(429, "RESOURCE_EXHAUSTED", "quota"),
            ApiError(429, "", ""),                      # solo el codigo
            ApiError(0, "RESOURCE_EXHAUSTED", ""),      # solo el status
        ]
        for err in casos:
            with self.subTest(err=str(err)):
                self.calls = 0
                self.script(err)
                with self.assertRaises(Exception):
                    await self.gen()
                self.assertEqual(self.calls, 1)

    async def test_el_error_relanzado_sigue_reconocible_como_limite_para_el_plan(self):
        from backend.core.autonomy import _is_provider_limit    # logica 4A
        self.script(Exception(QUOTA_TEXT))
        with self.assertRaises(Exception) as cm:
            await self.gen()
        self.assertTrue(_is_provider_limit(str(cm.exception)))

    async def test_cuota_se_registra_una_vez_como_no_reintentable(self):
        self.script(Exception(QUOTA_TEXT))
        with mock.patch.object(gem.logger, "warning") as warn:
            with self.assertRaises(Exception):
                await self.gen()
        self.assertEqual(warn.call_count, 1)
        self.assertIn("no se reintenta", warn.call_args[0][0])

    async def test_cuota_tras_un_503_deja_de_reintentar(self):
        self.script(ApiError(503, "UNAVAILABLE", "overloaded"), Exception(QUOTA_TEXT))
        with self.assertRaises(Exception):
            await self.gen()
        self.assertEqual(self.calls, 2)                # 503 reintentado, cuota corta
        self.assertEqual(len(self.sleeps), 1)


# =====================================================================
# 5xx y timeouts: siguen reintentando
# =====================================================================
class TestTransientRetry(_Base):
    async def test_503_reintenta_y_se_recupera(self):
        self.script(ApiError(503, "UNAVAILABLE", "overloaded"), _ok("hola"))
        r = await self.gen()
        self.assertEqual(r.text, "hola")
        self.assertEqual(self.calls, 2)
        self.assertEqual(len(self.sleeps), 1)
        self.assertTrue(1.0 <= self.sleeps[0] <= 2.0)  # backoff 2**0 + jitter(0..1)

    async def test_503_persistente_agota_los_reintentos_y_relanza(self):
        err = ApiError(503, "UNAVAILABLE", "overloaded")
        self.script(err)
        with self.assertRaises(ApiError) as cm:
            await self.gen()
        self.assertIs(cm.exception, err)
        self.assertEqual(self.calls, 3)                # GEMINI_MAX_RETRIES sin cambios
        self.assertEqual(len(self.sleeps), 2)
        self.assertTrue(2.0 <= self.sleeps[1] <= 3.0)  # 2**1 + jitter

    async def test_otros_5xx_y_estados_de_servidor(self):
        casos = [ApiError(500, "INTERNAL", ""), ApiError(502, "", "bad gateway"),
                 ApiError(504, "DEADLINE_EXCEEDED", ""), ApiError(0, "UNAVAILABLE", ""),
                 ApiError(408, "", "request timeout"),
                 Exception("503 UNAVAILABLE. The model is overloaded."),
                 Exception("504 DEADLINE_EXCEEDED"), Exception("502 Bad Gateway"),
                 Exception("SERVICE_UNAVAILABLE")]
        for err in casos:
            with self.subTest(err=str(err)):
                self.calls = 0
                self.script(err, _ok())
                self.assertEqual((await self.gen()).text, "respuesta")
                self.assertEqual(self.calls, 2)

    async def test_timeout_se_reintenta(self):
        for err in (gem.GeminiTimeout("Gemini timeout después de 60s"),
                    asyncio.TimeoutError(), TimeoutError("deadline"),
                    RuntimeError("Gemini timeout después de 60s")):
            with self.subTest(err=repr(err)):
                self.calls = 0
                self.script(err, _ok())
                self.assertEqual((await self.gen()).text, "respuesta")
                self.assertEqual(self.calls, 2)

    async def test_timeout_real_de_la_llamada_se_clasifica_y_reintenta(self):
        never = asyncio.Event()

        async def hang(**kw):
            await never.wait()                         # nunca responde

        self.provider._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=hang)))
        with mock.patch.object(gem, "REQUEST_TIMEOUT_SECONDS", 0.02):
            with self.assertRaises(gem.GeminiTimeout) as cm:
                await self.gen()
        self.assertIn("Gemini timeout", str(cm.exception))   # mismo mensaje de siempre
        self.assertIsInstance(cm.exception, RuntimeError)    # compatible con quien captura RuntimeError
        self.assertEqual(len(self.sleeps), 2)                # 3 intentos


# =====================================================================
# Errores no transitorios y clasificacion precisa
# =====================================================================
class TestNonTransientAndPrecision(_Base):
    async def test_errores_no_transitorios_no_se_reintentan(self):
        casos = [ApiError(400, "INVALID_ARGUMENT", "bad request"),
                 ApiError(401, "UNAUTHENTICATED", "api key not valid"),
                 ApiError(403, "PERMISSION_DENIED", "denied"),
                 ApiError(404, "NOT_FOUND", "model not found"),
                 RuntimeError("Gemini devolvió una respuesta vacía."),
                 ValueError("algo raro"), KeyError("x")]
        for err in casos:
            with self.subTest(err=repr(err)):
                self.calls = 0
                self.script(err)
                with self.assertRaises(type(err)):
                    await self.gen()
                self.assertEqual(self.calls, 1)
                self.assertEqual(self.sleeps, [])

    def test_numeros_que_contienen_500_502_503_no_son_errores_de_servidor(self):
        for msg in ("400 INVALID_ARGUMENT: la solicitud tiene 15003 tokens",
                    "max_output_tokens 65500 fuera de rango", "id 5030 invalido",
                    "request 1502 rechazada"):
            with self.subTest(msg=msg):
                self.assertFalse(gem._is_transient(Exception(msg)))
                self.assertEqual(gem._classify_error(Exception(msg)), "permanent")

    def test_codigo_estructurado_manda_sobre_el_texto(self):
        # 503 real aunque el mensaje hable de cuota; 429 real aunque el texto diga 503
        self.assertEqual(gem._classify_error(ApiError(503, "UNAVAILABLE", "quota quota")), "transient")
        self.assertEqual(gem._classify_error(ApiError(429, "RESOURCE_EXHAUSTED", "503")), "quota")
        # 4xx estructurado que no es cuota: permanente aunque diga "timeout" o "quota project"
        self.assertEqual(gem._classify_error(ApiError(400, "INVALID_ARGUMENT", "timeout 503")), "permanent")
        self.assertEqual(gem._classify_error(ApiError(403, "PERMISSION_DENIED", "set a quota project")), "permanent")

    def test_code_no_entero_se_ignora(self):
        e = Exception("503 UNAVAILABLE")
        e.code = "503"                                 # texto, no int
        self.assertEqual(gem._classify_error(e), "transient")   # cae al analisis de texto
        e2 = Exception("boom")
        e2.code = True                                 # bool no cuenta como codigo
        self.assertEqual(gem._classify_error(e2), "permanent")

    def test_is_transient_conserva_su_contrato(self):
        self.assertTrue(gem._is_transient(Exception("503 UNAVAILABLE")))
        self.assertTrue(gem._is_transient(Exception("Gemini timeout después de 60s")))
        self.assertFalse(gem._is_transient(Exception(QUOTA_TEXT)))
        self.assertFalse(gem._is_transient(Exception("429")))
        self.assertFalse(gem._is_transient(ValueError("x")))


class TestUnchangedBehavior(_Base):
    async def test_exito_a_la_primera_sin_esperas(self):
        self.script(_ok("directo"))
        self.assertEqual((await self.gen()).text, "directo")
        self.assertEqual((self.calls, self.sleeps), (1, []))

    async def test_sin_cliente_sigue_fallando_igual(self):
        self.provider._client = None
        with self.assertRaises(RuntimeError) as cm:
            await self.gen()
        self.assertIn("no está disponible", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
