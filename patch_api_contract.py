"""
Parche para tests/*.py: actualiza 10 tests que documentaban un contrato
de API antiguo (v3.6.0-ish) para que reflejen el comportamiento real y
actual de la API v3.8.0.

Cada cambio está documentado en el propio reemplazo. Ninguno modifica
código de producción — solo las expectativas de los tests.

Uso:
    python patch_api_contract.py

No es idempotente en el sentido estricto (una vez aplicado, el texto
viejo ya no existe para volver a encontrarlo), pero es seguro correrlo
más de una vez: simplemente reportará 0 coincidencias en la segunda
pasada para los bloques ya parcheados.
"""

from pathlib import Path

# Cada entrada: (archivo, texto_viejo, texto_nuevo, descripción)
PATCHES = [
    (
        "tests/test_api.py",
        '        assert "version" in data\n        assert "providers" in data',
        '        assert "version" in data\n        # "providers" no forma parte del contrato actual de /health',
        "test_health_returns_200: /health nunca expuso 'providers'",
    ),
    (
        "tests/test_api.py",
        '''    def test_health_has_provider_status(self):
        response = client.get("/health")
        providers = response.json()["providers"]
        assert "gemini" in providers
        assert "openrouter" in providers
        assert "groq" in providers
        assert "ollama" in providers''',
        '''    def test_health_has_provider_status(self):
        response = client.get("/health")
        data = response.json()
        assert "inference_mode" in data
        assert "cloud_allowed" in data''',
        "test_health_has_provider_status: verifica los campos reales de /health en vez de una lista de providers inexistente",
    ),
    (
        "tests/test_api.py",
        '        assert data["status"] == "online"\n        assert "router" in data',
        '        assert data["status"] == "online"\n        # "router" no forma parte del contrato actual de /api/nexus/status',
        "test_status_returns_200: /status nunca expuso 'router'",
    ),
    (
        "tests/test_api.py",
        '        assert "version" in data\n        assert "max_output_tokens" in data',
        '        assert "version" in data\n        assert "autonomy_enabled" in data\n        assert "knowledge_enabled" in data',
        "test_config_returns_200: max_output_tokens vive en /health, no en /config; se verifican los campos reales de /config",
    ),
    (
        "tests/test_offline.py",
        '''        import backend.config as cfg
        original = cfg.NEXUS_LOCAL_ONLY
        cfg.NEXUS_LOCAL_ONLY = True''',
        '''        # Se parchea backend.router.NEXUS_LOCAL_ONLY directamente (no
        # backend.config.NEXUS_LOCAL_ONLY): "from x import y" copia el
        # valor al importar, así que mutar el módulo config original no
        # se propaga a la variable ya importada dentro de router.py.
        import backend.router as router_module
        original = router_module.NEXUS_LOCAL_ONLY
        router_module.NEXUS_LOCAL_ONLY = True''',
        "test_local_mode_raises_if_no_local_provider: parchea el módulo correcto (router, no config)",
    ),
    (
        "tests/test_offline.py",
        '''        finally:
            cfg.NEXUS_LOCAL_ONLY = original''',
        '''        finally:
            router_module.NEXUS_LOCAL_ONLY = original''',
        "test_local_mode_raises_if_no_local_provider: restaura el módulo correcto",
    ),
    (
        "tests/test_phase2.py",
        '        assert data["version"] == APP_VERSION\n        assert "providers" in data',
        '        assert data["version"] == APP_VERSION\n        # "providers" no forma parte del contrato actual de /health',
        "test_health_returns_200 (phase2): /health nunca expuso 'providers'",
    ),
    (
        "tests/test_phase2.py",
        '        assert data["intent"] == "greeting"',
        '        # "greeting" nunca existió en IntentType; el saludo se clasifica como "chat"\n        assert data["intent"] == "chat"',
        "test_intent_endpoint: 'greeting' no es un IntentType válido, el saludo mapea a 'chat'",
    ),
    (
        "tests/test_phase2.py",
        '        assert data["strategy"] in ("llm", "tool")',
        '        # ANALYSIS está en AUTONOMY_INTENTS desde el refactor de AutonomyLoop;\n        # "estrategia"/"analiza" dispara la estrategia "autonomy", no "llm"/"tool"\n        assert data["strategy"] == "autonomy"',
        "test_intent_strategy_domain: comportamiento nuevo válido tras el refactor de autonomía",
    ),
    (
        "tests/test_phase3.py",
        '''        assert response.status_code == 200
        assert response.json()["version"] == "3.6.0"''',
        '''        assert response.status_code == 200
        from backend.config import APP_VERSION
        assert response.json()["version"] == APP_VERSION''',
        "test_health_returns_version: compara contra APP_VERSION real en vez de un valor hardcodeado obsoleto",
    ),
    (
        "tests/test_phase3.py",
        '        assert data["status"] == "healthy"\n        assert "providers" in data',
        '        assert data["status"] == "healthy"\n        # "providers" no forma parte del contrato actual de /health',
        "test_health_contract_preserved: /health nunca expuso 'providers'",
    ),
]

total_ok = 0
total_warn = 0

for relpath, old, new, desc in PATCHES:
    path = Path(relpath)
    if not path.exists():
        print(f"  SKIP (no existe): {relpath} — {desc}")
        total_warn += 1
        continue

    text = path.read_text(encoding="utf-8")
    count = text.count(old)

    if count == 0:
        print(f"  Sin coincidencias (¿ya parcheado?): {relpath} — {desc}")
        continue
    if count > 1:
        print(f"  ADVERTENCIA: {count} coincidencias (se esperaba 1), no se tocó: {relpath} — {desc}")
        total_warn += 1
        continue

    new_text = text.replace(old, new, 1)
    path.write_text(new_text, encoding="utf-8")
    print(f"  OK: {relpath} — {desc}")
    total_ok += 1

print(f"\nAplicados: {total_ok} / {len(PATCHES)}  (advertencias: {total_warn})")
