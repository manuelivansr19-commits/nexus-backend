"""
Segunda pasada del parche de contrato de API: corrige los 2 fallos que
quedaron después de la primera pasada.

1. tests/test_phase2.py::test_health_returns_200 — también esperaba
   "local_mode" en /health, campo que tampoco existe en el contrato
   actual (quedó oculto detrás del fallo de "providers" ya arreglado).

2. tests/test_offline.py::test_local_mode_raises_if_no_local_provider —
   el archivo tiene DOS tests que mutan NEXUS_LOCAL_ONLY; el otro
   (test_local_only_mode_skips_external) ya pasa y no debe tocarse.
   Este patch usa un bloque único (que incluye la clase
   "UnconfiguredLocal", exclusiva de este test) para no ambigüar con
   el otro.

Uso:
    python patch_api_contract_2.py
"""

from pathlib import Path

PATCHES = [
    (
        "tests/test_phase2.py",
        '        # "providers" no forma parte del contrato actual de /health\n        assert "local_mode" in data',
        '        # "providers" y "local_mode" no forman parte del contrato actual de /health',
        "test_health_returns_200 (phase2): /health tampoco expone 'local_mode'",
    ),
    (
        "tests/test_offline.py",
        '''        """LOCAL_ONLY sin provider local configurado → RuntimeError claro."""
        import backend.config as cfg
        original = cfg.NEXUS_LOCAL_ONLY
        cfg.NEXUS_LOCAL_ONLY = True

        try:
            from backend.providers.base import BaseModelProvider

            class UnconfiguredLocal(BaseModelProvider):
                is_local = True
                @property
                def name(self): return "local"
                @property
                def model(self): return "none"
                @property
                def is_configured(self): return False
                async def generate(self, req): raise RuntimeError("never")

            router = ModelRouter([UnconfiguredLocal()])
            with pytest.raises(RuntimeError, match="NEXUS_LOCAL_ONLY"):
                await router.generate(make_req())
        finally:
            cfg.NEXUS_LOCAL_ONLY = original''',
        '''        """LOCAL_ONLY sin provider local configurado → RuntimeError claro."""
        # Se parchea backend.router.NEXUS_LOCAL_ONLY directamente (no
        # backend.config.NEXUS_LOCAL_ONLY): "from x import y" copia el
        # valor al importar, así que mutar el módulo config original no
        # se propaga a la variable ya importada dentro de router.py.
        import backend.router as router_module
        original = router_module.NEXUS_LOCAL_ONLY
        router_module.NEXUS_LOCAL_ONLY = True

        try:
            from backend.providers.base import BaseModelProvider

            class UnconfiguredLocal(BaseModelProvider):
                is_local = True
                @property
                def name(self): return "local"
                @property
                def model(self): return "none"
                @property
                def is_configured(self): return False
                async def generate(self, req): raise RuntimeError("never")

            router = ModelRouter([UnconfiguredLocal()])
            with pytest.raises(RuntimeError, match="NEXUS_LOCAL_ONLY"):
                await router.generate(make_req())
        finally:
            router_module.NEXUS_LOCAL_ONLY = original''',
        "test_local_mode_raises_if_no_local_provider: parchea el módulo correcto (router, no config), sin ambigüar con el otro test",
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
