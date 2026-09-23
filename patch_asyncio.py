"""
Parche para tests/*.py: reemplaza el patrón obsoleto

    asyncio.get_event_loop().run_until_complete(...)

por el moderno y correcto

    asyncio.run(...)

En Python 3.10+ asyncio.get_event_loop() está deprecado cuando no hay
un loop corriendo en el hilo actual, y en Python 3.14 directamente
lanza RuntimeError en vez de crear uno silenciosamente. asyncio.run()
es la forma correcta de ejecutar una corrutina desde código síncrono
y funciona igual en todas las versiones recientes de Python.

Uso:
    python patch_asyncio.py

Es idempotente: si vuelves a correrlo, no vuelve a tocar líneas ya
parcheadas (ya no contienen el patrón viejo).
"""

from pathlib import Path

OLD = "asyncio.get_event_loop().run_until_complete("
NEW = "asyncio.run("

FILES = [
    "tests/test_phase3.py",
    "tests/test_phase4.py",
]

total_changes = 0

for relpath in FILES:
    path = Path(relpath)
    if not path.exists():
        print(f"  SKIP (no existe): {relpath}")
        continue

    text = path.read_text(encoding="utf-8")
    count = text.count(OLD)

    if count == 0:
        print(f"  Sin cambios (ya parcheado o sin coincidencias): {relpath}")
        continue

    new_text = text.replace(OLD, NEW)
    path.write_text(new_text, encoding="utf-8")
    print(f"  OK: {relpath} — {count} ocurrencia(s) reemplazada(s)")
    total_changes += count

print(f"\nTotal de reemplazos: {total_changes}")
