"""
Parche aislado para backend/tools/builtin.py: acota
StatusTool.intent_keywords, que usaba palabras sueltas muy comunes
("estado", "sistema", "operativo", "funcionando", "activo") capaces de
capturar conversación normal de negocios (dominio de NEXUS: gestión
empresarial, economía, estrategia) vía ToolRegistry.find_by_intent().

NO toca find_by_intent() ni ninguna otra herramienta: el algoritmo de
matching (substring simple) queda exactamente igual. Solo cambian los
strings candidatos de esta única herramienta, por frases compuestas
que exigen co-ocurrencia de palabras específicas ("estado" + "sistema"/
"nexus", "sistema"/"nexus" + "operativo", etc.), preservando las
consultas explícitas de estado y descartando palabras sueltas.

Uso:
    python patch_status_tool_keywords.py
"""

from pathlib import Path

PATH = Path("backend/tools/builtin.py")

OLD = '        return ["estado", "status", "sistema", "operativo", "funcionando", "activo"]'

NEW = '''        return [
            "estado del sistema", "estado de nexus", "estado de la ia",
            "estado operacional", "sistema operativo", "sistema está operativo",
            "sistema esta operativo", "nexus operativo", "nexus está operativo",
            "nexus esta operativo", "estás operativo", "estas operativo",
            "sigues activo", "sigues funcionando", "status del sistema",
        ]'''

if not PATH.exists():
    print(f"SKIP (no existe): {PATH}")
else:
    text = PATH.read_text(encoding="utf-8")
    count = text.count(OLD)

    if count == 0:
        print("Sin coincidencias (¿ya parcheado, o el archivo cambió?)")
    elif count > 1:
        print(f"ADVERTENCIA: {count} coincidencias (se esperaba 1), no se tocó nada.")
    else:
        new_text = text.replace(OLD, NEW, 1)
        PATH.write_text(new_text, encoding="utf-8")
        print("OK: StatusTool.intent_keywords acotado a frases compuestas.")
