"""
Parche mínimo y aislado para backend/tools/builtin.py: corrige la
versión hardcodeada "v3.5.0" en StatusTool.execute(), que quedó
desactualizada (la versión real actual es APP_VERSION = "3.8.0").

Alcance deliberadamente acotado (v3.8 Bloque 5, investigación de
"System v3.5 operativo"):
  - Solo cambia el string de versión hardcodeado por APP_VERSION real.
  - NO toca StatusTool.intent_keywords ni la lógica de matching de
    ToolRegistry.find_by_intent(): ese es un problema de diseño más
    amplio (palabras clave comunes del español que pueden capturar
    mensajes conversacionales normales), fuera del alcance autorizado
    de este bloque, y se reporta por separado en vez de tocarse aquí.

Uso:
    python patch_status_tool_version.py
"""

from pathlib import Path

PATH = Path("backend/tools/builtin.py")

OLD_IMPORT = "from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult"
NEW_IMPORT = (
    "from backend.config import APP_VERSION\n"
    "from backend.tools.base import BaseTool, RiskLevel, ToolInput, ToolResult"
)

OLD_OUTPUT = 'output="NEXUS Ω v3.5.0 operativo. Todos los subsistemas activos.",'
NEW_OUTPUT = 'output=f"NEXUS Ω v{APP_VERSION} operativo. Todos los subsistemas activos.",'

if not PATH.exists():
    print(f"SKIP (no existe): {PATH}")
else:
    text = PATH.read_text(encoding="utf-8")
    changes = 0

    if OLD_IMPORT in text and "from backend.config import APP_VERSION" not in text:
        text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)
        changes += 1
        print("OK: import de APP_VERSION agregado")
    else:
        print("Sin cambios en el import (ya presente o patrón no encontrado)")

    if OLD_OUTPUT in text:
        text = text.replace(OLD_OUTPUT, NEW_OUTPUT, 1)
        changes += 1
        print("OK: string de versión hardcodeado reemplazado por APP_VERSION")
    else:
        print("Sin cambios en el output (¿ya parcheado?)")

    if changes:
        PATH.write_text(text, encoding="utf-8")

    print(f"\nTotal de cambios aplicados: {changes} / 2")
