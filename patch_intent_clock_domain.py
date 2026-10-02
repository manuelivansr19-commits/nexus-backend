"""
Parche aislado para backend/core/intent.py: corrige la condición que
forzaba strategy=TOOL/clock ante cualquier mensaje clasificado como
Domain.TIME, incluyendo mensajes sobre "hoy", "mañana", "ayer", "fecha"
que no tienen relación con una consulta de reloj.

Causa raíz confirmada (ver diagnóstico de sesión):
  backend/core/intent.py, IntentRouter.route():

      elif domain == Domain.TIME or any(w in lower for w in ["hora","fecha"]):
          candidate_tools = ["clock"]
          requires_tool   = True
          strategy        = IntentStrategy.TOOL

  Domain.TIME se activa con cualquiera de ["hora","tiempo","fecha","dia",
  "hoy","manana"] — mucho más amplio que una consulta real de hora. Un
  mensaje como "¿Cuáles fueron las noticias del día de hoy?" terminaba
  forzado a strategy=TOOL/clock solo por contener "hoy".

Corrección (mínima, según regla arquitectónica: Domain.TIME ≠
automáticamente consulta de reloj):
  - Se elimina el disparador por `domain == Domain.TIME` (demasiado
    amplio).
  - Se elimina "fecha" de las palabras disparadoras (es temporalidad
    contextual, no consulta de reloj, según la propia categorización
    del plan).
  - Se conserva únicamente "hora" como disparador explícito de clock.

NO se toca:
  - _DOMAIN_KEYWORDS ni _score_domain(): Domain.TIME sigue existiendo
    y clasificando exactamente igual que antes.
  - TaskStateManager, ContextManager, NexusCore, ToolRegistry: ninguno
    interviene en este bug (confirmado en el diagnóstico).

Uso:
    python patch_intent_clock_domain.py
"""

from pathlib import Path

PATH = Path("backend/core/intent.py")

OLD = '        elif domain == Domain.TIME or any(w in lower for w in ["hora","fecha"]):'
NEW = '        elif "hora" in lower:'

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
        print("OK: condición de clock corregida (ya no se dispara por Domain.TIME amplio).")
