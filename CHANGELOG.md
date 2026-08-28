# Changelog

## [3.3.0] — 2026-08-25

### Fase 1.5: Estabilización + Modularización

**Corregido:**
- 🔴 BUG: `fallback` siempre era `false` aunque se usara un provider alternativo.
  Ahora el router detecta correctamente si la respuesta vino de un fallback.
- 🔴 BUG: Se creaba un nuevo `httpx.AsyncClient` por cada request HTTP.
  Ahora se usa un cliente global compartido con connection pooling.
- 🟡 BUG: Gemini no tenía timeout. Ahora usa `asyncio.wait_for()` con
  `REQUEST_TIMEOUT_SECONDS`.
- 🟡 BUG: Ollama usaba el endpoint legacy `/api/generate` con prompt
  concatenado. Ahora usa `/api/chat` con mensajes estructurados.

**Añadido:**
- Soporte para historial de conversación (`history` en ChatRequest).
  El frontend puede enviar los últimos N mensajes para contexto multi-turn.
- `BaseModelProvider`: interfaz abstracta para providers. El core no
  conoce SDKs específicos.
- `ModelRouter`: router desacoplado con fallback tracking, detección de
  providers configurados vs disponibles.
- `backend/config.py`: configuración centralizada. Ningún otro módulo
  lee `os.getenv()`.
- `.gitignore`: protege secrets y artefactos.
- `requirements.txt` con version ranges (builds reproducibles).
- Tests: `test_core.py` (router, fallback, rate limit) +
  `test_api.py` (endpoints HTTP).
- Campo `model` en respuesta de chat (qué modelo específico se usó).
- `CHANGELOG.md`.

**Refactorizado:**
- `main.py` monolítico (~800 líneas) → estructura modular:
  - `backend/config.py` — configuración
  - `backend/router.py` — model router
  - `backend/providers/base.py` — interfaz abstracta
  - `backend/providers/gemini.py` — provider Gemini
  - `backend/providers/openrouter.py` — provider OpenRouter
  - `backend/providers/groq.py` — provider Groq
  - `backend/providers/ollama.py` — provider Ollama
  - `backend/main.py` — solo endpoints y lifespan

**Sin cambios:**
- Todos los endpoints mantienen la misma ruta y schema de respuesta.
- El frontend existente sigue funcionando sin modificaciones.
- No se eliminaron providers.
- No se cambió el proveedor principal (Gemini).

## [3.2.0] — Pre-auditoría

Backend monolítico en un solo archivo `main.py`.
4 providers: Gemini, OpenRouter, Groq, Ollama.
Frontend PWA con reconocimiento de voz.
