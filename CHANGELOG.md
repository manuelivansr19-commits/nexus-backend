# Changelog

## [3.7.0] — 2026-08-30

### Fase 4: Knowledge Engine

**Nuevo — backend/knowledge/ (7 archivos):**
- `models.py` — KnowledgeEntry, KnowledgeType (FACT/OBSERVATION/ESTIMATE/FORECAST/SCENARIO/HYPOTHESIS), KnowledgeStatus (CURRENT/AGING/OUTDATED/REQUIRES_REVIEW), Domain (8 dominios), KnowledgeContext
- `store.py` — KnowledgeStore SQLite + FTS5 con índices por dominio/tipo/confianza/fecha
- `classifier.py` — KnowledgeClassifier: detección de dominio, tipo, subdominio, aging automático
- `retrieval.py` — KnowledgeRetriever: FTS, domain filter, recent, semantic (interfaz futura)
- `ingestion.py` — KnowledgeIngestion: validación → clasificación → deduplicación → guardado
- `engine.py` — KnowledgeEngine: orquestador principal, provider agnostic
- `__init__.py` — package init

**Actualizado:**
- `core/nexus.py` — integra KnowledgeEngine en el pipeline
- `core/context.py` — incluye conocimiento en el contexto del LLM
- `main.py` — 7 nuevos endpoints Knowledge
- `config.py` — KNOWLEDGE_DB_PATH, KNOWLEDGE_MAX_RESULTS, etc.

**Endpoints nuevos:**
- `POST /api/knowledge/add` — agregar conocimiento
- `POST /api/knowledge/search` — buscar con filtros
- `GET  /api/knowledge/domains` — dominios y subdominios
- `GET  /api/knowledge/stats` — estadísticas
- `GET  /api/knowledge/{id}` — obtener entrada
- `DELETE /api/knowledge/{id}` — eliminar entrada
- `POST /api/knowledge/refresh` — actualizar statuses por edad

**Tests A→Q:** `tests/test_phase4.py` — 30+ tests

**Sin cambios:**
- `/api/nexus/chat` y contrato JSON: inalterados
- Frontend: inalterado

---

## [3.6.0] — 2026-08-29 — Autonomy Core
## [3.5.0] — 2026-08-28 — NEXUS Core
## [3.4.0] — 2026-08-25 — AURA Brain
## [3.3.0] — 2026-08-25 — Fase 1.5
## [3.2.0] — Pre-auditoría
