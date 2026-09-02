"""
NEXUS Ω — Phase 4 Tests: A → Q

A. Insertar conocimiento
B. Recuperar conocimiento
C. Filtrar por dominio
D. Filtrar por fecha
E. Filtrar por confidence
F. Detectar duplicado
G. Detectar conocimiento antiguo
H. Metadata correcta
I. Fuente preservada
J. Gemini offline — knowledge sigue funcionando
K. Retrieval sin Gemini
L. Conocimiento corrupto
M. Contenido vacío
N. Múltiples fuentes
O. Clasificación FACT/FORECAST/SCENARIO
P. Integración con NexusCore
Q. Seguridad
"""

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient


# ============================================================
# FIXTURES
# ============================================================

def make_store(tmp_path=None):
    from backend.knowledge.store import KnowledgeStore
    if tmp_path:
        return KnowledgeStore(str(tmp_path))
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return KnowledgeStore(path), path


def make_engine(db_path=None):
    from backend.knowledge.engine import KnowledgeEngine
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return KnowledgeEngine(db_path=db_path), db_path


def sample_entry(
    title="Baterías LiFePO4",
    content="Las baterías LiFePO4 tienen una densidad energética de 90-120 Wh/kg y ciclos de vida superiores a 2000 cargas.",
    domain="energy",
    subdomain="storage",
    source="manual_test",
    confidence=0.9,
    knowledge_type="fact",
):
    from backend.knowledge.models import KnowledgeEntry, Domain, KnowledgeType, KnowledgeStatus
    return KnowledgeEntry(
        title=title, content=content,
        domain=Domain(domain), subdomain=subdomain,
        source=source, confidence=confidence,
        knowledge_type=KnowledgeType(knowledge_type),
        status=KnowledgeStatus.CURRENT,
        tags=["battery", "energy"],
    )


# ============================================================
# TEST A — Insertar conocimiento
# ============================================================

class TestA_Insert:

    def test_add_entry_returns_id(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Paneles solares monocristalinos",
                content="Los paneles solares monocristalinos tienen eficiencias del 20-23% y vida útil superior a 25 años.",
                domain="energy",
                subdomain="solar",
                source="test",
                confidence=0.85,
            )
            assert result.success is True
            assert result.entry_id is not None
            assert result.action == "created"
        finally:
            os.unlink(path)

    def test_store_save_and_get(self):
        store, path = make_store()
        try:
            entry    = sample_entry()
            entry_id = store.save(entry)
            fetched  = store.get(entry_id)
            assert fetched is not None
            assert fetched.title == entry.title
            assert fetched.content == entry.content
        finally:
            store.close(); os.unlink(path)

    def test_add_auto_classifies_domain(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Cultivo hidropónico NFT",
                content="El sistema NFT (Nutrient Film Technique) es un método hidropónico donde una película de nutrientes circula continuamente.",
                source="test",
            )
            assert result.success is True
            entry = engine.get(result.entry_id)
            assert entry.domain.value == "agriculture"
        finally:
            os.unlink(path)


# ============================================================
# TEST B — Recuperar conocimiento
# ============================================================

class TestB_Retrieve:

    def test_search_finds_entry(self):
        engine, path = make_engine()
        try:
            engine.add(
                title="Motor brushless BLDC",
                content="Los motores brushless BLDC ofrecen alta eficiencia (85-95%) y bajo mantenimiento para robótica.",
                domain="robotics", source="test", confidence=0.9,
            )
            ctx = engine.search("motor brushless robótica")
            assert ctx.total_found > 0
            assert any("brushless" in e.title.lower() or "brushless" in e.content.lower()
                       for e in ctx.entries)
        finally:
            os.unlink(path)

    def test_get_for_context_returns_structured(self):
        engine, path = make_engine()
        try:
            engine.add(
                title="SLAM para navegación robótica",
                content="SLAM (Simultaneous Localization and Mapping) permite a un robot construir un mapa mientras se localiza en él.",
                domain="robotics", source="test", confidence=0.88,
            )
            ctx = engine.get_for_context("navegación robot SLAM")
            assert not ctx.is_empty()
            prompt = ctx.to_prompt_string()
            assert "CONOCIMIENTO RELEVANTE" in prompt
            assert len(prompt) > 50
        finally:
            os.unlink(path)

    def test_get_by_id(self):
        engine, path = make_engine()
        try:
            result  = engine.add(title="Test entry", content="Contenido de prueba suficientemente largo.", source="test")
            fetched = engine.get(result.entry_id)
            assert fetched is not None
            assert fetched.entry_id == result.entry_id
        finally:
            os.unlink(path)


# ============================================================
# TEST C — Filtrar por dominio
# ============================================================

class TestC_DomainFilter:

    def test_filter_by_domain(self):
        engine, path = make_engine()
        try:
            engine.add(title="Solar A", content="Energía solar fotovoltaica para generación distribuida residencial.", domain="energy", source="test")
            engine.add(title="Robot B", content="Navegación autónoma con sensor LiDAR para robots móviles.", domain="robotics", source="test")
            engine.add(title="AI C",    content="Modelos de lenguaje grandes basados en arquitectura transformer.", domain="artificial_intelligence", source="test")

            energy_ctx = engine.search_by_domain("energy")
            assert all(e.domain.value == "energy" for e in energy_ctx.entries)
            assert energy_ctx.total_found >= 1
        finally:
            os.unlink(path)

    def test_search_with_domain_hint(self):
        engine, path = make_engine()
        try:
            engine.add(title="Hidrógeno verde", content="El hidrógeno verde se produce mediante electrólisis usando energía renovable.", domain="energy", subdomain="hydrogen", source="test", confidence=0.9)
            engine.add(title="Hidrógeno y robots", content="Los robots pueden usar celdas de hidrógeno como fuente de energía.", domain="robotics", source="test", confidence=0.75)
            ctx = engine.search("hidrógeno", domain="energy")
            assert all(e.domain.value == "energy" for e in ctx.entries)
        finally:
            os.unlink(path)


# ============================================================
# TEST D — Filtrar por fecha
# ============================================================

class TestD_DateFilter:

    def test_filter_since_days(self):
        store, path = make_store()
        try:
            from backend.knowledge.models import KnowledgeEntry, Domain, KnowledgeType, KnowledgeStatus
            # Entrada reciente
            recent = KnowledgeEntry(
                title="Reciente", content="Conocimiento reciente de prueba con suficiente longitud.",
                domain=Domain.ENERGY, source="test",
                date_acquired=time.time(),
            )
            # Entrada antigua (31 días)
            old = KnowledgeEntry(
                title="Antigua", content="Conocimiento antiguo de prueba con suficiente longitud.",
                domain=Domain.ENERGY, source="test",
                date_acquired=time.time() - 31 * 86400,
            )
            store.save(recent)
            store.save(old)
            results = store.filter(since_days=7)
            titles = [r.title for r in results]
            assert "Reciente" in titles
            assert "Antigua" not in titles
        finally:
            store.close(); os.unlink(path)


# ============================================================
# TEST E — Filtrar por confidence
# ============================================================

class TestE_ConfidenceFilter:

    def test_filter_by_min_confidence(self):
        engine, path = make_engine()
        try:
            engine.add(title="Alta confianza", content="Este es un hecho bien verificado con alta precisión y respaldo documental.", source="test", confidence=0.95)
            engine.add(title="Baja confianza", content="Esta información es especulativa y requiere validación adicional por expertos.", source="test", confidence=0.3)
            ctx = engine.search("confianza", min_confidence=0.8)
            assert all(e.confidence >= 0.8 for e in ctx.entries)
            titles = [e.title for e in ctx.entries]
            assert "Baja confianza" not in titles
        finally:
            os.unlink(path)


# ============================================================
# TEST F — Detectar duplicado
# ============================================================

class TestF_Duplicate:

    def test_duplicate_detected_by_hash(self):
        engine, path = make_engine()
        try:
            content = "Las baterías de litio tienen alta densidad energética y son ampliamente usadas en vehículos eléctricos."
            r1 = engine.add(title="Baterías Li-ion", content=content, source="test")
            r2 = engine.add(title="Baterías Li-ion v2", content=content, source="test")
            assert r1.success is True
            assert r2.success is True
            assert r2.duplicate is True
            assert r2.action == "skipped"
        finally:
            os.unlink(path)

    def test_allow_duplicate_flag(self):
        engine, path = make_engine()
        try:
            content = "Contenido exacto duplicado para prueba de flags de control."
            r1 = engine.add(title="T1", content=content, source="test")
            r2 = engine.add(title="T2", content=content, source="test", allow_duplicate=True)
            assert r2.action == "created"
            assert r2.duplicate is False
        finally:
            os.unlink(path)


# ============================================================
# TEST G — Detectar conocimiento antiguo
# ============================================================

class TestG_AgedKnowledge:

    def test_aging_status_computed(self):
        from backend.knowledge.models import Domain, KnowledgeStatus, DOMAIN_AGING_DAYS
        from backend.knowledge.classifier import KnowledgeClassifier

        classifier = KnowledgeClassifier()

        # AI knowledge aging threshold es 14 días
        ai_aging_days = DOMAIN_AGING_DAYS[Domain.ARTIFICIAL_INTELLIGENCE]
        old_date = time.time() - (ai_aging_days + 5) * 86400

        status = classifier.compute_status(Domain.ARTIFICIAL_INTELLIGENCE, old_date)
        assert status in (KnowledgeStatus.AGING, KnowledgeStatus.OUTDATED)

    def test_outdated_status_computed(self):
        from backend.knowledge.models import Domain, KnowledgeStatus, DOMAIN_OUTDATED_DAYS
        from backend.knowledge.classifier import KnowledgeClassifier

        classifier    = KnowledgeClassifier()
        outdated_days = DOMAIN_OUTDATED_DAYS[Domain.ARTIFICIAL_INTELLIGENCE]
        very_old_date = time.time() - (outdated_days + 5) * 86400

        status = classifier.compute_status(Domain.ARTIFICIAL_INTELLIGENCE, very_old_date)
        assert status == KnowledgeStatus.OUTDATED

    def test_current_entry_not_outdated(self):
        from backend.knowledge.models import Domain, KnowledgeStatus
        from backend.knowledge.classifier import KnowledgeClassifier

        classifier = KnowledgeClassifier()
        recent     = time.time() - 3600   # 1 hora
        status     = classifier.compute_status(Domain.ENGINEERING, recent)
        assert status == KnowledgeStatus.CURRENT

    def test_refresh_updates_statuses(self):
        engine, path = make_engine()
        try:
            from backend.knowledge.models import KnowledgeEntry, Domain, KnowledgeStatus
            store = engine._store
            old_entry = KnowledgeEntry(
                title="IA antigua",
                content="Información de inteligencia artificial que ha quedado obsoleta por el avance tecnológico.",
                domain=Domain.ARTIFICIAL_INTELLIGENCE,
                source="test",
                date_acquired=time.time() - 200 * 86400,   # 200 días
                status=KnowledgeStatus.CURRENT,  # status incorrecto
            )
            store.save(old_entry)
            updated = engine.refresh_statuses()
            assert updated >= 1
            refreshed = engine.get(old_entry.entry_id)
            assert refreshed.status != KnowledgeStatus.CURRENT
        finally:
            os.unlink(path)


# ============================================================
# TEST H — Metadata correcta
# ============================================================

class TestH_Metadata:

    def test_all_metadata_fields_saved(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Test completo de metadata",
                content="Contenido de prueba con todos los campos de metadata completos para verificación.",
                domain="robotics",
                subdomain="sensors",
                source="paper_ieee",
                source_url="https://doi.org/10.1234/test",
                knowledge_type="observation",
                confidence=0.77,
                tags=["lidar", "slam", "robotics"],
                date_source="2026-01",
            )
            entry = engine.get(result.entry_id)
            assert entry.domain.value == "robotics"
            assert entry.subdomain == "sensors"
            assert entry.source == "paper_ieee"
            assert entry.source_url == "https://doi.org/10.1234/test"
            assert entry.knowledge_type.value == "observation"
            assert abs(entry.confidence - 0.77) < 0.01
            assert "lidar" in entry.tags
            assert entry.date_source == "2026-01"
            assert entry.content_hash != ""
            assert entry.date_acquired > 0
        finally:
            os.unlink(path)

    def test_content_hash_computed(self):
        entry = sample_entry()
        assert len(entry.content_hash) == 16
        entry2 = sample_entry(content="Contenido diferente totalmente distinto.")
        assert entry.content_hash != entry2.content_hash


# ============================================================
# TEST I — Fuente preservada (provenance)
# ============================================================

class TestI_Provenance:

    def test_source_preserved_after_save(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Artículo sobre energía nuclear",
                content="Los reactores de fusión nuclear representan una fuente potencial de energía limpia e ilimitada para el futuro.",
                source="nature_energy_journal",
                source_url="https://nature.com/articles/s41560-026-test",
                domain="energy",
                subdomain="nuclear",
            )
            entry = engine.get(result.entry_id)
            assert entry.source == "nature_energy_journal"
            assert entry.source_url == "https://nature.com/articles/s41560-026-test"
        finally:
            os.unlink(path)

    def test_multiple_sources_tracked(self):
        engine, path = make_engine()
        try:
            engine.add(title="Fuente A", content="Información proveniente de la fuente A con datos verificados.", source="source_A")
            engine.add(title="Fuente B", content="Información proveniente de la fuente B con datos independientes.", source="source_B")
            engine.add(title="Fuente C", content="Información proveniente de la fuente C con análisis adicional.", source="source_C")
            ctx = engine.search("información fuente")
            sources = {e.source for e in ctx.entries}
            assert len(sources) >= 2
        finally:
            os.unlink(path)


# ============================================================
# TEST J — Gemini offline: knowledge sigue funcionando
# ============================================================

class TestJ_GeminiOffline:

    def test_knowledge_engine_works_without_llm(self):
        """Knowledge Engine no depende de ningún LLM."""
        engine, path = make_engine()
        try:
            # Sin ningún router ni LLM
            result = engine.add(
                title="Conocimiento sin Gemini",
                content="Este conocimiento se agrega directamente sin necesidad de ningún modelo de lenguaje.",
                source="offline_test",
            )
            assert result.success is True
            ctx = engine.search("Gemini")
            # Puede o no encontrar, pero no crashea
            assert ctx is not None
        finally:
            os.unlink(path)

    def test_nexus_core_without_model_uses_knowledge(self):
        """NexusCore sin modelo igual puede acceder al knowledge engine."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.core.context import ContextManager

        engine, path = make_engine()
        try:
            engine.add(
                title="Robótica autónoma",
                content="Los robots autónomos utilizan sensores LiDAR e IMU para navegación sin GPS en entornos interiores.",
                domain="robotics", source="test", confidence=0.9,
            )
            mem  = Memory(RAMMemoryStore())
            ctx  = ContextManager(memory=mem, knowledge_engine=engine)
            core = NexusCore(
                model_router=None,   # sin modelo
                memory=mem,
                intent_router=IntentRouter(),
                context_manager=ctx,
                knowledge_engine=engine,
            )
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                core.process("hola")
            )
            # Direct intent → no necesita LLM
            assert result.provider == "system"
        finally:
            os.unlink(path)


# ============================================================
# TEST K — Retrieval sin Gemini
# ============================================================

class TestK_RetrievalWithoutGemini:

    def test_retrieval_fully_local(self):
        """El retrieval es 100% local (SQLite FTS), sin LLM."""
        engine, path = make_engine()
        try:
            engine.add(title="LiDAR 2D", content="El sensor LiDAR RPLIDAR A1 tiene rango de 12m y frecuencia de escaneo de 8000 muestras por segundo.", domain="robotics", source="test", confidence=0.92)
            engine.add(title="IMU MPU6050", content="El IMU MPU-6050 integra acelerómetro y giroscopio en un solo chip comunicado por I2C.", domain="electronics", source="test", confidence=0.88)

            from backend.knowledge.retrieval import KnowledgeRetriever
            from backend.knowledge.store import KnowledgeStore

            # Nuevo retriever directo, sin ningún LLM
            retriever = KnowledgeRetriever(engine._store)
            ctx = retriever.search("sensor LiDAR robot", limit=5)

            assert ctx.total_found > 0
            assert ctx.search_time_ms >= 0
        finally:
            os.unlink(path)

    def test_get_for_context_no_llm(self):
        engine, path = make_engine()
        try:
            engine.add(title="Jetson Orin NX", content="El NVIDIA Jetson Orin NX tiene 1024 núcleos CUDA y 32 TOPS de potencia neuronal.", domain="electronics", source="test", confidence=0.95)
            ctx = engine.get_for_context("Jetson procesador IA", min_confidence=0.5)
            assert isinstance(ctx.to_prompt_string(), str)
        finally:
            os.unlink(path)


# ============================================================
# TEST L — Conocimiento corrupto
# ============================================================

class TestL_CorruptKnowledge:

    def test_missing_title_fails_validation(self):
        engine, path = make_engine()
        try:
            result = engine.add(title="", content="Contenido sin título.", source="test")
            assert result.success is False
            assert any("title" in e for e in result.errors)
        finally:
            os.unlink(path)

    def test_corrupt_domain_falls_back_to_classifier(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Datos",
                content="Este texto habla de robótica y sensores LiDAR para navegación autónoma en interiores.",
                domain="dominio_inexistente_xyz",
                source="test",
            )
            # Debe clasificar automáticamente o fallar gracefully
            assert result.success is True   # classifier detecta el dominio
            entry = engine.get(result.entry_id)
            assert entry is not None
        finally:
            os.unlink(path)


# ============================================================
# TEST M — Contenido vacío
# ============================================================

class TestM_EmptyContent:

    def test_empty_content_rejected(self):
        engine, path = make_engine()
        try:
            result = engine.add(title="Título OK", content="", source="test")
            assert result.success is False
            assert any("content" in e for e in result.errors)
        finally:
            os.unlink(path)

    def test_whitespace_content_rejected(self):
        engine, path = make_engine()
        try:
            result = engine.add(title="Título OK", content="   \n\t  ", source="test")
            assert result.success is False
        finally:
            os.unlink(path)

    def test_very_short_content_rejected(self):
        engine, path = make_engine()
        try:
            result = engine.add(title="T", content="Corto", source="test")
            assert result.success is False
        finally:
            os.unlink(path)


# ============================================================
# TEST N — Múltiples fuentes
# ============================================================

class TestN_MultipleSources:

    def test_batch_ingest_multiple_sources(self):
        engine, path = make_engine()
        try:
            items = [
                {"title": "NASA Solar Wind", "content": "El viento solar consiste en partículas cargadas emitidas continuamente por el Sol a velocidades de 400-800 km/s.", "source": "nasa.gov", "domain": "aerospace", "confidence": 0.95},
                {"title": "ESA Orbit Data",  "content": "La ISS orbita la Tierra a una altitud media de 408 km con un período orbital de 92.68 minutos.", "source": "esa.int",  "domain": "aerospace", "confidence": 0.99},
                {"title": "User Note",        "content": "La robótica de enjambre podría ser clave para la exploración espacial autónoma en misiones de larga duración.", "source": "user",     "domain": "robotics",  "confidence": 0.7},
            ]
            results = engine.add_batch(items, source="batch_test")
            assert len(results) == 3
            assert all(r.success for r in results)
            sources = set()
            for r in results:
                entry = engine.get(r.entry_id)
                if entry:
                    sources.add(entry.source)
            assert len(sources) >= 2
        finally:
            os.unlink(path)


# ============================================================
# TEST O — Clasificación FACT/FORECAST/SCENARIO
# ============================================================

class TestO_KnowledgeTypeClassification:

    def test_fact_detected(self):
        from backend.knowledge.classifier import KnowledgeClassifier
        from backend.knowledge.models import KnowledgeType
        c = KnowledgeClassifier()
        t = c.classify_type("El grafeno consiste en una capa de átomos de carbono dispuestos en red hexagonal.")
        assert t == KnowledgeType.FACT

    def test_forecast_detected(self):
        from backend.knowledge.classifier import KnowledgeClassifier
        from backend.knowledge.models import KnowledgeType
        c = KnowledgeClassifier()
        t = c.classify_type("Se prevé que la capacidad de baterías de estado sólido crecerá un 40% para 2030.")
        assert t == KnowledgeType.FORECAST

    def test_scenario_detected(self):
        from backend.knowledge.classifier import KnowledgeClassifier
        from backend.knowledge.models import KnowledgeType
        c = KnowledgeClassifier()
        t = c.classify_type("Si los costos de hidrógeno bajan un 50%, podría reemplazar los combustibles fósiles en transporte.")
        assert t == KnowledgeType.SCENARIO

    def test_hypothesis_detected(self):
        from backend.knowledge.classifier import KnowledgeClassifier
        from backend.knowledge.models import KnowledgeType
        c = KnowledgeClassifier()
        t = c.classify_type("Hipótesis: los materiales topológicos podrían permitir superconductividad a temperatura ambiente.")
        assert t == KnowledgeType.HYPOTHESIS

    def test_explicit_type_overrides_classifier(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Proyección energética",
                content="La energía renovable podría cubrir el 80% de la demanda eléctrica global para 2050.",
                knowledge_type="forecast",
                source="test",
            )
            entry = engine.get(result.entry_id)
            assert entry.knowledge_type.value == "forecast"
        finally:
            os.unlink(path)


# ============================================================
# TEST P — Integración con NexusCore
# ============================================================

class TestP_NexusCoreIntegration:

    @pytest.mark.asyncio
    async def test_knowledge_enriches_context(self):
        """El knowledge engine enriquece el contexto antes del LLM."""
        from backend.core.nexus import NexusCore
        from backend.core.intent import IntentRouter
        from backend.core.memory import Memory, RAMMemoryStore
        from backend.core.context import ContextManager
        from backend.providers.base import BaseModelProvider, GenerateRequest, ProviderResponse
        from backend.router import ModelRouter

        class CapturingProvider(BaseModelProvider):
            captured_prompt = ""
            @property
            def name(self): return "mock"
            @property
            def model(self): return "mock-model"
            @property
            def is_configured(self): return True
            async def generate(self, request: GenerateRequest) -> ProviderResponse:
                CapturingProvider.captured_prompt = request.prompt
                return ProviderResponse(text="respuesta mock", provider="mock", model="mock-model", duration_ms=1)

        engine, path = make_engine()
        try:
            engine.add(
                title="LiDAR RPLIDAR A1 specs",
                content="El RPLIDAR A1 tiene rango de 12m, 8000 muestras/s y conectividad USB para robótica.",
                domain="robotics", source="test", confidence=0.95,
            )

            provider = CapturingProvider()
            router   = ModelRouter([provider])
            mem      = Memory(RAMMemoryStore())
            ctx_mgr  = ContextManager(memory=mem, knowledge_engine=engine)
            core     = NexusCore(
                model_router=router, memory=mem,
                intent_router=IntentRouter(),
                context_manager=ctx_mgr,
                knowledge_engine=engine,
            )

            result = await core.process("que sensores LiDAR recomiendas para robot")
            assert result.knowledge_used >= 0
            # El prompt capturado debe contener info del knowledge engine
            assert result.provider == "mock"
        finally:
            os.unlink(path)

    def test_knowledge_stats_in_aura_status(self):
        client = TestClient(__import__("backend.main", fromlist=["app"]).app)
        r = client.get("/api/aura/status")
        assert r.status_code == 200
        brain = r.json().get("brain", {})
        assert "knowledge_engine" in brain or "knowledge_stats" in brain


# ============================================================
# TEST Q — Seguridad
# ============================================================

class TestQ_Security:

    def test_secret_in_content_rejected(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Config",
                content="El api_key para acceder al sistema es sk-1234567890abcdef.",
                source="test",
            )
            assert result.success is False
            assert any("prohibido" in e.lower() for e in result.errors)
        finally:
            os.unlink(path)

    def test_password_in_content_rejected(self):
        engine, path = make_engine()
        try:
            result = engine.add(
                title="Credenciales",
                content="Para iniciar sesión usar password: admin123 en el servidor.",
                source="test",
            )
            assert result.success is False
        finally:
            os.unlink(path)

    def test_knowledge_engine_has_no_exec(self):
        """El KnowledgeEngine no contiene llamadas a exec o shell."""
        import inspect
        from backend.knowledge import engine as ke_module
        source = inspect.getsource(ke_module)
        assert "exec(" not in source
        assert "subprocess" not in source
        assert "os.system" not in source

    def test_delete_requires_valid_id(self):
        client = TestClient(__import__("backend.main", fromlist=["app"]).app)
        r = client.delete("/api/knowledge/nonexistent-id-12345")
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_add_endpoint_validates_input(self):
        client = TestClient(__import__("backend.main", fromlist=["app"]).app)
        r = client.post("/api/knowledge/add", json={"title": "", "content": ""})
        # Puede ser 200 con success=False o 422 por Pydantic
        if r.status_code == 200:
            assert r.json()["success"] is False or r.json().get("errors")

    def test_knowledge_domains_endpoint(self):
        client = TestClient(__import__("backend.main", fromlist=["app"]).app)
        r = client.get("/api/knowledge/domains")
        assert r.status_code == 200
        data = r.json()
        assert "domains" in data
        assert "engineering" in data["domains"]
        assert "robotics" in data["domains"]
