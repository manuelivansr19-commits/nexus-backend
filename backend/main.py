"""NEXUS Omega -- main.py v3.8.0"""
from __future__ import annotations
import time, uuid
from contextlib import asynccontextmanager
from typing import Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from backend.config import (
    ALL_SECRETS, ALLOW_CLOUD_INFERENCE, APP_VERSION, AUTONOMY_ENABLED,
    DEFAULT_SYSTEM, INFERENCE_MODE, KNOWLEDGE_DB_PATH, KNOWLEDGE_CONTEXT_LIMIT,
    KNOWLEDGE_MIN_CONFIDENCE, MAX_EXECUTION_LOOPS, MAX_HISTORY_TURNS,
    MAX_OUTPUT_TOKENS, MAX_PLAN_STEPS, MAX_RETRIES_PER_STEP,
    MEMORY_DB_PATH, REQUEST_TIMEOUT_SECONDS, USE_OLLAMA_ONLY, logger,
)
from backend.inference import (
    InferenceLayer, InferenceMode, InferencePolicy, InferenceRegistry,
    OllamaRuntime, LlamaCppRuntime, GeminiRuntime, GroqRuntime, OpenRouterRuntime,
)
from backend.core.nexus import NexusCore
from backend.core.intent import IntentRouter
from backend.core.context import ContextManager
from backend.core.memory import Memory, MemoryType, SQLiteMemoryStore, RAMMemoryStore
from backend.core.executor import Executor
from backend.core.evaluation import Evaluator
from backend.core.planner import Planner
from backend.core.evaluator import StepEvaluator, PlanEvaluator
from backend.core.autonomy import AutonomyLoop
fr
@'
"""NEXUS Omega -- main.py v3.8.0"""
from __future__ import annotations
import time, uuid
from contextlib import asynccontextmanager
from typing import Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from backend.config import (
    ALL_SECRETS, ALLOW_CLOUD_INFERENCE, APP_VERSION, AUTONOMY_ENABLED,
    DEFAULT_SYSTEM, INFERENCE_MODE, KNOWLEDGE_DB_PATH, KNOWLEDGE_CONTEXT_LIMIT,
    KNOWLEDGE_MIN_CONFIDENCE, MAX_EXECUTION_LOOPS, MAX_HISTORY_TURNS,
    MAX_OUTPUT_TOKENS, MAX_PLAN_STEPS, MAX_RETRIES_PER_STEP,
    MEMORY_DB_PATH, REQUEST_TIMEOUT_SECONDS, USE_OLLAMA_ONLY, logger,
)
from backend.inference import (
    InferenceLayer, InferenceMode, InferencePolicy, InferenceRegistry,
    OllamaRuntime, LlamaCppRuntime, GeminiRuntime, GroqRuntime, OpenRouterRuntime,
)
from backend.core.nexus import NexusCore
from backend.core.intent import IntentRouter
from backend.core.context import ContextManager
from backend.core.memory import Memory, MemoryType, SQLiteMemoryStore, RAMMemoryStore
from backend.core.executor import Executor
from backend.core.evaluation import Evaluator
from backend.core.planner import Planner
from backend.core.evaluator import StepEvaluator, PlanEvaluator
from backend.core.autonomy import AutonomyLoop
from backend.core.task_state import TaskStateManager, TaskStatus
from backend.core.perception import Modality, Perception, PerceptionEvent
from backend.simulation.engine import SimulationEngine, SimulationScenario
from backend.tools.builtin import create_default_registry
from backend.tools.interfaces import register_future_tools
from backend.knowledge.engine import KnowledgeEngine
from backend.knowledge.models import Domain, KnowledgeType, KnowledgeStatus

class HistoryMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=30000)

class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=30000)
    system:  str = Field(default=DEFAULT_SYSTEM, max_length=12000)
    history: Optional[list[HistoryMessage]] = None
    project: str = Field(default="", max_length=100)

class TaskRequest(BaseModel):
    goal:         str  = Field(..., max_length=5000)
    system:       str  = Field(default=DEFAULT_SYSTEM, max_length=12000)
    project:      str  = Field(default="", max_length=100)
    use_llm_plan: bool = Field(default=True)

class KnowledgeAddRequest(BaseModel):
    title:          str           = Field(..., max_length=500)
    content:        str           = Field(..., max_length=50000)
    domain:         Optional[str] = None
    subdomain:      Optional[str] = None
    source:         str           = Field(default="user", max_length=200)
    source_url:     str           = Field(default="", max_length=1000)
    knowledge_type: Optional[str] = None
    confidence:     float         = Field(default=0.8, ge=0.0, le=1.0)
    tags:           list          = Field(default_factory=list)
    date_source:    str           = Field(default="", max_length=50)
    allow_duplicate: bool         = False

class KnowledgeSearchRequest(BaseModel):
    query:          str           = Field(..., max_length=1000)
    domain:         Optional[str] = None
    subdomain:      Optional[str] = None
    knowledge_type: Optional[str] = None
    status:         Optional[str] = None
    min_confidence: float         = Field(default=0.0, ge=0.0, le=1.0)
    since_days:     Optional[float] = None
    limit:          int           = Field(default=10, ge=1, le=50)
    exclude_outdated: bool        = False

class PerceiveRequest(BaseModel):
    modality:   str = Field(...)
    data:       Any = Field(...)
    source:     str = Field(default="api")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class SimulateRequest(BaseModel):
    scenario: str = Field(default="idle")
    ticks:    int = Field(default=1, ge=1, le=20)

class IntentDebugRequest(BaseModel):
    message: str = Field(..., max_length=5000)

def safe_error_message(error):
    text = str(error).strip() or "Error desconocido."
    for s in ALL_SECRETS:
        text = text.replace(s, "[REDACTED]")
    return text[:500]

def new_request_id(): return str(uuid.uuid4())

inference_layer:  Optional[InferenceLayer]   = None
nexus_core:       Optional[NexusCore]        = None
autonomy_loop_g:  Optional[AutonomyLoop]     = None
knowledge_engine: Optional[KnowledgeEngine]  = None
task_state_mgr:   Optional[TaskStateManager] = None
http_client:      Optional[httpx.AsyncClient] = None
perception:  Perception       = Perception()
simulation:  SimulationEngine = SimulationEngine()

@asynccontextmanager
async def lifespan(application: FastAPI):
    global inference_layer, nexus_core, autonomy_loop_g
    global knowledge_engine, task_state_mgr, http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(
        connect=15.0, read=REQUEST_TIMEOUT_SECONDS, write=15.0, pool=15.0))
    registry = InferenceRegistry()
    registry.register(OllamaRuntime(http_client=http_client))
    registry.register(LlamaCppRuntime())
    registry.register(GeminiRuntime())
    registry.register(GroqRuntime(http_client=http_client))
    registry.register(OpenRouterRuntime(http_client=http_client))
    try: mode = InferenceMode(INFERENCE_MODE)
    except ValueError: mode = InferenceMode.LOCAL_FIRST
    policy = InferencePolicy(mode=mode, allow_cloud=ALLOW_CLOUD_INFERENCE)
    inference_layer = InferenceLayer(registry=registry, policy=policy)
    try:
        store  = SQLiteMemoryStore(MEMORY_DB_PATH)
        memory = Memory(store)
    except Exception:
        memory = Memory(RAMMemoryStore())
    try: knowledge_engine = KnowledgeEngine(db_path=KNOWLEDGE_DB_PATH)
    except Exception: knowledge_engine = None
    try: task_state_mgr = TaskStateManager("nexus_tasks.db")
    except Exception: task_state_mgr = None
    tool_registry = create_default_registry(memory)
    register_future_tools(tool_registry)
    intent_router   = IntentRouter(registry=tool_registry)
    context_manager = ContextManager(memory=memory, knowledge_engine=knowledge_engine)
    executor        = Executor(registry=tool_registry, memory=memory)
    evaluator       = Evaluator()
    planner         = Planner(inference_layer=inference_layer)
    autonomy_loop_g = AutonomyLoop(planner=planner, executor=executor,
        step_evaluator=StepEvaluator(), plan_evaluator=PlanEvaluator(),
        memory=memory, inference_layer=inference_layer,
        max_loops=MAX_EXECUTION_LOOPS, max_steps=MAX_PLAN_STEPS,
        max_retries=MAX_RETRIES_PER_STEP)
    nexus_core = NexusCore(inference=inference_layer, memory=memory,
        intent_router=intent_router, context_manager=context_manager,
        executor=executor, evaluator=evaluator, autonomy_loop=autonomy_loop_g,
        knowledge_engine=knowledge_engine, task_state=task_state_mgr)
    logger.info("="*55)
    logger.info("NEXUS Omega v%s | Mode: %s | Cloud: %s", APP_VERSION, mode.value,
        "allowed" if ALLOW_CLOUD_INFERENCE else "BLOCKED")
    logger.info("="*55)
    yield
    await inference_layer.shutdown()
    await http_client.aclose()
    if task_state_mgr:
        try: task_state_mgr.close()
        except Exception: pass
    logger.info("NEXUS Omega detenido.")

app = FastAPI(title="NEXUS Omega", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def home(): return FileResponse("index.html")

@app.head("/")
async def home_head(): return Response(status_code=200)

@app.get("/health")
async def health():
    return {"status":"healthy","system":"NEXUS","version":APP_VERSION,
            "inference_mode":INFERENCE_MODE,"cloud_allowed":ALLOW_CLOUD_INFERENCE,
            "max_output_tokens":MAX_OUTPUT_TOKENS}

@app.get("/api/nexus/status")
async def nexus_status():
    return {"status":"online","system":"NEXUS","version":APP_VERSION,
            "inference_mode":INFERENCE_MODE,"cloud_allowed":ALLOW_CLOUD_INFERENCE}

@app.get("/api/nexus/config")
async def nexus_config():
    return {"version":APP_VERSION,"inference_mode":INFERENCE_MODE,
            "cloud_allowed":ALLOW_CLOUD_INFERENCE,"autonomy_enabled":AUTONOMY_ENABLED,
            "knowledge_enabled":bool(knowledge_engine),"max_plan_steps":MAX_PLAN_STEPS,
            "max_execution_loops":MAX_EXECUTION_LOOPS,"multi_provider":True}

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    assert nexus_core is not None
    request_id = new_request_id()
    started    = time.perf_counter()
    remote_ip  = req.client.host if req.client else "unknown"
    logger.info("[%s] Chat | ip=%s | chars=%d", request_id, remote_ip, len(request.message))
    if not request.message.strip():
        return {"success":True,"response":"NEXUS: Escuchando. Adelante.",
                "provider":"system","fallback":False,"request_id":request_id}
    history = [{"role":m.role,"content":m.content}
               for m in (request.history or [])[-MAX_HISTORY_TURNS:]]
    try:
        result = await nexus_core.process(message=request.message,
            system_prompt=request.system.strip() or DEFAULT_SYSTEM,
            history=history, project=request.project, request_id=request_id)
        elapsed = time.perf_counter() - started
        logger.info("[%s] OK | provider=%s | intent=%s | %.2fs",
            request_id, result.provider, result.intent, elapsed)
        return {"success":True,"response":result.text,"provider":result.provider,
                "model":result.model,"fallback":result.fallback,"local_mode":result.is_local,
                "is_local":result.is_local,"request_id":request_id,
                "duration_ms":result.duration_ms,"intent":result.intent,
                "domain":result.domain,"tools_used":result.tools_used,
                "knowledge_used":result.knowledge_used,"task_id":result.task_id}
    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception("[%s] Error | %.2fs", request_id, elapsed)
        raise HTTPException(status_code=503, detail={
            "message":"NEXUS no pudo obtener respuesta.",
            "error":safe_error_message(error),"provider":"none","request_id":request_id
        }) from error

@app.get("/sw.js")
async def service_worker():
    return Response(
        content='self.addEventListener("install",e=>self.skipWaiting());\nself.addEventListener("activate",e=>self.clients.claim());\n',
        media_type="application/javascript",
        headers={"Cache-Control":"no-cache, no-store, must-revalidate"})

@app.get("/api/nexus/tasks")
async def nexus_tasks_list():
    if not task_state_mgr:
        return {"success":False,"error":"TaskStateManager no disponible."}
    tasks = task_state_mgr.list_recent(limit=20)
    return {"success":True,"tasks":[t.to_dict() for t in tasks],"stats":task_state_mgr.stats()}

@app.get("/api/nexus/tasks/pending")
async def nexus_tasks_pending():
    if not task_state_mgr:
        return {"success":False,"error":"TaskStateManager no disponible."}
    tasks = task_state_mgr.list_pending()
    return {"success":True,"tasks":[t.to_dict() for t in tasks]}

@app.get("/api/nexus/tasks/{task_id}")
async def nexus_task_get(task_id: str):
    if not task_state_mgr:
        raise HTTPException(status_code=503, detail="TaskStateManager no disponible.")
    task = task_state_mgr.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Tarea {task_id} no encontrada.")
    return {"success":True,"task":task.to_dict()}

@app.get("/api/inference/health")
async def inference_health():
    assert inference_layer is not None
    status = await inference_layer.status()
    return {"success":True,**status}

@app.get("/api/nexus/memory")
async def nexus_memory():
    assert nexus_core is not None
    return {"success":True,"memory":nexus_core._memory.stats() if nexus_core._memory else {}}

@app.post("/api/nexus/memory/clear")
async def nexus_memory_clear(body: dict = {}):
    assert nexus_core is not None
    if not nexus_core._memory: return {"success":False}
    try:
        mt    = MemoryType(body.get("type","working"))
        count = nexus_core._memory._store.clear(mt)
    except ValueError:
        count = nexus_core._memory._store.clear()
    return {"success":True,"cleared":count}

@app.get("/api/nexus/tools")
async def nexus_tools():
    assert nexus_core is not None
    if not nexus_core._executor or not nexus_core._executor._registry:
        return {"success":True,"tools":[]}
    return {"success":True,"tools":nexus_core._executor._registry.describe(),
            "stats":nexus_core._executor._registry.stats()}

@app.post("/api/nexus/intent")
async def nexus_intent(body: IntentDebugRequest):
    assert nexus_core is not None
    if not nexus_core._intent: return {"success":False}
    result = nexus_core._intent.route(body.message)
    return {"success":True,"intent":result.intent.value,"domain":result.domain.value,
            "confidence":result.confidence,"strategy":result.strategy.value,
            "requires_tool":result.requires_tool,"candidate_tools":result.candidate_tools,
            "requires_memory":result.requires_memory,"requires_planning":result.requires_planning,
            "is_follow_up":result.is_follow_up}

@app.post("/api/nexus/task")
async def nexus_task_create(request: TaskRequest, req: Request):
    assert autonomy_loop_g is not None
    request_id = new_request_id()
    started    = time.perf_counter()
    task_id    = None
    if task_state_mgr:
        task    = task_state_mgr.create(description=request.goal, intent="task", domain="general")
        task_id = task.task_id
        task_state_mgr.update(task_id, TaskStatus.IN_PROGRESS)
    try:
        result  = await autonomy_loop_g.run(goal=request.goal, intent_type="task",
            context=request.goal, request_id=request_id, use_llm_plan=request.use_llm_plan)
        elapsed = time.perf_counter() - started
        plan    = result.plan
        if task_id and task_state_mgr:
            task_state_mgr.update(task_id, TaskStatus.COMPLETED, result=result.text)
        return {"success":True,"response":result.text,"request_id":request_id,
                "task_id":task_id,"duration_ms":int(elapsed*1000),
                "needs_input":result.needs_input,"input_question":result.input_question,
                "trace":{"run_id":result.trace.run_id,"status":result.trace.status.value,
                         "loops":result.trace.loops,"score":result.trace.plan_score,
                         "steps":result.trace.steps_log},
                "plan":plan.summary() if plan else None}
    except Exception as error:
        if task_id and task_state_mgr:
            task_state_mgr.update(task_id, TaskStatus.FAILED, error=safe_error_message(error))
        raise HTTPException(status_code=503, detail={
            "message":"Error en tarea autonoma.","error":safe_error_message(error),
            "request_id":request_id,"task_id":task_id}) from error

@app.get("/api/nexus/autonomy")
async def nexus_autonomy():
    return {"enabled":AUTONOMY_ENABLED,"version":APP_VERSION,
            "max_plan_steps":MAX_PLAN_STEPS,"max_execution_loops":MAX_EXECUTION_LOOPS,
            "max_retries_per_step":MAX_RETRIES_PER_STEP,
            "safe_actions":["read","analyze","plan","calculate","search","remember"],
            "forbidden_actions":["exec","shell","file_write","file_delete","network_raw"]}

@app.post("/api/knowledge/add")
async def knowledge_add(request: KnowledgeAddRequest):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    result = knowledge_engine.add(title=request.title, content=request.content,
        domain=request.domain, subdomain=request.subdomain, source=request.source,
        source_url=request.source_url, knowledge_type=request.knowledge_type,
        confidence=request.confidence, tags=request.tags, date_source=request.date_source,
        allow_duplicate=request.allow_duplicate)
    return {"success":result.success,"entry_id":result.entry_id,
            "duplicate":result.duplicate,"action":result.action,"errors":result.errors}

@app.post("/api/knowledge/search")
async def knowledge_search(request: KnowledgeSearchRequest):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    ctx = knowledge_engine.search(query=request.query, domain=request.domain,
        subdomain=request.subdomain, knowledge_type=request.knowledge_type,
        status=request.status, min_confidence=request.min_confidence,
        since_days=request.since_days, limit=request.limit,
        exclude_outdated=request.exclude_outdated)
    return {"success":True,"query":ctx.query,"total_found":ctx.total_found,
            "domains_covered":ctx.domains_covered,"entries":[e.to_dict() for e in ctx.entries]}

@app.get("/api/knowledge/domains")
async def knowledge_domains():
    from backend.knowledge.models import SUBDOMAINS
    if knowledge_engine is None:
        return {"success":True,"domains":[d.value for d in Domain]}
    return {"success":True,"domains":knowledge_engine.domains(),
            "subdomains":{k.value:v for k,v in SUBDOMAINS.items()}}

@app.get("/api/knowledge/stats")
async def knowledge_stats():
    if knowledge_engine is None: return {"success":False}
    return {"success":True,**knowledge_engine.stats()}

@app.get("/api/knowledge/{entry_id}")
async def knowledge_get(entry_id: str):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    entry = knowledge_engine.get(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entrada {entry_id} no encontrada.")
    return {"success":True,"entry":entry.to_dict()}

@app.delete("/api/knowledge/{entry_id}")
async def knowledge_delete(entry_id: str):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    return {"success":knowledge_engine.delete(entry_id),"entry_id":entry_id}

@app.post("/api/knowledge/refresh")
async def knowledge_refresh():
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    return {"success":True,"updated_entries":knowledge_engine.refresh_statuses()}

@app.get("/api/aura/status")
async def aura_status():
    core_status = nexus_core.status() if nexus_core else {}
    return {"system":"AURA","version":APP_VERSION,"brain":core_status,
            "hardware":{"simulation_scenario":simulation.scenario.value}}

@app.post("/api/aura/perceive")
async def aura_perceive(request: PerceiveRequest):
    try: modality = Modality(request.modality)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Modality invalida.")
    event = PerceptionEvent(modality=modality, data=request.data,
                            source=request.source, confidence=request.confidence)
    perception.receive(event)
    return {"success":True,"event_id":event.event_id,"summary":event.to_text()}

@app.post("/api/aura/simulate")
async def aura_simulate(request: SimulateRequest):
    try: scenario = SimulationScenario(request.scenario)
    except ValueError:
        raise HTTPException(status_code=422, detail="Scenario invalido.")
    simulation.set_scenario(scenario)
    all_events = []
    for _ in range(request.ticks):
        events = simulation.tick()
        for e in events: perception.receive(e)
        all_events.extend(events)
    return {"success":True,"scenario":scenario.value,"events":len(all_events),
            "summaries":[e.to_text() for e in all_events[:20]]}

@asynccontextmanager
async def lifespan(application: FastAPI):
    global inference_layer, nexus_core, autonomy_loop_g
    global knowledge_engine, task_state_mgr, http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(
        connect=15.0, read=REQUEST_TIMEOUT_SECONDS, write=15.0, pool=15.0))
    registry = InferenceRegistry()
    registry.register(OllamaRuntime(http_client=http_client))
    registry.register(LlamaCppRuntime())
    registry.register(GeminiRuntime())
    registry.register(GroqRuntime(http_client=http_client))
    registry.register(OpenRouterRuntime(http_client=http_client))
    try: mode = InferenceMode(INFERENCE_MODE)
    except ValueError: mode = InferenceMode.LOCAL_FIRST
    policy = InferencePolicy(mode=mode, allow_cloud=ALLOW_CLOUD_INFERENCE)
    inference_layer = InferenceLayer(registry=registry, policy=policy)
    try:
        store  = SQLiteMemoryStore(MEMORY_DB_PATH)
        memory = Memory(store)
    except Exception:
        memory = Memory(RAMMemoryStore())
    try: knowledge_engine = KnowledgeEngine(db_path=KNOWLEDGE_DB_PATH)
    except Exception: knowledge_engine = None
    try: task_state_mgr = TaskStateManager("nexus_tasks.db")
    except Exception: task_state_mgr = None
    tool_registry = create_default_registry(memory)
    register_future_tools(tool_registry)
    intent_router   = IntentRouter(registry=tool_registry)
    context_manager = ContextManager(memory=memory, knowledge_engine=knowledge_engine)
    executor        = Executor(registry=tool_registry, memory=memory)
    evaluator       = Evaluator()
    planner         = Planner(inference_layer=inference_layer)
    autonomy_loop_g = AutonomyLoop(planner=planner, executor=executor,
        step_evaluator=StepEvaluator(), plan_evaluator=PlanEvaluator(),
        memory=memory, inference_layer=inference_layer,
        max_loops=MAX_EXECUTION_LOOPS, max_steps=MAX_PLAN_STEPS,
        max_retries=MAX_RETRIES_PER_STEP)
    nexus_core = NexusCore(inference=inference_layer, memory=memory,
        intent_router=intent_router, context_manager=context_manager,
        executor=executor, evaluator=evaluator, autonomy_loop=autonomy_loop_g,
        knowledge_engine=knowledge_engine, task_state=task_state_mgr)
    logger.info("="*55)
    logger.info("NEXUS Omega v%s | Mode: %s | Cloud: %s", APP_VERSION, mode.value,
        "allowed" if ALLOW_CLOUD_INFERENCE else "BLOCKED")
    logger.info("="*55)
    yield
    await inference_layer.shutdown()
    await http_client.aclose()
    if task_state_mgr:
        try: task_state_mgr.close()
        except Exception: pass
    logger.info("NEXUS Omega detenido.")

app = FastAPI(title="NEXUS Omega", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def home(): return FileResponse("index.html")

@app.head("/")
async def home_head(): return Response(status_code=200)

@app.get("/health")
async def health():
    return {"status":"healthy","system":"NEXUS","version":APP_VERSION,
            "inference_mode":INFERENCE_MODE,"cloud_allowed":ALLOW_CLOUD_INFERENCE,
            "max_output_tokens":MAX_OUTPUT_TOKENS}

@app.get("/api/nexus/status")
async def nexus_status():
    return {"status":"online","system":"NEXUS","version":APP_VERSION,
            "inference_mode":INFERENCE_MODE,"cloud_allowed":ALLOW_CLOUD_INFERENCE}

@app.get("/api/nexus/config")
async def nexus_config():
    return {"version":APP_VERSION,"inference_mode":INFERENCE_MODE,
            "cloud_allowed":ALLOW_CLOUD_INFERENCE,"autonomy_enabled":AUTONOMY_ENABLED,
            "knowledge_enabled":bool(knowledge_engine),"max_plan_steps":MAX_PLAN_STEPS,
            "max_execution_loops":MAX_EXECUTION_LOOPS,"multi_provider":True}

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    assert nexus_core is not None
    request_id = new_request_id()
    started    = time.perf_counter()
    remote_ip  = req.client.host if req.client else "unknown"
    logger.info("[%s] Chat | ip=%s | chars=%d", request_id, remote_ip, len(request.message))
    if not request.message.strip():
        return {"success":True,"response":"NEXUS: Escuchando. Adelante.",
                "provider":"system","fallback":False,"request_id":request_id}
    history = [{"role":m.role,"content":m.content}
               for m in (request.history or [])[-MAX_HISTORY_TURNS:]]
    try:
        result = await nexus_core.process(message=request.message,
            system_prompt=request.system.strip() or DEFAULT_SYSTEM,
            history=history, project=request.project, request_id=request_id)
        elapsed = time.perf_counter() - started
        logger.info("[%s] OK | provider=%s | intent=%s | %.2fs",
            request_id, result.provider, result.intent, elapsed)
        return {"success":True,"response":result.text,"provider":result.provider,
                "model":result.model,"fallback":result.fallback,"local_mode":result.is_local,
                "is_local":result.is_local,"request_id":request_id,
                "duration_ms":result.duration_ms,"intent":result.intent,
                "domain":result.domain,"tools_used":result.tools_used,
                "knowledge_used":result.knowledge_used,"task_id":result.task_id}
    except Exception as error:
        elapsed = time.perf_counter() - started
        logger.exception("[%s] Error | %.2fs", request_id, elapsed)
        raise HTTPException(status_code=503, detail={
            "message":"NEXUS no pudo obtener respuesta.",
            "error":safe_error_message(error),"provider":"none","request_id":request_id
        }) from error

@app.get("/sw.js")
async def service_worker():
    return Response(
        content='self.addEventListener("install",e=>self.skipWaiting());\nself.addEventListener("activate",e=>self.clients.claim());\n',
        media_type="application/javascript",
        headers={"Cache-Control":"no-cache, no-store, must-revalidate"})

@app.get("/api/nexus/tasks")
async def nexus_tasks_list():
    if not task_state_mgr:
        return {"success":False,"error":"TaskStateManager no disponible."}
    tasks = task_state_mgr.list_recent(limit=20)
    return {"success":True,"tasks":[t.to_dict() for t in tasks],"stats":task_state_mgr.stats()}

@app.get("/api/nexus/tasks/pending")
async def nexus_tasks_pending():
    if not task_state_mgr:
        return {"success":False,"error":"TaskStateManager no disponible."}
    tasks = task_state_mgr.list_pending()
    return {"success":True,"tasks":[t.to_dict() for t in tasks]}

@app.get("/api/nexus/tasks/{task_id}")
async def nexus_task_get(task_id: str):
    if not task_state_mgr:
        raise HTTPException(status_code=503, detail="TaskStateManager no disponible.")
    task = task_state_mgr.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Tarea {task_id} no encontrada.")
    return {"success":True,"task":task.to_dict()}

@app.get("/api/inference/health")
async def inference_health():
    assert inference_layer is not None
    status = await inference_layer.status()
    return {"success":True,**status}

@app.get("/api/nexus/memory")
async def nexus_memory():
    assert nexus_core is not None
    return {"success":True,"memory":nexus_core._memory.stats() if nexus_core._memory else {}}

@app.post("/api/nexus/memory/clear")
async def nexus_memory_clear(body: dict = {}):
    assert nexus_core is not None
    if not nexus_core._memory: return {"success":False}
    try:
        mt    = MemoryType(body.get("type","working"))
        count = nexus_core._memory._store.clear(mt)
    except ValueError:
        count = nexus_core._memory._store.clear()
    return {"success":True,"cleared":count}

@app.get("/api/nexus/tools")
async def nexus_tools():
    assert nexus_core is not None
    if not nexus_core._executor or not nexus_core._executor._registry:
        return {"success":True,"tools":[]}
    return {"success":True,"tools":nexus_core._executor._registry.describe(),
            "stats":nexus_core._executor._registry.stats()}

@app.post("/api/nexus/intent")
async def nexus_intent(body: IntentDebugRequest):
    assert nexus_core is not None
    if not nexus_core._intent: return {"success":False}
    result = nexus_core._intent.route(body.message)
    return {"success":True,"intent":result.intent.value,"domain":result.domain.value,
            "confidence":result.confidence,"strategy":result.strategy.value,
            "requires_tool":result.requires_tool,"candidate_tools":result.candidate_tools,
            "requires_memory":result.requires_memory,"requires_planning":result.requires_planning,
            "is_follow_up":result.is_follow_up}

@app.post("/api/nexus/task")
async def nexus_task_create(request: TaskRequest, req: Request):
    assert autonomy_loop_g is not None
    request_id = new_request_id()
    started    = time.perf_counter()
    task_id    = None
    if task_state_mgr:
        task    = task_state_mgr.create(description=request.goal, intent="task", domain="general")
        task_id = task.task_id
        task_state_mgr.update(task_id, TaskStatus.IN_PROGRESS)
    try:
        result  = await autonomy_loop_g.run(goal=request.goal, intent_type="task",
            context=request.goal, request_id=request_id, use_llm_plan=request.use_llm_plan)
        elapsed = time.perf_counter() - started
        plan    = result.plan
        if task_id and task_state_mgr:
            task_state_mgr.update(task_id, TaskStatus.COMPLETED, result=result.text)
        return {"success":True,"response":result.text,"request_id":request_id,
                "task_id":task_id,"duration_ms":int(elapsed*1000),
                "needs_input":result.needs_input,"input_question":result.input_question,
                "trace":{"run_id":result.trace.run_id,"status":result.trace.status.value,
                         "loops":result.trace.loops,"score":result.trace.plan_score,
                         "steps":result.trace.steps_log},
                "plan":plan.summary() if plan else None}
    except Exception as error:
        if task_id and task_state_mgr:
            task_state_mgr.update(task_id, TaskStatus.FAILED, error=safe_error_message(error))
        raise HTTPException(status_code=503, detail={
            "message":"Error en tarea autonoma.","error":safe_error_message(error),
            "request_id":request_id,"task_id":task_id}) from error

@app.get("/api/nexus/autonomy")
async def nexus_autonomy():
    return {"enabled":AUTONOMY_ENABLED,"version":APP_VERSION,
            "max_plan_steps":MAX_PLAN_STEPS,"max_execution_loops":MAX_EXECUTION_LOOPS,
            "max_retries_per_step":MAX_RETRIES_PER_STEP,
            "safe_actions":["read","analyze","plan","calculate","search","remember"],
            "forbidden_actions":["exec","shell","file_write","file_delete","network_raw"]}

@app.post("/api/knowledge/add")
async def knowledge_add(request: KnowledgeAddRequest):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    result = knowledge_engine.add(title=request.title, content=request.content,
        domain=request.domain, subdomain=request.subdomain, source=request.source,
        source_url=request.source_url, knowledge_type=request.knowledge_type,
        confidence=request.confidence, tags=request.tags, date_source=request.date_source,
        allow_duplicate=request.allow_duplicate)
    return {"success":result.success,"entry_id":result.entry_id,
            "duplicate":result.duplicate,"action":result.action,"errors":result.errors}

@app.post("/api/knowledge/search")
async def knowledge_search(request: KnowledgeSearchRequest):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    ctx = knowledge_engine.search(query=request.query, domain=request.domain,
        subdomain=request.subdomain, knowledge_type=request.knowledge_type,
        status=request.status, min_confidence=request.min_confidence,
        since_days=request.since_days, limit=request.limit,
        exclude_outdated=request.exclude_outdated)
    return {"success":True,"query":ctx.query,"total_found":ctx.total_found,
            "domains_covered":ctx.domains_covered,"entries":[e.to_dict() for e in ctx.entries]}

@app.get("/api/knowledge/domains")
async def knowledge_domains():
    from backend.knowledge.models import SUBDOMAINS
    if knowledge_engine is None:
        return {"success":True,"domains":[d.value for d in Domain]}
    return {"success":True,"domains":knowledge_engine.domains(),
            "subdomains":{k.value:v for k,v in SUBDOMAINS.items()}}

@app.get("/api/knowledge/stats")
async def knowledge_stats():
    if knowledge_engine is None: return {"success":False}
    return {"success":True,**knowledge_engine.stats()}

@app.get("/api/knowledge/{entry_id}")
async def knowledge_get(entry_id: str):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    entry = knowledge_engine.get(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entrada {entry_id} no encontrada.")
    return {"success":True,"entry":entry.to_dict()}

@app.delete("/api/knowledge/{entry_id}")
async def knowledge_delete(entry_id: str):
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    return {"success":knowledge_engine.delete(entry_id),"entry_id":entry_id}

@app.post("/api/knowledge/refresh")
async def knowledge_refresh():
    if knowledge_engine is None:
        raise HTTPException(status_code=503, detail="KnowledgeEngine no disponible.")
    return {"success":True,"updated_entries":knowledge_engine.refresh_statuses()}

@app.get("/api/aura/status")
async def aura_status():
    core_status = nexus_core.status() if nexus_core else {}
    return {"system":"AURA","version":APP_VERSION,"brain":core_status,
            "hardware":{"simulation_scenario":simulation.scenario.value}}

@app.post("/api/aura/perceive")
async def aura_perceive(request: PerceiveRequest):
    try: modality = Modality(request.modality)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Modality invalida.")
    event = PerceptionEvent(modality=modality, data=request.data,
                            source=request.source, confidence=request.confidence)
    perception.receive(event)
    return {"success":True,"event_id":event.event_id,"summary":event.to_text()}

@app.post("/api/aura/simulate")
async def aura_simulate(request: SimulateRequest):
    try: scenario = SimulationScenario(request.scenario)
    except ValueError:
        raise HTTPException(status_code=422, detail="Scenario invalido.")
    simulation.set_scenario(scenario)
    all_events = []
    for _ in range(request.ticks):
        events = simulation.tick()
        for e in events: perception.receive(e)
        all_events.extend(events)
    return {"success":True,"scenario":scenario.value,"events":len(all_events),
            "summaries":[e.to_text() for e in all_events[:20]]}
