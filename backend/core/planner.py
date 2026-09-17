"""NEXUS Omega -- Planner v3.8.0"""
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from backend.config import MAX_PLAN_STEPS, PLANNER_SYSTEM, logger

class StepStatus(str, Enum):
    PENDING="pending"; RUNNING="running"; DONE="done"; FAILED="failed"; SKIPPED="skipped"

@dataclass
class PlanStep:
    description: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:6])
    tool: Optional[str] = None
    requires_llm: bool = True
    expected_output: str = ""
    depends_on: list = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    duration_ms: int = 0
    def to_dict(self):
        return {"step_id":self.step_id,"description":self.description,"tool":self.tool,
                "requires_llm":self.requires_llm,"status":self.status.value,
                "result":self.result[:100] if self.result else None,
                "error":self.error[:100] if self.error else None,"retries":self.retries}

@dataclass
class Plan:
    goal: str
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps: list = field(default_factory=list)
    done: bool = False
    created_at: float = field(default_factory=time.time)
    source: str = "local"
    metadata: dict = field(default_factory=dict)
    def next_step(self):
        completed = {s.step_id for s in self.steps if s.status==StepStatus.DONE}
        for step in self.steps:
            if step.status!=StepStatus.PENDING: continue
            if all(dep in completed for dep in step.depends_on): return step
        return None
    def mark_done(self, step_id, result=""):
        for s in self.steps:
            if s.step_id==step_id: s.status=StepStatus.DONE; s.result=result
        if all(s.status in (StepStatus.DONE,StepStatus.SKIPPED) for s in self.steps): self.done=True
    def mark_failed(self, step_id, reason=""):
        for s in self.steps:
            if s.step_id==step_id: s.status=StepStatus.FAILED; s.error=reason
    def mark_running(self, step_id):
        for s in self.steps:
            if s.step_id==step_id: s.status=StepStatus.RUNNING
    def progress(self):
        return f"{sum(1 for s in self.steps if s.status==StepStatus.DONE)}/{len(self.steps)}"
    def collect_results(self):
        parts=[f"[{s.description}]\n{s.result}" for s in self.steps if s.status==StepStatus.DONE and s.result]
        return "\n\n".join(parts) if parts else ""
    def summary(self):
        return {"plan_id":self.plan_id,"goal":self.goal,"source":self.source,
                "steps":len(self.steps),"done":self.done,"progress":self.progress(),
                "steps_detail":[s.to_dict() for s in self.steps]}

_TEMPLATES = {
    "analysis":[{"description":"Identificar el problema central y sus variables"},
                {"description":"Analizar causas y factores del contexto"},
                {"description":"Identificar riesgos y oportunidades"},
                {"description":"Formular conclusiones y recomendaciones"}],
    "research":[{"description":"Definir alcance y preguntas clave de la investigacion"},
                {"description":"Analizar el mercado objetivo y segmentos"},
                {"description":"Identificar competencia y canales alternativos"},
                {"description":"Evaluar oportunidades y barreras de entrada"},
                {"description":"Sintetizar hallazgos y recomendaciones accionables"}],
    "design":  [{"description":"Definir requisitos y restricciones"},
                {"description":"Proponer arquitectura general"},
                {"description":"Especificar componentes e interfaces"},
                {"description":"Identificar dependencias y riesgos"},
                {"description":"Producir diseno preliminar"}],
    "task":    [{"description":"Clarificar objetivo y criterios de exito"},
                {"description":"Identificar recursos disponibles"},
                {"description":"Definir pasos de implementacion"},
                {"description":"Identificar riesgos y contingencias"},
                {"description":"Producir plan de accion"}],
}

class LocalPlanner:
    def __init__(self, max_steps=MAX_PLAN_STEPS):
        self._max_steps=max_steps
    def plan(self, goal, intent_type="task"):
        tmpl=_TEMPLATES.get(intent_type,_TEMPLATES["task"])
        steps=[PlanStep(description=s["description"],tool=s.get("tool"),
                        requires_llm=s.get("requires_llm",True)) for s in tmpl[:self._max_steps]]
        return Plan(goal=goal,steps=steps,source="local")
    def single_step(self, description, tool=None):
        return Plan(goal=description,steps=[PlanStep(description=description,tool=tool,
                    requires_llm=(tool is None))],source="local")

class LLMPlanner:
    def __init__(self, inference_layer, max_steps=MAX_PLAN_STEPS):
        self._inference=inference_layer; self._max_steps=max_steps
    async def plan(self, goal, intent_type="task", context=""):
        from backend.inference.models import InferenceRequest
        prompt=(f"Crea un plan para: {goal}\nTipo: {intent_type}\n"
                f"Responde SOLO JSON valido sin markdown:\n"
                f"{{\"goal\":\"\",\"steps\":[{{\"description\":\"\",\"tool\":null,\"requires_llm\":true}}]}}\n"
                f"Maximo {min(self._max_steps,8)} pasos. Solo analisis y planificacion.")
        result=await self._inference.generate(InferenceRequest(prompt=prompt,system=PLANNER_SYSTEM,max_tokens=1500))
        raw=result.text.strip().replace("```json","").replace("```","").strip()
        data=json.loads(raw)
        steps=[PlanStep(description=s.get("description","paso"),tool=s.get("tool"),
                        requires_llm=s.get("requires_llm",True))
               for s in data.get("steps",[])[:self._max_steps]]
        if not steps: raise ValueError("Plan LLM vacio.")
        return Plan(goal=data.get("goal",goal),steps=steps,source="llm")

class _LegacyPlanner:
    def __init__(self, router, max_steps):
        self._router=router; self._max_steps=max_steps
    async def plan(self, goal, intent_type="task", context=""):
        from backend.providers.base import GenerateRequest
        prompt=f"Plan para: {goal}\nJSON: {{\"goal\":\"\",\"steps\":[{{\"description\":\"\"}}]}}"
        result=await self._router.generate(GenerateRequest(prompt=prompt,system=PLANNER_SYSTEM,max_tokens=1500))
        data=json.loads(result.response.text.strip().replace("```json","").replace("```","").strip())
        steps=[PlanStep(description=s.get("description","paso"),requires_llm=True)
               for s in data.get("steps",[])[:self._max_steps]]
        return Plan(goal=data.get("goal",goal),steps=steps,source="llm_legacy")

class Planner:
    def __init__(self, inference_layer=None, model_router=None, max_steps=MAX_PLAN_STEPS):
        self._local=LocalPlanner(max_steps=max_steps); self._max_steps=max_steps
        if inference_layer is not None: self._llm=LLMPlanner(inference_layer,max_steps)
        elif model_router is not None:  self._llm=_LegacyPlanner(model_router,max_steps)
        else: self._llm=None
    async def plan(self, goal, intent_type="task", use_llm=True, context=""):
        if use_llm and self._llm is not None:
            try: return await self._llm.plan(goal,intent_type,context)
            except Exception as e: logger.warning("Planner LLM fallo: %s",str(e)[:80])
        return self._local.plan(goal,intent_type)
    def plan_sync(self, goal, intent_type="task"):
        return self._local.plan(goal,intent_type)
    def single_step(self, description, tool=None):
        return self._local.single_step(description,tool)
