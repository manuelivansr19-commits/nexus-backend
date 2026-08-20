from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
import os
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEXUS-BACKEND")

app = FastAPI(title="NEXUS AI", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
USE_OLLAMA_ONLY = os.getenv("USE_OLLAMA_ONLY", "false").lower() == "true"
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and not USE_OLLAMA_ONLY else None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    system: str = Field(
        default=(
            "Eres NEXUS, un asistente de inteligencia artificial "
            "personal. Responde de forma corta, directa, precisa y útil. "
            "Máximo 3 oraciones salvo que el usuario solicite más detalle."
        ),
        max_length=5000
    )

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.head("/")
async def home_head():
    return {"status": "ok"}

@app.get("/sw.js")
async def service_worker():
    return JSONResponse(status_code=404, content={"error": "Service worker no configurado."})

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "NEXUS",
        "gemini_active": client is not None,
        "ollama_configured": bool(OLLAMA_URL)
    }

@app.get("/api/nexus/status")
async def status():
    return {
        "status": "online",
        "system": "NEXUS",
        "gemini_configured": client is not None,
        "use_ollama_only": USE_OLLAMA_ONLY
    }

async def call_gemini_with_retry(prompt: str, system_instruction: str):
    if not client:
        raise Exception("Cliente Gemini no inicializado.")
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7
        )
    )
    if response and getattr(response, "text", None):
        return response.text.strip()
    raise Exception("Gemini devolvió respuesta vacía.")

async def call_ollama(prompt: str, system_instruction: str):
    if not OLLAMA_URL:
        raise Exception("OLLAMA_URL no está configurada en Render.")
    
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
        
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System: {system_instruction}\n\nUser: {prompt}\nNEXUS:",
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=30) as http_client:
        r = await http_client.post(OLLAMA_URL, json=payload, headers=headers)
        if r.status_code == 200:
            data = r.json()
            return data.get("response", "").strip()
        else:
            raise Exception(f"Ollama remoto respondió con HTTP {r.status_code}")

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    request_id = str(random.randint(10000, 99999))
    text = None
    provider = "gemini"
    fallback = False
    error_log = ""

    if USE_OLLAMA_ONLY:
        try:
            text = await call_ollama(request.message, request.system)
            provider = "ollama"
        except Exception as oe:
            error_log = str(oe)
    else:
        try:
            text = await call_gemini_with_retry(request.message, request.system)
            provider = "gemini"
        except Exception as ge:
            error_log = str(ge)
            logger.warning(f"[{request_id}] Gemini falló (Cuota/Error): {error_log}. Intentando respaldo...")
            
            # Intentar Ollama si está configurado
            if OLLAMA_URL:
                try:
                    text = await call_ollama(request.message, request.system)
                    provider = "ollama"
                    fallback = True
                except Exception as oe:
                    error_log += f" | Ollama Error: {str(oe)}"
            
            # RESPALDO DE EMERGENCIA LOCAL EN RENDER (Evita el pantallazo rojo si no hay Ollama externo)
            if not text:
                text = "NEXUS se encuentra recargando energía (Límite de cuota de Gemini alcanzado y sin servidor Ollama secundario configurado). Intenta de nuevo en unos minutos."
                provider = "emergency_fallback"
                fallback = True

    return {
        "success": True,
        "response": text,
        "provider": provider,
        "fallback": fallback
    }
    
