from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import httpx
import os
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEXUS-BACKEND")

app = FastAPI(title="NEXUS AI", version="2.4.0")

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
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "600"))

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and not USE_OLLAMA_ONLY else None

class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=10000)
    system: str = Field(
        default=(
            "Eres NEXUS, un sistema avanzado de inteligencia artificial enfocado en "
            "estrategia, psicología aplicada, PNL y desarrollo de proyectos. "
            "Responde siempre en español, de forma analítica, clara, directa y estructurada. "
            "Nunca dejes frases a medias y completa tus explicaciones de manera profesional."
        ),
        max_length=5000
    )

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "gemini_active": client is not None}

# Endpoints requeridos por el frontend para desbloquear la pantalla de carga inicial
@app.get("/api/nexus/status")
async def nexus_status():
    return {
        "status": "online",
        "ollama_active": bool(OLLAMA_URL),
        "gemini_active": client is not None
    }

@app.get("/api/nexus/config")
async def nexus_config():
    return {
        "model": "gemini-1.5-flash",
        "ollama_model": OLLAMA_MODEL,
        "use_ollama_only": USE_OLLAMA_ONLY
    }

async def call_gemini_async(prompt: str, system_instruction: str):
    if not client:
        raise Exception("Gemini no inicializado.")
    
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-1.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.6
        )
    )
    if response and getattr(response, "text", None):
        return response.text.strip()
    raise Exception("Respuesta vacía de Gemini.")

async def call_ollama(prompt: str, system_instruction: str):
    if not OLLAMA_URL:
        raise Exception("OLLAMA_URL no configurada.")
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System: {system_instruction}\n\nUser: {prompt}\nNEXUS:",
        "stream": False
    }
    async with httpx.AsyncClient(timeout=30) as http_client:
        r = await http_client.post(OLLAMA_URL, json=payload, headers=headers)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        raise Exception(f"Error {r.status_code}")

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest, req: Request):
    try:
        if not request.message.strip():
            return {
                "success": True, 
                "response": "NEXUS: Escuchando. Adelante con tu consulta.", 
                "provider": "system", 
                "fallback": False
            }

        text = None
        provider = "gemini"
        fallback = False
        
        if USE_OLLAMA_ONLY:
            try:
                text = await call_ollama(request.message, request.system)
                provider = "ollama"
            except Exception:
                pass
        else:
            try:
                text = await call_gemini_async(request.message, request.system)
            except Exception as ge:
                logger.warning(f"Gemini falló: {str(ge)}")
                if OLLAMA_URL:
                    try:
                        text = await call_ollama(request.message, request.system)
                        provider = "ollama"
                        fallback = True
                    except Exception:
                        pass
                
                if not text:
                    text = "NEXUS: El sistema principal se encuentra en pausa temporal por cuota. Intenta de nuevo en unos momentos."
                    provider = "emergency_fallback"
                    fallback = True

        return {
            "success": True,
            "response": text,
            "provider": provider,
            "fallback": fallback
        }

    except Exception as global_e:
        logger.error(f"Error crítico global: {str(global_e)}")
        return {
            "success": True,
            "response": "NEXUS: Operación procesada con reanudación de enlace interno.",
            "provider": "error_recovery",
            "fallback": True
    }
                                                 
