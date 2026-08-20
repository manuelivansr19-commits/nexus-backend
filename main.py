from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import httpx
import os
import random
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEXUS-BACKEND")

app = FastAPI(title="NEXUS AI", version="2.2.0")

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
    # Evita que el servidor colapse si el micrófono manda un texto vacío
    message: str = Field(default="", max_length=10000)
    system: str = Field(
        default=(
            "Eres NEXUS, un asistente de inteligencia artificial personal. "
            "Responde de forma corta, directa, precisa y útil."
        ),
        max_length=5000
    )

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "gemini_active": client is not None}

async def call_gemini_async(prompt: str, system_instruction: str):
    if not client:
        raise Exception("Gemini no inicializado.")
    
    # Envolvemos la llamada en un hilo asíncrono para NO congelar el servidor de Render
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-1.5-flash",  # Versión oficial estable
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7
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
        # 1. Si el micrófono captó ruido o vacío, respondemos suavemente
        if not request.message.strip():
            return {
                "success": True, 
                "response": "No escuché bien, ¿puedes repetir?", 
                "provider": "system", 
                "fallback": False
            }

        text = None
        provider = "gemini"
        fallback = False
        
        # 2. Lógica principal sin bloqueos
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
                
                # Respaldo en caso de que Gemini alcance su límite
                if not text:
                    text = "NEXUS: Mi conexión con la red principal está recargándose. Intenta en un momento."
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
        # 3. Blindaje total: Nunca devolver un error 500 que cause "Error de conexión"
        return {
            "success": False,
            "response": "NEXUS: Ocurrió un error interno, pero sigo en línea.",
            "provider": "error",
            "fallback": True
        }
        
