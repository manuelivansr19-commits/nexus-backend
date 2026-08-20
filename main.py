from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import httpx
import os

app = FastAPI(title="NEXUS AI", version="1.0.0")

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ ERROR: GEMINI_API_KEY no está configurada.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ============================================================
# MODELOS
# ============================================================

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


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


# ============================================================
# STATUS
# ============================================================

@app.get("/api/nexus/status")
async def status():

    return {
        "status": "online",
        "system": "NEXUS",
        "gemini_configured": client is not None
    }


# ============================================================
# CHAT NEXUS CON FALLBACK AUTOMÁTICO A OLLAMA
# ============================================================

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest):

    text = None

    # 1. Intentar con Gemini
    if client is not None:
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=request.message,
                config=types.GenerateContentConfig(
                    system_instruction=request.system,
                    max_output_tokens=300,
                    temperature=0.7
                )
            )
            if response and getattr(response, "text", None):
                text = response.text.strip()
        except Exception:
            # Si Gemini falla (cuota 429, red, etc.), simplemente pasamos a Ollama
            pass

    # 2. Si no hay texto de Gemini, usar Ollama local como respaldo limpio
    if not text:
        try:
            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434/api/generate")
            payload = {
                "model": "llama3",
                "prompt": f"{request.system}\n\nUsuario: {request.message}\nNEXUS:",
                "stream": False
            }
            async with httpx.AsyncClient(timeout=30) as http_client:
                r = await http_client.post(ollama_url, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    text = data.get("response", "").strip()
        except Exception as oe:
            print("⚠️ Ollama local no disponible:", repr(oe))

    # 3. Si ambos motores fallan, retornamos el error limpio
    if not text:
        raise HTTPException(
            status_code=500,
            detail="NEXUS no pudo procesar la solicitud: Límite de Gemini alcanzado y Ollama local inalcanzable."
        )

    return {
        "success": True,
        "response": text
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "system": "NEXUS"
    }
    
