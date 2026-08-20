from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import os


# ============================================================
# NEXUS AI
# ============================================================

app = FastAPI(
    title="NEXUS AI",
    version="1.0.0"
)


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
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY no está configurada.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    system: str = (
        "Eres NEXUS. "
        "Responde de forma corta, directa y precisa. "
        "Máximo 3 oraciones salvo que el usuario pida más detalle."
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


# ============================================================
# NEXUS STATUS
# ============================================================

@app.get("/api/nexus/status")
async def status():

    return {
        "status": "online",
        "system": "NEXUS",
        "gemini_configured": client is not None
    }


# ============================================================
# NEXUS CHAT
# ============================================================

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY no está configurada en el servidor."
        )

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                max_output_tokens=300
            )
        )

        if not response:
            raise Exception("Gemini no devolvió respuesta.")

        text = response.text

        if not text:
            raise Exception("Gemini devolvió una respuesta vacía.")

        return {
            "response": text.strip()
        }

    except Exception as e:

        print("===================================")
        print("ERROR EN NEXUS")
        print(repr(e))
        print("===================================")

        raise HTTPException(
            status_code=500,
            detail=f"NEXUS no pudo procesar la solicitud: {str(e)}"
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "system": "NEXUS"
    }
