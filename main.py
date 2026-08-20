from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
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
# CHAT NEXUS
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
            model="gemini-3.6-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                max_output_tokens=300,
                temperature=0.7
            )
        )

        # Protección contra respuestas vacías
        if not response:
            raise Exception("Gemini no devolvió respuesta.")

        text = getattr(response, "text", None)

        if not text:
            raise Exception("Gemini devolvió una respuesta sin texto.")

        return {
            "success": True,
            "response": text.strip()
        }

    except Exception as e:

        print("===================================")
        print("ERROR NEXUS:")
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
    
