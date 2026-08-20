from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa el cliente oficial de Gemini (toma automáticamente la variable GEMINI_API_KEY del entorno)
client = genai.Client()

class ChatRequest(BaseModel):
    message: str
    system: str = "Eres NEXUS. Responde de forma corta, directa y precisa. Maximo 3 oraciones salvo que el usuario pida mas detalle."

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/api/nexus/status")
async def status():
    return {"status": "online", "system": "NEXUS"}

@app.post("/api/nexus/chat")
async def chat(request: ChatRequest):
    try:
        # Petición utilizando el SDK oficial con soporte de system_instruction integrado
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                max_output_tokens=300
            )
        )
        return {"response": response.text}
    except Exception as e:
        print("Error en NEXUS:", e)
        raise HTTPException(status_code=500, detail=str(e))
        
