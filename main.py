from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Puedes colocar tu API key directamente aquí si no usas variables de entorno en Windows
API_KEY = os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")

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
        # Usando el modelo estándar actual gemini-2.5-flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": request.system}]},
            "contents": [{"parts": [{"text": request.message}]}],
            "generationConfig": {"maxOutputTokens": 300}
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=payload)
            data = r.json()
            
        if "candidates" in data:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": text}
        else:
            print("Gemini error:", data)
            raise HTTPException(status_status=500, detail=str(data))
    except HTTPException:
        raise
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
        
