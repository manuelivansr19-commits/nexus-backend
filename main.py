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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                max_output_tokens=300
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))