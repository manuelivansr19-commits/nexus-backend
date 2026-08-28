# NEXUS Ω — Arquitectura Objetivo

**Versión:** 3.4.0  
**Fecha:** 2026-08-25  
**Estado:** En construcción — Fase A-H completada

---

## Visión

NEXUS Ω es el sistema de inteligencia central de AURA.
Debe funcionar sin Internet. Los proveedores externos son recursos opcionales, no el cerebro.

---

## Mapa completo

```
                        AURA
                          │
               ┌──────────┴──────────┐
               │      NEXUS CORE     │
               │   (backend/main.py) │
               └──────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    PERCEPTION          MEMORY           REASONING
  (core/perception)  (core/memory)   (core/reasoning)
          │                │                │
          └────────────────┼────────────────┘
                           │
                       PLANNER
                   (core/planning)
                           │
                       EXECUTOR
                    (core/action)
                           │
                      EVALUATOR
                  (core/evaluation)
                           │
               ┌───────────┴───────────┐
               │      MODEL ROUTER     │
               │    (backend/router)   │
               └───────────┬───────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
  LOCAL ENGINE        EXTERNAL             FUTURE
(providers/local)    PROVIDERS           PROVIDERS
       │                   │
  ┌────┴────┐    ┌──────────┼──────────┐
  │llama.cpp│    │ Gemini   │OpenRouter│
  │ GGUF   │    │ (3 retry)│  Groq    │
  └─────────┘    │  Ollama  │          │
                 └──────────┴──────────┘
                           │
              ┌────────────┴────────────┐
              │    HARDWARE LAYER       │
              │  (backend/hardware/)    │
              └────────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │         │        │        │         │
    Camera      Lidar    Audio     IMU    Servo/Motor
  (simulated) (simulated)(simul.) (simul.) (simul.)
        │         │        │        │         │
        └──────────────────┼──────────────────┘
                           │
              ┌────────────┴────────────┐
              │   SIMULATION ENGINE     │
              │ (backend/simulation/)   │
              │  idle | exploring |     │
              │  conversation | obstacle│
              └─────────────────────────┘
```

---

## Estado por fase

| Fase | Descripción | Estado |
|---|---|---|
| 1.5 | Estabilización + modularización | ✅ Completo |
| A | Local Engine Abstraction | ✅ Completo |
| B | Local Model Runtime (llama.cpp + Ollama local) | ✅ Completo |
| C | Offline Mode (NEXUS_LOCAL_ONLY) | ✅ Completo |
| D | Model Router con disponibilidad real | ✅ Completo |
| E | AURA Brain API (6 subsistemas) | ✅ Completo |
| F | Hardware Interfaces (7 dispositivos) | ✅ Completo |
| G | Head Computer Requirements | ✅ Documentado |
| H | Simulation Layer | ✅ Completo |

---

## Árbol de archivos actual

```
nexus-backend/
│
├── index.html                      Frontend PWA (sin cambios de contrato)
├── requirements.txt                Dependencias pinneadas
├── .gitignore
├── CHANGELOG.md
├── NEXUS_ARCHITECTURE_CURRENT.md
├── NEXUS_ARCHITECTURE_TARGET.md    ← este archivo
├── AURA_HARDWARE_REQUIREMENTS.md
│
└── backend/
    ├── __init__.py
    ├── config.py                   Toda la configuración centralizada
    ├── router.py                   ModelRouter con LOCAL_ONLY + fallback
    ├── main.py                     Endpoints (existentes + AURA Brain)
    │
    ├── providers/
    │   ├── base.py                 BaseModelProvider (interfaz abstracta)
    │   ├── local.py                LocalProvider — llama.cpp + Ollama local
    │   ├── gemini.py               Gemini + retry/backoff/jitter (hotfix)
    │   ├── openrouter.py           OpenRouter
    │   ├── groq.py                 Groq
    │   └── ollama.py               Ollama externo
    │
    ├── core/                       AURA Brain API
    │   ├── perception.py           Recepción de señales (texto, IMU, LiDAR…)
    │   ├── memory.py               Memoria working / episódica / semántica
    │   ├── reasoning.py            Clasificador intent + ensamblador contexto
    │   ├── planning.py             Descomposición de objetivos en pasos
    │   ├── action.py               Ejecución segura de acciones
    │   └── evaluation.py           Evaluación de calidad de respuestas
    │
    ├── hardware/                   Interfaces de hardware (todas simuladas)
    │   ├── base.py                 HardwareInterface abstracta
    │   ├── camera.py               Cámara RGB
    │   ├── lidar.py                LiDAR 2D/3D
    │   ├── microphone.py           Micrófono
    │   ├── imu.py                  IMU (acelerómetro + giroscopio)
    │   ├── servo.py                Servo motor
    │   ├── motor.py                Motor DC
    │   └── sensor.py               Sensor genérico
    │
    └── simulation/
        └── engine.py               Generador de sensores virtuales
                                    Escenarios: idle, exploring,
                                    conversation, obstacle

tests/
    ├── test_core.py                Router, fallback, providers
    ├── test_api.py                 Endpoints HTTP
    └── test_offline.py             Modo LOCAL_ONLY + AURA Brain
```

---

## Contratos de API

### Existentes (sin cambios)

```
POST /api/nexus/chat
Body:  { message, system, history? }
Resp:  { success, response, provider, model, fallback, local_mode,
         request_id, duration_ms }

GET  /health
Resp:  { status, system, version, providers, models,
         max_output_tokens, ollama_only, local_mode }
```

### Nuevos (AURA Brain)

```
GET  /api/aura/status
Resp: { system, version, local_mode, providers, brain, hardware }

POST /api/aura/perceive
Body: { modality, data, source, confidence }
Resp: { success, event_id, modality, summary }

POST /api/aura/simulate
Body: { scenario, ticks }
Resp: { success, scenario, ticks, events, summaries[] }
```

---

## Variables de entorno

### Producción actual (Render)

```bash
GEMINI_API_KEY=...          # requerido
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_RETRIES=3
MAX_OUTPUT_TOKENS=8192
REQUEST_TIMEOUT_SECONDS=180
LOG_LEVEL=INFO
```

### Hardware físico AURA (futuro)

```bash
NEXUS_LOCAL_ONLY=true
LOCAL_RUNTIME=llama_cpp
LOCAL_MODEL_PATH=/models/llama-3.2-3b-q4_k_m.gguf
LOCAL_OLLAMA_URL=http://localhost:11434
LOCAL_OLLAMA_MODEL=llama3.2
MAX_OUTPUT_TOKENS=2048
REQUEST_TIMEOUT_SECONDS=60
```

---

## Criterio de terminación — Fase A-H

### ✅ Demostrable ahora (en Render / sin hardware)

| Criterio | Cómo verificar |
|---|---|
| NEXUS inicia | `GET /health` → 200 |
| Frontend funciona | `GET /` → index.html |
| Chat funciona | `POST /api/nexus/chat` → respuesta |
| Gemini con retry | Logs muestran reintentos en error 503 |
| Fallback correcto | `fallback: true` cuando Gemini falla |
| LOCAL_ONLY mode | `NEXUS_LOCAL_ONLY=true` → solo local provider |
| Providers intercambiables | BaseModelProvider implementado × 5 |
| Memoria abstraída | `POST /api/aura/perceive` + `GET /api/aura/status` |
| Simulación funciona | `POST /api/aura/simulate` → eventos virtuales |
| Tests pasan | `pytest tests/ -v` → 35+ tests OK |

### 🔜 Requiere hardware físico

| Criterio | Prerequisito |
|---|---|
| LLM local responde | Jetson + modelo GGUF descargado |
| Cámara funciona | Camera Module 3 conectada |
| LiDAR detecta objetos | RPLIDAR A1 conectado |
| IMU reporta orientación | MPU-6050 vía I2C |
| Servo se mueve | PCA9685 + servo + sandbox activado |

---

## Próximos pasos recomendados

### Fase 2 — NEXUS Ω Core (siguiente)

1. **Tool Registry** — registro de herramientas invocables (web search, file ops)
2. **Intent Router** — conectar Reasoning al ciclo completo: Perceive → Reason → Plan → Act
3. **Memory persistence** — migrar MemoryStore de RAM a SQLite
4. **Conversation history real** — almacenar en Memory, no solo en localStorage
5. **Streaming responses** — SSE para respuestas largas

### Fase 3 — Capability Engine

6. **Tool calling** — Gemini/local function calling
7. **Code execution sandbox** — ejecutar código generado en entorno aislado
8. **Capability Registry** — registro de capacidades nuevas

### Fase 4 — Hardware integration

9. Adquirir Jetson Orin NX
10. Instalar modelo GGUF local
11. Verificar NEXUS_LOCAL_ONLY funciona sin Internet
12. Conectar cámara → pipeline visión
13. Conectar LiDAR → detección obstáculos
14. Conectar IMU → odometría básica
15. Conectar MCU → control servos (sandbox)
