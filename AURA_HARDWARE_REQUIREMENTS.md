# AURA — Requisitos de Hardware del Head Computer

**Versión:** 1.0  
**Estado:** ESPECIFICACIÓN — No comprar todavía  
**Propósito:** Ejecutar NEXUS Ω localmente (modelo LLM + percepción + control)

---

## 1. Resumen de requisitos

AURA necesita un computador embebido capaz de:
- Ejecutar un modelo LLM de 7B–13B parámetros en local (cuantizado GGUF)
- Procesar video en tiempo real (640×480 @ 30fps mínimo)
- Leer sensores IMU, LiDAR, audio simultáneamente
- Comunicarse con microcontroladores (servos, motores) vía UART/I2C/SPI
- Operar sin conexión a internet

---

## 2. CPU

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Arquitectura | ARM64 / x86_64 | ARM64 (eficiencia energética) |
| Cores | 4 | 8+ |
| Frecuencia | 1.8 GHz | 2.4 GHz+ |
| TDP | < 15W | < 10W (móvil) |

**Candidatos:**
- Raspberry Pi 5 (Cortex-A76 @ 2.4GHz, 4 cores) — económico, ecosistema amplio
- NVIDIA Jetson Orin NX — recomendado para visión + LLM
- Orange Pi 5 Plus (RK3588, 8 cores) — alto rendimiento, bajo costo
- Intel NUC (x86_64) — máxima compatibilidad, mayor consumo

---

## 3. RAM

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Capacidad | 8 GB | 16 GB |
| Tipo | LPDDR4 | LPDDR5 |

**Justificación:**
- Modelo LLM 7B Q4 ocupa ~4-5 GB en RAM (CPU inference)
- Sistema operativo + NEXUS: ~1.5 GB
- Buffers de percepción (video, LiDAR): ~500 MB
- Margen: ~2 GB

Con 8 GB es viable para modelos 7B.
Con 16 GB se pueden ejecutar modelos 13B.

---

## 4. GPU / NPU

| Opción | Capacidad | Ideal para |
|---|---|---|
| Sin GPU (CPU only) | llama.cpp en CPU | modelos ≤ 7B Q4, lento |
| NVIDIA Jetson Orin NX 8GB | 1024 CUDA cores + 32 TOPS | visión + LLM acelerado |
| NPU integrada (RK3588) | 6 TOPS | inferencia ligera |
| Apple M-series (Mac Mini M4) | Neural Engine 38 TOPS | máximo rendimiento local |

**Recomendación para AURA:**  
Jetson Orin NX 8GB o 16GB — es el punto óptimo entre potencia, consumo y ecosistema robótico (ROS2, CUDA, JetPack).

---

## 5. Almacenamiento

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Tipo | microSD UHS-I | NVMe SSD (M.2) |
| Capacidad | 64 GB | 256 GB |
| Velocidad | 80 MB/s | 400 MB/s+ |

**Uso:**
- SO + NEXUS: ~10 GB
- Modelo LLM (7B Q4): ~4 GB
- Modelo LLM (13B Q4): ~8 GB
- Logs, memoria episódica, capturas: ~20 GB

---

## 6. Consumo energético

| Componente | Consumo estimado |
|---|---|
| Head Computer (Jetson Orin NX) | 10–25W |
| Cámara RGB | 1–2W |
| LiDAR (RPLIDAR A1) | 2–3W |
| IMU | < 0.1W |
| Micrófono | < 0.5W |
| **Total (sin motores)** | **~15–30W** |

Para operación autónoma con batería: LiPo 5000mAh @ 5V ≈ ~1-2h de autonomía sin motores.

---

## 7. Interfaces de comunicación

| Interface | Uso |
|---|---|
| USB 3.0 × 2+ | Cámara, LiDAR, hub |
| UART / RS232 | Comunicación con MCU (Arduino, ESP32) |
| I2C | IMU (MPU-6050, BNO085) |
| SPI | Sensores de alta velocidad |
| GPIO | Señales digitales, triggers |
| Ethernet / WiFi 6 | Comunicación con servidor / nube cuando disponible |
| Bluetooth 5.0 | Control remoto, sensores inalámbricos |

---

## 8. Cámara

| Especificación | Mínimo | Recomendado |
|---|---|---|
| Resolución | 640×480 | 1920×1080 |
| FPS | 30 | 60 |
| Interface | USB / CSI | CSI (menor latencia) |
| Profundidad | No | Opcional (Intel RealSense D435i) |

**Candidatos:**
- Raspberry Pi Camera Module 3 (CSI, 12MP, HDR)
- Intel RealSense D435i (RGB + depth + IMU integrada)
- Logitech C920 (USB, 1080p) — económica

---

## 9. LiDAR

| Especificación | Valor |
|---|---|
| Tipo recomendado | 2D (para navegación básica) |
| Rango | 6–12m |
| Frecuencia de escaneo | 5–10 Hz |
| Interface | USB / UART |

**Candidatos:**
- RPLIDAR A1M8 (12m, 8000 samples/s, ~$100) — punto de entrada
- RPLIDAR C1 (12m, 360°, compacto, ~$80)
- Livox Mid-360 (3D, 40m, industrial, ~$500) — futuro

---

## 10. Audio

| Componente | Especificación |
|---|---|
| Micrófono | Array de 2–4 micrófonos (dirección de sonido) |
| Sample rate | 16 kHz mínimo para STT |
| Interface | USB o I2S |

**Candidato:** ReSpeaker USB Mic Array v2 (4 mics, detección de dirección)

---

## 11. Comunicación con MCU

AURA necesita un microcontrolador para el control de bajo nivel:

| MCU | Ventajas |
|---|---|
| ESP32-S3 | WiFi/BT integrado, FPU, bajo costo |
| Arduino Mega | Ecosistema amplio, muchos pines |
| Teensy 4.1 | Alta velocidad, USB nativo |
| STM32 | Industrial, alta precisión |

**Protocolo de comunicación Head Computer ↔ MCU:**
- UART @ 115200–921600 bps
- Protocolo: JSON lines o protobuf
- Comandos: servo_move, motor_speed, sensor_read
- Heartbeat: cada 500ms para detectar desconexión

---

## 12. Modelos LLM locales compatibles

### Formato requerido: GGUF (llama.cpp)

| Modelo | Parámetros | RAM Q4 | Velocidad (CPU) | Velocidad (GPU) |
|---|---|---|---|---|
| Llama 3.2 3B | 3B | ~2 GB | ~8 tok/s | ~25 tok/s |
| Llama 3.1 8B | 8B | ~5 GB | ~4 tok/s | ~15 tok/s |
| Mistral 7B v0.3 | 7B | ~4.5 GB | ~4 tok/s | ~18 tok/s |
| Qwen 2.5 7B | 7B | ~4.5 GB | ~4 tok/s | ~18 tok/s |
| Phi-3 Mini 3.8B | 3.8B | ~2.3 GB | ~7 tok/s | ~22 tok/s |
| Gemma 2 9B | 9B | ~5.5 GB | ~3 tok/s | ~12 tok/s |

**Recomendación inicial:** Llama 3.2 3B (Q4_K_M) — mínimo 8GB RAM, rápido en CPU.  
**Recomendación producción:** Llama 3.1 8B (Q4_K_M) — requiere 8GB RAM mínimo.

### Fuentes de descarga:
- https://huggingface.co/bartowski (GGUF optimizados)
- https://huggingface.co/lmstudio-community
- Comando: `huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF`

---

## 13. Software stack del Head Computer

```
OS:       Ubuntu 22.04 LTS (ARM64) o JetPack (Jetson)
Runtime:  Python 3.11+
LLM:      llama-cpp-python (con soporte CUDA en Jetson)
Vision:   OpenCV 4.x
Audio:    whisper.cpp (STT), espeak-ng (TTS)
ROS:      ROS2 Humble (futuro, para robótica)
Comm:     pyserial (MCU), asyncio (interno)
Deploy:   Docker o systemd service
```

---

## 14. Configuración de NEXUS para el Head Computer

Variables de entorno en el hardware físico:

```bash
NEXUS_LOCAL_ONLY=true
LOCAL_RUNTIME=llama_cpp
LOCAL_MODEL_PATH=/models/llama-3.2-3b-q4_k_m.gguf
LOCAL_OLLAMA_URL=http://localhost:11434
LOG_LEVEL=INFO
MAX_OUTPUT_TOKENS=2048
REQUEST_TIMEOUT_SECONDS=60
```

---

## 15. Próximos pasos de hardware

1. **Adquirir kit de desarrollo:** Jetson Orin NX 8GB Developer Kit
2. **Instalar JetPack** y configurar NEXUS
3. **Descargar modelo:** Llama 3.2 3B Q4_K_M
4. **Verificar:** `NEXUS_LOCAL_ONLY=true` funciona sin internet
5. **Integrar cámara** + probar pipeline de visión
6. **Integrar LiDAR** + probar detección de obstáculos
7. **Integrar IMU** + probar orientación
8. **Conectar MCU** + probar control de servos (en modo simulado primero)

**NO comprar hardware hasta verificar que el software está listo.**
