# ============================================================
# NEXUS Ω v3.4.0 — Deploy Script PowerShell
# ============================================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  NEXUS Omega v3.4.0 - AURA Local Brain" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# --- PASO 1: Repo ---
Write-Host "[1/7] Preparando repositorio..." -ForegroundColor Yellow
if (-not (Test-Path "nexus-backend")) {
    git clone https://github.com/manuelivansr19-commits/nexus-backend.git
}
Set-Location nexus-backend

# --- PASO 2: Estructura ---
Write-Host "[2/7] Creando estructura de carpetas..." -ForegroundColor Yellow
$folders = @(
    "backend", "backend/providers", "backend/core",
    "backend/hardware", "backend/simulation", "tests"
)
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force | Out-Null
        Write-Host "  + $f" -ForegroundColor Green
    }
}

# --- PASO 3: Copiar archivos ---
Write-Host ""
Write-Host "[3/7] COPIAR ARCHIVOS DESCARGADOS" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Raiz:" -ForegroundColor White
Write-Host "    .gitignore  requirements.txt  CHANGELOG.md  index.html" -ForegroundColor Gray
Write-Host "    NEXUS_ARCHITECTURE_TARGET.md" -ForegroundColor Gray
Write-Host "    AURA_HARDWARE_REQUIREMENTS.md" -ForegroundColor Gray
Write-Host ""
Write-Host "  backend/" -ForegroundColor White
Write-Host "    __init__.py  config.py  main.py  router.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  backend/providers/" -ForegroundColor White
Write-Host "    __init__.py  base.py  local.py  gemini.py" -ForegroundColor Gray
Write-Host "    openrouter.py  groq.py  ollama.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  backend/core/" -ForegroundColor White
Write-Host "    __init__.py  perception.py  memory.py  reasoning.py" -ForegroundColor Gray
Write-Host "    planning.py  action.py  evaluation.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  backend/hardware/" -ForegroundColor White
Write-Host "    __init__.py  base.py  camera.py  lidar.py" -ForegroundColor Gray
Write-Host "    microphone.py  imu.py  servo.py  motor.py  sensor.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  backend/simulation/" -ForegroundColor White
Write-Host "    __init__.py  engine.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  tests/" -ForegroundColor White
Write-Host "    __init__.py  test_core.py  test_api.py  test_offline.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  Presiona ENTER cuando hayas copiado todos los archivos..." -ForegroundColor Cyan
Read-Host

# --- PASO 4: Limpiar obsoletos ---
Write-Host "[4/7] Limpiando archivos obsoletos..." -ForegroundColor Yellow
$remove = @("main.py", "data", "frontend")
foreach ($item in $remove) {
    if (Test-Path $item) {
        Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  - Eliminado: $item" -ForegroundColor Red
    }
}

# --- PASO 5: Verificar ---
Write-Host ""
Write-Host "[5/7] Verificando archivos..." -ForegroundColor Yellow

$required = @(
    ".gitignore","requirements.txt","CHANGELOG.md","index.html",
    "NEXUS_ARCHITECTURE_TARGET.md","AURA_HARDWARE_REQUIREMENTS.md",
    "backend/__init__.py","backend/config.py","backend/main.py","backend/router.py",
    "backend/providers/__init__.py","backend/providers/base.py",
    "backend/providers/local.py","backend/providers/gemini.py",
    "backend/providers/openrouter.py","backend/providers/groq.py",
    "backend/providers/ollama.py",
    "backend/core/__init__.py","backend/core/perception.py",
    "backend/core/memory.py","backend/core/reasoning.py",
    "backend/core/planning.py","backend/core/action.py",
    "backend/core/evaluation.py",
    "backend/hardware/__init__.py","backend/hardware/base.py",
    "backend/hardware/camera.py","backend/hardware/lidar.py",
    "backend/hardware/microphone.py","backend/hardware/imu.py",
    "backend/hardware/servo.py","backend/hardware/motor.py",
    "backend/hardware/sensor.py",
    "backend/simulation/__init__.py","backend/simulation/engine.py",
    "tests/__init__.py","tests/test_core.py",
    "tests/test_api.py","tests/test_offline.py"
)

$missing = @()
foreach ($file in $required) {
    if (Test-Path $file) {
        Write-Host "  OK   $file" -ForegroundColor Green
    } else {
        Write-Host "  FALTA $file" -ForegroundColor Red
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Faltan $($missing.Count) archivos. Revisa antes de continuar." -ForegroundColor Red
    Write-Host "ENTER para continuar de todos modos, Ctrl+C para cancelar..." -ForegroundColor Yellow
    Read-Host
}

# --- PASO 6: Commits atómicos ---
Write-Host ""
Write-Host "[6/7] Commits atómicos..." -ForegroundColor Yellow

git add .gitignore requirements.txt
git commit -m "chore: update gitignore and pin dependencies v3.4.0"

git add backend/config.py
git commit -m "feat: add LOCAL_ENGINE config (NEXUS_LOCAL_ONLY, LOCAL_MODEL_PATH)"

git add backend/providers/local.py backend/providers/__init__.py
git commit -m "feat(providers): add LocalProvider (llama.cpp + Ollama local)"

git add backend/providers/gemini.py
git commit -m "fix(gemini): add retry with exponential backoff + jitter (hotfix 503)"

git add backend/router.py
git commit -m "feat(router): support NEXUS_LOCAL_ONLY mode and is_local providers"

git add backend/core/
git commit -m "feat(core): add AURA Brain API (Perception, Memory, Reasoning, Planning, Action, Evaluation)"

git add backend/hardware/
git commit -m "feat(hardware): add abstract hardware interfaces (Camera, Lidar, IMU, Servo, Motor) - simulated"

git add backend/simulation/
git commit -m "feat(simulation): add SimulationEngine with 4 scenarios (idle, exploring, conversation, obstacle)"

git add backend/main.py
git commit -m "feat(api): add /api/aura/status, /api/aura/perceive, /api/aura/simulate endpoints"

git add tests/test_offline.py
git commit -m "test: add offline mode and AURA brain tests"

git add CHANGELOG.md NEXUS_ARCHITECTURE_TARGET.md AURA_HARDWARE_REQUIREMENTS.md
git commit -m "docs: update changelog, add target architecture and hardware requirements"

git add -A
git commit -m "chore: remove obsolete files, clean repo structure" --allow-empty

# --- PASO 7: Push ---
Write-Host ""
Write-Host "[7/7] Push a GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  NEXUS Omega v3.4.0 desplegado!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Render — Start Command:" -ForegroundColor Yellow
Write-Host '  uvicorn backend.main:app --host 0.0.0.0 --port $PORT' -ForegroundColor White
Write-Host ""
Write-Host "Verificar deploy:" -ForegroundColor Yellow
Write-Host "  /health          → providers + local_mode" -ForegroundColor Cyan
Write-Host "  /api/aura/status → cerebro AURA" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para activar modo offline en Render (test):" -ForegroundColor Yellow
Write-Host "  NEXUS_LOCAL_ONLY=true (Render env vars)" -ForegroundColor White
Write-Host "  Luego: GET /health → todos los externos = false" -ForegroundColor Gray
Write-Host ""
