"""
Sondeo de la Fase 1 (NO es una prueba: lo ejecutan las pruebas en un subproceso).

Arranca la aplicación REAL (lifespan incluido, política construida por
backend/main.py) con el entorno que le pase el proceso padre, bajo una guarda de
red que REGISTRA y BLOQUEA cualquier destino que no sea loopback. Así nunca
sale tráfico real, aunque el código bajo prueba intente salir.

Imprime una sola línea final:  PROBE_JSON:{...}

Debe ejecutarse con un directorio actual temporal: las bases SQLite con rutas
relativas se crean allí, y el sondeo informa qué archivos aparecieron.
"""
import json
import os
import socket
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ATTEMPTS = []


def _is_loopback(host):
    h = str(host).lower()
    return h in ("localhost", "::1") or h.startswith("127.")


_real_getaddrinfo = socket.getaddrinfo


def _guard_getaddrinfo(host, *args, **kwargs):
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host is not None and not _is_loopback(host):
        ATTEMPTS.append(str(host))
        raise socket.gaierror(-2, "bloqueado por la guarda de red de prueba")
    return _real_getaddrinfo(host, *args, **kwargs)


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _guard_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if not _is_loopback(host):
        ATTEMPTS.append(str(host))
        raise OSError("bloqueado por la guarda de red de prueba")
    return _real_connect(self, address)


def _guard_connect_ex(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if not _is_loopback(host):
        ATTEMPTS.append(str(host))
        return 1
    return _real_connect_ex(self, address)


socket.getaddrinfo = _guard_getaddrinfo
socket.socket.connect = _guard_connect
socket.socket.connect_ex = _guard_connect_ex


def _endpoint_view(client, path):
    r = client.get(path)
    j = r.json()
    return {
        "status_code": r.status_code,
        "inference_mode": j.get("inference_mode", j.get("mode")),
        "cloud_allowed": j.get("cloud_allowed"),
        "cloud_enabled": j.get("cloud_enabled", "n/a"),
        "config_warnings": j.get("config_warnings", "n/a"),
    }


def main():
    out = {"cwd": os.getcwd()}
    from fastapi.testclient import TestClient
    import backend.main as m
    import backend.config as cfg
    from backend.inference import InferenceRequest

    out["config_paths"] = {
        "memory": cfg.MEMORY_DB_PATH,
        "knowledge": cfg.KNOWLEDGE_DB_PATH,
        "tasks": getattr(cfg, "TASKS_DB_PATH", None),
    }

    with TestClient(m.app) as client:
        out["endpoints"] = {
            "health": _endpoint_view(client, "/health"),
            "status": _endpoint_view(client, "/api/nexus/status"),
            "config": _endpoint_view(client, "/api/nexus/config"),
            "inference": _endpoint_view(client, "/api/inference/health"),
        }

        # 1) Capa de inferencia real, tal como la construyó main.py.
        # Se ejecuta en el MISMO bucle de eventos de la app (portal del
        # TestClient): otro bucle haría fallar el cliente HTTP compartido antes
        # de intentar salir a la red y daría un falso "cero tráfico".
        try:
            res = client.portal.call(
                m.inference_layer.generate, InferenceRequest(prompt="ping de prueba"))
            out["generate"] = {"ok": True, "runtime": res.runtime_name}
        except Exception as e:  # noqa: BLE001
            out["generate"] = {"ok": False, "error": type(e).__name__}

        # 2) Flujo completo por la API.
        try:
            r = client.post("/api/nexus/chat", json={"message": "hola"})
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            text = str(body.get("response", ""))
            out["chat"] = {"status_code": r.status_code,
                           "degraded_text": "modo degradado" in text.lower()}
        except Exception as e:  # noqa: BLE001
            out["chat"] = {"status_code": None, "error": type(e).__name__}

    out["attempts"] = sorted(set(ATTEMPTS))
    out["db_files_in_cwd"] = sorted(f for f in os.listdir(os.getcwd()) if f.endswith(".db"))
    out["db_exists"] = {
        k: (os.path.exists(v) if v else None) for k, v in out["config_paths"].items()
    }
    return out


if __name__ == "__main__":
    try:
        result = main()
    except Exception:  # noqa: BLE001
        result = {"fatal": traceback.format_exc()[-1500:], "attempts": sorted(set(ATTEMPTS))}
    print("PROBE_JSON:" + json.dumps(result))
