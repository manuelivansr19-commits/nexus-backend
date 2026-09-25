"""
NEXUS Ω — Tests de API (endpoints HTTP).

Ejecutar: python -m pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)
client.__enter__()  # dispara lifespan (fix)


class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["system"] == "NEXUS"
        assert "version" in data
        # "providers" no forma parte del contrato actual de /health

    def test_health_has_provider_status(self):
        response = client.get("/health")
        data = response.json()
        assert "inference_mode" in data
        assert "cloud_allowed" in data

    def test_head_returns_200(self):
        response = client.head("/")
        assert response.status_code == 200


class TestStatus:

    def test_status_returns_200(self):
        response = client.get("/api/nexus/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        # "router" no forma parte del contrato actual de /api/nexus/status


class TestConfig:

    def test_config_returns_200(self):
        response = client.get("/api/nexus/config")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "autonomy_enabled" in data
        assert "knowledge_enabled" in data
        assert data["multi_provider"] is True


class TestChat:

    def test_empty_message_returns_static(self):
        response = client.post(
            "/api/nexus/chat",
            json={"message": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["provider"] == "system"
        assert "request_id" in data

    def test_whitespace_message_returns_static(self):
        response = client.post(
            "/api/nexus/chat",
            json={"message": "   "},
        )
        assert response.status_code == 200
        assert response.json()["provider"] == "system"

    def test_chat_validates_max_length(self):
        response = client.post(
            "/api/nexus/chat",
            json={"message": "x" * 30001},
        )
        assert response.status_code == 422  # Pydantic validation

    def test_chat_accepts_history(self):
        response = client.post(
            "/api/nexus/chat",
            json={
                "message": "",
                "history": [
                    {"role": "user", "content": "Hola"},
                    {"role": "assistant", "content": "Hola"},
                ],
            },
        )
        assert response.status_code == 200

    def test_chat_rejects_invalid_role(self):
        response = client.post(
            "/api/nexus/chat",
            json={
                "message": "test",
                "history": [
                    {"role": "admin", "content": "hack"},
                ],
            },
        )
        assert response.status_code == 422


class TestServiceWorker:

    def test_sw_returns_javascript(self):
        response = client.get("/sw.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]
        assert "skipWaiting" in response.text
