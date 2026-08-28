"""
NEXUS Ω — Tests de API (endpoints HTTP).

Ejecutar: python -m pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["system"] == "NEXUS"
        assert "version" in data
        assert "providers" in data

    def test_health_has_provider_status(self):
        response = client.get("/health")
        providers = response.json()["providers"]
        assert "gemini" in providers
        assert "openrouter" in providers
        assert "groq" in providers
        assert "ollama" in providers

    def test_head_returns_200(self):
        response = client.head("/")
        assert response.status_code == 200


class TestStatus:

    def test_status_returns_200(self):
        response = client.get("/api/nexus/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "router" in data


class TestConfig:

    def test_config_returns_200(self):
        response = client.get("/api/nexus/config")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "max_output_tokens" in data
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
