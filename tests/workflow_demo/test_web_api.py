"""HTTP contract for the local coach bridge. Coach startup is mocked."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.workflow_demo import web_api

pytestmark = pytest.mark.integration


class _FakeCoach:
    def initial_greeting(self) -> str:
        return "hello"

    def process_turn(self, message: str) -> str:
        return f"echo:{message}"

    def get_pedagogy_snapshot_for_api(self):
        return {"phase": "test"}

    def tutor_session_active_for_api(self) -> bool:
        return False


@pytest.fixture
def client(monkeypatch):
    web_api._SESSIONS.clear()
    monkeypatch.setattr(web_api, "build_coach_runtime", lambda **kwargs: _FakeCoach())
    with TestClient(web_api.app) as test_client:
        yield test_client
    web_api._SESSIONS.clear()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_then_chat_returns_mocked_reply(client):
    created = client.post("/api/session")
    assert created.status_code == 200
    body = created.json()
    assert body["error"] is None
    assert body["greeting"] == "hello"
    assert body["session_id"]

    chat = client.post(
        "/api/chat",
        json={"session_id": body["session_id"], "message": "hi"},
    )
    assert chat.status_code == 200
    assert chat.json()["response"] == "echo:hi"
    assert chat.json()["pedagogy_snapshot"] == {"phase": "test"}
    assert chat.json()["tutor_session_active"] is False


def test_failed_create_returns_error_and_chat_is_404(client, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("missing key")

    monkeypatch.setattr(web_api, "build_coach_runtime", _boom)
    created = client.post("/api/session")
    assert created.status_code == 200
    body = created.json()
    assert body["error"] == "missing key"
    assert body["greeting"] == ""
    assert body["session_id"] not in web_api._SESSIONS

    chat = client.post(
        "/api/chat",
        json={"session_id": body["session_id"], "message": "hi"},
    )
    assert chat.status_code == 404
