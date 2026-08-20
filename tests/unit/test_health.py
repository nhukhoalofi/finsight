from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.routes import health
from app.main import app

client = TestClient(app)


def test_liveness() -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "finsight-api"}


def test_readiness_when_dependencies_are_healthy(monkeypatch) -> None:
    monkeypatch.setattr(health, "check_postgres_connection", AsyncMock())
    monkeypatch.setattr(health, "check_qdrant_connection", AsyncMock())

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"postgres": "ok", "qdrant": "ok"},
    }


def test_readiness_when_postgres_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        health,
        "check_postgres_connection",
        AsyncMock(side_effect=ConnectionError()),
    )
    monkeypatch.setattr(health, "check_qdrant_connection", AsyncMock())

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"] == {"postgres": "error", "qdrant": "ok"}


def test_readiness_when_qdrant_fails(monkeypatch) -> None:
    monkeypatch.setattr(health, "check_postgres_connection", AsyncMock())
    monkeypatch.setattr(
        health,
        "check_qdrant_connection",
        AsyncMock(side_effect=ConnectionError()),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"] == {"postgres": "ok", "qdrant": "error"}
