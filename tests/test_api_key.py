"""Testes de autenticação por API Key."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"
TEST_KEY = "test-cardioprenatal-api-key-32chars!!"


@pytest.fixture()
def client_with_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DATASET_PATH", str(FIXTURES / "dataset_min.csv"))
    monkeypatch.setenv("METRICS_PATH", str(FIXTURES / "metricas_min.csv"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing_model.pkl"))
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("API_KEY", TEST_KEY)

    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def test_health_public_without_key(client_with_auth: TestClient):
    r = client_with_auth.get("/health")
    assert r.status_code == 200


def test_predict_rejected_without_key(client_with_auth: TestClient):
    r = client_with_auth.post(
        "/predict",
        json={"idade": 30, "diabetes": 0, "hipertensao": 0, "imc": 24, "idade_gestacional": 20, "tabagismo": 0, "paridade": 0},
    )
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_predict_accepted_with_x_api_key(client_with_auth: TestClient):
    r = client_with_auth.post(
        "/predict",
        headers={"X-API-Key": TEST_KEY},
        json={"idade": 30, "diabetes": 0, "hipertensao": 0, "imc": 24, "idade_gestacional": 20, "tabagismo": 0, "paridade": 0},
    )
    assert r.status_code == 200
    assert "probabilidade_risco" in r.json()


def test_gestantes_accepted_with_bearer(client_with_auth: TestClient):
    r = client_with_auth.get(
        "/gestantes",
        headers={"Authorization": f"Bearer {TEST_KEY}"},
        params={"limit": 1},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_wrong_key_rejected(client_with_auth: TestClient):
    r = client_with_auth.get("/gestantes", headers={"X-API-Key": "chave-errada"})
    assert r.status_code == 401
