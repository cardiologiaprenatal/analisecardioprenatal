"""Testes de contrato da API canônica CardioPrenatal-ML."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # aponta settings para fixtures mínimas
    monkeypatch.setenv("DATASET_PATH", str(FIXTURES / "dataset_min.csv"))
    monkeypatch.setenv("METRICS_PATH", str(FIXTURES / "metricas_min.csv"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing_model.pkl"))
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("API_KEY", "")  # auth off nos testes de contrato

    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["dataset_loaded"] is True
    assert data["gestantes"] == 3
    assert data["metrics_loaded"] is True
    assert data["model_loaded"] is False  # heuristic


def test_modelos(client: TestClient):
    r = client.get("/modelos")
    assert r.status_code == 200
    data = r.json()
    assert data["disponivel"] is True
    assert data["total_modelos"] >= 2
    assert "treino_cv" in data["modelos"][0]
    assert "teste" in data["modelos"][0]


def test_gestantes_list(client: TestClient):
    r = client.get("/gestantes", params={"limit": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert len(data["gestantes"]) == 1
    assert data["gestantes"][0]["id_gestante"].startswith("GEST-")


def test_gestantes_sync_before_id_route(client: TestClient):
    r = client.get("/gestantes/sync", params={"page": 1, "per_page": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["page"] == 1
    assert data["per_page"] == 2
    assert len(data["gestantes"]) == 2


def test_gestante_by_id(client: TestClient):
    r = client.get("/gestantes/GEST-00001")
    assert r.status_code == 200
    data = r.json()
    assert data["id_gestante"] == "GEST-00001"
    assert "dados_clinicos" in data
    assert "features_modelo" in data
    assert len(data["features_modelo"]) >= 30


def test_gestante_case_insensitive(client: TestClient):
    r = client.get("/gestantes/gest-00001")
    assert r.status_code == 200
    assert r.json()["id_gestante"] == "GEST-00001"


def test_predict_json(client: TestClient):
    r = client.post(
        "/predict",
        json={
            "idade": 32,
            "diabetes": 1,
            "hipertensao": 0,
            "imc": 31,
            "idade_gestacional": 22,
            "tabagismo": 0,
            "paridade": 1,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "classificacao_risco" in data
    assert "probabilidade_risco" in data
    assert "risk" in data
    assert "probability" in data
    assert 0 <= data["probabilidade_risco"] <= 1
    assert isinstance(data["shap"], list)


def test_predict_by_id(client: TestClient):
    r = client.post("/predict/GEST-00001")
    assert r.status_code == 200
    data = r.json()
    assert data["id_gestante"] == "GEST-00001"
    assert data["classificacao_risco"] in {"Baixo", "Moderado", "Alto"}
    assert 0 <= data["probabilidade_risco"] <= 1


def test_analisar_completo(client: TestClient):
    r = client.post(
        "/analisar/completo",
        params={"persistir": "false"},
        json={
            "id_gestante": "GEST-TEST",
            "idade_materna": 36,
            "etnia": "Branca",
            "imc_pre_gestacional": 29,
            "paridade": 1,
            "diabetes_pre_gestacional": 1,
            "hipertensao_cronica": 1,
            "tabagismo": 0,
            "idade_gestacional": 20,
            "translucencia_nucal_aumentada": 1,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "predicao" in data
    assert "analise" in data
    assert "recomendacoes" in data
    assert "shap" in data
    assert data["metadados"]["persistido"] is False


def test_graficos_catalog(client: TestClient):
    r = client.get("/graficos")
    assert r.status_code == 200
    data = r.json()
    assert "graficos" in data
