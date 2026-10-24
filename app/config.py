"""Configuração via variáveis de ambiente (.env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings do microsserviço CardioPrenatal-ML."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "CardioPrenatal-ML"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8010
    model_path: str = "models/best_model.joblib"
    dataset_path: str = "dataset_com_id_gestante.csv"
    metrics_path: str = "data/metricas_modelos.csv"
    log_level: str = "info"
    cors_origins: str = (
        "http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # Segurança: chave compartilhada com o Laravel (X-API-Key / Bearer)
    # Se vazio → auth desligada (dev). Em produção, DEFINA um valor forte.
    api_key: str | None = None

    # Interpretação clínica (Gemini / Vertex) — Fase 1
    interpretacao_vertex_enabled: bool = False
    google_cloud_project: str | None = None
    vertex_location: str = "us-central1"
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.2
    google_api_key: str | None = None
    interpretacao_fallback_mock: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
