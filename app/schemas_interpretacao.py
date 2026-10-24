"""Schemas PatientContext + interpretação clínica (Fase 2 — longitudinal)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InterpretacaoRequest(BaseModel):
    """PatientContext v1.1 enviado pelo Laravel (ClinicalContextBuilder)."""

    schema_version: str = "1.1"
    gestante: dict[str, Any] = Field(default_factory=dict)
    consulta_atual: dict[str, Any] = Field(default_factory=dict)
    consulta_anterior: dict[str, Any] | None = None
    historico_consultas: list[dict[str, Any]] = Field(default_factory=list)
    predicao_catboost: dict[str, Any] = Field(default_factory=dict)
    predicao_anterior: dict[str, Any] | None = None
    deltas_numericos: list[dict[str, Any]] = Field(default_factory=list)
    interpretacao_anterior: dict[str, Any] | None = None
    politica: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class FatorExplicado(BaseModel):
    feature: str
    impacto: float | None = None
    explicacao: str


class InterpretacaoResponse(BaseModel):
    mock: bool = True
    model_id: str = "mock-fase2"
    prompt_version: str = "system_v2"
    schema_version: str = "1.1"
    cache_hit: bool = False
    resumo_medico: str
    resumo_gestante: str | None = None
    fatores_explicados: list[FatorExplicado] = Field(default_factory=list)
    recomendacoes: list[str] = Field(default_factory=list)
    deltas: list[str] = Field(default_factory=list)
    # Eco da predição CatBoost — Laravel valida que não foi alterada
    predicao_ecoada: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    aviso: str | None = None
