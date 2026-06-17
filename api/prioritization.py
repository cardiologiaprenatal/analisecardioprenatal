"""Algoritmo clínico de priorização para ecocardiograma fetal."""

from __future__ import annotations

from typing import Any

MARCADORES_FETAIS_CRITICOS = {
    "translucencia_nucal_aumentada",
    "alteracao_ducto_venoso",
    "regurgitacao_tricuspide",
    "malformacao_extracardiaca",
}

INTERCORRENCIA_GRAVE_PESO = 1.5


def _contar_graves(intercorrencias: list[dict[str, Any]]) -> int:
    count = 0
    for item in intercorrencias:
        if item.get("categoria") == "marcadores_fetais":
            count += 1
        elif item.get("peso_risco", 0) >= INTERCORRENCIA_GRAVE_PESO:
            count += 1
    return count


def _contar_marcadores_criticos(intercorrencias: list[dict[str, Any]]) -> int:
    return sum(1 for i in intercorrencias if i.get("campo") in MARCADORES_FETAIS_CRITICOS)


def calcular_priorizacao_clinica(
    probabilidade: float,
    intercorrencias: list[dict[str, Any]],
    idade_gestacional: float,
) -> dict[str, Any]:
    """
    Score 0–100 considerando:
    1. risco previsto (35%)
    2. intercorrências graves (25%)
    3. marcadores fetais críticos (20%)
    4. idade gestacional — janela ideal 18–24 sem (10%)
    5. volume total de intercorrências (10%)
    """
    graves = _contar_graves(intercorrencias)
    criticos = _contar_marcadores_criticos(intercorrencias)
    total = len(intercorrencias)

    score_risco = probabilidade * 35
    score_graves = min(graves * 8, 25)
    score_criticos = min(criticos * 10, 20)

    if 18 <= idade_gestacional <= 24:
        score_ig = 10
    elif 11 <= idade_gestacional < 18 or 24 < idade_gestacional <= 28:
        score_ig = 6
    else:
        score_ig = 2

    score_volume = min(total * 2, 10)

    score = round(min(score_risco + score_graves + score_criticos + score_ig + score_volume, 100), 2)
    prioridade = classificar_prioridade(score)

    return {
        "score_prioridade": score,
        "prioridade_ecocardiograma": prioridade,
        "fatores": {
            "componente_risco": round(score_risco, 2),
            "componente_intercorrencias_graves": round(score_graves, 2),
            "componente_marcadores_criticos": round(score_criticos, 2),
            "componente_idade_gestacional": round(score_ig, 2),
            "componente_volume_intercorrencias": round(score_volume, 2),
            "intercorrencias_graves": graves,
            "marcadores_fetais_criticos": criticos,
        },
    }


def classificar_prioridade(score: float) -> str:
    if score >= 75:
        return "urgente"
    if score >= 50:
        return "alta"
    if score >= 30:
        return "moderada"
    return "baixa"


def classificar_risco(probabilidade: float) -> str:
    if probabilidade < 0.15:
        return "baixo"
    if probabilidade < 0.35:
        return "moderado"
    return "alto"


def recomendar_ecocardiograma(
    probabilidade: float,
    prioridade: str,
    intercorrencias: list[dict[str, Any]],
) -> bool:
    if prioridade in ("urgente", "alta"):
        return True
    if probabilidade >= 0.20:
        return True
    if _contar_marcadores_criticos(intercorrencias) > 0:
        return True
    if _contar_graves(intercorrencias) >= 2:
        return True
    return False


def prazo_sugerido(prioridade: str) -> str:
    return {
        "urgente": "imediato",
        "alta": "7_dias",
        "moderada": "14_dias",
        "baixa": "rotina",
    }.get(prioridade, "rotina")
