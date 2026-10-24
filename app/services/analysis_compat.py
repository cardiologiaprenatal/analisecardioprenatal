"""Análise clínica completa (compat CcfApiClient) — sem MySQL / sem persistência."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.services.gestante_repository import map_row_to_model_features
from app.services.predictor import PredictorService

logger = logging.getLogger(__name__)

API_VERSION = "2.0.0"


def _risk_lower(label: str) -> str:
    mapping = {"Baixo": "baixo", "Moderado": "moderado", "Alto": "alto"}
    return mapping.get(label, label.lower())


def _prioridade_eco(probability: float, data: dict[str, Any]) -> tuple[str, float, bool]:
    score = probability * 100.0
    if int(data.get("translucencia_nucal_aumentada") or 0):
        score += 25
    if int(data.get("alteracao_ducto_venoso") or 0):
        score += 20
    if int(data.get("regurgitacao_tricuspide") or 0):
        score += 15
    if int(data.get("malformacao_extracardiaca") or 0):
        score += 15
    if int(data.get("historia_familiar_cardiopatia") or 0):
        score += 10

    if score >= 70:
        prioridade = "urgente"
    elif score >= 50:
        prioridade = "alta"
    elif score >= 30:
        prioridade = "moderada"
    else:
        prioridade = "baixa"

    recomenda = prioridade != "baixa" or probability >= 0.15
    return prioridade, round(score, 2), recomenda


def _intercorrencias(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = [
        ("diabetes_gestacional", "Doenças maternas", "Diabetes gestacional", "alto", 0.18),
        ("diabetes_pre_gestacional", "Doenças maternas", "Diabetes pré-gestacional", "alto", 0.22),
        ("hipertensao_cronica", "Doenças maternas", "Hipertensão crônica", "medio", 0.12),
        ("tabagismo", "Hábitos", "Tabagismo", "medio", 0.10),
        ("alcool", "Hábitos", "Álcool", "medio", 0.08),
        ("translucencia_nucal_aumentada", "Marcadores fetais", "TN aumentada", "alto", 0.25),
        ("alteracao_ducto_venoso", "Marcadores fetais", "Ducto venoso alterado", "alto", 0.20),
        ("regurgitacao_tricuspide", "Marcadores fetais", "Regurgitação tricúspide", "alto", 0.15),
        ("malformacao_extracardiaca", "Marcadores fetais", "Malformação extracardíaca", "alto", 0.18),
        ("historia_familiar_cardiopatia", "Dados maternos", "História familiar CCF", "alto", 0.16),
        ("restricao_crescimento_fetal", "Marcadores fetais", "RCF/IUGR", "medio", 0.10),
    ]
    out: list[dict[str, Any]] = []
    for campo, cat, nome, impacto, peso in flags:
        if int(data.get(campo) or 0) >= 1:
            out.append({
                "categoria": cat.lower().replace(" ", "_"),
                "categoria_label": cat,
                "campo": campo,
                "nome": nome,
                "valor": 1,
                "alteracao": "presente",
                "impacto": impacto,
                "peso_risco": peso,
                "relevancia_modelo": peso,
            })
    return out


def analisar_completo(
    predictor: PredictorService,
    payload: dict[str, Any],
    *,
    persistir: bool = False,
    top_n_shap: int = 36,
) -> dict[str, Any]:
    """
    Resposta rica compatível com CcfApiClient::analisarCompleto.
    persistir é ignorado (API não grava MySQL / histórico local).
    """
    if persistir:
        logger.warning(
            "analisar_completo persistir=true ignorado — API ML não persiste pacientes."
        )

    t0 = time.perf_counter()
    data = dict(payload)
    id_gestante = data.get("id_gestante")
    if isinstance(id_gestante, str):
        id_gestante = id_gestante.strip() or None

    # mescla payload rico + 36 features normalizadas
    features = map_row_to_model_features(data)
    infer_payload = {**data, **features}
    pred = predictor.predict(
        infer_payload,
        quiet=False,
        id_gestante=id_gestante,
    )

    probabilidade = pred.probabilidade_risco
    risco_title = pred.classificacao_risco
    risco = _risk_lower(risco_title)
    pred_class = pred.class_
    prioridade, score_prio, recomenda = _prioridade_eco(probabilidade, data)
    intercorrencias = _intercorrencias(data)
    por_cat: dict[str, int] = {}
    for item in intercorrencias:
        por_cat[item["categoria_label"]] = por_cat.get(item["categoria_label"], 0) + 1

    shap_top = []
    for s in pred.shap[:top_n_shap]:
        shap_top.append({
            "campo": s.feature,
            "nome": s.nome or s.feature,
            "shap_value": s.impact,
            "impacto_absoluto": s.impacto_absoluto or abs(s.impact),
            "percentual_contribuicao": round(abs(s.impact) / max(probabilidade, 1e-6) * 100, 2),
            "direcao": "aumenta_risco" if s.impact >= 0 else "diminui_risco",
            "valor_paciente": data.get(s.feature),
        })

    prazo = {
        "urgente": "até 48h",
        "alta": "até 7 dias",
        "moderada": "até 14 dias",
        "baixa": "rotina",
    }.get(prioridade, "rotina")

    settings = get_settings()
    tempo_ms = round((time.perf_counter() - t0) * 1000, 1)
    framework = predictor.model.framework or "heuristic"

    return {
        "id_gestante": id_gestante,
        "predicao": {
            "probabilidade_ccf": probabilidade,
            "classificacao_risco": risco,
            "cardiopatia_congenita_prevista": pred_class,
            "modelo": framework,
            "experimento": "cardio_prenatal_ml",
        },
        "analise": {
            "probabilidade_ccf": probabilidade,
            "classificacao_risco": risco,
            "score_prioridade": score_prio,
            "prioridade_ecocardiograma": prioridade,
            "posicao_fila": None,
            "recomenda_ecocardiograma": recomenda,
            "resumo_clinico": {
                "tecnico": (
                    f"Paciente classificada como risco {risco} para CCF "
                    f"(probabilidade estimada: {probabilidade * 100:.1f}%). "
                    f"Prioridade para ecocardiograma fetal: {prioridade}."
                ),
                "simplificado": (
                    f"Risco {risco} detectado ({probabilidade * 100:.0f}%). "
                    "Avaliar ecocardiograma conforme protocolo."
                ),
            },
            "resumo_intercorrencias": {
                "total": len(intercorrencias),
                "por_categoria": por_cat,
                "marcadores_fetais_detectados": sum(
                    1 for i in intercorrencias if i["categoria_label"] == "Marcadores fetais"
                ),
                "possui_alteracao_ecografica": any(
                    i["categoria_label"] == "Marcadores fetais" for i in intercorrencias
                ),
            },
            "fatores_priorizacao": {
                "componente_risco": round(probabilidade * 35, 2),
                "prioridade": prioridade,
            },
        },
        "intercorrencias": intercorrencias,
        "shap": {
            "metodo": "heuristic_impacts" if not predictor.model.loaded else "model_impacts",
            "classe_alvo": "cardiopatia_congenita",
            "valor_base": 0.07,
            "soma_shap": round(sum(abs(s.impact) for s in pred.shap), 4),
            "resumo": "Principais contribuições para a probabilidade estimada.",
            "probabilidade_ccf": probabilidade,
            "top_contribuicoes": shap_top,
            "importancia_global": [],
        },
        "recomendacoes": {
            "ecocardiograma_fetal": recomenda,
            "prioridade": prioridade,
            "prazo_sugerido": prazo,
            "acoes": [
                "Registrar resultado da análise no prontuário Laravel.",
                f"Priorizar ecocardiograma fetal: {prioridade} ({prazo}).",
            ],
        },
        "qualidade_dados": {
            "status": "ok",
            "campos_ausentes": [
                k for k, v in features.items() if v is None
            ],
        },
        "metadados": {
            "analise_id": str(uuid.uuid4()),
            "versao_api": API_VERSION,
            "versao_modelo": f"{framework}/{settings.app_version}",
            "data_analise": datetime.now(timezone.utc).isoformat(),
            "tempo_processamento_ms": tempo_ms,
            "persistido": False,
        },
        # espelho unitário útil para jobs
        "probabilidade_risco": probabilidade,
        "classificacao_risco": risco_title,
        "score": probabilidade,
        "class": pred_class,
        "execution_time_ms": tempo_ms,
    }
