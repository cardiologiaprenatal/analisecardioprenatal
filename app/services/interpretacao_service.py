"""Interpretação clínica individualizada — Fase 2 (longitudinal + texto gestante).

Não altera probabilidade/classe/SHAP do CatBoost. Apenas narra o contexto recebido.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import get_settings
from app.prompts.interpretacao_system_v2 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.schemas_interpretacao import FatorExplicado, InterpretacaoRequest, InterpretacaoResponse
from app.services.gemini_client import (
    GeminiUnavailableError,
    gemini_disponivel,
    gerar_narrativa,
)

logger = logging.getLogger(__name__)

_LABELS_CAMPO = {
    "idade_gestacional": "Idade gestacional",
    "pressao_sistolica": "PA sistólica",
    "bpm_materno": "BPM materno",
    "saturacao": "Saturação",
    "imc_pre_gestacional": "IMC pré-gestacional",
    "probabilidade_risco": "Probabilidade CatBoost",
    "classificacao_risco": "Classificação de risco",
}


def _ecoar_predicao(pred: dict[str, Any]) -> dict[str, Any]:
    """Força eco da verdade CatBoost (ignora qualquer alteração do LLM)."""
    return {
        "probabilidade_risco": pred.get("probabilidade_risco"),
        "classificacao_risco": pred.get("classificacao_risco"),
        "analise_id": pred.get("analise_id"),
    }


def _fatores_from_shap(shap: Any, limit: int = 8) -> list[FatorExplicado]:
    if not isinstance(shap, list):
        return []
    fatores: list[FatorExplicado] = []
    for item in shap[:limit]:
        if not isinstance(item, dict):
            continue
        feat = str(item.get("feature") or item.get("nome") or "variavel")
        impact = item.get("impact", item.get("impacto"))
        try:
            impact_f = float(impact) if impact is not None else None
        except (TypeError, ValueError):
            impact_f = None
        direcao = "aumenta" if (impact_f or 0) >= 0 else "reduz"
        fatores.append(
            FatorExplicado(
                feature=feat,
                impacto=impact_f,
                explicacao=f"Contribuição SHAP ({direcao} risco) segundo o CatBoost — sem recálculo.",
            )
        )
    return fatores


def _fmt_valor(campo: str, valor: Any) -> str:
    if valor is None:
        return "N/D"
    if campo == "probabilidade_risco":
        try:
            return f"{float(valor) * 100:.1f}%"
        except (TypeError, ValueError):
            return str(valor)
    if isinstance(valor, float):
        return f"{valor:.1f}" if abs(valor) >= 1 else f"{valor:.4f}"
    return str(valor)


def _deltas_from_numericos(ctx: InterpretacaoRequest) -> list[str]:
    raw = getattr(ctx, "deltas_numericos", None)
    if raw is None and isinstance(getattr(ctx, "__dict__", None), dict):
        raw = ctx.__dict__.get("deltas_numericos")
    # Pydantic extra / model_dump
    data = ctx.model_dump(mode="json") if hasattr(ctx, "model_dump") else {}
    items = data.get("deltas_numericos") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = raw if isinstance(raw, list) else []

    frases: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        campo = str(item.get("campo") or "")
        if not campo:
            continue
        label = _LABELS_CAMPO.get(campo, campo.replace("_", " "))
        ant = item.get("anterior")
        atu = item.get("atual")
        unidade = item.get("unidade") or ""
        uni_txt = f" {unidade}" if unidade and unidade != "fracao" else ""
        if campo == "classificacao_risco":
            frases.append(f"{label}: {ant or 'N/D'} → {atu or 'N/D'}")
            continue
        frases.append(
            f"{label}: {_fmt_valor(campo, ant)} → {_fmt_valor(campo, atu)}{uni_txt}".rstrip()
        )
    return frases


def _texto_gestante(risco: str, tem_longitudinal: bool) -> str:
    base = (
        "Seu pré-natal continua sendo acompanhado pela equipe. "
        f"O sistema de apoio classificou o nível de atenção atual como «{risco}». "
        "Isso não substitui a avaliação médica — continue as consultas e siga as orientações "
        "da sua equipe de saúde."
    )
    if tem_longitudinal:
        return (
            base
            + " Comparando com a consulta anterior, a equipe já tem informações para "
            "acompanhar a evolução dos seus sinais clínicos."
        )
    return (
        base
        + " Ainda há só uma consulta registrada neste acompanhamento; "
        "a comparação entre consultas ficará disponível nas próximas visitas."
    )


def interpretar_gestante_mock(ctx: InterpretacaoRequest) -> InterpretacaoResponse:
    started = time.perf_counter()
    pred = ctx.predicao_catboost or {}
    consulta = ctx.consulta_atual or {}
    gestante = ctx.gestante or {}
    data = ctx.model_dump(mode="json")

    risco = str(pred.get("classificacao_risco") or "baixo")
    prob = pred.get("probabilidade_risco")
    try:
        prob_f = float(prob) if prob is not None else None
    except (TypeError, ValueError):
        prob_f = None

    fatores = _fatores_from_shap(pred.get("shap"))
    deltas = _deltas_from_numericos(ctx)
    tem_longitudinal = len(deltas) > 0

    if not tem_longitudinal:
        deltas = [
            "Sem base longitudinal: há apenas uma consulta (ou sem dados comparáveis) "
            "neste contexto."
        ]

    id_publico = gestante.get("id_publico") or gestante.get("id_interno") or "?"
    ig = consulta.get("idade_gestacional")
    idade = gestante.get("idade_materna") or consulta.get("idade_materna")
    prob_txt = f"{prob_f * 100:.1f}%" if prob_f is not None else "N/D"

    evolucao_txt = (
        "Evolução entre consultas disponível (ver deltas)."
        if tem_longitudinal
        else "Sem comparação longitudinal ainda."
    )

    resumo_medico = (
        f"[MOCK v2] Gestante {id_publico}: CatBoost «{risco}» ({prob_txt}). "
        f"Idade materna={idade}, IG={ig}. {evolucao_txt} "
        f"Consultas no histórico: {len(ctx.historico_consultas)}. "
        "Narrativa sem alteração do score."
    )

    pred_ant = data.get("predicao_anterior") if isinstance(data, dict) else None
    if isinstance(pred_ant, dict) and pred_ant.get("classificacao_risco"):
        resumo_medico += (
            f" Predição anterior: «{pred_ant.get('classificacao_risco')}» "
            f"({_fmt_valor('probabilidade_risco', pred_ant.get('probabilidade_risco'))})."
        )

    recomendacoes = [
        "Manter o resultado CatBoost como referência de triagem.",
        "Revisar fatores SHAP de maior impacto na consulta clínica.",
    ]
    if tem_longitudinal:
        recomendacoes.append(
            "Confrontar deltas clínicos (PA, BPM, SpO2, IG) com o exame atual."
        )
    if str(risco).lower() in {"alto", "moderado"} or pred.get("recomenda_ecocardiograma"):
        recomendacoes.append(
            "Considerar priorização de ecocardiograma fetal conforme protocolo local "
            f"(prioridade informada: {pred.get('prioridade_ecocardiograma') or 'N/D'})."
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return InterpretacaoResponse(
        mock=True,
        model_id="mock-fase2",
        prompt_version=PROMPT_VERSION,
        schema_version=str(ctx.schema_version or "1.1"),
        cache_hit=False,
        resumo_medico=resumo_medico,
        resumo_gestante=_texto_gestante(risco, tem_longitudinal),
        fatores_explicados=fatores,
        recomendacoes=recomendacoes,
        deltas=deltas,
        predicao_ecoada=_ecoar_predicao(pred),
        tokens_in=0,
        tokens_out=0,
        latency_ms=round(elapsed_ms, 3),
        aviso="Fallback mock Fase 2: Vertex/Gemini indisponível ou desabilitado.",
    )


def _parse_fatores_llm(raw: list[dict[str, Any]], shap_fallback: Any) -> list[FatorExplicado]:
    fatores: list[FatorExplicado] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        feat = str(item.get("feature") or item.get("nome") or "").strip()
        if not feat:
            continue
        impact = item.get("impacto", item.get("impact"))
        try:
            impact_f = float(impact) if impact is not None else None
        except (TypeError, ValueError):
            impact_f = None
        expl = str(item.get("explicacao") or item.get("explanation") or "").strip()
        if not expl:
            expl = "Fator contextualizado a partir do SHAP CatBoost."
        fatores.append(FatorExplicado(feature=feat, impacto=impact_f, explicacao=expl))
    return fatores if fatores else _fatores_from_shap(shap_fallback)


def interpretar_gestante_gemini(ctx: InterpretacaoRequest) -> InterpretacaoResponse:
    settings = get_settings()
    started = time.perf_counter()
    pred = ctx.predicao_catboost or {}

    payload = ctx.model_dump(mode="json")
    user_prompt = build_user_prompt(json.dumps(payload, ensure_ascii=False, default=str))

    narrativa, meta = gerar_narrativa(SYSTEM_PROMPT, user_prompt, settings)

    fatores = _parse_fatores_llm(narrativa.fatores_explicados, pred.get("shap"))
    deltas = [str(d).strip() for d in (narrativa.deltas or []) if str(d).strip()]
    if not deltas:
        deltas = _deltas_from_numericos(ctx) or [
            "Sem base longitudinal: há apenas uma consulta neste contexto."
        ]

    resumo_gestante = (narrativa.resumo_gestante or "").strip() or None
    if not resumo_gestante:
        risco = str(pred.get("classificacao_risco") or "baixo")
        resumo_gestante = _texto_gestante(risco, len(_deltas_from_numericos(ctx)) > 0)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return InterpretacaoResponse(
        mock=False,
        model_id=str(meta.get("model_id") or settings.gemini_model),
        prompt_version=PROMPT_VERSION,
        schema_version=str(ctx.schema_version or "1.1"),
        cache_hit=False,
        resumo_medico=narrativa.resumo_medico.strip(),
        resumo_gestante=resumo_gestante,
        fatores_explicados=fatores,
        recomendacoes=[str(r).strip() for r in (narrativa.recomendacoes or []) if str(r).strip()],
        deltas=deltas,
        predicao_ecoada=_ecoar_predicao(pred),
        tokens_in=int(meta.get("tokens_in") or 0),
        tokens_out=int(meta.get("tokens_out") or 0),
        latency_ms=round(elapsed_ms, 3),
        aviso=None,
    )


def interpretar_gestante(ctx: InterpretacaoRequest) -> InterpretacaoResponse:
    """Entrada única: tenta Gemini; fallback mock se configurado."""
    settings = get_settings()

    if gemini_disponivel(settings):
        try:
            return interpretar_gestante_gemini(ctx)
        except GeminiUnavailableError as exc:
            logger.warning("Gemini indisponível: %s", exc)
            if not settings.interpretacao_fallback_mock:
                raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro inesperado no Gemini: %s", exc)
            if not settings.interpretacao_fallback_mock:
                raise

    return interpretar_gestante_mock(ctx)
