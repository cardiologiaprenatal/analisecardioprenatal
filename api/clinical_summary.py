"""Geração de resumos clínicos técnicos e simplificados."""

from __future__ import annotations

from typing import Any

RISCO_LABEL = {"baixo": "baixo risco", "moderado": "risco moderado", "alto": "alto risco"}
PRIORIDADE_LABEL = {
    "urgente": "urgente",
    "alta": "alta",
    "moderada": "moderada",
    "baixa": "baixa",
}


def classificar_impacto_intercorrencia(item: dict[str, Any]) -> str:
    peso = item.get("peso_risco", 0)
    rel = item.get("relevancia_modelo", 0)
    if item.get("categoria") == "marcadores_fetais" or peso >= 2.0 or rel >= 0.10:
        return "alto"
    if peso >= 1.0 or rel >= 0.05:
        return "medio"
    return "baixo"


def enriquecer_intercorrencias(intercorrencias: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resultado = []
    for item in intercorrencias:
        resultado.append(
            {
                "categoria": item["categoria"],
                "categoria_label": item["categoria_label"],
                "campo": item["campo"],
                "nome": item["nome"],
                "valor": item["valor"],
                "alteracao": item["alteracao"],
                "impacto": classificar_impacto_intercorrencia(item),
                "peso_risco": item.get("peso_risco", 0),
                "relevancia_modelo": item.get("relevancia_modelo", 0),
            }
        )
    return resultado


def gerar_resumo_clinico(
    classificacao_risco: str,
    probabilidade: float,
    intercorrencias: list[dict[str, Any]],
    shap_top: list[dict[str, Any]],
    prioridade_eco: str,
) -> dict[str, str]:
    risco_txt = RISCO_LABEL.get(classificacao_risco, classificacao_risco)
    pct = probabilidade * 100

    fatores_shap = [c["nome"] for c in shap_top if c.get("direcao") == "aumenta_risco"][:4]
    fatores_inter = [i["nome"] for i in intercorrencias if i.get("impacto") in ("alto", "medio")][:4]

    if fatores_shap:
        fatores_txt = ", ".join(fatores_shap)
    elif fatores_inter:
        fatores_txt = ", ".join(fatores_inter)
    else:
        fatores_txt = "nenhum fator de alto impacto isolado identificado"

    tecnico = (
        f"Paciente classificada como {risco_txt} para cardiopatia congênita fetal "
        f"(probabilidade estimada: {pct:.1f}%). "
        f"Os principais fatores associados ao aumento do risco foram {fatores_txt}. "
        f"Prioridade para ecocardiograma fetal: {PRIORIDADE_LABEL[prioridade_eco]}."
    )

    if classificacao_risco == "alto":
        simplificado = (
            f"Há indícios importantes de risco para cardiopatia no bebê ({pct:.0f}%). "
            f"Recomenda-se ecocardiograma fetal com prioridade {PRIORIDADE_LABEL[prioridade_eco]}."
        )
    elif classificacao_risco == "moderado":
        simplificado = (
            f"Risco moderado detectado ({pct:.0f}%). "
            "A equipe deve avaliar ecocardiograma fetal conforme protocolo."
        )
    else:
        simplificado = (
            f"Risco baixo nesta avaliação ({pct:.0f}%). "
            "Manter acompanhamento pré-natal de rotina."
        )

    if fatores_inter:
        simplificado += f" Atenção a: {', '.join(fatores_inter[:3]).lower()}."

    return {"tecnico": tecnico, "simplificado": simplificado}


def gerar_recomendacoes(
    prioridade: str,
    recomenda_eco: bool,
    intercorrencias: list[dict[str, Any]],
    classificacao_risco: str,
) -> dict[str, Any]:
    acoes: list[str] = []

    if recomenda_eco:
        acoes.append("Solicitar ecocardiograma fetal")
        if prioridade == "urgente":
            acoes.append("Encaminhamento imediato ao especialista em cardiologia fetal")
        elif prioridade == "alta":
            acoes.append("Agendar ecocardiograma em até 7 dias")

    marcadores = [i for i in intercorrencias if i.get("categoria") == "marcadores_fetais"]
    if marcadores:
        acoes.append("Revisar ultrassonografia morfológica e marcadores de 1º/2º trimestre")

    if classificacao_risco in ("moderado", "alto"):
        acoes.append("Discussão do caso em pré-natal de alto risco")

    if not acoes:
        acoes.append("Manter seguimento pré-natal de rotina")

    return {
        "ecocardiograma_fetal": recomenda_eco,
        "prioridade": prioridade,
        "prazo_sugerido": prazo_sugerido(prioridade) if recomenda_eco else None,
        "acoes": acoes,
    }


def prazo_sugerido(prioridade: str) -> str:
    return {
        "urgente": "imediato",
        "alta": "7_dias",
        "moderada": "14_dias",
        "baixa": "rotina",
    }.get(prioridade, "rotina")
