"""Validação e alertas de qualidade dos dados clínicos de entrada."""

from __future__ import annotations

from typing import Any

CAMPOS_OBRIGATORIOS = [
    "idade_materna",
    "etnia",
    "imc_pre_gestacional",
    "paridade",
    "abortamentos_previos",
    "idade_gestacional",
]

CAMPOS_BINARIOS = [
    "historia_natimorto",
    "filho_anterior_cardiopatia",
    "historia_familiar_cardiopatia",
    "diabetes_pre_gestacional",
    "diabetes_gestacional",
    "lupus",
    "fenilcetonuria",
    "doenca_tireoide",
    "hipertensao_cronica",
    "rubeola",
    "citomegalovirus",
    "toxoplasmose",
    "sifilis",
    "isotretinoina",
    "acido_valproico",
    "litio",
    "alcool",
    "tabagismo",
    "drogas_ilicitas",
    "solventes",
    "pesticidas",
    "gestacao_gemelar",
    "translucencia_nucal_aumentada",
    "alteracao_ducto_venoso",
    "regurgitacao_tricuspide",
    "malformacao_extracardiaca",
    "restricao_crescimento_fetal",
    "polidramnio",
    "oligodramnio",
]


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, float) and val != val:
        return True
    return False


def validar_qualidade_dados(data: dict[str, Any]) -> dict[str, Any]:
    alertas: list[dict[str, str]] = []

    for campo in CAMPOS_OBRIGATORIOS:
        if _is_missing(data.get(campo)):
            alertas.append(
                {
                    "tipo": "campo_faltante",
                    "campo": campo,
                    "severidade": "alta",
                    "mensagem": f"Campo obrigatório ausente: {campo}",
                }
            )

    for campo in CAMPOS_BINARIOS:
        val = data.get(campo)
        if not _is_missing(val) and val not in (0, 1, 0.0, 1.0, "0", "1"):
            alertas.append(
                {
                    "tipo": "valor_inconsistente",
                    "campo": campo,
                    "severidade": "media",
                    "mensagem": f"Valor binário inválido em {campo}: esperado 0 ou 1",
                }
            )

    idade = data.get("idade_materna")
    if not _is_missing(idade):
        try:
            idade_f = float(idade)
            if idade_f < 14 or idade_f > 48:
                alertas.append(
                    {
                        "tipo": "valor_inconsistente",
                        "campo": "idade_materna",
                        "severidade": "alta",
                        "mensagem": f"Idade materna fora da faixa plausível: {idade_f}",
                    }
                )
        except (TypeError, ValueError):
            alertas.append(
                {
                    "tipo": "valor_inconsistente",
                    "campo": "idade_materna",
                    "severidade": "alta",
                    "mensagem": "Idade materna não numérica",
                }
            )

    ig = data.get("idade_gestacional")
    if not _is_missing(ig):
        try:
            ig_f = float(ig)
            if ig_f < 11 or ig_f > 28:
                alertas.append(
                    {
                        "tipo": "idade_gestacional_incompativel",
                        "campo": "idade_gestacional",
                        "severidade": "alta",
                        "mensagem": f"Idade gestacional fora da janela de triagem (11–28 sem): {ig_f}",
                    }
                )
        except (TypeError, ValueError):
            alertas.append(
                {
                    "tipo": "valor_inconsistente",
                    "campo": "idade_gestacional",
                    "severidade": "alta",
                    "mensagem": "Idade gestacional não numérica",
                }
            )

    gemelar = data.get("gestacao_gemelar", 0)
    corion = data.get("corionicidade", "NA")
    if int(float(gemelar or 0)) == 0 and str(corion) not in ("NA", "", "nan"):
        alertas.append(
            {
                "tipo": "dado_conflitante",
                "campo": "corionicidade",
                "severidade": "media",
                "mensagem": "Corionicidade informada sem gestação gemelar",
            }
        )
    if int(float(gemelar or 0)) == 1 and str(corion) in ("NA", "", "nan"):
        alertas.append(
            {
                "tipo": "dado_conflitante",
                "campo": "corionicidade",
                "severidade": "media",
                "mensagem": "Gestação gemelar sem definição de corionicidade",
            }
        )

    if int(float(data.get("polidramnio", 0) or 0)) == 1 and int(float(data.get("oligodramnio", 0) or 0)) == 1:
        alertas.append(
            {
                "tipo": "dado_conflitante",
                "campo": "liquido_amniotico",
                "severidade": "alta",
                "mensagem": "Polidrâmnio e oligoidrâmnio simultâneos — dados conflitantes",
            }
        )

    penalidade = sum(30 if a["severidade"] == "alta" else 10 for a in alertas)
    score = max(0, 100 - penalidade)

    return {
        "valido": not any(a["severidade"] == "alta" for a in alertas),
        "score_qualidade": score,
        "total_alertas": len(alertas),
        "alertas": alertas,
    }
