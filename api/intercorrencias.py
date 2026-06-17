"""Catálogo e análise de intercorrências gestacionais para CCF."""

from __future__ import annotations

from typing import Any

from src.config import BINARY_COLS, FEATURE_GROUPS
from src.data_generator import RISK_WEIGHTS

# Pesos SHAP do melhor modelo (experimento_3_completo / CatBoost)
SHAP_RELEVANCE: dict[str, float] = {
    "translucencia_nucal_aumentada": 0.2955,
    "restricao_crescimento_fetal": 0.1878,
    "regurgitacao_tricuspide": 0.1391,
    "imc_pre_gestacional": 0.1379,
    "idade_materna": 0.1289,
    "alteracao_ducto_venoso": 0.1180,
    "malformacao_extracardiaca": 0.1110,
    "diabetes_gestacional": 0.0898,
    "historia_familiar_cardiopatia": 0.0869,
    "tabagismo": 0.0755,
    "diabetes_pre_gestacional": 0.0699,
    "alcool": 0.0488,
    "polidramnio": 0.0448,
    "hipertensao_cronica": 0.0404,
    "idade_gestacional": 0.0369,
    "citomegalovirus": 0.0279,
    "filho_anterior_cardiopatia": 0.0250,
    "gestacao_gemelar": 0.0220,
    "isotretinoina": 0.0200,
    "rubeola": 0.0180,
}

CATEGORIA_LABELS = {
    "maternal": "Dados maternos",
    "doencas": "Doenças maternas",
    "infeccoes": "Infecções",
    "medicacoes": "Medicações teratogênicas",
    "habitos": "Hábitos",
    "exposicao": "Exposição ambiental",
    "gestacionais": "Condições gestacionais",
    "marcadores_fetais": "Marcadores fetais",
}

FIELD_META: dict[str, dict[str, str]] = {
    "idade_materna": {"nome": "Idade materna", "categoria": "maternal"},
    "etnia": {"nome": "Etnia", "categoria": "maternal"},
    "imc_pre_gestacional": {"nome": "IMC pré-gestacional", "categoria": "maternal"},
    "paridade": {"nome": "Paridade", "categoria": "maternal"},
    "abortamentos_previos": {"nome": "Abortamentos prévios", "categoria": "maternal"},
    "historia_natimorto": {"nome": "História de natimorto", "categoria": "maternal"},
    "filho_anterior_cardiopatia": {"nome": "Filho anterior com cardiopatia", "categoria": "maternal"},
    "historia_familiar_cardiopatia": {"nome": "História familiar de cardiopatia", "categoria": "maternal"},
    "diabetes_pre_gestacional": {"nome": "Diabetes pré-gestacional", "categoria": "doencas"},
    "diabetes_gestacional": {"nome": "Diabetes gestacional", "categoria": "doencas"},
    "lupus": {"nome": "Lúpus", "categoria": "doencas"},
    "fenilcetonuria": {"nome": "Fenilcetonúria", "categoria": "doencas"},
    "doenca_tireoide": {"nome": "Doença da tireoide", "categoria": "doencas"},
    "hipertensao_cronica": {"nome": "Hipertensão crônica", "categoria": "doencas"},
    "rubeola": {"nome": "Rubéola", "categoria": "infeccoes"},
    "citomegalovirus": {"nome": "Citomegalovírus", "categoria": "infeccoes"},
    "toxoplasmose": {"nome": "Toxoplasmose", "categoria": "infeccoes"},
    "sifilis": {"nome": "Sífilis", "categoria": "infeccoes"},
    "isotretinoina": {"nome": "Isotretinoína", "categoria": "medicacoes"},
    "acido_valproico": {"nome": "Ácido valpróico", "categoria": "medicacoes"},
    "litio": {"nome": "Lítio", "categoria": "medicacoes"},
    "alcool": {"nome": "Álcool", "categoria": "habitos"},
    "tabagismo": {"nome": "Tabagismo", "categoria": "habitos"},
    "drogas_ilicitas": {"nome": "Drogas ilícitas", "categoria": "habitos"},
    "solventes": {"nome": "Exposição a solventes", "categoria": "exposicao"},
    "pesticidas": {"nome": "Exposição a pesticidas", "categoria": "exposicao"},
    "idade_gestacional": {"nome": "Idade gestacional", "categoria": "gestacionais"},
    "gestacao_gemelar": {"nome": "Gestação gemelar", "categoria": "gestacionais"},
    "corionicidade": {"nome": "Corionicidade", "categoria": "gestacionais"},
    "translucencia_nucal_aumentada": {"nome": "Translucência nucal aumentada", "categoria": "marcadores_fetais"},
    "alteracao_ducto_venoso": {"nome": "Alteração do ducto venoso", "categoria": "marcadores_fetais"},
    "regurgitacao_tricuspide": {"nome": "Regurgitação tricúspide", "categoria": "marcadores_fetais"},
    "malformacao_extracardiaca": {"nome": "Malformação extracardíaca", "categoria": "marcadores_fetais"},
    "restricao_crescimento_fetal": {"nome": "Restrição de crescimento fetal", "categoria": "marcadores_fetais"},
    "polidramnio": {"nome": "Polidrâmnio", "categoria": "marcadores_fetais"},
    "oligodramnio": {"nome": "Oligoidrâmnio", "categoria": "marcadores_fetais"},
}

FIELD_TO_CATEGORY = {
    field: group
    for group, fields in FEATURE_GROUPS.items()
    for field in fields
}


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and (value != value):  # NaN
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "na", "nan", "none")
    if isinstance(value, (int, float)):
        return int(value) == 1
    return bool(value)


def _describe_alteracao(campo: str, valor: Any) -> str | None:
    if campo == "idade_materna":
        try:
            idade = float(valor)
        except (TypeError, ValueError):
            return None
        if idade < 18:
            return f"Idade materna abaixo de 18 anos ({idade:.0f} anos)"
        if idade > 35:
            return f"Idade materna avançada acima de 35 anos ({idade:.0f} anos)"
        return None

    if campo == "imc_pre_gestacional":
        try:
            imc = float(valor)
        except (TypeError, ValueError):
            return None
        if imc < 18.5:
            return f"Baixo peso pré-gestacional (IMC {imc:.1f})"
        if imc >= 30:
            return f"Obesidade pré-gestacional (IMC {imc:.1f})"
        return None

    if campo == "paridade" and _is_numeric_positive(valor, min_value=3):
        return f"Multiparidade elevada ({int(float(valor))} partos)"

    if campo == "abortamentos_previos" and _is_numeric_positive(valor, min_value=2):
        return f"Múltiplos abortamentos prévios ({int(float(valor))})"

    if campo == "corionicidade" and str(valor) == "Monocoriônica":
        return "Gestação gemelar monocoriônica"

    if campo in BINARY_COLS and _is_present(valor):
        return f"{FIELD_META[campo]['nome']} presente/alterado"

    return None


def _is_numeric_positive(valor: Any, min_value: int = 1) -> bool:
    try:
        return float(valor) >= min_value
    except (TypeError, ValueError):
        return False


def analisar_intercorrencias(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Detecta e descreve cada intercorrência/alteração presente na gestante."""
    intercorrencias: list[dict[str, Any]] = []

    for campo, meta in FIELD_META.items():
        valor = data.get(campo)
        alteracao = _describe_alteracao(campo, valor)
        if not alteracao:
            continue

        categoria = meta["categoria"]
        intercorrencias.append(
            {
                "campo": campo,
                "nome": meta["nome"],
                "categoria": categoria,
                "categoria_label": CATEGORIA_LABELS[categoria],
                "valor": valor,
                "alteracao": alteracao,
                "presente": True,
                "peso_risco": round(RISK_WEIGHTS.get(campo, 0.0), 2),
                "relevancia_modelo": round(SHAP_RELEVANCE.get(campo, 0.0), 4),
            }
        )

    intercorrencias.sort(
        key=lambda item: (item["relevancia_modelo"], item["peso_risco"]),
        reverse=True,
    )
    return intercorrencias


def resumir_intercorrencias(intercorrencias: list[dict[str, Any]]) -> dict[str, Any]:
    por_categoria: dict[str, int] = {}
    for item in intercorrencias:
        cat = item["categoria_label"]
        por_categoria[cat] = por_categoria.get(cat, 0) + 1

    marcadores_fetais = [i for i in intercorrencias if i["categoria"] == "marcadores_fetais"]

    return {
        "total": len(intercorrencias),
        "por_categoria": por_categoria,
        "marcadores_fetais_detectados": len(marcadores_fetais),
        "possui_alteracao_ecografica": len(marcadores_fetais) > 0,
    }


def calcular_score_ecocardiograma(
    probabilidade: float,
    intercorrencias: list[dict[str, Any]],
) -> float:
    """Score de priorização para ecocardiograma fetal (0–100)."""
    score = probabilidade * 55

    for item in intercorrencias:
        if item["categoria"] == "marcadores_fetais":
            score += 12 + item["relevancia_modelo"] * 20
        else:
            score += item["peso_risco"] * 2.5 + item["relevancia_modelo"] * 8

    return round(min(score, 100.0), 2)


def classificar_prioridade_eco(score: float) -> str:
    if score >= 70:
        return "urgente"
    if score >= 45:
        return "alta"
    if score >= 25:
        return "moderada"
    return "baixa"


def recomendar_ecocardiograma(probabilidade: float, intercorrencias: list[dict[str, Any]]) -> bool:
    if probabilidade >= 0.15:
        return True
    if any(i["categoria"] == "marcadores_fetais" for i in intercorrencias):
        return True
    if len(intercorrencias) >= 3:
        return True
    return False
