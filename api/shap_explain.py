"""Explicações SHAP por predição e importância global do modelo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from api.intercorrencias import FIELD_META
from src.config import CATEGORICAL_COLS
from src.interpretability import get_feature_names_from_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_SHAP_PATH = PROJECT_ROOT / "data" / "shap_importance.csv"


def _base_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name[5:]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[5:]
        for col in CATEGORICAL_COLS:
            prefix = f"{col}_"
            if rest.startswith(prefix):
                return col
    return transformed_name


def _prepare_row(data: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in feature_cols:
        val = data.get(col)
        if col == "corionicidade":
            row[col] = "NA" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
        elif col == "etnia":
            row[col] = "Branca" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
        elif col in ("idade_materna", "imc_pre_gestacional", "idade_gestacional"):
            row[col] = float(val) if val is not None else 0.0
        else:
            row[col] = int(float(val)) if val is not None else 0
    return pd.DataFrame([row])[feature_cols]


def _aggregate_shap_values(
    feature_names: list[str],
    shap_row: np.ndarray,
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for name, value in zip(feature_names, shap_row):
        base = _base_feature_name(name)
        aggregated[base] = aggregated.get(base, 0.0) + float(value)
    return aggregated


def explain_prediction(
    pipeline: Any,
    data: dict[str, Any],
    feature_cols: list[str],
    top_n: int = 10,
) -> dict[str, Any]:
    """Calcula SHAP para uma gestante e retorna fatores que aumentam/diminuem o risco."""
    row = _prepare_row(data, feature_cols)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(row)
    feature_names = get_feature_names_from_pipeline(pipeline, feature_cols)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_row = np.asarray(shap_values[0], dtype=float)
    aggregated = _aggregate_shap_values(feature_names, shap_row)

    contribuicoes = []
    for campo, valor in aggregated.items():
        meta = FIELD_META.get(campo, {"nome": campo.replace("_", " ").title()})
        contribuicoes.append(
            {
                "campo": campo,
                "nome": meta["nome"],
                "shap_value": round(valor, 5),
                "impacto_absoluto": round(abs(valor), 5),
                "direcao": "aumenta_risco" if valor > 0 else "diminui_risco",
                "valor_paciente": data.get(campo),
            }
        )

    contribuicoes.sort(key=lambda x: x["impacto_absoluto"], reverse=True)

    total_abs = sum(c["impacto_absoluto"] for c in contribuicoes) or 1.0
    for c in contribuicoes:
        c["percentual_contribuicao"] = round(c["impacto_absoluto"] / total_abs * 100, 2)

    top = contribuicoes[:top_n]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])
    else:
        base_value = float(base_value)

    return {
        "metodo": "TreeExplainer",
        "classe_alvo": "cardiopatia_congenita",
        "valor_base": round(base_value, 5),
        "soma_shap": round(float(sum(aggregated.values())), 5),
        "total_features": len(aggregated),
        "resumo": _gerar_resumo_texto(top),
        "top_contribuicoes": top,
        "todas_contribuicoes": contribuicoes,
        "importancia_global": get_global_shap_importance(top_n=15).get("features", []),
    }


def _gerar_resumo_texto(contribuicoes: list[dict[str, Any]]) -> str:
    if not contribuicoes:
        return "Sem contribuições SHAP disponíveis."

    aumentam = [c for c in contribuicoes if c["direcao"] == "aumenta_risco"]
    diminuem = [c for c in contribuicoes if c["direcao"] == "diminui_risco"]

    partes = []
    if aumentam:
        nomes = ", ".join(c["nome"] for c in aumentam[:3])
        partes.append(f"Fatores que mais aumentam o risco: {nomes}.")
    if diminuem:
        nomes = ", ".join(c["nome"] for c in diminuem[:3])
        partes.append(f"Fatores que mais reduzem o risco: {nomes}.")
    return " ".join(partes)


def get_global_shap_importance(top_n: int = 15) -> dict[str, Any]:
    """Importância global SHAP do pipeline (arquivo gerado no treinamento)."""
    if not GLOBAL_SHAP_PATH.exists():
        return {"disponivel": False, "features": [], "mensagem": "Arquivo shap_importance.csv não encontrado."}

    df = pd.read_csv(GLOBAL_SHAP_PATH).head(top_n).copy()

    features = []
    for _, row in df.iterrows():
        raw = str(row["feature"])
        base = _base_feature_name(raw if raw.startswith(("num__", "cat__")) else f"num__{raw}")
        meta = FIELD_META.get(base, {"nome": base.replace("_", " ").title()})
        features.append(
            {
                "feature": base,
                "nome": meta["nome"],
                "shap_importance": round(float(row["shap_importance"]), 4),
            }
        )

    return {
        "disponivel": True,
        "modelo": df["modelo"].iloc[0] if "modelo" in df.columns else "CatBoost",
        "experimento": df["experimento"].iloc[0] if "experimento" in df.columns else "experimento_3_completo",
        "total_features": len(features),
        "features": features,
    }
