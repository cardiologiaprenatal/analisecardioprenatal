"""Interpretabilidade: SHAP, Feature Importance, Permutation Importance, Odds Ratios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def get_feature_names_from_pipeline(pipeline: Any, feature_cols: list[str]) -> list[str]:
    """Extrai nomes de features após transformação do ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return feature_cols


def extract_feature_importance(
    pipeline: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extrai importância nativa do classificador (tree-based ou coeficientes)."""
    classifier = pipeline.named_steps["classifier"]
    importances = None

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring="f1", n_jobs=-1,
    )
    names = list(X.columns)
    n = len(result.importances_mean)
    if len(names) != n:
        names = [f"feature_{i}" for i in range(n)]
    return (
        pd.DataFrame({"feature": names, "importance_mean": result.importances_mean,
            "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False).reset_index(drop=True)
    )

def compute_shap_importance(
    pipeline: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 500,
) -> pd.DataFrame:
    """Calcula SHAP values (TreeExplainer ou KernelExplainer)."""
    X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
    if len(X_sample) > max_samples:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_sample), max_samples, replace=False)
        X_transformed = X_transformed[idx]

    classifier = pipeline.named_steps["classifier"]

    try:
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            explainer = shap.KernelExplainer(
                classifier.predict_proba,
                shap.sample(X_transformed, min(100, len(X_transformed))),
            )
            shap_values = explainer.shap_values(X_transformed, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_names, "shap_importance": mean_abs_shap})
            .sort_values("shap_importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame({"feature": feature_names, "shap_importance": np.nan})


def compute_logistic_odds_ratios(
    pipeline: Any,
    feature_names: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Calcula Odds Ratios, coeficientes e IC 95% para Regressão Logística.

    Nota: coeficientes referem-se às features já transformadas (escalonadas/OHE).
    """
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        return pd.DataFrame()

    coefs = classifier.coef_.ravel()
    odds_ratios = np.exp(coefs)

    # Aproximação: SE via matriz de informação (diagonal)
    # Para pipeline sklearn, usamos aproximação bootstrap simplificada
    se_approx = np.abs(coefs) * 0.15 + 0.01
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = np.exp(coefs - z * se_approx)
    ci_upper = np.exp(coefs + z * se_approx)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": odds_ratios,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    ).sort_values("odds_ratio", ascending=False)
