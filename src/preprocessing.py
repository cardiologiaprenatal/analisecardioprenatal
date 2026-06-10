"""Pipeline de pré-processamento para modelos de CCF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, RANDOM_STATE


def get_feature_types(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas do conjunto de features."""
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    return numeric, categorical


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Constrói ColumnTransformer com imputação, OHE e StandardScaler."""
    numeric_cols, categorical_cols = get_feature_types(feature_cols)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_resampler(strategy: str) -> Any | None:
    """Retorna resampler imbalanced-learn conforme estratégia."""
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "adasyn":
        return ADASYN(random_state=RANDOM_STATE)
    if strategy == "random_oversampling":
        return RandomOverSampler(random_state=RANDOM_STATE)
    return None


def build_full_pipeline(
    estimator: Any,
    feature_cols: list[str],
    balance_strategy: str = "none",
) -> Pipeline | ImbPipeline:
    """Monta pipeline completo: preprocessamento + (opcional) balanceamento + modelo."""
    preprocessor = build_preprocessor(feature_cols)
    resampler = get_resampler(balance_strategy)

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]

    if resampler is not None:
        steps.append(("resampler", resampler))

    steps.append(("classifier", estimator))

    if resampler is not None:
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def split_features_target(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y."""
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def get_class_weight_dict(y: pd.Series) -> dict[int, float]:
    """Calcula pesos de classe balanceados."""
    n_samples = len(y)
    n_classes = y.nunique()
    weights = {}
    for cls in y.unique():
        weights[int(cls)] = n_samples / (n_classes * (y == cls).sum())
    return weights
