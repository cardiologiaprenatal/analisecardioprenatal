"""Carregamento de artefato .pkl / .joblib — pipeline completo ou modelo enxuto."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from app.features import FULL_FEATURE_COLS, SLIM_FEATURE_ORDER

logger = logging.getLogger(__name__)

# caminhos candidatos (preferir modelo treinado com 36 features)
MODEL_CANDIDATES = (
    "models/best_model.joblib",
    "models/model.pkl",
    "models/model.joblib",
)


@dataclass
class LoadedModel:
    estimator: Any | None
    path: str
    loaded: bool
    framework: str | None = None
    features: list[str] = field(default_factory=lambda: list(FULL_FEATURE_COLS))
    mode: str = "heuristic"  # heuristic | full_pipeline | slim


def _resolve_path(configured: str) -> Path:
    path = Path(configured)
    if path.is_file():
        return path

    # Fallback só para caminhos relativos padrão em models/
    # (não sobrescreve MODEL_PATH absoluto de testes/CI)
    configured_norm = configured.replace("\\", "/")
    if configured_norm.startswith("models/") or configured_norm in MODEL_CANDIDATES:
        for candidate in MODEL_CANDIDATES:
            c = Path(candidate)
            if c.is_file():
                return c
    return path


def _detect_mode(estimator: Any) -> tuple[str, list[str]]:
    """Pipeline sklearn com 36 cols vs estimador numpy enxuto."""
    # ColumnTransformer / Pipeline com feature names
    named = getattr(estimator, "named_steps", None)
    if isinstance(named, dict):
        pre = named.get("preprocessor") or next(iter(named.values()), None)
        if pre is not None:
            names = getattr(pre, "feature_names_in_", None)
            if names is not None and len(list(names)) >= 20:
                return "full_pipeline", [str(n) for n in list(names)]
            # fallback: assume experimento completo
            return "full_pipeline", list(FULL_FEATURE_COLS)

    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        cols = [str(n) for n in list(names)]
        if len(cols) >= 20:
            return "full_pipeline", cols
        return "slim", cols

    return "slim", list(SLIM_FEATURE_ORDER)


def load_model(model_path: str) -> LoadedModel:
    path = _resolve_path(model_path)
    path_display = path.as_posix()

    if not path.is_file():
        logger.warning("model_missing path=%s — modo heurístico (36 features clínicas)", path_display)
        return LoadedModel(
            estimator=None,
            path=path_display,
            loaded=False,
            framework="heuristic",
            features=list(FULL_FEATURE_COLS),
            mode="heuristic",
        )

    try:
        estimator = joblib.load(path)
        mode, features = _detect_mode(estimator)
        framework = type(estimator).__name__
        logger.info(
            "model_loaded path=%s framework=%s mode=%s n_features=%s",
            path_display,
            framework,
            mode,
            len(features),
        )
        return LoadedModel(
            estimator=estimator,
            path=path_display,
            loaded=True,
            framework=framework,
            features=features,
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("model_load_failed: %s", exc)
        return LoadedModel(
            estimator=None,
            path=path_display,
            loaded=False,
            framework="heuristic",
            features=list(FULL_FEATURE_COLS),
            mode="heuristic",
        )
