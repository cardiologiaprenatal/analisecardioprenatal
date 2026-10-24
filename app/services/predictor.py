"""Pré-processamento + predição com 36 features (pipeline completo ou heurística)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.features import FULL_FEATURE_COLS, SLIM_FEATURE_ORDER, build_full_clinical_row, to_slim_features
from app.model_loader import LoadedModel
from app.schemas import ShapItem, UnifiedInferenceResponse

logger = logging.getLogger(__name__)

RiskLabel = Literal["Baixo", "Moderado", "Alto"]


@dataclass
class PredictorMetrics:
    total_predictions: int = 0
    total_execution_ms: float = 0.0
    last_prediction_at: str | None = None

    def record(self, elapsed_ms: float) -> None:
        self.total_predictions += 1
        self.total_execution_ms += elapsed_ms
        self.last_prediction_at = datetime.now(timezone.utc).isoformat()

    @property
    def avg_execution_time_ms(self) -> float | None:
        if self.total_predictions == 0:
            return None
        return round(self.total_execution_ms / self.total_predictions, 3)


@dataclass
class PredictorService:
    model: LoadedModel
    metrics: PredictorMetrics = field(default_factory=PredictorMetrics)

    def predict(
        self,
        payload: dict[str, Any],
        *,
        quiet: bool = False,
        id_gestante: str | None = None,
    ) -> UnifiedInferenceResponse:
        started = time.perf_counter()
        full_row = build_full_clinical_row(payload)

        if self.model.loaded and self.model.estimator is not None and self.model.mode == "full_pipeline":
            probability, class_label, shap_items = self._predict_full_pipeline(full_row)
            mode = "full_pipeline"
        elif self.model.loaded and self.model.estimator is not None:
            probability, class_label, shap_items = self._predict_slim(full_row)
            mode = "slim"
        else:
            probability, class_label, shap_items = self._predict_heuristic(full_row)
            mode = "heuristic"

        risk = self._risk_label(probability)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record(elapsed_ms)

        if not quiet:
            logger.info(
                "prediction_done mode=%s risk=%s probability=%.4f n_features=%s ms=%.3f",
                mode,
                risk,
                float(probability),
                len(FULL_FEATURE_COLS),
                elapsed_ms,
            )

        gid = id_gestante or (str(payload.get("id_gestante")) if payload.get("id_gestante") else None)
        return UnifiedInferenceResponse(
            id_gestante=gid,
            probabilidade_risco=round(float(probability), 4),
            classificacao_risco=risk,
            score=round(float(probability), 4),
            class_=int(class_label),
            shap=shap_items,
            execution_time_ms=round(elapsed_ms, 3),
            risk=risk,
            probability=round(float(probability), 4),
        )

    def _predict_full_pipeline(
        self,
        full_row: dict[str, float | int | str],
    ) -> tuple[float, int, list[ShapItem]]:
        cols = self.model.features or FULL_FEATURE_COLS
        frame = pd.DataFrame([{c: full_row.get(c) for c in cols}])
        estimator = self.model.estimator
        if hasattr(estimator, "predict_proba"):
            proba = float(estimator.predict_proba(frame)[0][-1])
        else:
            pred = float(estimator.predict(frame)[0])
            proba = min(max(pred, 0.0), 1.0)
        proba = min(max(proba, 0.0), 1.0)
        label = 1 if proba >= 0.5 else 0
        return proba, label, self._feature_impacts(full_row, proba)

    def _predict_slim(
        self,
        full_row: dict[str, float | int | str],
    ) -> tuple[float, int, list[ShapItem]]:
        slim = to_slim_features(full_row)
        cols = self.model.features or SLIM_FEATURE_ORDER
        x = np.asarray([[float(slim.get(c, 0.0)) for c in cols]], dtype=float)
        estimator = self.model.estimator
        if hasattr(estimator, "predict_proba"):
            proba = float(estimator.predict_proba(x)[0][-1])
        else:
            pred = float(estimator.predict(x)[0])
            proba = min(max(pred, 0.0), 1.0)
        proba = min(max(proba, 0.0), 1.0)
        label = 1 if proba >= 0.5 else 0
        return proba, label, self._feature_impacts(full_row, proba)

    def _predict_heuristic(
        self,
        full_row: dict[str, float | int | str],
    ) -> tuple[float, int, list[ShapItem]]:
        """Heurística usando o máximo de variáveis clínicas disponíveis."""
        idade = float(full_row["idade_materna"])
        imc = float(full_row["imc_pre_gestacional"])
        ig = float(full_row["idade_gestacional"])

        score = 0.06
        # idade: atenção <18 e >35
        if idade < 18:
            score += 0.08
        elif idade >= 40:
            score += 0.18
        elif idade >= 35:
            score += 0.12

        score += 0.16 if imc >= 30 else (0.06 if imc >= 25 else (0.06 if imc < 18.5 else 0.0))
        score += 0.20 if int(full_row["diabetes_pre_gestacional"]) else 0.0
        score += 0.14 if int(full_row["diabetes_gestacional"]) else 0.0
        score += 0.12 if int(full_row["hipertensao_cronica"]) else 0.0
        score += 0.10 if int(full_row["lupus"]) else 0.0
        score += 0.08 if int(full_row["doenca_tireoide"]) else 0.0
        score += 0.10 if int(full_row["fenilcetonuria"]) else 0.0

        score += 0.08 if int(full_row["historia_natimorto"]) else 0.0
        score += 0.14 if int(full_row["filho_anterior_cardiopatia"]) else 0.0
        score += 0.16 if int(full_row["historia_familiar_cardiopatia"]) else 0.0
        score += 0.04 * min(int(full_row["abortamentos_previos"]), 3)

        score += 0.10 if int(full_row["rubeola"]) else 0.0
        score += 0.10 if int(full_row["citomegalovirus"]) else 0.0
        score += 0.08 if int(full_row["toxoplasmose"]) else 0.0
        score += 0.08 if int(full_row["sifilis"]) else 0.0

        score += 0.18 if int(full_row["isotretinoina"]) else 0.0
        score += 0.16 if int(full_row["acido_valproico"]) else 0.0
        score += 0.14 if int(full_row["litio"]) else 0.0

        score += 0.08 if int(full_row["alcool"]) else 0.0
        score += 0.08 if int(full_row["tabagismo"]) else 0.0
        score += 0.10 if int(full_row["drogas_ilicitas"]) else 0.0
        score += 0.06 if int(full_row["solventes"]) else 0.0
        score += 0.06 if int(full_row["pesticidas"]) else 0.0

        score += 0.05 if ig < 14 else 0.0
        score += 0.12 if int(full_row["gestacao_gemelar"]) else 0.0

        score += 0.22 if int(full_row["translucencia_nucal_aumentada"]) else 0.0
        score += 0.18 if int(full_row["alteracao_ducto_venoso"]) else 0.0
        score += 0.16 if int(full_row["regurgitacao_tricuspide"]) else 0.0
        score += 0.16 if int(full_row["malformacao_extracardiaca"]) else 0.0
        score += 0.10 if int(full_row["restricao_crescimento_fetal"]) else 0.0
        score += 0.06 if int(full_row["polidramnio"]) else 0.0
        score += 0.06 if int(full_row["oligodramnio"]) else 0.0

        probability = float(min(max(score, 0.01), 0.99))
        label = 1 if probability >= 0.5 else 0
        return probability, label, self._feature_impacts(full_row, probability)

    def _feature_impacts(
        self,
        full_row: dict[str, float | int | str],
        probability: float,
    ) -> list[ShapItem]:
        weights: dict[str, float] = {}
        for col in FULL_FEATURE_COLS:
            if col in ("etnia", "corionicidade"):
                weights[col] = 0.02
                continue
            if col == "idade_materna":
                idade = float(full_row[col])
                weights[col] = 0.14 if (idade < 18 or idade >= 35) else 0.04
            elif col == "imc_pre_gestacional":
                imc = float(full_row[col])
                weights[col] = 0.12 if (imc >= 30 or imc < 18.5) else 0.03
            else:
                val = full_row[col]
                active = int(val) >= 1 if not isinstance(val, str) else False
                weights[col] = 0.12 if active else 0.01

        total = sum(weights.values()) or 1.0
        scale = probability / total
        items = [
            ShapItem(
                feature=name,
                impact=round(w * scale, 4),
                nome=name,
                impacto_absoluto=round(abs(w * scale), 4),
            )
            for name, w in weights.items()
        ]
        items.sort(key=lambda i: abs(i.impact), reverse=True)
        # Análise completa: todas as features do modelo (não truncar top-N).
        return items

    @staticmethod
    def _risk_label(probability: float) -> RiskLabel:
        if probability >= 0.7:
            return "Alto"
        if probability >= 0.4:
            return "Moderado"
        return "Baixo"
