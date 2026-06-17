"""Serviço de carregamento, inferência e análise clínica por gestante."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from api.intercorrencias import (
    analisar_intercorrencias,
    calcular_score_ecocardiograma,
    classificar_prioridade_eco,
    recomendar_ecocardiograma,
    resumir_intercorrencias,
)
from api.shap_explain import explain_prediction, get_global_shap_importance
from src.config import (
    EXPERIMENTS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.models import get_model_registry
from src.preprocessing import build_full_pipeline, get_class_weight_dict, split_features_target

FEATURE_COLS = EXPERIMENTS["experimento_3_completo"]
MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "model_meta.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CANDIDATES = [
    PROJECT_ROOT / "dataset_com_id_gestante.csv",
    PROJECT_ROOT / "dataset.csv",
    PROJECT_ROOT / "data" / "dataset.csv",
]

DEFAULT_META = {
    "modelo": "CatBoost",
    "experimento": "experimento_3_completo",
    "balanceamento": "class_weight",
}


def _resolve_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return DATASET_CANDIDATES[-1]


DATASET_PATH = _resolve_dataset_path()


class ModelService:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.meta: dict[str, Any] = dict(DEFAULT_META)
        self._gestantes_df: pd.DataFrame | None = None
        self._analises_cache: dict[str, dict[str, Any]] = {}
        self._priorizacao_cache: list[dict[str, Any]] | None = None
        self._shap_cache: dict[str, dict[str, Any]] = {}

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load_or_train(self) -> None:
        if MODEL_PATH.exists():
            self.pipeline = joblib.load(MODEL_PATH)
            if META_PATH.exists():
                self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        elif DATASET_PATH.exists():
            self._train_and_save()
        else:
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH} e dataset ausente. "
                "Execute `py train_model.py` ou `py main.py` primeiro."
            )

        self._load_gestantes()

    def _train_and_save(self) -> None:
        df = pd.read_csv(DATASET_PATH)
        X, y = split_features_target(df, FEATURE_COLS, TARGET_COL)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        cw = get_class_weight_dict(y_train)
        estimator = get_model_registry(class_weight=cw)["CatBoost"]
        pipeline = build_full_pipeline(estimator, FEATURE_COLS, "none")
        pipeline.fit(X_train, y_train)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        META_PATH.write_text(json.dumps(DEFAULT_META, indent=2, ensure_ascii=False), encoding="utf-8")

        self.pipeline = pipeline
        self.meta = dict(DEFAULT_META)

    def _load_gestantes(self) -> None:
        if not DATASET_PATH.exists():
            self._gestantes_df = pd.DataFrame()
            return

        df = pd.read_csv(DATASET_PATH)
        if "id_gestante" not in df.columns:
            df.insert(0, "id_gestante", [f"GEST-{i:05d}" for i in range(1, len(df) + 1)])
        self._gestantes_df = df
        self._priorizacao_cache = None
        self._build_analises_cache()

    def _build_analises_cache(self) -> None:
        if self._gestantes_df is None or self._gestantes_df.empty or not self.pipeline:
            self._analises_cache = {}
            return

        X = self._gestantes_df[FEATURE_COLS].copy()
        for col in FEATURE_COLS:
            if col == "corionicidade":
                X[col] = X[col].fillna("NA")
            elif col == "etnia":
                X[col] = X[col].fillna("Branca")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        probas = self.pipeline.predict_proba(X)[:, 1]
        preds = self.pipeline.predict(X)

        cache: dict[str, dict[str, Any]] = {}
        for pos, (_, row) in enumerate(self._gestantes_df.iterrows()):
            id_gestante = str(row["id_gestante"])
            data = self._row_to_dict(row)
            proba = float(probas[pos])
            pred = int(preds[pos])

            predicao = {
                "cardiopatia_congenita": pred,
                "probabilidade": round(proba, 4),
                "risco": self._classificar_risco(proba),
                "modelo": self.meta.get("modelo", "CatBoost"),
                "experimento": self.meta.get("experimento", "experimento_3_completo"),
            }
            intercorrencias = analisar_intercorrencias(data)
            resumo = resumir_intercorrencias(intercorrencias)
            score_eco = calcular_score_ecocardiograma(proba, intercorrencias)

            analise = {
                "id_gestante": id_gestante,
                "predicao": predicao,
                "analise": {
                    "probabilidade_ccf": predicao["probabilidade"],
                    "risco": predicao["risco"],
                    "cardiopatia_congenita_prevista": predicao["cardiopatia_congenita"],
                    "recomenda_ecocardiograma": recomendar_ecocardiograma(proba, intercorrencias),
                    "score_prioridade_eco": score_eco,
                    "prioridade_ecocardiograma": classificar_prioridade_eco(score_eco),
                },
                "intercorrencias": intercorrencias,
                "resumo_intercorrencias": resumo,
                "dados_clinicos": {k: data.get(k) for k in FEATURE_COLS if k in data},
            }
            if TARGET_COL in row.index and not pd.isna(row[TARGET_COL]):
                analise["desfecho_real"] = {
                    "cardiopatia_congenita": int(row[TARGET_COL]),
                    "fonte": "dataset",
                }
            cache[id_gestante] = analise

        self._analises_cache = cache

    def _classificar_risco(self, proba: float) -> str:
        if proba < 0.15:
            return "baixo"
        if proba < 0.35:
            return "moderado"
        return "alto"

    def predict_one(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError("Modelo não carregado.")

        row = pd.DataFrame([data])[FEATURE_COLS]
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        pred = int(self.pipeline.predict(row)[0])

        return {
            "cardiopatia_congenita": pred,
            "probabilidade": round(proba, 4),
            "risco": self._classificar_risco(proba),
            "modelo": self.meta.get("modelo", "CatBoost"),
            "experimento": self.meta.get("experimento", "experimento_3_completo"),
        }

    def analisar_gestante(
        self,
        data: dict[str, Any],
        id_gestante: str | None = None,
    ) -> dict[str, Any]:
        predicao = self.predict_one(data)
        intercorrencias = analisar_intercorrencias(data)
        resumo = resumir_intercorrencias(intercorrencias)
        score_eco = calcular_score_ecocardiograma(predicao["probabilidade"], intercorrencias)

        return {
            "id_gestante": id_gestante,
            "predicao": predicao,
            "analise": {
                "probabilidade_ccf": predicao["probabilidade"],
                "risco": predicao["risco"],
                "cardiopatia_congenita_prevista": predicao["cardiopatia_congenita"],
                "recomenda_ecocardiograma": recomendar_ecocardiograma(
                    predicao["probabilidade"], intercorrencias
                ),
                "score_prioridade_eco": score_eco,
                "prioridade_ecocardiograma": classificar_prioridade_eco(score_eco),
            },
            "intercorrencias": intercorrencias,
            "resumo_intercorrencias": resumo,
            "dados_clinicos": {k: data.get(k) for k in FEATURE_COLS if k in data},
        }

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for col in FEATURE_COLS:
            val = row.get(col)
            if pd.isna(val):
                if col == "corionicidade":
                    data[col] = "NA"
                elif col == "etnia":
                    data[col] = "Branca"
                else:
                    data[col] = 0
            elif col in ("etnia", "corionicidade"):
                data[col] = str(val) if str(val).strip() else "NA"
            elif col in ("idade_materna", "imc_pre_gestacional", "idade_gestacional"):
                data[col] = float(val)
            else:
                data[col] = int(float(val))
        return data

    def listar_gestantes(
        self,
        offset: int = 0,
        limit: int = 50,
        apenas_com_intercorrencia: bool = False,
    ) -> dict[str, Any]:
        if not self._analises_cache:
            return {"total": 0, "gestantes": [], "dataset": str(DATASET_PATH)}

        items = list(self._analises_cache.values())
        if apenas_com_intercorrencia:
            items = [i for i in items if i["resumo_intercorrencias"]["total"] > 0]

        total = len(items)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "dataset": str(DATASET_PATH),
            "gestantes": items[offset : offset + limit],
        }

    def obter_gestante(self, id_gestante: str) -> dict[str, Any] | None:
        return self._analises_cache.get(id_gestante)

    def priorizacao_ecocardiograma(
        self,
        limite: int = 100,
        apenas_recomendadas: bool = True,
    ) -> dict[str, Any]:
        if self._priorizacao_cache is None:
            fila: list[dict[str, Any]] = []
            for analise in self._analises_cache.values():
                if apenas_recomendadas and not analise["analise"]["recomenda_ecocardiograma"]:
                    continue
                fila.append(
                    {
                        "posicao": 0,
                        "id_gestante": analise["id_gestante"],
                        "score_prioridade_eco": analise["analise"]["score_prioridade_eco"],
                        "prioridade_ecocardiograma": analise["analise"]["prioridade_ecocardiograma"],
                        "probabilidade_ccf": analise["analise"]["probabilidade_ccf"],
                        "risco": analise["analise"]["risco"],
                        "total_intercorrencias": analise["resumo_intercorrencias"]["total"],
                        "marcadores_fetais": analise["resumo_intercorrencias"]["marcadores_fetais_detectados"],
                        "intercorrencias_principais": [
                            i["alteracao"] for i in analise["intercorrencias"][:5]
                        ],
                    }
                )
            fila.sort(key=lambda x: x["score_prioridade_eco"], reverse=True)
            for idx, item in enumerate(fila, start=1):
                item["posicao"] = idx
            self._priorizacao_cache = fila

        lista = (self._priorizacao_cache or [])[:limite]
        return {
            "total_na_fila": len(self._priorizacao_cache or []),
            "limite_retornado": limite,
            "priorizacao": lista,
        }

    def get_summary_metrics(self) -> dict[str, Any] | None:
        summary_path = REPORTS_DIR / "pipeline_summary.json"
        if not summary_path.exists():
            return None
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def explicar_shap(
        self,
        data: dict[str, Any],
        id_gestante: str | None = None,
        top_n: int = 10,
        usar_cache: bool = True,
    ) -> dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError("Modelo não carregado.")

        cache_key = id_gestante or "__adhoc__"
        if usar_cache and cache_key in self._shap_cache:
            return self._shap_cache[cache_key]

        shap_result = explain_prediction(self.pipeline, data, FEATURE_COLS, top_n=top_n)
        predicao = self.predict_one(data)

        resultado = {
            "id_gestante": id_gestante,
            "probabilidade_ccf": predicao["probabilidade"],
            **shap_result,
        }

        if usar_cache and id_gestante:
            self._shap_cache[id_gestante] = resultado
        return resultado

    def explicar_shap_gestante(self, id_gestante: str, top_n: int = 10) -> dict[str, Any] | None:
        analise = self._analises_cache.get(id_gestante)
        if not analise:
            return None
        return self.explicar_shap(
            analise["dados_clinicos"],
            id_gestante=id_gestante,
            top_n=top_n,
        )

    def get_shap_global(self, top_n: int = 15) -> dict[str, Any]:
        return get_global_shap_importance(top_n=top_n)


model_service = ModelService()
