"""Carrega e agrega métricas CV vs teste dos modelos (CSV do pipeline)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "experimento_3_completo"

METRICS_CANDIDATES = (
    "data/metricas_modelos.csv",
    "metricas_modelos.csv",
)

DISPLAY_NAMES: dict[str, str] = {
    "Logistic Regression": "Regressão Logística",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "SVM": "SVM",
    "KNN": "KNN",
    "Naive Bayes": "Naive Bayes",
    "MLP Neural Network": "MLP (Rede Neural)",
    "Extra Trees": "Extra Trees",
    "Gradient Boosting": "Gradient Boosting",
}

MODEL_ORDER = list(DISPLAY_NAMES.keys())


def _f(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        return round(float(raw), 4)
    except ValueError:
        return None


def _pack_split(row: dict[str, str], *, prefix: str) -> dict[str, float | None]:
    """prefix '' for CV means, 'test_' for hold-out."""
    if prefix == "test_":
        return {
            "acuracia": _f(row, "test_accuracy"),
            "f1": _f(row, "test_f1_score"),
            "roc_auc": _f(row, "test_roc_auc"),
            "sensibilidade": _f(row, "test_sensibilidade"),
            "especificidade": _f(row, "test_especificidade"),
            "precisao": _f(row, "test_precision"),
            "pr_auc": _f(row, "test_pr_auc"),
        }
    return {
        "acuracia": _f(row, "accuracy_mean"),
        "f1": _f(row, "f1_score_mean"),
        "roc_auc": _f(row, "roc_auc_mean"),
        "sensibilidade": _f(row, "sensibilidade_mean"),
        "especificidade": _f(row, "especificidade_mean"),
        "precisao": _f(row, "precision_mean"),
        "pr_auc": _f(row, "pr_auc_mean"),
    }


class ModelMetricsService:
    def __init__(self) -> None:
        self.path: str = ""
        self._rows: list[dict[str, str]] = []
        self._cache: dict[tuple[str, bool], dict[str, Any]] = {}

    @property
    def loaded(self) -> bool:
        return bool(self._rows)

    def resolve_path(self, configured: str) -> Path:
        for candidate in [Path(configured), *[Path(p) for p in METRICS_CANDIDATES]]:
            if candidate.is_file():
                return candidate
        return Path(configured)

    def load(self, configured_path: str) -> None:
        path = self.resolve_path(configured_path)
        self.path = path.as_posix()
        self._cache.clear()
        if not path.is_file():
            logger.warning("metrics_csv_missing path=%s", self.path)
            self._rows = []
            return
        with path.open(newline="", encoding="utf-8") as handle:
            self._rows = list(csv.DictReader(handle))
        logger.info("metrics_csv_loaded path=%s rows=%s", self.path, len(self._rows))

    def listar_experimentos(self) -> list[str]:
        seen: list[str] = []
        for row in self._rows:
            exp = row.get("experimento") or ""
            if exp and exp not in seen:
                seen.append(exp)
        return seen

    def comparacao(
        self,
        experimento: str = DEFAULT_EXPERIMENT,
        *,
        apenas_melhor_por_modelo: bool = True,
    ) -> dict[str, Any]:
        key = (experimento, apenas_melhor_por_modelo)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._comparacao_impl(experimento, apenas_melhor_por_modelo)
        self._cache[key] = result
        return result

    def _comparacao_impl(
        self,
        experimento: str,
        apenas_melhor_por_modelo: bool,
    ) -> dict[str, Any]:
        if not self._rows:
            return {
                "disponivel": False,
                "experimento": experimento,
                "fonte": self.path,
                "modelos": [],
                "destaques": {},
                "mensagem": "Arquivo data/metricas_modelos.csv não encontrado.",
            }

        subset = [r for r in self._rows if (r.get("experimento") or "") == experimento]
        if not subset:
            return {
                "disponivel": False,
                "experimento": experimento,
                "fonte": self.path,
                "experimentos_disponiveis": self.listar_experimentos(),
                "modelos": [],
                "destaques": {},
                "mensagem": f"Experimento '{experimento}' não encontrado.",
            }

        if apenas_melhor_por_modelo:
            best_by_model: dict[str, dict[str, str]] = {}
            for row in subset:
                modelo = row.get("modelo") or ""
                f1 = _f(row, "f1_score_mean") or -1.0
                prev = best_by_model.get(modelo)
                prev_f1 = _f(prev, "f1_score_mean") if prev else -1.0
                if prev is None or f1 > (prev_f1 or -1.0):
                    best_by_model[modelo] = row
            selected = list(best_by_model.values())
        else:
            selected = subset

        def sort_key(row: dict[str, str]) -> tuple[int, str]:
            nome = row.get("modelo") or ""
            try:
                return (MODEL_ORDER.index(nome), nome)
            except ValueError:
                return (999, nome)

        selected.sort(key=sort_key)

        modelos: list[dict[str, Any]] = []
        for row in selected:
            nome = row.get("modelo") or ""
            modelos.append({
                "modelo": nome,
                "nome_exibicao": DISPLAY_NAMES.get(nome, nome),
                "balanceamento": row.get("balanceamento"),
                "treino_cv": _pack_split(row, prefix=""),
                "teste": _pack_split(row, prefix="test_"),
            })

        destaques = self._destaques(modelos)
        return {
            "disponivel": True,
            "experimento": experimento,
            "fonte": self.path,
            "criterio_selecao": "melhor_f1_cv" if apenas_melhor_por_modelo else "todas_combinacoes",
            "total_modelos": len(modelos),
            "destaques": destaques,
            "modelos": modelos,
            "legendas": {
                "treino_cv": "Média Stratified 10-Fold Cross-Validation",
                "teste": "Hold-out 20% (conjunto de teste)",
                "acuracia": "Cuidado: com CCF ~7%, acurácia alta pode mascarar baixa sensibilidade",
            },
        }

    @staticmethod
    def _destaques(modelos: list[dict[str, Any]]) -> dict[str, Any]:
        if not modelos:
            return {}

        def pick(path: tuple[str, str], higher: bool = True) -> dict[str, Any]:
            best = None
            best_val = None
            for m in modelos:
                split = m.get(path[0]) or {}
                val = split.get(path[1])
                if val is None:
                    continue
                if best_val is None or (higher and val > best_val) or (not higher and val < best_val):
                    best_val = val
                    best = m
            if best is None:
                return {}
            return {
                "modelo": best["modelo"],
                "nome_exibicao": best["nome_exibicao"],
                "valor": best_val,
                "balanceamento": best.get("balanceamento"),
            }

        return {
            "melhor_f1_cv": pick(("treino_cv", "f1")),
            "melhor_f1_teste": pick(("teste", "f1")),
            "melhor_roc_auc_teste": pick(("teste", "roc_auc")),
            "maior_acuracia_teste": pick(("teste", "acuracia")),
        }
