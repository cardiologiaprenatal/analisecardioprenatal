"""Metricas, validacao cruzada e visualizacoes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.config import FIGURES_DIR, N_FOLDS, RANDOM_STATE


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "sensibilidade": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "especificidade": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def _evaluate_fold(pipeline, X, y, train_idx, val_idx, fold_idx):
    from sklearn.base import clone as sk_clone
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    try:
        fold_pipe = sk_clone(pipeline)
    except Exception:
        import pickle
        fold_pipe = pickle.loads(pickle.dumps(pipeline))
    fold_pipe.fit(X_train, y_train)
    y_pred = fold_pipe.predict(X_val)
    if hasattr(fold_pipe, "predict_proba"):
        y_prob = fold_pipe.predict_proba(X_val)[:, 1]
    else:
        y_prob = fold_pipe.decision_function(X_val)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    metrics = compute_metrics(y_val.values, y_pred, y_prob)
    metrics["fold"] = fold_idx + 1
    return metrics


def cross_validate_model(pipeline, X, y, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    fold_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_evaluate_fold)(pipeline, X, y, tr, va, i)
        for i, (tr, va) in enumerate(splits)
    )
    if not fold_results:
        raise RuntimeError("Cross-validation falhou em todos os folds.")
    return pd.DataFrame(fold_results)


def aggregate_cv_results(cv_df):
    metric_cols = [c for c in cv_df.columns if c != "fold"]
    return {f"{col}_mean": cv_df[col].mean() for col in metric_cols} | {
        f"{col}_std": cv_df[col].std() for col in metric_cols
    }


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("1 - Especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, title, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(pipeline, X, y, title, save_path):
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=3, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.3, 1.0, 5), shuffle=True, random_state=RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validacao")
    ax.set_xlabel("Amostras de treino")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(y_true, y_prob, title, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Modelo")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfeita")
    ax.set_xlabel("Probabilidade predita")
    ax.set_ylabel("Frequencia observada")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR