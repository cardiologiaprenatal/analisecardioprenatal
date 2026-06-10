"""Finaliza pipeline a partir das metricas ja calculadas."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, EXPERIMENTS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COL, TEST_SIZE
from src.evaluation import (
    compute_metrics,
    ensure_figures_dir,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_pr_curve,
    plot_roc_curve,
)
from src.interpretability import (
    compute_logistic_odds_ratios,
    compute_permutation_importance,
    get_feature_names_from_pipeline,
)
from src.models import get_model_registry
from src.preprocessing import build_full_pipeline, get_class_weight_dict, split_features_target
from src.report_generator import generate_scientific_report


def main():
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    metrics_df = pd.read_csv(DATA_DIR / "metricas_modelos.csv")
    fi_df = pd.read_csv(DATA_DIR / "feature_importance.csv") if (DATA_DIR / "feature_importance.csv").exists() else pd.DataFrame()
    shap_df = pd.read_csv(DATA_DIR / "shap_importance.csv") if (DATA_DIR / "shap_importance.csv").exists() else pd.DataFrame()

    best_row = metrics_df.loc[metrics_df["f1_score_mean"].idxmax()]
    exp_name = best_row["experimento"]
    balance = best_row["balanceamento"]
    model_name = best_row["modelo"]
    feature_cols = EXPERIMENTS[exp_name]

    print(f"Melhor: {model_name} + {balance} ({exp_name}) F1={best_row['f1_score_mean']:.3f}")

    X, y = split_features_target(df, feature_cols, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    use_cw = balance == "class_weight"
    cw = get_class_weight_dict(y_train) if use_cw else None
    actual_balance = "none" if use_cw else balance
    pipeline = build_full_pipeline(get_model_registry(cw)[model_name], feature_cols, actual_balance)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    figures_dir = ensure_figures_dir()
    prefix = f"best_{exp_name}_{model_name}".replace(" ", "_")
    plot_confusion_matrix(y_test.values, y_pred, f"Confusion - {model_name}", figures_dir / f"{prefix}_confusion_matrix.png")
    plot_roc_curve(y_test.values, y_prob, f"ROC - {model_name}", figures_dir / f"{prefix}_roc_curve.png")
    plot_pr_curve(y_test.values, y_prob, f"PR - {model_name}", figures_dir / f"{prefix}_pr_curve.png")
    plot_learning_curves(pipeline, X_train, y_train, f"Learning - {model_name}", figures_dir / f"{prefix}_learning_curves.png")
    plot_calibration_curve(y_test.values, y_prob, f"Calibration - {model_name}", figures_dir / f"{prefix}_calibration_curve.png")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "best_model.joblib")

    feat_names = get_feature_names_from_pipeline(pipeline, feature_cols)
    perm_df = compute_permutation_importance(pipeline, X_test, y_test, feat_names, n_repeats=5)
    perm_df.to_csv(DATA_DIR / "permutation_importance.csv", index=False)

    lr_pipeline = build_full_pipeline(get_model_registry()["Logistic Regression"], EXPERIMENTS["experimento_3_completo"], "smote")
    X_full, y_full = split_features_target(df, EXPERIMENTS["experimento_3_completo"], TARGET_COL)
    X_tr, _, y_tr, _ = train_test_split(X_full, y_full, test_size=TEST_SIZE, stratify=y_full, random_state=RANDOM_STATE)
    lr_pipeline.fit(X_tr, y_tr)
    odds_df = compute_logistic_odds_ratios(lr_pipeline, get_feature_names_from_pipeline(lr_pipeline, EXPERIMENTS["experimento_3_completo"]))
    if not odds_df.empty:
        odds_df.to_csv(DATA_DIR / "odds_ratios_logistic.csv", index=False)

    experiment_summaries = {}
    for exp in metrics_df["experimento"].unique():
        sub = metrics_df[metrics_df["experimento"] == exp]
        b = sub.loc[sub["f1_score_mean"].idxmax()]
        experiment_summaries[exp] = {
            "melhor_modelo": b["modelo"], "melhor_balanceamento": b["balanceamento"],
            "f1_cv_mean": b["f1_score_mean"], "roc_auc_cv_mean": b["roc_auc_mean"],
            "pr_auc_cv_mean": b["pr_auc_mean"], "test_f1": b["test_f1_score"], "test_roc_auc": b["test_roc_auc"],
        }

    best_overall = best_row.to_dict()
    best_overall.update(compute_metrics(y_test.values, y_pred, y_prob))

    report_path = generate_scientific_report(
        df=df, metrics_df=metrics_df, fi_df=fi_df, shap_df=shap_df,
        experiment_summaries=experiment_summaries, best_overall=best_overall,
    )

    summary = {
        "report_path": str(report_path),
        "best_model": model_name,
        "best_balance": balance,
        "best_experiment": exp_name,
        "f1_cv": float(best_row["f1_score_mean"]),
    }
    with open(REPORTS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Concluido! Relatorio: {report_path}")


if __name__ == "__main__":
    main()
