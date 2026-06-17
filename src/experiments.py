"""Orquestracao dos experimentos de predicao de CCF."""

from __future__ import annotations
import json, time, warnings
from typing import Any
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import BALANCE_STRATEGIES, DATA_DIR, EXPERIMENTS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COL, TEST_SIZE
from src.data_generator import generate_synthetic_dataset, save_dataset
from src.evaluation import aggregate_cv_results, compute_metrics, cross_validate_model, ensure_figures_dir
from src.evaluation import plot_calibration_curve, plot_confusion_matrix, plot_learning_curves, plot_pr_curve, plot_roc_curve
from src.interpretability import compute_logistic_odds_ratios, compute_permutation_importance, compute_shap_importance, extract_feature_importance, get_feature_names_from_pipeline
from src.models import get_model_registry
from src.preprocessing import build_full_pipeline, get_class_weight_dict, split_features_target

warnings.filterwarnings("ignore")

def run_full_pipeline():
    t0 = time.time()
    print("=" * 60)
    print("GestRisk CCF - Pipeline de Predicao de Cardiopatia Congenita Fetal")
    print("=" * 60)

    print("\n[1/5] Gerando dataset sintetico...")
    df = generate_synthetic_dataset()
    dataset_path = save_dataset(df)
    prevalence = df[TARGET_COL].mean()
    print(f"  -> Dataset: {dataset_path}")
    print(f"  -> Registros: {len(df):,} | Prevalencia CCF: {prevalence:.2%}")

    all_metrics = []
    experiment_summaries = {}
    best_overall = {"f1_score_mean": -1}
    figures_dir = ensure_figures_dir()
    total = len(EXPERIMENTS) * len(BALANCE_STRATEGIES) * 12
    done = 0

    for exp_name, feature_cols in EXPERIMENTS.items():
        print(f"\n[Experimento] {exp_name} ({len(feature_cols)} features)")
        X, y = split_features_target(df, feature_cols, TARGET_COL)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        exp_metrics = []

        for balance in BALANCE_STRATEGIES:
            use_cw = balance == "class_weight"
            cw = get_class_weight_dict(y_train) if use_cw else None
            actual_balance = "none" if use_cw else balance
            models = get_model_registry(class_weight=cw)

            for model_name, estimator in models.items():
                done += 1
                t_model = time.time()
                pipeline = build_full_pipeline(estimator, feature_cols, actual_balance)
                try:
                    cv_results = cross_validate_model(pipeline, X_train, y_train)
                    cv_agg = aggregate_cv_results(cv_results)
                    pipeline.fit(X_train, y_train)
                except Exception as exc:
                    print(f"  [{done}/{total}] {balance}/{model_name} ERRO: {exc}", flush=True)
                    continue

                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(X_test)
                if not hasattr(pipeline, "predict_proba"):
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

                test_metrics = compute_metrics(y_test.values, y_pred, y_prob)
                record = {"experimento": exp_name, "balanceamento": balance, "modelo": model_name,
                    **{k: round(v, 4) for k, v in cv_agg.items()},
                    **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()}}
                all_metrics.append(record)
                exp_metrics.append(record)

                if cv_agg.get("f1_score_mean", 0) > best_overall.get("f1_score_mean", -1):
                    best_overall = {**record, "pipeline": pipeline, "X_train": X_train, "X_test": X_test,
                        "y_train": y_train, "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob, "feature_cols": feature_cols}

                print(f"  [{done}/{total}] {balance}/{model_name} F1={cv_agg['f1_score_mean']:.3f} ({time.time()-t_model:.1f}s)", flush=True)

        exp_df = pd.DataFrame(exp_metrics)
        best_exp = exp_df.loc[exp_df["f1_score_mean"].idxmax()]
        experiment_summaries[exp_name] = {
            "melhor_modelo": best_exp["modelo"], "melhor_balanceamento": best_exp["balanceamento"],
            "f1_cv_mean": best_exp["f1_score_mean"], "roc_auc_cv_mean": best_exp["roc_auc_mean"],
            "pr_auc_cv_mean": best_exp["pr_auc_mean"], "test_f1": best_exp["test_f1_score"], "test_roc_auc": best_exp["test_roc_auc"],
        }
        print(f"  -> Melhor: {best_exp['modelo']} + {best_exp['balanceamento']} (F1={best_exp['f1_score_mean']:.3f})")

    print("\n[3/5] Salvando metricas...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = DATA_DIR / "metricas_modelos.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fi_df = pd.DataFrame()
    shap_df = pd.DataFrame()
    if "pipeline" in best_overall:
        print("[3b] Interpretabilidade do melhor modelo...")
        bp = best_overall
        feat_names = get_feature_names_from_pipeline(bp["pipeline"], bp["feature_cols"])
        fi_df = extract_feature_importance(bp["pipeline"], feat_names)
        fi_df["experimento"] = bp["experimento"]
        fi_df["modelo"] = bp["modelo"]
        fi_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
        shap_df = compute_shap_importance(bp["pipeline"], bp["X_test"], feat_names, max_samples=300)
        shap_df["experimento"] = bp["experimento"]
        shap_df["modelo"] = bp["modelo"]
        shap_df.to_csv(DATA_DIR / "shap_importance.csv", index=False)
        perm_df = compute_permutation_importance(bp["pipeline"], bp["X_test"], bp["y_test"], feat_names, n_repeats=5)
        perm_df.to_csv(DATA_DIR / "permutation_importance.csv", index=False)

    print("\n[4/5] Visualizacoes...")
    if "pipeline" in best_overall:
        bp = best_overall
        prefix = f"best_{bp['experimento']}_{bp['modelo']}".replace(" ", "_")
        plot_confusion_matrix(bp["y_test"].values, bp["y_pred"], f"Confusion Matrix - {bp['modelo']}", figures_dir / f"{prefix}_confusion_matrix.png")
        plot_roc_curve(bp["y_test"].values, bp["y_prob"], f"ROC - {bp['modelo']}", figures_dir / f"{prefix}_roc_curve.png")
        plot_pr_curve(bp["y_test"].values, bp["y_prob"], f"PR - {bp['modelo']}", figures_dir / f"{prefix}_pr_curve.png")
        plot_learning_curves(bp["pipeline"], bp["X_train"], bp["y_train"], f"Learning - {bp['modelo']}", figures_dir / f"{prefix}_learning_curves.png")
        plot_calibration_curve(bp["y_test"].values, bp["y_prob"], f"Calibration - {bp['modelo']}", figures_dir / f"{prefix}_calibration_curve.png")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bp["pipeline"], MODELS_DIR / "best_model.joblib")

        lr_pipeline = build_full_pipeline(get_model_registry()["Logistic Regression"], EXPERIMENTS["experimento_3_completo"], "smote")
        X_full, y_full = split_features_target(df, EXPERIMENTS["experimento_3_completo"], TARGET_COL)
        X_tr, _, y_tr, _ = train_test_split(X_full, y_full, test_size=TEST_SIZE, stratify=y_full, random_state=RANDOM_STATE)
        lr_pipeline.fit(X_tr, y_tr)
        odds_df = compute_logistic_odds_ratios(lr_pipeline, get_feature_names_from_pipeline(lr_pipeline, EXPERIMENTS["experimento_3_completo"]))
        if not odds_df.empty:
            odds_df.to_csv(DATA_DIR / "odds_ratios_logistic.csv", index=False)

    print("\n[5/5] Relatorio cientifico...")
    from src.report_generator import generate_scientific_report
    report_path = generate_scientific_report(df=df, metrics_df=metrics_df, fi_df=fi_df, shap_df=shap_df, experiment_summaries=experiment_summaries, best_overall=best_overall)

    summary = {"dataset_path": str(dataset_path), "metrics_path": str(metrics_path), "report_path": str(report_path),
        "prevalence": prevalence, "experiment_summaries": experiment_summaries,
        "best_model": {k: v for k, v in best_overall.items() if k not in ("pipeline", "X_train", "X_test", "y_train", "y_test", "y_pred", "y_prob", "feature_cols")},
        "elapsed_seconds": round(time.time() - t0, 1)}
    with open(REPORTS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print(f"Pipeline concluido em {summary['elapsed_seconds']}s!")
    print(f"  Relatorio: {report_path}")
    print("=" * 60)
    return summary