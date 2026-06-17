#!/usr/bin/env python3
"""Gera gráficos de análise descritiva e resumo do modelo GestRisk CCF."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.intercorrencias import CATEGORIA_LABELS, FIELD_META
from api.services import model_service
from src.config import BINARY_COLS, FEATURE_GROUPS, TARGET_COL

OUTPUT_DIR = PROJECT_ROOT / "analise_descritiva"
FIGURES_DIR = OUTPUT_DIR / "figuras"
DATASET_PATH = PROJECT_ROOT / "dataset_com_id_gestante.csv"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def grafico_prevalencia_ccf(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Sem CCF", "Com CCF"]
    colors = ["#4CAF50", "#E53935"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, edgecolor="white")
    prev = df[TARGET_COL].mean() * 100
    ax.set_title(f"Prevalência de Cardiopatia Congênita Fetal\n({prev:.1f}% dos casos)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Número de gestantes")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{int(h):,}", ha="center", fontsize=11)
    return _save(fig, "01_prevalencia_ccf.png")


def grafico_distribuicao_idade_imc(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["idade_materna"].dropna(), kde=True, ax=axes[0], color="#1976D2")
    axes[0].axvline(18, color="red", linestyle="--", alpha=0.7, label="18 anos")
    axes[0].axvline(35, color="red", linestyle="--", alpha=0.7, label="35 anos")
    axes[0].set_title("Idade materna")
    axes[0].set_xlabel("Anos")
    axes[0].legend(fontsize=8)

    sns.histplot(df["imc_pre_gestacional"].dropna(), kde=True, ax=axes[1], color="#7B1FA2")
    axes[1].axvline(18.5, color="red", linestyle="--", alpha=0.7)
    axes[1].axvline(30, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("IMC pré-gestacional")
    axes[1].set_xlabel("kg/m²")

    sns.histplot(df["idade_gestacional"].dropna(), kde=True, ax=axes[2], color="#00897B")
    axes[2].set_title("Idade gestacional")
    axes[2].set_xlabel("Semanas")
    fig.suptitle("Distribuição de variáveis contínuas", fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "02_distribuicao_idade_imc.png")


def grafico_ccf_por_fatores(df: pd.DataFrame) -> str:
    fatores = [
        ("translucencia_nucal_aumentada", "TN aumentada"),
        ("alteracao_ducto_venoso", "Ducto venoso"),
        ("gestacao_gemelar", "Gestação gemelar"),
        ("diabetes_pre_gestacional", "Diabetes pré-gest."),
        ("historia_familiar_cardiopatia", "Hist. familiar CCF"),
    ]
    rows = []
    for campo, label in fatores:
        for val, nome in [(0, "Ausente"), (1, "Presente")]:
            sub = df[df[campo].fillna(0) == val]
            if len(sub) == 0:
                continue
            rows.append({"Fator": label, "Condição": nome, "Prevalência CCF (%)": sub[TARGET_COL].mean() * 100})

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Fator", y="Prevalência CCF (%)", hue="Condição", ax=ax, palette=["#81C784", "#E57373"])
    ax.set_title("Prevalência de CCF por fator de risco", fontsize=13, fontweight="bold")
    ax.set_ylabel("Prevalência (%)")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "03_ccf_por_fatores_risco.png")


def grafico_intercorrencias_categoria(analises: list[dict]) -> str:
    counter: Counter[str] = Counter()
    for a in analises:
        for item in a["intercorrencias"]:
            counter[item["categoria_label"]] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    cats = list(counter.keys())
    vals = [counter[c] for c in cats]
    order = np.argsort(vals)
    cats = [cats[i] for i in order]
    vals = [vals[i] for i in order]
    ax.barh(cats, vals, color=sns.color_palette("viridis", len(cats)))
    ax.set_title("Intercorrências detectadas por categoria", fontsize=13, fontweight="bold")
    ax.set_xlabel("Quantidade de ocorrências")
    return _save(fig, "04_intercorrencias_por_categoria.png")


def grafico_top_intercorrencias(analises: list[dict], top_n: int = 12) -> str:
    counter: Counter[str] = Counter()
    for a in analises:
        for item in a["intercorrencias"]:
            counter[item["nome"]] += 1

    top = counter.most_common(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    nomes = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    ax.barh(nomes, vals, color="#FF7043")
    ax.set_title(f"Top {top_n} intercorrências mais frequentes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de gestantes")
    return _save(fig, "05_top_intercorrencias.png")


def grafico_distribuicao_risco(analises: list[dict]) -> str:
    riscos = Counter(a["analise"]["risco"] for a in analises)
    ordem = ["baixo", "moderado", "alto"]
    labels = [r for r in ordem if r in riscos]
    vals = [riscos[r] for r in labels]
    cores = {"baixo": "#66BB6A", "moderado": "#FFA726", "alto": "#EF5350"}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(vals, labels=[r.capitalize() for r in labels], autopct="%1.1f%%",
           colors=[cores[r] for r in labels], startangle=90, textprops={"fontsize": 11})
    ax.set_title("Distribuição de risco predito (modelo CatBoost)", fontsize=13, fontweight="bold")
    return _save(fig, "06_distribuicao_risco.png")


def grafico_probabilidade_modelo(analises: list[dict]) -> str:
    probas = [a["analise"]["probabilidade_ccf"] for a in analises]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(probas, bins=40, kde=True, ax=ax, color="#5C6BC0")
    ax.axvline(0.15, color="orange", linestyle="--", label="Limite moderado (15%)")
    ax.axvline(0.35, color="red", linestyle="--", label="Limite alto (35%)")
    ax.set_title("Distribuição da probabilidade predita de CCF", fontsize=13, fontweight="bold")
    ax.set_xlabel("Probabilidade")
    ax.set_ylabel("Gestantes")
    ax.legend()
    return _save(fig, "07_distribuicao_probabilidade.png")


def grafico_prioridade_ecocardiograma(analises: list[dict]) -> str:
    prioridades = Counter(a["analise"]["prioridade_ecocardiograma"] for a in analises)
    ordem = ["urgente", "alta", "moderada", "baixa"]
    labels = [p.capitalize() for p in ordem if p in prioridades]
    vals = [prioridades[p] for p in ordem if p in prioridades]
    cores = {"urgente": "#B71C1C", "alta": "#E65100", "moderada": "#F9A825", "baixa": "#2E7D32"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=[cores.get(p.lower(), "#888") for p in labels], edgecolor="white")
    ax.set_title("Priorização para ecocardiograma fetal", fontsize=13, fontweight="bold")
    ax.set_ylabel("Número de gestantes")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{int(h):,}", ha="center", fontsize=10)
    return _save(fig, "08_prioridade_ecocardiograma.png")


def grafico_importancia_features() -> str | None:
    fi_path = PROJECT_ROOT / "data" / "feature_importance.csv"
    if not fi_path.exists():
        return None

    fi = pd.read_csv(fi_path).head(15).copy()
    fi["feature"] = fi["feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi, y="feature", x="importance", ax=ax, palette="rocket")
    ax.set_title("Top 15 features — importância CatBoost", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    return _save(fig, "09_importancia_features.png")


def grafico_metricas_experimentos() -> str | None:
    summary_path = PROJECT_ROOT / "reports" / "pipeline_summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exp = summary.get("experiment_summaries", {})
    if not exp:
        return None

    rows = []
    for nome, dados in exp.items():
        rows.append({
            "Experimento": nome.replace("experimento_", "Exp. ").replace("_", " ").title(),
            "F1 CV": dados["f1_cv_mean"],
            "ROC-AUC CV": dados["roc_auc_cv_mean"],
            "PR-AUC CV": dados["pr_auc_cv_mean"],
            "Melhor modelo": dados["melhor_modelo"],
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["F1 CV"], w, label="F1", color="#42A5F5")
    ax.bar(x, df["ROC-AUC CV"], w, label="ROC-AUC", color="#66BB6A")
    ax.bar(x + w, df["PR-AUC CV"], w, label="PR-AUC", color="#AB47BC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Experimento"], rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Comparativo de experimentos do modelo", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    return _save(fig, "10_comparativo_experimentos.png")


def grafico_etnia_ccf(df: pd.DataFrame) -> str:
    et = df.groupby("etnia")[TARGET_COL].agg(["mean", "count"]).reset_index()
    et = et[et["count"] >= 20].sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=et, x="etnia", y="mean", ax=ax, palette="muted")
    ax.set_title("Prevalência de CCF por etnia", fontsize=13, fontweight="bold")
    ax.set_ylabel("Prevalência")
    ax.set_ylim(0, et["mean"].max() * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    return _save(fig, "11_prevalencia_por_etnia.png")


def gerar_resumo_markdown(
    df: pd.DataFrame,
    analises: list[dict],
    figuras: dict[str, str],
) -> str:
    n = len(df)
    prev = df[TARGET_COL].mean()
    riscos = Counter(a["analise"]["risco"] for a in analises)
    prioridades = Counter(a["analise"]["prioridade_ecocardiograma"] for a in analises)
    recomenda_eco = sum(1 for a in analises if a["analise"]["recomenda_ecocardiograma"])
    media_inter = np.mean([a["resumo_intercorrencias"]["total"] for a in analises])

    counter_cat: Counter[str] = Counter()
    for a in analises:
        for item in a["intercorrencias"]:
            counter_cat[item["categoria_label"]] += 1

    lines = [
        "# Resumo da Análise Descritiva — GestRisk CCF",
        "",
        "## Modelo em uso",
        "",
        "| Item | Valor |",
        "|------|-------|",
        "| **Modelo** | CatBoost |",
        "| **Experimento** | experimento_3_completo (36 features) |",
        "| **Balanceamento** | class_weight |",
        f"| **Dataset** | `{DATASET_PATH.name}` |",
        f"| **Gestantes** | {n:,} |",
        f"| **Prevalência real de CCF** | {prev:.1%} |",
        "",
        "## 1. Análise do dataset",
        "",
        f"- **Idade materna média:** {df['idade_materna'].mean():.1f} anos (DP {df['idade_materna'].std():.1f})",
        f"- **IMC médio:** {df['imc_pre_gestacional'].mean():.1f} kg/m²",
        f"- **Idade gestacional média:** {df['idade_gestacional'].mean():.1f} semanas",
        f"- **Gestação gemelar:** {df['gestacao_gemelar'].fillna(0).mean():.1%}",
        "",
        f"![Prevalência](figuras/01_prevalencia_ccf.png)",
        "",
        f"![Distribuição](figuras/02_distribuicao_idade_imc.png)",
        "",
        "**Resumo:** O dataset é desbalanceado (~{prev:.0%} CCF), típico de triagem fetal. A maioria das gestantes tem idade e IMC dentro da faixa habitual.",
        "",
        "## 2. Análise de fatores de risco",
        "",
        f"![CCF por fatores](figuras/03_ccf_por_fatores_risco.png)",
        "",
        f"![Prevalência por etnia](figuras/11_prevalencia_por_etnia.png)",
        "",
        "**Resumo:** Marcadores fetais (TN aumentada, ducto venoso) e histórico familiar concentram maior prevalência de CCF quando presentes.",
        "",
        "## 3. Análise de intercorrências (API)",
        "",
        f"- **Média de intercorrências por gestante:** {media_inter:.1f}",
        f"- **Total de intercorrências detectadas:** {sum(counter_cat.values()):,}",
        "",
        "**Por categoria:**",
        "",
    ]
    for cat, qtd in counter_cat.most_common():
        lines.append(f"- {cat}: {qtd:,}")

    lines += [
        "",
        f"![Intercorrências por categoria](figuras/04_intercorrencias_por_categoria.png)",
        "",
        f"![Top intercorrências](figuras/05_top_intercorrencias.png)",
        "",
        "**Resumo:** Cada gestante é analisada campo a campo. As intercorrências são as alterações clínicas detectadas — é nelas que estão os sinais de risco individual.",
        "",
        "## 4. Análise preditiva do modelo",
        "",
        f"- **Risco baixo:** {riscos.get('baixo', 0):,} ({riscos.get('baixo', 0)/n:.1%})",
        f"- **Risco moderado:** {riscos.get('moderado', 0):,} ({riscos.get('moderado', 0)/n:.1%})",
        f"- **Risco alto:** {riscos.get('alto', 0):,} ({riscos.get('alto', 0)/n:.1%})",
        "",
        f"![Distribuição de risco](figuras/06_distribuicao_risco.png)",
        "",
        f"![Probabilidade](figuras/07_distribuicao_probabilidade.png)",
        "",
        "**Resumo:** O modelo CatBoost estima probabilidade contínua de CCF. Limiares: <15% baixo, 15–35% moderado, ≥35% alto.",
        "",
        "## 5. Priorização de ecocardiograma fetal",
        "",
        f"- **Recomendam ecocardiograma:** {recomenda_eco:,} ({recomenda_eco/n:.1%})",
        f"- **Urgente:** {prioridades.get('urgente', 0):,}",
        f"- **Alta:** {prioridades.get('alta', 0):,}",
        f"- **Moderada:** {prioridades.get('moderada', 0):,}",
        f"- **Baixa:** {prioridades.get('baixa', 0):,}",
        "",
        f"![Priorização eco](figuras/08_prioridade_ecocardiograma.png)",
        "",
        "**Resumo:** A fila prioriza gestantes com marcadores fetais, alta probabilidade do modelo ou múltiplas intercorrências.",
        "",
        "## 6. Performance do modelo (pipeline)",
        "",
    ]

    summary_path = PROJECT_ROOT / "reports" / "pipeline_summary.json"
    if summary_path.exists():
        best = json.loads(summary_path.read_text(encoding="utf-8")).get("best_model", {})
        lines += [
            f"| Métrica | Validação cruzada | Teste |",
            f"|---------|-------------------|-------|",
            f"| F1-Score | {best.get('f1_score_mean', 0):.3f} | {best.get('test_f1_score', 0):.3f} |",
            f"| ROC-AUC | {best.get('roc_auc_mean', 0):.3f} | {best.get('test_roc_auc', 0):.3f} |",
            f"| Sensibilidade | {best.get('sensibilidade_mean', 0):.3f} | {best.get('test_sensibilidade', 0):.3f} |",
            f"| Especificidade | {best.get('especificidade_mean', 0):.3f} | {best.get('test_especificidade', 0):.3f} |",
            "",
        ]

    if figuras.get("importancia"):
        lines += [
            f"![Importância features](figuras/09_importancia_features.png)",
            "",
        ]
    if figuras.get("experimentos"):
        lines += [
            f"![Experimentos](figuras/10_comparativo_experimentos.png)",
            "",
            "**Resumo:** O experimento completo (36 variáveis) supera os demais. Marcadores fetais são as features mais importantes.",
            "",
        ]

    lines += [
        "## Gráficos gerados",
        "",
    ]
    for nome, path in sorted(figuras.items()):
        if path:
            lines.append(f"- `{Path(path).name}` — {nome}")

    return "\n".join(lines)


def main() -> None:
    print("Carregando modelo e análises...")
    model_service.load_or_train()
    analises = list(model_service._analises_cache.values())

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {len(df)} gestantes | Análises: {len(analises)}")

    figuras: dict[str, str] = {}
    print("Gerando gráficos...")
    figuras["prevalencia"] = grafico_prevalencia_ccf(df)
    figuras["distribuicao"] = grafico_distribuicao_idade_imc(df)
    figuras["fatores_risco"] = grafico_ccf_por_fatores(df)
    figuras["intercorrencias_cat"] = grafico_intercorrencias_categoria(analises)
    figuras["top_intercorrencias"] = grafico_top_intercorrencias(analises)
    figuras["risco"] = grafico_distribuicao_risco(analises)
    figuras["probabilidade"] = grafico_probabilidade_modelo(analises)
    figuras["prioridade_eco"] = grafico_prioridade_ecocardiograma(analises)
    figuras["etnia"] = grafico_etnia_ccf(df)

    imp = grafico_importancia_features()
    if imp:
        figuras["importancia"] = imp
    exp = grafico_metricas_experimentos()
    if exp:
        figuras["experimentos"] = exp

    resumo = gerar_resumo_markdown(df, analises, figuras)
    resumo_path = OUTPUT_DIR / "resumo_analise_descritiva.md"
    resumo_path.write_text(resumo, encoding="utf-8")

    meta = {
        "total_gestantes": len(df),
        "prevalencia_ccf": round(float(df[TARGET_COL].mean()), 4),
        "graficos": figuras,
        "resumo": str(resumo_path),
    }
    (OUTPUT_DIR / "analise_descritiva.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nConcluído! {len(figuras)} gráficos em: {FIGURES_DIR}")
    print(f"Resumo: {resumo_path}")


if __name__ == "__main__":
    main()
