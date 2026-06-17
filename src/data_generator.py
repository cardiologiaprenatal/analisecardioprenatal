"""
Gerador de base sintética para predição de Cardiopatia Congênita Fetal (CCF).

Os pesos de risco foram calibrados com base em associações descritas na literatura
obstétrica e de cardiologia fetal, simulando um cenário epidemiológico plausível.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    N_SAMPLES,
    RANDOM_STATE,
    TARGET_PREVALENCE_MAX,
    TARGET_PREVALENCE_MIN,
)

# Pesos log-odds para fatores de risco (literatura médica)
RISK_WEIGHTS: dict[str, float] = {
    # Marcadores fetais — associação forte
    "translucencia_nucal_aumentada": 2.80,
    "alteracao_ducto_venoso": 2.50,
    "regurgitacao_tricuspide": 2.20,
    "malformacao_extracardiaca": 2.60,
    # Histórico reprodutivo/familiar
    "historia_familiar_cardiopatia": 1.90,
    "filho_anterior_cardiopatia": 2.10,
    "historia_natimorto": 0.90,
    # Doenças maternas
    "diabetes_pre_gestacional": 1.70,
    "diabetes_gestacional": 0.60,
    "lupus": 1.20,
    "fenilcetonuria": 1.50,
    "doenca_tireoide": 0.50,
    "hipertensao_cronica": 0.70,
    # Infecções
    "rubeola": 2.00,
    "citomegalovirus": 1.80,
    "toxoplasmose": 0.40,
    "sifilis": 0.80,
    # Medicações teratogênicas
    "isotretinoina": 2.30,
    "acido_valproico": 2.00,
    "litio": 1.10,
    # Hábitos e exposição
    "alcool": 1.00,
    "tabagismo": 0.60,
    "drogas_ilicitas": 0.90,
    "solventes": 0.70,
    "pesticidas": 0.50,
    # Gestacionais
    "gestacao_gemelar": 1.40,
    # Outros marcadores fetais
    "restricao_crescimento_fetal": 1.30,
    "polidramnio": 0.80,
    "oligodramnio": 1.00,
}

# Probabilidades marginais para variáveis independentes
MARGINAL_PROBS: dict[str, float] = {
    "historia_natimorto": 0.03,
    "filho_anterior_cardiopatia": 0.008,
    "historia_familiar_cardiopatia": 0.025,
    "diabetes_pre_gestacional": 0.025,
    "diabetes_gestacional": 0.08,
    "lupus": 0.005,
    "fenilcetonuria": 0.001,
    "doenca_tireoide": 0.04,
    "hipertensao_cronica": 0.05,
    "rubeola": 0.002,
    "citomegalovirus": 0.015,
    "toxoplasmose": 0.01,
    "sifilis": 0.008,
    "isotretinoina": 0.002,
    "acido_valproico": 0.003,
    "litio": 0.004,
    "alcool": 0.05,
    "tabagismo": 0.12,
    "drogas_ilicitas": 0.02,
    "solventes": 0.03,
    "pesticidas": 0.025,
    "gestacao_gemelar": 0.025,
    "translucencia_nucal_aumentada": 0.04,
    "alteracao_ducto_venoso": 0.025,
    "regurgitacao_tricuspide": 0.03,
    "malformacao_extracardiaca": 0.02,
    "restricao_crescimento_fetal": 0.06,
    "polidramnio": 0.04,
    "oligodramnio": 0.05,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_binary(rng: np.random.Generator, n: int, key: str) -> np.ndarray:
    return rng.binomial(1, MARGINAL_PROBS[key], size=n)


def generate_synthetic_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Gera dataset sintético com variável alvo baseada em combinação probabilística."""
    rng = np.random.default_rng(seed)

    # Variáveis contínuas e categóricas maternas
    idade_materna = rng.normal(28.5, 6.5, n_samples).clip(14, 48).round(1)
    etnia = rng.choice(
        ["Branca", "Parda", "Preta", "Amarela", "Indigena"],
        size=n_samples,
        p=[0.45, 0.35, 0.12, 0.05, 0.03],
    )
    imc_pre_gestacional = rng.normal(25.5, 5.0, n_samples).clip(16, 45).round(1)
    paridade = rng.poisson(1.2, n_samples).clip(0, 6)
    abortamentos_previos = rng.poisson(0.3, n_samples).clip(0, 4)

    data: dict[str, np.ndarray] = {
        "idade_materna": idade_materna,
        "etnia": etnia,
        "imc_pre_gestacional": imc_pre_gestacional,
        "paridade": paridade,
        "abortamentos_previos": abortamentos_previos,
    }

    # Variáveis binárias
    for key in MARGINAL_PROBS:
        data[key] = _generate_binary(rng, n_samples, key)

    # Idade gestacional (semanas)
    data["idade_gestacional"] = rng.normal(20, 3.5, n_samples).clip(11, 28).round(1)

    # Corionicidade (apenas para gemelares)
    corionicidade = np.full(n_samples, "NA", dtype=object)
    twin_mask = data["gestacao_gemelar"] == 1
    n_twins = twin_mask.sum()
    if n_twins > 0:
        corionicidade[twin_mask] = rng.choice(
            ["Monocoriônica", "Dicoriônica"],
            size=n_twins,
            p=[0.30, 0.70],
        )
    data["corionicidade"] = corionicidade

    df = pd.DataFrame(data)

    # Efeitos contínuos no log-odds
    log_odds = np.full(n_samples, -3.8)  # intercepto calibrado

    # Idade extrema (<18 ou >35)
    log_odds += np.where(df["idade_materna"] < 18, 0.50, 0.0)
    log_odds += np.where(df["idade_materna"] > 35, 0.60, 0.0)

    # IMC extremos
    log_odds += np.where(df["imc_pre_gestacional"] < 18.5, 0.30, 0.0)
    log_odds += np.where(df["imc_pre_gestacional"] >= 30, 0.45, 0.0)

    # Paridade e abortamentos
    log_odds += df["paridade"].values * 0.08
    log_odds += df["abortamentos_previos"].values * 0.12

    # Corionicidade monocoriônica
    mono_mask = df["corionicidade"] == "Monocoriônica"
    log_odds += mono_mask.astype(float).values * 0.80

    # Fatores binários ponderados
    for var, weight in RISK_WEIGHTS.items():
        log_odds += df[var].values * weight

    # Co-ocorrência clínica: marcadores fetais tendem a aparecer juntos
    fetal_cluster = (
        df["translucencia_nucal_aumentada"]
        + df["alteracao_ducto_venoso"]
        + df["regurgitacao_tricuspide"]
    )
    log_odds += (fetal_cluster >= 2).astype(float) * 0.70

    probs = _sigmoid(log_odds)

    # Calibrar intercepto para prevalência alvo (3–8%)
    target_prev = (TARGET_PREVALENCE_MIN + TARGET_PREVALENCE_MAX) / 2
    current_prev = probs.mean()
    if current_prev > 0:
        adjustment = np.log(target_prev / (1 - target_prev)) - np.log(
            current_prev / (1 - current_prev)
        )
        probs = _sigmoid(np.log(probs / (1 - probs + 1e-9)) + adjustment * 0.5)

    # Garantir faixa de prevalência
    for _ in range(20):
        prev = probs.mean()
        if TARGET_PREVALENCE_MIN <= prev <= TARGET_PREVALENCE_MAX:
            break
        scale = 1.05 if prev < TARGET_PREVALENCE_MIN else 0.95
        probs = np.clip(probs * scale, 0.001, 0.999)
        probs = _sigmoid(np.log(probs / (1 - probs)) + (0.1 if prev < TARGET_PREVALENCE_MIN else -0.1))

    df["cardiopatia_congenita"] = rng.binomial(1, probs)

    # Introduzir ~2% de missing values aleatórios (exceto target)
    for col in df.columns:
        if col == "cardiopatia_congenita":
            continue
        missing_mask = rng.random(n_samples) < 0.02
        df.loc[missing_mask, col] = np.nan

    return df


def save_dataset(df: pd.DataFrame, path: str | None = None) -> str:
    """Salva dataset em CSV."""
    from src.config import DATA_DIR

    output_path = path or str(DATA_DIR / "dataset.csv")
    df.to_csv(output_path, index=False)
    return output_path
