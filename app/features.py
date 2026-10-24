"""Features clínicas canônicas — experimento 3 completo (36 variáveis)."""

from __future__ import annotations

# Mesma ordem de src.config.EXPERIMENTS["experimento_3_completo"]
FULL_FEATURE_COLS: list[str] = [
    # Maternas
    "idade_materna",
    "etnia",
    "imc_pre_gestacional",
    "paridade",
    "abortamentos_previos",
    "historia_natimorto",
    "filho_anterior_cardiopatia",
    "historia_familiar_cardiopatia",
    # Doenças
    "diabetes_pre_gestacional",
    "diabetes_gestacional",
    "lupus",
    "fenilcetonuria",
    "doenca_tireoide",
    "hipertensao_cronica",
    # Infecções
    "rubeola",
    "citomegalovirus",
    "toxoplasmose",
    "sifilis",
    # Medicações
    "isotretinoina",
    "acido_valproico",
    "litio",
    # Hábitos
    "alcool",
    "tabagismo",
    "drogas_ilicitas",
    # Exposição
    "solventes",
    "pesticidas",
    # Gestacionais
    "idade_gestacional",
    "gestacao_gemelar",
    "corionicidade",
    # Marcadores fetais
    "translucencia_nucal_aumentada",
    "alteracao_ducto_venoso",
    "regurgitacao_tricuspide",
    "malformacao_extracardiaca",
    "restricao_crescimento_fetal",
    "polidramnio",
    "oligodramnio",
]

CATEGORICAL_COLS = {"etnia", "corionicidade"}

# Contrato enxuto Laravel (PythonPredictPayloadMapper) — fallback heurístico
SLIM_FEATURE_ORDER: list[str] = [
    "idade",
    "diabetes",
    "hipertensao",
    "imc",
    "idade_gestacional",
    "tabagismo",
    "paridade",
]

DEFAULTS: dict[str, float | str] = {
    "idade_materna": 28.0,
    "etnia": "Branca",
    "imc_pre_gestacional": 24.0,
    "paridade": 0,
    "abortamentos_previos": 0,
    "historia_natimorto": 0,
    "filho_anterior_cardiopatia": 0,
    "historia_familiar_cardiopatia": 0,
    "diabetes_pre_gestacional": 0,
    "diabetes_gestacional": 0,
    "lupus": 0,
    "fenilcetonuria": 0,
    "doenca_tireoide": 0,
    "hipertensao_cronica": 0,
    "rubeola": 0,
    "citomegalovirus": 0,
    "toxoplasmose": 0,
    "sifilis": 0,
    "isotretinoina": 0,
    "acido_valproico": 0,
    "litio": 0,
    "alcool": 0,
    "tabagismo": 0,
    "drogas_ilicitas": 0,
    "solventes": 0,
    "pesticidas": 0,
    "idade_gestacional": 20.0,
    "gestacao_gemelar": 0,
    "corionicidade": "NA",
    "translucencia_nucal_aumentada": 0,
    "alteracao_ducto_venoso": 0,
    "regurgitacao_tricuspide": 0,
    "malformacao_extracardiaca": 0,
    "restricao_crescimento_fetal": 0,
    "polidramnio": 0,
    "oligodramnio": 0,
}


def _to_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_bin(value: object) -> int:
    return 1 if _to_float(value, 0.0) >= 0.5 else 0


def build_full_clinical_row(payload: dict) -> dict[str, float | int | str]:
    """
    Normaliza payload (CSV, CcfPayloadMapper ou /predict enxuto) para as 36 features.
    """
    p = dict(payload)

    # aliases do contrato enxuto Laravel
    if p.get("idade_materna") is None and p.get("idade") is not None:
        p["idade_materna"] = p["idade"]
    if p.get("imc_pre_gestacional") is None and p.get("imc") is not None:
        p["imc_pre_gestacional"] = p["imc"]
    if p.get("hipertensao_cronica") is None and p.get("hipertensao") is not None:
        p["hipertensao_cronica"] = p["hipertensao"]
    if p.get("diabetes") is not None:
        d = _to_bin(p["diabetes"])
        if p.get("diabetes_gestacional") is None:
            p["diabetes_gestacional"] = d
        if p.get("diabetes_pre_gestacional") is None and d:
            p["diabetes_pre_gestacional"] = d

    row: dict[str, float | int | str] = {}
    for col in FULL_FEATURE_COLS:
        default = DEFAULTS[col]
        raw = p.get(col, default)
        if raw is None or raw == "":
            raw = default
        if col in CATEGORICAL_COLS:
            s = str(raw).strip() or str(default)
            if col == "etnia" and s.lower() in {"indígena", "indigena"}:
                s = "Indigena"
            row[col] = s
        elif col in ("idade_materna", "imc_pre_gestacional", "idade_gestacional"):
            row[col] = _to_float(raw, float(default))  # type: ignore[arg-type]
        elif col in ("paridade", "abortamentos_previos"):
            row[col] = int(_to_float(raw, float(default)))  # type: ignore[arg-type]
        else:
            row[col] = _to_bin(raw)
    return row


def to_slim_features(full_row: dict) -> dict[str, float | int]:
    """Compatível com listagem enxuta / PythonPredictPayloadMapper."""
    return {
        "idade": float(full_row["idade_materna"]),
        "diabetes": max(
            int(full_row["diabetes_pre_gestacional"]),
            int(full_row["diabetes_gestacional"]),
        ),
        "hipertensao": int(full_row["hipertensao_cronica"]),
        "imc": float(full_row["imc_pre_gestacional"]),
        "idade_gestacional": float(full_row["idade_gestacional"]),
        "tabagismo": int(full_row["tabagismo"]),
        "paridade": int(full_row["paridade"]),
    }
