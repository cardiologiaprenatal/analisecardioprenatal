"""Normalização de dados clínicos de entrada."""

from __future__ import annotations

from typing import Any

from api.services import FEATURE_COLS


def normalizar_dados_clinicos(data: dict[str, Any]) -> dict[str, Any]:
    """Garante todas as features do modelo com defaults seguros."""
    normalizado: dict[str, Any] = {}
    for col in FEATURE_COLS:
        val = data.get(col)
        if col == "corionicidade":
            normalizado[col] = "NA" if val is None or str(val).strip() in ("", "nan") else str(val)
        elif col == "etnia":
            normalizado[col] = "Branca" if val is None or str(val).strip() == "" else str(val)
        elif col in ("idade_materna", "imc_pre_gestacional", "idade_gestacional"):
            normalizado[col] = float(val) if val is not None else 0.0
        else:
            normalizado[col] = int(float(val)) if val is not None else 0
    return normalizado
