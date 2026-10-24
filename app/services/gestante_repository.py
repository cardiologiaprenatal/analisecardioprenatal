"""Leitura do CSV de gestantes — sem MySQL e sem persistência."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_CANDIDATES = (
    "dataset_com_id_gestante.csv",
    "dataset.csv",
    "data/dataset.csv",
)


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def map_row_to_model_features(row: dict[str, Any]) -> dict[str, float | int | str]:
    """Mapeia colunas do CSV para as 36 features do experimento completo."""
    from app.features import build_full_clinical_row

    return build_full_clinical_row(row)


def row_to_list_item(row: dict[str, Any]) -> dict[str, Any]:
    from app.features import to_slim_features

    features = map_row_to_model_features(row)
    slim = to_slim_features(features)
    return {
        "id_gestante": str(row.get("id_gestante") or ""),
        "idade_materna": _to_float(row.get("idade_materna")),
        "etnia": (str(row["etnia"]) if row.get("etnia") not in (None, "") else None),
        "imc_pre_gestacional": _to_float(row.get("imc_pre_gestacional")),
        "paridade": _to_float(row.get("paridade")),
        "idade_gestacional": _to_float(row.get("idade_gestacional")),
        "diabetes": int(slim["diabetes"]),
        "hipertensao": int(slim["hipertensao"]),
        "tabagismo": int(slim["tabagismo"]),
    }


def sanitize_clinical_row(row: dict[str, Any]) -> dict[str, Any]:
    """Converte valores do CSV para tipos JSON-friendly (só este registro)."""
    out: dict[str, Any] = {}
    for key, raw in row.items():
        if raw is None or raw == "":
            out[key] = None
            continue
        if key in {"id_gestante", "etnia", "corionicidade"}:
            out[key] = str(raw)
            continue
        num = _to_float(raw)
        out[key] = num if num is not None else str(raw)
    return out


class GestanteRepository:
    """
    Repositório em memória indexado por id_gestante.

    - listagem: fatia do CSV, sem ML
    - get_by_id: O(1) via índice — nunca varre nem prediz o cohort inteiro
    """

    def __init__(self) -> None:
        self.path: str = ""
        self._rows: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    @property
    def total(self) -> int:
        return len(self._rows)

    @property
    def loaded(self) -> bool:
        return bool(self._rows)

    def resolve_path(self, configured: str) -> Path:
        candidates = [Path(configured), *[Path(p) for p in DATASET_CANDIDATES]]
        for path in candidates:
            if path.is_file():
                return path
        return Path(configured)

    def load(self, configured_path: str) -> None:
        path = self.resolve_path(configured_path)
        self.path = path.as_posix()
        if not path.is_file():
            logger.warning("dataset_missing path=%s", self.path)
            self._rows = []
            self._index = {}
            return

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if rows and "id_gestante" not in rows[0]:
            for i, row in enumerate(rows, start=1):
                row["id_gestante"] = f"GEST-{i:05d}"

        index: dict[str, int] = {}
        for i, row in enumerate(rows):
            gid = str(row.get("id_gestante") or "").strip()
            if gid and gid not in index:
                index[gid] = i

        self._rows = rows
        self._index = index
        logger.info("dataset_loaded path=%s total=%s", self.path, len(rows))

    def list_rows(self, offset: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        offset = max(0, offset)
        limit = max(1, min(limit, 1000))
        return self._rows[offset : offset + limit]

    def iter_all_rows(self) -> list[dict[str, Any]]:
        """Retorna referência às linhas em memória (sem cópia profunda)."""
        return self._rows

    def get_by_id(self, id_gestante: str) -> dict[str, Any] | None:
        key = str(id_gestante).strip()
        idx = self._index.get(key)
        if idx is None:
            # tentativa case-insensitive pontual (não reconstrói catálogo)
            for candidate, pos in self._index.items():
                if candidate.lower() == key.lower():
                    return self._rows[pos]
            return None
        return self._rows[idx]
