#!/usr/bin/env python3
"""Adiciona coluna id_gestante fictícia ao dataset.csv."""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
BACKUP_PATH = PROJECT_ROOT / "dataset.csv"


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "id_gestante" in df.columns:
        return df
    df = df.copy()
    df.insert(0, "id_gestante", [f"GEST-{i:05d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    source = DATASET_PATH if DATASET_PATH.exists() else BACKUP_PATH
    df = pd.read_csv(source)
    df = add_ids(df)

    if "id_gestante" in pd.read_csv(source, nrows=0).columns:
        print(f"id_gestante já existe em {source}")
        return

    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8", newline="") as tmp:
            tmp_path = Path(tmp.name)
        df.to_csv(tmp_path, index=False)
        shutil.move(str(tmp_path), DATASET_PATH)
        print(f"Atualizado: {DATASET_PATH} ({len(df)} registros)")
    except PermissionError:
        df.to_csv(BACKUP_PATH, index=False)
        print("Sem permissão para gravar em data/dataset.csv.")
        print(f"Arquivo gerado em: {BACKUP_PATH}")
        print("Feche o CSV no editor e execute novamente, ou copie manualmente para data/.")
        sys.exit(1)

    print(df[["id_gestante", "idade_materna", "etnia"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
