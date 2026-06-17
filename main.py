#!/usr/bin/env python3
"""GestRisk CCF - Pipeline principal."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
