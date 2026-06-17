#!/usr/bin/env python3
"""Treina e salva o melhor modelo para uso na API."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.services import model_service

if __name__ == "__main__":
    model_service.load_or_train()
    print(f"Modelo salvo e pronto em: {model_service.meta}")
