"""Catálogo e entrega de gráficos descritivos (PNG) da análise GestRisk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "analise_descritiva" / "figuras"
META_JSON = PROJECT_ROOT / "analise_descritiva" / "analise_descritiva.json"

GRAFICOS_TITULOS: dict[str, str] = {
    "prevalencia": "Prevalência de CCF",
    "distribuicao": "Idade, IMC e idade gestacional",
    "fatores_risco": "CCF por fatores de risco",
    "intercorrencias_cat": "Intercorrências por categoria",
    "top_intercorrencias": "Top intercorrências",
    "risco": "Distribuição de risco (cohort)",
    "probabilidade": "Distribuição de probabilidade",
    "prioridade_eco": "Priorização ecocardiograma",
    "etnia": "Prevalência por etnia",
    "importancia": "Importância das features",
    "experimentos": "Comparativo de experimentos",
}


def _resolver_caminho_arquivo(slug: str) -> Path | None:
    if META_JSON.is_file():
        try:
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            graficos = meta.get("graficos") or {}
            if slug in graficos:
                path = Path(str(graficos[slug]))
                if path.is_file():
                    return path
        except (json.JSONDecodeError, OSError):
            pass

    if not FIGURES_DIR.is_dir():
        return None

    padroes = sorted(FIGURES_DIR.glob(f"*{slug}*.png"))
    if padroes:
        return padroes[0]

    fallback = FIGURES_DIR / f"{slug}.png"
    return fallback if fallback.is_file() else None


def listar_graficos_descritivos() -> dict[str, Any]:
    catalogo: list[dict[str, Any]] = []

    slugs = list(GRAFICOS_TITULOS.keys())
    if META_JSON.is_file():
        try:
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            slugs_meta = list((meta.get("graficos") or {}).keys())
            slugs = list(dict.fromkeys([*slugs_meta, *slugs]))
        except (json.JSONDecodeError, OSError):
            pass

    for slug in slugs:
        path = _resolver_caminho_arquivo(slug)
        catalogo.append({
            "slug": slug,
            "titulo": GRAFICOS_TITULOS.get(slug, slug.replace("_", " ").title()),
            "arquivo": path.name if path else None,
            "disponivel": path is not None,
        })

    return {
        "total": len(catalogo),
        "disponiveis": sum(1 for g in catalogo if g["disponivel"]),
        "graficos": catalogo,
        "diretorio": str(FIGURES_DIR),
    }


def obter_grafico_descritivo(slug: str) -> FileResponse:
    if slug not in GRAFICOS_TITULOS and _resolver_caminho_arquivo(slug) is None:
        raise HTTPException(status_code=404, detail=f"Gráfico '{slug}' não reconhecido.")

    path = _resolver_caminho_arquivo(slug)
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Arquivo PNG para '{slug}' não encontrado. "
                "Execute: py gerar_analise_descritiva.py"
            ),
        )

    return FileResponse(path, media_type="image/png", filename=path.name)
