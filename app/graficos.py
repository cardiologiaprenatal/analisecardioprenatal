"""Catálogo e entrega das análises estáticas (PNG + metadados do CSV)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "analise_descritiva" / "figuras"
META_JSON = PROJECT_ROOT / "analise_descritiva" / "analise_descritiva.json"
RESUMO_MD = PROJECT_ROOT / "analise_descritiva" / "resumo_analise_descritiva.md"

# slug -> arquivo canônico gerado por gerar_analise_descritiva.py
SLUG_TO_FILE: dict[str, str] = {
    "prevalencia": "01_prevalencia_ccf.png",
    "distribuicao": "02_distribuicao_idade_imc.png",
    "fatores_risco": "03_ccf_por_fatores_risco.png",
    "intercorrencias_cat": "04_intercorrencias_por_categoria.png",
    "top_intercorrencias": "05_top_intercorrencias.png",
    "risco": "06_distribuicao_risco.png",
    "probabilidade": "07_distribuicao_probabilidade.png",
    "prioridade_eco": "08_prioridade_ecocardiograma.png",
    "importancia": "09_importancia_features.png",
    "experimentos": "10_comparativo_experimentos.png",
    "etnia": "11_prevalencia_por_etnia.png",
}

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


def _load_meta() -> dict[str, Any]:
    if not META_JSON.is_file():
        return {}
    try:
        data = json.loads(META_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("falha ao ler analise_descritiva.json: %s", exc)
        return {}


def _resolver_caminho_arquivo(slug: str) -> Path | None:
    """Resolve PNG local; ignora caminhos absolutos de outras máquinas no JSON."""
    canonical = SLUG_TO_FILE.get(slug)
    if canonical:
        path = FIGURES_DIR / canonical
        if path.is_file():
            return path

    meta = _load_meta()
    graficos = meta.get("graficos") or {}
    if slug in graficos:
        candidate = Path(str(graficos[slug]))
        if candidate.is_file():
            return candidate
        # se o meta aponta para outro host, tenta só o nome do arquivo
        local = FIGURES_DIR / candidate.name
        if local.is_file():
            return local

    if FIGURES_DIR.is_dir():
        matches = sorted(FIGURES_DIR.glob(f"*{slug}*.png"))
        if matches:
            return matches[0]

    return None


def listar_graficos_descritivos() -> dict[str, Any]:
    catalogo: list[dict[str, Any]] = []
    slugs = list(dict.fromkeys([*SLUG_TO_FILE.keys(), *GRAFICOS_TITULOS.keys()]))

    meta = _load_meta()
    for slug_meta in (meta.get("graficos") or {}):
        if slug_meta not in slugs:
            slugs.append(slug_meta)

    for slug in slugs:
        path = _resolver_caminho_arquivo(slug)
        catalogo.append({
            "slug": slug,
            "titulo": GRAFICOS_TITULOS.get(slug, slug.replace("_", " ").title()),
            "arquivo": path.name if path else None,
            "disponivel": path is not None,
            "url": f"/graficos/{slug}" if path else None,
        })

    return {
        "total": len(catalogo),
        "disponiveis": sum(1 for g in catalogo if g["disponivel"]),
        "graficos": catalogo,
        "diretorio": FIGURES_DIR.as_posix(),
    }


def obter_grafico_descritivo(slug: str) -> FileResponse:
    path = _resolver_caminho_arquivo(slug)
    if path is None or not path.is_file():
        if slug not in GRAFICOS_TITULOS and slug not in SLUG_TO_FILE:
            raise HTTPException(status_code=404, detail=f"Gráfico '{slug}' não reconhecido.")
        raise HTTPException(
            status_code=404,
            detail=(
                f"Arquivo PNG para '{slug}' não encontrado. "
                "Execute: python gerar_analise_descritiva.py"
            ),
        )
    return FileResponse(path, media_type="image/png", filename=path.name)


def obter_analise_descritiva() -> dict[str, Any]:
    """Metadados da análise estática do CSV + catálogo de gráficos."""
    meta = _load_meta()
    catalogo = listar_graficos_descritivos()

    resumo_texto: str | None = None
    if RESUMO_MD.is_file():
        try:
            resumo_texto = RESUMO_MD.read_text(encoding="utf-8")
        except OSError:
            resumo_texto = None

    return {
        "fonte": "analise_descritiva",
        "total_gestantes": meta.get("total_gestantes"),
        "prevalencia_ccf": meta.get("prevalencia_ccf"),
        "resumo_path": RESUMO_MD.as_posix() if RESUMO_MD.is_file() else None,
        "resumo_markdown": resumo_texto,
        "graficos": catalogo["graficos"],
        "total_graficos": catalogo["total"],
        "graficos_disponiveis": catalogo["disponiveis"],
    }
