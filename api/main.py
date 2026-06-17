"""API GestRisk CCF v2.0 — fonte única de inteligência clínica."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.analysis_service import API_VERSION, analysis_service
from api.database import init_db
from api.graficos import listar_graficos_descritivos, obter_grafico_descritivo
from api.history_service import listar_historico_gestante, obter_analise_por_id
from api.schemas import (
    AnaliseCompletaOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    HistoricoDetalheOutput,
    HistoricoGestanteOutput,
    ModelInfo,
    PredictionInput,
    PredictionOutput,
    PriorizacaoEcoOutput,
)
from api.services import DATASET_PATH, FEATURE_COLS, MODEL_PATH, model_service

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173",
).split(",")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    model_service.load_or_train()
    analysis_service.construir_catalogo_laravel()
    yield


app = FastAPI(
    title="GestRisk CCF API",
    description=(
        "Fonte única de inteligência clínica para predição de Cardiopatia Congênita Fetal. "
        "Toda análise, priorização, SHAP e histórico são responsabilidade desta API."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Sistema"])
def root():
    return {
        "nome": "GestRisk CCF API",
        "versao": API_VERSION,
        "papel": "fonte_unica_inteligencia_clinica",
        "docs": "/docs",
        "endpoints_principais": {
            "analisar_completo": "POST /analisar/completo",
            "gestantes": "GET /gestantes",
            "gestantes_sync": "GET /gestantes/sync?page=1&per_page=500",
            "gestantes_todas": "GET /gestantes/todas",
            "gestantes_export": "GET /gestantes/export",
            "gestantes_ids": "GET /gestantes/ids",
            "gestante_por_id": "GET /gestantes/{id_gestante}",
            "priorizacao_eco": "GET /ecocardiograma/priorizacao",
            "historico_gestante": "GET /historico/{id_gestante}",
            "historico_analise": "GET /historico/analise/{analise_id}",
            "graficos": "GET /graficos",
            "grafico_png": "GET /graficos/{slug}",
        },
        "endpoints_legado": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "health": "GET /health",
            "model_info": "GET /model/info",
            "features": "GET /features",
            "metrics": "GET /metrics",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    return HealthResponse(status="ok", modelo_carregado=model_service.is_loaded, versao=API_VERSION)


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    return ModelInfo(
        modelo=model_service.meta.get("modelo", "CatBoost"),
        experimento=model_service.meta.get("experimento", "experimento_3_completo"),
        balanceamento=model_service.meta.get("balanceamento", "class_weight"),
        features=FEATURE_COLS,
        model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
        carregado=model_service.is_loaded,
        dataset_path=str(DATASET_PATH) if DATASET_PATH.exists() else None,
        versao_api=API_VERSION,
    )


@app.get("/features", tags=["Modelo"])
def features():
    return {
        "experimento": "experimento_3_completo",
        "total_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "categorias_intercorrencia": [
            "maternal", "doencas", "infeccoes", "medicacoes",
            "habitos", "exposicao", "gestacionais", "marcadores_fetais",
        ],
    }


@app.get("/metrics", tags=["Modelo"])
def metrics():
    summary = model_service.get_summary_metrics()
    if not summary:
        return {"mensagem": "Execute `py main.py` para gerar métricas do pipeline."}
    return summary.get("best_model", summary)


# ── Endpoint principal ──────────────────────────────────────────────────────

@app.post("/analisar/completo", response_model=AnaliseCompletaOutput, tags=["Análise Clínica"])
def analisar_completo(
    payload: PredictionInput,
    top_n_shap: int = Query(10, ge=1, le=36),
    persistir: bool = Query(True, description="Salvar no histórico auditável"),
):
    """
    **Endpoint principal** — predição, intercorrências, SHAP, priorização,
  recomendações, qualidade de dados e metadados em uma única resposta.
    """
    try:
        data = payload.model_dump(exclude={"id_gestante"})
        return analysis_service.analisar_completo(
            data,
            id_gestante=payload.id_gestante,
            top_n_shap=top_n_shap,
            persistir=persistir,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Gestantes do CSV (Laravel sincroniza daqui) ───────────────────────────

@app.get("/gestantes/sync", tags=["Gestantes"])
def sincronizar_gestantes(
    page: int = Query(1, ge=1, description="Página 1-based para o Laravel"),
    per_page: int = Query(500, ge=1, le=1000),
):
    """
    Sincronização paginada — 10 páginas de 500 = 5000 gestantes.
    Com per_page=500 (padrão), entrega arquivo JSON pré-gerado (mais estável).
    """
    if per_page == 500:
        path = analysis_service.caminho_sync_pagina(page=page, per_page=per_page)
        if path:
            return FileResponse(
                path,
                media_type="application/json",
                filename=f"gestantes_page_{page:02d}.json",
            )
    return analysis_service.sincronizar_gestantes_laravel(page=page, per_page=per_page)


@app.get("/gestantes/ids", tags=["Gestantes"])
def listar_ids_gestantes():
    """Lista todos os id_gestante do CSV (verificação pós-sync no Laravel)."""
    return analysis_service.listar_ids_gestantes()


@app.get("/gestantes/export", tags=["Gestantes"])
def exportar_gestantes():
    """
    Download do JSON com TODAS as gestantes (~14 MB).
    Arquivo gerado no startup da API — mais estável que serializar em memória.
    """
    path = analysis_service.caminho_export_laravel()
    return FileResponse(
        path,
        media_type="application/json",
        filename="gestantes_laravel.json",
    )


@app.get("/gestantes/todas", tags=["Gestantes"])
def listar_todas_gestantes():
    """
    Retorna TODAS as gestantes do CSV para sincronização com Laravel.
    Entrega o arquivo JSON pré-gerado (5000 gestantes com análise completa).
    """
    path = analysis_service.caminho_export_laravel()
    return FileResponse(
        path,
        media_type="application/json",
        filename="gestantes_laravel.json",
    )


@app.get("/gestantes", tags=["Gestantes"])
def listar_gestantes(
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    apenas_com_intercorrencia: bool = Query(False),
):
    """
    Lista gestantes do CSV com análise clínica completa (sem SHAP).
    Use paginação: offset/limit. Total = 5000 gestantes.
    """
    return analysis_service.listar_gestantes_laravel(
        offset=offset,
        limit=limit,
        apenas_com_intercorrencia=apenas_com_intercorrencia,
    )


@app.get("/gestantes/{id_gestante}", tags=["Gestantes"])
def obter_gestante(id_gestante: str, incluir_shap: bool = Query(False)):
    """Análise de uma gestante do CSV. Com incluir_shap=true adiciona explicação SHAP."""
    gestante = analysis_service.obter_gestante_laravel(id_gestante)
    if not gestante:
        raise HTTPException(status_code=404, detail=f"Gestante {id_gestante} não encontrada no dataset.")

    if incluir_shap:
        shap = model_service.explicar_shap(
            gestante["dados_clinicos"],
            id_gestante=id_gestante,
            top_n=10,
        )
        return {**gestante, "shap": shap}

    return gestante


# ── Priorização (Laravel não recalcula) ───────────────────────────────────

@app.get("/ecocardiograma/priorizacao", response_model=PriorizacaoEcoOutput, tags=["Ecocardiograma"])
def priorizacao_ecocardiograma(
    limite: int = Query(100, ge=1, le=5000),
    apenas_recomendadas: bool = Query(True),
):
    return analysis_service.construir_fila_ecocardiograma(
        limite=limite,
        apenas_recomendadas=apenas_recomendadas,
    )


# ── Histórico auditável ───────────────────────────────────────────────────

@app.get("/historico/{id_gestante}", response_model=HistoricoGestanteOutput, tags=["Histórico"])
def historico_gestante(
    id_gestante: str,
    limite: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Lista análises anteriores — nunca sobrescritas."""
    return listar_historico_gestante(id_gestante, limite=limite, offset=offset)


@app.get("/historico/analise/{analise_id}", response_model=HistoricoDetalheOutput, tags=["Histórico"])
def historico_analise(analise_id: str):
    resultado = obter_analise_por_id(analise_id)
    if not resultado:
        raise HTTPException(status_code=404, detail=f"Análise {analise_id} não encontrada.")
    return resultado


# ── Gráficos descritivos (PNG) ─────────────────────────────────────────────

@app.get("/graficos", tags=["Gráficos"])
def listar_graficos():
    """Lista gráficos estáticos da análise descritiva (PNG em analise_descritiva/figuras)."""
    return listar_graficos_descritivos()


@app.get("/graficos/{slug}", tags=["Gráficos"])
def grafico_descritivo(slug: str):
    """Retorna PNG do gráfico descritivo (gerado por gerar_analise_descritiva.py)."""
    return obter_grafico_descritivo(slug)


# ── Endpoints legado (compatibilidade) ────────────────────────────────────

@app.post("/predict", response_model=PredictionOutput, tags=["Legado"])
def predict(payload: PredictionInput):
    try:
        return model_service.predict_one(payload.model_dump(exclude={"id_gestante"}))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Legado"])
def predict_batch(payload: BatchPredictionInput):
    try:
        resultados = [
            PredictionOutput(**model_service.predict_one(p.model_dump(exclude={"id_gestante"})))
            for p in payload.pacientes
        ]
        return BatchPredictionOutput(resultados=resultados, total=len(resultados))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
