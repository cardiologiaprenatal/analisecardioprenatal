"""Rotas FastAPI — fonte canônica ML CardioPrenatal (porta 8010)."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.graficos import (
    listar_graficos_descritivos,
    obter_analise_descritiva,
    obter_grafico_descritivo,
)
from app.features import FULL_FEATURE_COLS
from app.model_loader import load_model
from app.schemas import (
    AnalisarCompletoRequest,
    BatchPredictResponse,
    GestanteDetailResponse,
    GestanteIdsResponse,
    GestanteListResponse,
    GestanteSyncResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelosComparacaoResponse,
    PredictRequest,
    UnifiedInferenceResponse,
)
from app.schemas_interpretacao import InterpretacaoRequest, InterpretacaoResponse
from app.security import ApiKeyMiddleware
from app.services.analysis_compat import analisar_completo
from app.services.gestante_repository import GestanteRepository
from app.services.gestante_service import GestanteService
from app.services.interpretacao_service import interpretar_gestante as interpretar_gestante_svc
from app.services.model_metrics import DEFAULT_EXPERIMENT, ModelMetricsService
from app.services.predictor import PredictorService

settings = get_settings()


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    model = load_model(settings.model_path)
    predictor = PredictorService(model=model)

    repository = GestanteRepository()
    repository.load(settings.dataset_path)
    if not repository.loaded:
        logger.error(
            "Dataset ausente em %s — /gestantes e /predict/{id} retornarão 503.",
            settings.dataset_path,
        )

    metrics_service = ModelMetricsService()
    metrics_service.load(settings.metrics_path)
    if not metrics_service.loaded:
        logger.warning(
            "Métricas ausentes em %s — /modelos retornará disponivel=false.",
            settings.metrics_path,
        )

    app.state.predictor = predictor
    app.state.gestante_service = GestanteService(
        repository=repository,
        predictor=predictor,
    )
    app.state.model_metrics = metrics_service
    logger.info(
        "startup | app=%s v%s | model_loaded=%s | framework=%s | gestantes=%s | metricas_rows=%s | api_key_enabled=%s",
        settings.app_name,
        settings.app_version,
        model.loaded,
        model.framework,
        repository.total,
        len(metrics_service._rows),
        bool((settings.api_key or "").strip()),
    )
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Fonte canônica de Machine Learning do CardioPrenatal/GestRisk. "
        "Inferência + dataset CSV + métricas/gráficos. "
        "**Não** acessa nem grava no MySQL do Laravel. Porta **8010**."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Auth depois do CORS (Starlette: último add = primeiro a executar no request)
app.add_middleware(ApiKeyMiddleware)


def _predictor(request: Request) -> PredictorService:
    return request.app.state.predictor


def _gestantes(request: Request) -> GestanteService:
    return request.app.state.gestante_service


def _model_metrics(request: Request) -> ModelMetricsService:
    return request.app.state.model_metrics


@app.get("/", tags=["ops"])
def root() -> dict[str, Any]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "docs": "/docs",
        "porta": 8010,
        "endpoints": {
            "health": "GET /health",
            "metrics": "GET /metrics",
            "model_info": "GET /model/info",
            "modelos": "GET /modelos",
            "modelos_metricas": "GET /modelos/metricas",
            "gestantes": "GET /gestantes",
            "gestantes_sync": "GET /gestantes/sync",
            "gestantes_ids": "GET /gestantes/ids",
            "gestante_por_id": "GET /gestantes/{id_gestante}",
            "predict": "POST /predict",
            "predict_por_id": "POST /predict/{id_gestante}",
            "predict_all": "POST /predict/all",
            "analisar_completo": "POST /analisar/completo",
            "interpretar_gestante": "POST /interpretar/gestante",
            "graficos": "GET /graficos",
            "grafico_png": "GET /graficos/{slug}",
            "analise_descritiva": "GET /analise/descritiva",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    predictor = _predictor(request)
    repo = _gestantes(request).repository
    metrics = _model_metrics(request)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        model_loaded=predictor.model.loaded,
        dataset_loaded=repo.loaded,
        gestantes=repo.total,
        metrics_loaded=metrics.loaded,
        framework=predictor.model.framework,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics(request: Request) -> MetricsResponse:
    predictor = _predictor(request)
    m = predictor.metrics
    repo = _gestantes(request).repository
    return MetricsResponse(
        model_loaded=predictor.model.loaded,
        total_predictions=m.total_predictions,
        avg_execution_time_ms=m.avg_execution_time_ms,
        last_prediction_at=m.last_prediction_at,
        extra={
            "framework": predictor.model.framework,
            "features": predictor.model.features,
            "app_version": settings.app_version,
            "gestantes_carregadas": repo.total,
            "dataset_loaded": repo.loaded,
            "metrics_loaded": _model_metrics(request).loaded,
        },
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    response_model_by_alias=True,
    tags=["model"],
)
def model_info(request: Request) -> ModelInfoResponse:
    predictor = _predictor(request)
    model = predictor.model
    return ModelInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        loaded=model.loaded,
        model_path=model.path,
        features=model.features or list(FULL_FEATURE_COLS),
        framework=model.framework,
    )


@app.get("/modelos", response_model=ModelosComparacaoResponse, tags=["model"])
def listar_modelos(
    request: Request,
    experimento: str = Query(DEFAULT_EXPERIMENT),
) -> ModelosComparacaoResponse:
    data = _model_metrics(request).comparacao(experimento, apenas_melhor_por_modelo=True)
    return ModelosComparacaoResponse(**data)


@app.get("/modelos/metricas", response_model=ModelosComparacaoResponse, tags=["model"])
def metricas_modelos_detalhadas(
    request: Request,
    experimento: str = Query(DEFAULT_EXPERIMENT),
    todas_combinacoes: bool = Query(False),
) -> ModelosComparacaoResponse:
    data = _model_metrics(request).comparacao(
        experimento,
        apenas_melhor_por_modelo=not todas_combinacoes,
    )
    return ModelosComparacaoResponse(**data)


# --- Gestantes: rotas estáticas ANTES de /{id} ---

@app.get("/gestantes/sync", response_model=GestanteSyncResponse, tags=["gestantes"])
def sincronizar_gestantes(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(500, ge=1, le=1000),
) -> GestanteSyncResponse:
    """Sync paginado (1-based) — PythonApiService::syncGestantesPage."""
    return _gestantes(request).sincronizar(page=page, per_page=per_page)


@app.get("/gestantes/ids", response_model=GestanteIdsResponse, tags=["gestantes"])
def listar_ids_gestantes(request: Request) -> GestanteIdsResponse:
    return _gestantes(request).listar_ids()


@app.get("/gestantes", response_model=GestanteListResponse, tags=["gestantes"])
def listar_gestantes(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> GestanteListResponse:
    """Lista paginada do CSV — sem ML."""
    return _gestantes(request).listar(offset=offset, limit=limit)


@app.get(
    "/gestantes/{id_gestante}",
    response_model=GestanteDetailResponse,
    tags=["gestantes"],
)
def obter_gestante(id_gestante: str, request: Request) -> GestanteDetailResponse:
    """Dados de uma gestante — sem inferência."""
    return _gestantes(request).obter_por_id(id_gestante)


# --- Inferência: /predict/all ANTES de /predict/{id} ---

@app.post(
    "/predict",
    response_model=UnifiedInferenceResponse,
    response_model_by_alias=True,
    tags=["inference"],
)
def predict_json(body: PredictRequest, request: Request) -> UnifiedInferenceResponse:
    """Inferência a partir de JSON clínico (PythonPredictPayloadMapper / legado)."""
    predictor = _predictor(request)
    payload: dict[str, Any] = body.model_dump(exclude_none=False)
    if body.model_extra:
        payload.update(body.model_extra)
    try:
        return predictor.predict(payload, id_gestante=body.id_gestante)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction_error: %s", exc)
        raise HTTPException(status_code=500, detail="Falha ao executar predição.") from exc


@app.post(
    "/predict/all",
    response_model=BatchPredictResponse,
    response_model_by_alias=True,
    tags=["inference"],
)
def predict_all(request: Request) -> BatchPredictResponse:
    """
    Inferência de **todas** as gestantes, uma a uma.
    Custo alto — timeout Laravel recomendado ≥ 180s.
    """
    return _gestantes(request).prever_todas()


@app.post(
    "/predict/{id_gestante}",
    response_model=UnifiedInferenceResponse,
    response_model_by_alias=True,
    tags=["inference"],
)
def predict_por_id(id_gestante: str, request: Request) -> UnifiedInferenceResponse:
    """Inferência apenas da gestante informada (dataset CSV)."""
    return _gestantes(request).prever_por_id(id_gestante)


@app.post("/analisar/completo", tags=["inference"])
def analisar_completo_endpoint(
    body: AnalisarCompletoRequest,
    request: Request,
    persistir: bool = Query(
        False,
        description="Ignorado: esta API não persiste no MySQL. Sempre false.",
    ),
    top_n_shap: int = Query(36, ge=1, le=36),
) -> dict[str, Any]:
    """
    Análise clínica rica (CcfApiClient). Sem escrita em banco.
    `persistir` default **false**; true é ignorado com log.
    """
    predictor = _predictor(request)
    payload = body.model_dump(exclude_none=False)
    if body.model_extra:
        payload.update(body.model_extra)
    try:
        return analisar_completo(
            predictor,
            payload,
            persistir=persistir,
            top_n_shap=top_n_shap,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("analisar_completo_error: %s", exc)
        raise HTTPException(status_code=500, detail="Falha em /analisar/completo.") from exc


@app.post(
    "/interpretar/gestante",
    response_model=InterpretacaoResponse,
    tags=["interpretacao"],
)
def interpretar_gestante(body: InterpretacaoRequest) -> InterpretacaoResponse:
    """
    Interpretação clínica individualizada (Fase 1 = Gemini + fallback mock).

    Recebe PatientContext do Laravel. **Não** recalcula CatBoost/SHAP.
    """
    pred = body.predicao_catboost or {}
    if pred.get("probabilidade_risco") is None and pred.get("classificacao_risco") is None:
        raise HTTPException(
            status_code=422,
            detail="predicao_catboost obrigatória (verdade do CatBoost).",
        )
    try:
        return interpretar_gestante_svc(body)
    except Exception as exc:  # noqa: BLE001
        logger.exception("interpretar_gestante_error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Falha na interpretação: {exc}") from exc


@app.get("/graficos", tags=["analise"])
def listar_graficos() -> dict[str, Any]:
    return listar_graficos_descritivos()


@app.get("/graficos/{slug}", tags=["analise"])
def obter_grafico(slug: str):
    return obter_grafico_descritivo(slug)


@app.get("/analise/descritiva", tags=["analise"])
def analise_descritiva() -> dict[str, Any]:
    return obter_analise_descritiva()
