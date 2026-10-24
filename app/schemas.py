"""Contratos Pydantic v2 — API canônica CardioPrenatal-ML (Laravel)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLabel = Literal["Baixo", "Moderado", "Alto"]
RiskLabelLower = Literal["baixo", "moderado", "alto"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str | None = None
    version: str | None = None
    model_loaded: bool = False
    dataset_loaded: bool = False
    gestantes: int = 0
    metrics_loaded: bool = False
    framework: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    loaded: bool
    model_path: str = Field(serialization_alias="path")
    features: list[str]
    framework: str | None = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class ShapItem(BaseModel):
    """Explicação por feature — aliases estáveis para Laravel."""

    feature: str
    impact: float
    nome: str | None = None
    impacto_absoluto: float | None = None

    @model_validator(mode="after")
    def _aliases(self) -> ShapItem:
        if self.nome is None:
            self.nome = self.feature
        if self.impacto_absoluto is None:
            self.impacto_absoluto = abs(float(self.impact))
        return self


class PredictRequest(BaseModel):
    """Payload clínico enxuto (PythonPredictPayloadMapper) + extras."""

    model_config = ConfigDict(extra="allow")

    id_gestante: str | None = None
    idade: float | None = Field(default=None, description="Idade materna (anos)")
    diabetes: int | None = Field(default=None, ge=0, le=1)
    hipertensao: int | None = Field(default=None, ge=0, le=1)
    imc: float | None = Field(default=None, gt=0)
    idade_gestacional: float | None = None
    tabagismo: int | None = Field(default=None, ge=0, le=1)
    paridade: int | None = Field(default=None, ge=0)
    # aliases CSV / CcfPayloadMapper
    idade_materna: float | None = None
    imc_pre_gestacional: float | None = None
    diabetes_gestacional: int | None = Field(default=None, ge=0, le=1)
    diabetes_pre_gestacional: int | None = Field(default=None, ge=0, le=1)
    hipertensao_cronica: int | None = Field(default=None, ge=0, le=1)


class UnifiedInferenceResponse(BaseModel):
    """
    Contrato canônico de inferência unitária.
    Laravel: PythonApiService::normalizeInferenceResult / assertUnitaryInference.
    """

    id_gestante: str | None = None
    probabilidade_risco: float = Field(ge=0, le=1)
    classificacao_risco: RiskLabel
    score: float
    class_: int = Field(alias="class", ge=0, le=1)
    shap: list[ShapItem]
    execution_time_ms: float
    # aliases legado
    risk: RiskLabel | None = None
    probability: float | None = Field(default=None, ge=0, le=1)

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    @model_validator(mode="after")
    def _fill_aliases(self) -> UnifiedInferenceResponse:
        if self.risk is None:
            self.risk = self.classificacao_risco
        if self.probability is None:
            self.probability = self.probabilidade_risco
        return self


# aliases de nome para código legado interno
PredictResponse = UnifiedInferenceResponse
GestanteInferenceResult = UnifiedInferenceResponse


class MetricsResponse(BaseModel):
    model_loaded: bool
    total_predictions: int
    avg_execution_time_ms: float | None
    last_prediction_at: str | None
    extra: dict[str, Any] = Field(default_factory=dict)


class GestanteListItem(BaseModel):
    id_gestante: str
    idade_materna: float | None = None
    etnia: str | None = None
    imc_pre_gestacional: float | None = None
    paridade: float | None = None
    idade_gestacional: float | None = None
    diabetes: int = 0
    hipertensao: int = 0
    tabagismo: int = 0


class GestanteListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    dataset: str
    gestantes: list[GestanteListItem]


class GestanteSyncResponse(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int
    dataset: str
    gestantes: list[GestanteListItem]


class GestanteIdsResponse(BaseModel):
    total: int
    ids: list[str]


class MetricSplit(BaseModel):
    acuracia: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    sensibilidade: float | None = None
    especificidade: float | None = None
    precisao: float | None = None
    pr_auc: float | None = None


class ModeloMetricasItem(BaseModel):
    modelo: str
    nome_exibicao: str
    balanceamento: str | None = None
    treino_cv: MetricSplit
    teste: MetricSplit


class ModelosComparacaoResponse(BaseModel):
    disponivel: bool
    experimento: str
    fonte: str
    criterio_selecao: str | None = None
    total_modelos: int = 0
    destaques: dict[str, Any] = Field(default_factory=dict)
    modelos: list[ModeloMetricasItem] = Field(default_factory=list)
    legendas: dict[str, str] = Field(default_factory=dict)
    mensagem: str | None = None
    experimentos_disponiveis: list[str] | None = None


class GestanteDetailResponse(BaseModel):
    id_gestante: str
    dados_clinicos: dict[str, Any]
    features_modelo: dict[str, float | int | str]


class BatchPredictResponse(BaseModel):
    total: int
    processadas: int
    falhas: int
    tempo_total_ms: float
    resultados: list[UnifiedInferenceResponse]
    aviso: str | None = (
        "Processa uma gestante por vez. Timeout Laravel recomendado ≥ 180s."
    )

    model_config = ConfigDict(populate_by_name=True)


class AnalisarCompletoRequest(BaseModel):
    """Payload clínico rico (CcfPayloadMapper) — campos opcionais com defaults seguros."""

    model_config = ConfigDict(extra="allow")

    id_gestante: str | None = None
    idade_materna: float | None = Field(default=None, ge=14, le=48)
    idade: float | None = None
    etnia: str | None = "Branca"
    imc_pre_gestacional: float | None = Field(default=None, ge=16, le=45)
    imc: float | None = None
    paridade: int | None = Field(default=0, ge=0, le=6)
    abortamentos_previos: int | None = Field(default=0, ge=0, le=4)
    historia_natimorto: int = Field(0, ge=0, le=1)
    filho_anterior_cardiopatia: int = Field(0, ge=0, le=1)
    historia_familiar_cardiopatia: int = Field(0, ge=0, le=1)
    diabetes_pre_gestacional: int = Field(0, ge=0, le=1)
    diabetes_gestacional: int = Field(0, ge=0, le=1)
    diabetes: int | None = Field(default=None, ge=0, le=1)
    lupus: int = Field(0, ge=0, le=1)
    fenilcetonuria: int = Field(0, ge=0, le=1)
    doenca_tireoide: int = Field(0, ge=0, le=1)
    hipertensao_cronica: int = Field(0, ge=0, le=1)
    hipertensao: int | None = Field(default=None, ge=0, le=1)
    rubeola: int = Field(0, ge=0, le=1)
    citomegalovirus: int = Field(0, ge=0, le=1)
    toxoplasmose: int = Field(0, ge=0, le=1)
    sifilis: int = Field(0, ge=0, le=1)
    isotretinoina: int = Field(0, ge=0, le=1)
    acido_valproico: int = Field(0, ge=0, le=1)
    litio: int = Field(0, ge=0, le=1)
    alcool: int = Field(0, ge=0, le=1)
    tabagismo: int = Field(0, ge=0, le=1)
    drogas_ilicitas: int = Field(0, ge=0, le=1)
    solventes: int = Field(0, ge=0, le=1)
    pesticidas: int = Field(0, ge=0, le=1)
    idade_gestacional: float | None = Field(default=20.0, ge=11, le=28)
    gestacao_gemelar: int = Field(0, ge=0, le=1)
    corionicidade: str | None = "NA"
    translucencia_nucal_aumentada: int = Field(0, ge=0, le=1)
    alteracao_ducto_venoso: int = Field(0, ge=0, le=1)
    regurgitacao_tricuspide: int = Field(0, ge=0, le=1)
    malformacao_extracardiaca: int = Field(0, ge=0, le=1)
    restricao_crescimento_fetal: int = Field(0, ge=0, le=1)
    polidramnio: int = Field(0, ge=0, le=1)
    oligodramnio: int = Field(0, ge=0, le=1)
