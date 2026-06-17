"""Schemas Pydantic v2.0 — GestRisk CCF API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Etnia = Literal["Branca", "Parda", "Preta", "Amarela", "Indigena"]
Corionicidade = Literal["Monocoriônica", "Dicoriônica", "NA"]
PrioridadeEco = Literal["urgente", "alta", "moderada", "baixa"]
Risco = Literal["baixo", "moderado", "alto"]
Impacto = Literal["alto", "medio", "baixo"]
DirecaoShap = Literal["aumenta_risco", "diminui_risco"]


class PredictionInput(BaseModel):
    id_gestante: str | None = Field(None, description="Identificador da gestante")
    idade_materna: float = Field(..., ge=14, le=48)
    etnia: Etnia
    imc_pre_gestacional: float = Field(..., ge=16, le=45)
    paridade: int = Field(..., ge=0, le=6)
    abortamentos_previos: int = Field(..., ge=0, le=4)
    historia_natimorto: int = Field(0, ge=0, le=1)
    filho_anterior_cardiopatia: int = Field(0, ge=0, le=1)
    historia_familiar_cardiopatia: int = Field(0, ge=0, le=1)
    diabetes_pre_gestacional: int = Field(0, ge=0, le=1)
    diabetes_gestacional: int = Field(0, ge=0, le=1)
    lupus: int = Field(0, ge=0, le=1)
    fenilcetonuria: int = Field(0, ge=0, le=1)
    doenca_tireoide: int = Field(0, ge=0, le=1)
    hipertensao_cronica: int = Field(0, ge=0, le=1)
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
    idade_gestacional: float = Field(..., ge=11, le=28)
    gestacao_gemelar: int = Field(0, ge=0, le=1)
    corionicidade: Corionicidade = "NA"
    translucencia_nucal_aumentada: int = Field(0, ge=0, le=1)
    alteracao_ducto_venoso: int = Field(0, ge=0, le=1)
    regurgitacao_tricuspide: int = Field(0, ge=0, le=1)
    malformacao_extracardiaca: int = Field(0, ge=0, le=1)
    restricao_crescimento_fetal: int = Field(0, ge=0, le=1)
    polidramnio: int = Field(0, ge=0, le=1)
    oligodramnio: int = Field(0, ge=0, le=1)


class PredicaoOutput(BaseModel):
    probabilidade_ccf: float
    classificacao_risco: Risco
    cardiopatia_congenita_prevista: int
    modelo: str
    experimento: str


class ResumoClinico(BaseModel):
    tecnico: str
    simplificado: str


class AnaliseOutput(BaseModel):
    probabilidade_ccf: float
    classificacao_risco: Risco
    score_prioridade: float
    prioridade_ecocardiograma: PrioridadeEco
    posicao_fila: int | None = None
    recomenda_ecocardiograma: bool
    resumo_clinico: ResumoClinico
    resumo_intercorrencias: dict[str, Any]
    fatores_priorizacao: dict[str, Any] | None = None


class IntercorrenciaOutput(BaseModel):
    categoria: str
    categoria_label: str
    campo: str
    nome: str
    valor: Any
    alteracao: str
    impacto: Impacto
    peso_risco: float = 0
    relevancia_modelo: float = 0


class ShapContribuicaoOutput(BaseModel):
    campo: str
    nome: str
    shap_value: float
    impacto_absoluto: float
    percentual_contribuicao: float
    direcao: DirecaoShap
    valor_paciente: Any = None


class ShapOutput(BaseModel):
    metodo: str
    classe_alvo: str
    valor_base: float
    soma_shap: float
    resumo: str
    probabilidade_ccf: float | None = None
    top_contribuicoes: list[ShapContribuicaoOutput]
    importancia_global: list[dict[str, Any]] = []


class RecomendacoesOutput(BaseModel):
    ecocardiograma_fetal: bool
    prioridade: PrioridadeEco
    prazo_sugerido: str | None = None
    acoes: list[str]


class QualidadeDadosOutput(BaseModel):
    valido: bool
    score_qualidade: int
    total_alertas: int
    alertas: list[dict[str, str]]


class MetadadosOutput(BaseModel):
    analise_id: str
    versao_api: str
    versao_modelo: str
    data_analise: str
    tempo_processamento_ms: float


class AnaliseCompletaOutput(BaseModel):
    """Resposta principal — fonte única de inteligência clínica."""

    id_gestante: str | None
    predicao: PredicaoOutput
    analise: AnaliseOutput
    intercorrencias: list[IntercorrenciaOutput]
    shap: ShapOutput
    recomendacoes: RecomendacoesOutput
    qualidade_dados: QualidadeDadosOutput
    metadados: MetadadosOutput


class PriorizacaoEcoItem(BaseModel):
    posicao_fila: int
    id_gestante: str
    score_prioridade: float
    prioridade_ecocardiograma: PrioridadeEco
    probabilidade_ccf: float
    classificacao_risco: Risco
    idade_gestacional: float | None = None
    total_intercorrencias: int
    marcadores_fetais_criticos: int
    intercorrencias_principais: list[str]
    resumo_simplificado: str


class PriorizacaoEcoOutput(BaseModel):
    total_na_fila: int
    limite_retornado: int
    gerado_em: str
    priorizacao: list[PriorizacaoEcoItem]


class HistoricoItem(BaseModel):
    analise_id: str
    data_analise: str
    versao_api: str
    versao_modelo: str
    classificacao_risco: str | None = None
    probabilidade_ccf: float | None = None
    prioridade_ecocardiograma: str | None = None
    score_prioridade: float | None = None


class HistoricoGestanteOutput(BaseModel):
    id_gestante: str
    total: int
    offset: int
    limite: int
    historico: list[HistoricoItem]


class HistoricoDetalheOutput(BaseModel):
    analise_id: str
    id_gestante: str
    data_analise: str
    versao_api: str
    versao_modelo: str
    resultado: dict[str, Any]
    shap: dict[str, Any]


class BatchPredictionInput(BaseModel):
    pacientes: list[PredictionInput] = Field(..., min_length=1)


class PredictionOutput(BaseModel):
    cardiopatia_congenita: int
    probabilidade: float
    risco: Risco
    modelo: str
    experimento: str


class BatchPredictionOutput(BaseModel):
    resultados: list[PredictionOutput]
    total: int


class ModelInfo(BaseModel):
    modelo: str
    experimento: str
    balanceamento: str
    features: list[str]
    model_path: str | None
    carregado: bool
    dataset_path: str | None = None
    versao_api: str = "2.0.0"


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    versao: str = "2.0.0"
