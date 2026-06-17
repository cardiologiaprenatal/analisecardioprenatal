"""Orquestrador central — fonte única de inteligência clínica GestRisk CCF."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "export"
EXPORT_JSON_PATH = EXPORT_DIR / "gestantes_laravel.json"

from api.clinical_data import normalizar_dados_clinicos
from api.clinical_summary import enriquecer_intercorrencias, gerar_recomendacoes, gerar_resumo_clinico
from api.data_quality import validar_qualidade_dados
from api.history_service import salvar_analise
from api.intercorrencias import analisar_intercorrencias, resumir_intercorrencias
from api.prioritization import (
    calcular_priorizacao_clinica,
    classificar_risco,
    recomendar_ecocardiograma,
)
from api.services import DATASET_PATH, FEATURE_COLS, model_service
from api.shap_explain import explain_prediction, get_global_shap_importance

API_VERSION = "2.0.0"


class AnalysisService:
    """Toda lógica clínica de análise reside aqui — Laravel apenas consome."""

    def __init__(self) -> None:
        self._fila_cache: list[dict[str, Any]] | None = None
        self._fila_posicoes: dict[str, int] = {}
        self._catalogo_laravel: list[dict[str, Any]] | None = None
        self._catalogo_por_id: dict[str, dict[str, Any]] = {}

    def construir_catalogo_laravel(self) -> None:
        """Pré-analisa TODAS as gestantes do CSV para consumo pelo Laravel."""
        if self._catalogo_laravel is not None:
            return

        if not model_service._analises_cache:
            model_service.load_or_train()

        catalogo: list[dict[str, Any]] = []
        for id_gestante, cache in model_service._analises_cache.items():
            data = cache["dados_clinicos"]
            proba = cache["analise"]["probabilidade_ccf"]
            pred = cache["predicao"]["cardiopatia_congenita"]
            item = self._montar_gestante_catalogo(data, id_gestante, proba, pred)
            catalogo.append(item)
            self._catalogo_por_id[id_gestante] = item

        catalogo.sort(key=lambda g: g["id_gestante"])
        self._catalogo_laravel = catalogo
        self._persistir_export_laravel()

    def _persistir_export_laravel(self) -> Path:
        """Grava JSON com todas as gestantes para download estável pelo Laravel."""
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        catalogo = self._catalogo_laravel or []
        payload = {
            "total": len(catalogo),
            "dataset": str(DATASET_PATH),
            "gerado_em": datetime.now(timezone.utc).isoformat(),
            "gestantes": catalogo,
        }
        EXPORT_JSON_PATH.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        per_page = 500
        total_pages = max(1, (len(catalogo) + per_page - 1) // per_page)
        for page in range(1, total_pages + 1):
            offset = (page - 1) * per_page
            pagina = catalogo[offset : offset + per_page]
            page_payload = {
                "total": len(catalogo),
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "dataset": str(DATASET_PATH),
                "gerado_em": payload["gerado_em"],
                "gestantes": pagina,
            }
            page_path = EXPORT_DIR / f"gestantes_page_{page:02d}.json"
            page_path.write_text(
                json.dumps(page_payload, ensure_ascii=False),
                encoding="utf-8",
            )

        return EXPORT_JSON_PATH

    def caminho_export_laravel(self) -> Path:
        self.construir_catalogo_laravel()
        if not EXPORT_JSON_PATH.exists():
            self._persistir_export_laravel()
        return EXPORT_JSON_PATH

    def caminho_sync_pagina(self, page: int, per_page: int = 500) -> Path | None:
        self.construir_catalogo_laravel()
        total = len(self._catalogo_laravel or [])
        total_pages = max(1, (total + per_page - 1) // per_page)
        if page < 1 or page > total_pages or per_page != 500:
            return None
        path = EXPORT_DIR / f"gestantes_page_{page:02d}.json"
        if not path.exists():
            self._persistir_export_laravel()
        return path if path.exists() else None

    def listar_ids_gestantes(self) -> dict[str, Any]:
        self.construir_catalogo_laravel()
        ids = [g["id_gestante"] for g in (self._catalogo_laravel or [])]
        return {
            "total": len(ids),
            "dataset": str(DATASET_PATH),
            "ids": ids,
        }

    def sincronizar_gestantes_laravel(self, page: int = 1, per_page: int = 500) -> dict[str, Any]:
        """Paginação por página (1-based) — ideal para sync incremental no Laravel."""
        self.construir_catalogo_laravel()
        items = self._catalogo_laravel or []
        total = len(items)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(page, 1), total_pages)
        offset = (page - 1) * per_page
        pagina = items[offset : offset + per_page]
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "dataset": str(DATASET_PATH),
            "export_disponivel": str(EXPORT_JSON_PATH),
            "gestantes": pagina,
        }

    def listar_gestantes_laravel(
        self,
        offset: int = 0,
        limit: int = 500,
        apenas_com_intercorrencia: bool = False,
    ) -> dict[str, Any]:
        self.construir_catalogo_laravel()
        items = self._catalogo_laravel or []

        if apenas_com_intercorrencia:
            items = [g for g in items if g["analise"]["resumo_intercorrencias"]["total"] > 0]

        total = len(items)
        pagina = items[offset : offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "dataset": str(DATASET_PATH),
            "gestantes": pagina,
        }

    def listar_todas_gestantes_laravel(self) -> dict[str, Any]:
        """Retorna as 5000 gestantes do CSV — endpoint de sincronização Laravel."""
        self.construir_catalogo_laravel()
        catalogo = self._catalogo_laravel or []
        return {
            "total": len(catalogo),
            "dataset": str(DATASET_PATH),
            "gerado_em": datetime.now(timezone.utc).isoformat(),
            "gestantes": catalogo,
        }

    def obter_gestante_laravel(self, id_gestante: str) -> dict[str, Any] | None:
        self.construir_catalogo_laravel()
        return self._catalogo_por_id.get(id_gestante)

    def _montar_gestante_catalogo(
        self,
        data: dict[str, Any],
        id_gestante: str,
        probabilidade: float,
        pred: int,
    ) -> dict[str, Any]:
        data = normalizar_dados_clinicos(data)
        classificacao = classificar_risco(probabilidade)

        intercorrencias_raw = analisar_intercorrencias(data)
        intercorrencias = enriquecer_intercorrencias(intercorrencias_raw)
        resumo_inter = resumir_intercorrencias(intercorrencias_raw)

        idade_gestacional = float(data.get("idade_gestacional", 20))
        priorizacao = calcular_priorizacao_clinica(probabilidade, intercorrencias_raw, idade_gestacional)
        prioridade_eco = priorizacao["prioridade_ecocardiograma"]
        recomenda_eco = recomendar_ecocardiograma(probabilidade, prioridade_eco, intercorrencias_raw)

        resumo_clinico = gerar_resumo_clinico(
            classificacao, probabilidade, intercorrencias, [], prioridade_eco
        )
        recomendacoes = gerar_recomendacoes(prioridade_eco, recomenda_eco, intercorrencias, classificacao)
        qualidade = validar_qualidade_dados(data)

        posicao_fila = self._fila_posicoes.get(id_gestante)

        return {
            "id_gestante": id_gestante,
            "dados_clinicos": {k: data.get(k) for k in FEATURE_COLS},
            "predicao": {
                "probabilidade_ccf": probabilidade,
                "classificacao_risco": classificacao,
                "cardiopatia_congenita_prevista": pred,
                "modelo": model_service.meta.get("modelo", "CatBoost"),
                "experimento": model_service.meta.get("experimento", "experimento_3_completo"),
            },
            "analise": {
                "probabilidade_ccf": probabilidade,
                "classificacao_risco": classificacao,
                "score_prioridade": priorizacao["score_prioridade"],
                "prioridade_ecocardiograma": prioridade_eco,
                "posicao_fila": posicao_fila,
                "recomenda_ecocardiograma": recomenda_eco,
                "resumo_clinico": resumo_clinico,
                "resumo_intercorrencias": resumo_inter,
                "fatores_priorizacao": priorizacao["fatores"],
            },
            "intercorrencias": intercorrencias,
            "recomendacoes": recomendacoes,
            "qualidade_dados": qualidade,
        }

    def analisar_completo(
        self,
        data: dict[str, Any],
        id_gestante: str | None = None,
        top_n_shap: int = 10,
        persistir: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        if not model_service.is_loaded:
            raise RuntimeError("Modelo não carregado.")

        data = normalizar_dados_clinicos(data)
        analise_id = str(uuid.uuid4())
        qualidade = validar_qualidade_dados(data)

        predicao_raw = model_service.predict_one(data)
        probabilidade = predicao_raw["probabilidade"]
        classificacao = classificar_risco(probabilidade)

        intercorrencias_raw = analisar_intercorrencias(data)
        intercorrencias = enriquecer_intercorrencias(intercorrencias_raw)
        resumo_inter = resumir_intercorrencias(intercorrencias_raw)

        idade_gestacional = float(data.get("idade_gestacional", 20))
        priorizacao = calcular_priorizacao_clinica(probabilidade, intercorrencias_raw, idade_gestacional)
        prioridade_eco = priorizacao["prioridade_ecocardiograma"]
        score_prioridade = priorizacao["score_prioridade"]
        recomenda_eco = recomendar_ecocardiograma(probabilidade, prioridade_eco, intercorrencias_raw)

        shap = explain_prediction(model_service.pipeline, data, FEATURE_COLS, top_n=top_n_shap)
        shap["probabilidade_ccf"] = probabilidade

        resumo_clinico = gerar_resumo_clinico(
            classificacao,
            probabilidade,
            intercorrencias,
            shap["top_contribuicoes"],
            prioridade_eco,
        )

        recomendacoes = gerar_recomendacoes(prioridade_eco, recomenda_eco, intercorrencias, classificacao)

        tempo_ms = round((time.perf_counter() - t0) * 1000, 1)
        versao_modelo = f"{model_service.meta.get('modelo', 'CatBoost')}/{model_service.meta.get('experimento', '')}"

        resultado: dict[str, Any] = {
            "id_gestante": id_gestante,
            "predicao": {
                "probabilidade_ccf": probabilidade,
                "classificacao_risco": classificacao,
                "cardiopatia_congenita_prevista": predicao_raw["cardiopatia_congenita"],
                "modelo": predicao_raw["modelo"],
                "experimento": predicao_raw["experimento"],
            },
            "analise": {
                "probabilidade_ccf": probabilidade,
                "classificacao_risco": classificacao,
                "score_prioridade": score_prioridade,
                "prioridade_ecocardiograma": prioridade_eco,
                "posicao_fila": None,
                "recomenda_ecocardiograma": recomenda_eco,
                "resumo_clinico": resumo_clinico,
                "resumo_intercorrencias": resumo_inter,
                "fatores_priorizacao": priorizacao["fatores"],
            },
            "intercorrencias": intercorrencias,
            "shap": shap,
            "recomendacoes": recomendacoes,
            "qualidade_dados": qualidade,
            "metadados": {
                "analise_id": analise_id,
                "versao_api": API_VERSION,
                "versao_modelo": versao_modelo,
                "data_analise": datetime.now(timezone.utc).isoformat(),
                "tempo_processamento_ms": tempo_ms,
            },
        }

        if persistir and id_gestante:
            salvar_analise(resultado, API_VERSION, versao_modelo)

        if id_gestante and id_gestante in self._fila_posicoes:
            resultado["analise"]["posicao_fila"] = self._fila_posicoes[id_gestante]

        return resultado

    def construir_fila_ecocardiograma(
        self,
        limite: int = 100,
        apenas_recomendadas: bool = True,
    ) -> dict[str, Any]:
        """Fila ordenada pela API — Laravel não recalcula."""
        if not model_service._analises_cache:
            model_service.load_or_train()

        fila: list[dict[str, Any]] = []

        for id_gestante, cache in model_service._analises_cache.items():
            data = cache["dados_clinicos"]
            resultado = self._analisar_sem_shap(data, id_gestante=id_gestante)
            if apenas_recomendadas and not resultado["recomendacoes"]["ecocardiograma_fetal"]:
                continue

            fila.append(
                {
                    "id_gestante": id_gestante,
                    "score_prioridade": resultado["analise"]["score_prioridade"],
                    "prioridade_ecocardiograma": resultado["analise"]["prioridade_ecocardiograma"],
                    "probabilidade_ccf": resultado["predicao"]["probabilidade_ccf"],
                    "classificacao_risco": resultado["predicao"]["classificacao_risco"],
                    "idade_gestacional": data.get("idade_gestacional"),
                    "total_intercorrencias": resultado["analise"]["resumo_intercorrencias"]["total"],
                    "marcadores_fetais_criticos": resultado["analise"]["fatores_priorizacao"]["marcadores_fetais_criticos"],
                    "intercorrencias_principais": [i["nome"] for i in resultado["intercorrencias"][:5]],
                    "resumo_simplificado": resultado["analise"]["resumo_clinico"]["simplificado"],
                }
            )

        fila.sort(key=lambda x: x["score_prioridade"], reverse=True)

        for pos, item in enumerate(fila, start=1):
            item["posicao_fila"] = pos
            self._fila_posicoes[item["id_gestante"]] = pos

        self._fila_cache = fila

        return {
            "total_na_fila": len(fila),
            "limite_retornado": limite,
            "gerado_em": datetime.now(timezone.utc).isoformat(),
            "priorizacao": fila[:limite],
        }

    def _analisar_sem_shap(
        self,
        data: dict[str, Any],
        id_gestante: str | None = None,
    ) -> dict[str, Any]:
        """Análise clínica sem SHAP — usada para montagem da fila em lote."""
        data = normalizar_dados_clinicos(data)
        predicao_raw = model_service.predict_one(data)
        probabilidade = predicao_raw["probabilidade"]
        classificacao = classificar_risco(probabilidade)

        intercorrencias_raw = analisar_intercorrencias(data)
        intercorrencias = enriquecer_intercorrencias(intercorrencias_raw)
        resumo_inter = resumir_intercorrencias(intercorrencias_raw)

        idade_gestacional = float(data.get("idade_gestacional", 20))
        priorizacao = calcular_priorizacao_clinica(probabilidade, intercorrencias_raw, idade_gestacional)
        prioridade_eco = priorizacao["prioridade_ecocardiograma"]
        recomenda_eco = recomendar_ecocardiograma(probabilidade, prioridade_eco, intercorrencias_raw)

        resumo_clinico = gerar_resumo_clinico(
            classificacao, probabilidade, intercorrencias, [], prioridade_eco
        )
        recomendacoes = gerar_recomendacoes(prioridade_eco, recomenda_eco, intercorrencias, classificacao)

        return {
            "id_gestante": id_gestante,
            "predicao": {
                "probabilidade_ccf": probabilidade,
                "classificacao_risco": classificacao,
            },
            "analise": {
                "score_prioridade": priorizacao["score_prioridade"],
                "prioridade_ecocardiograma": prioridade_eco,
                "resumo_clinico": resumo_clinico,
                "resumo_intercorrencias": resumo_inter,
                "fatores_priorizacao": priorizacao["fatores"],
            },
            "intercorrencias": intercorrencias,
            "recomendacoes": recomendacoes,
        }

    def get_shap_global(self, top_n: int = 15) -> dict[str, Any]:
        return get_global_shap_importance(top_n=top_n)


analysis_service = AnalysisService()
