"""Casos de uso de gestantes: listagem, sync, detalhe e inferência."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.schemas import (
    BatchPredictResponse,
    GestanteDetailResponse,
    GestanteIdsResponse,
    GestanteListItem,
    GestanteListResponse,
    GestanteSyncResponse,
    UnifiedInferenceResponse,
)
from app.services.gestante_repository import (
    GestanteRepository,
    map_row_to_model_features,
    row_to_list_item,
    sanitize_clinical_row,
)
from app.services.predictor import PredictorService

logger = logging.getLogger(__name__)


@dataclass
class GestanteService:
    repository: GestanteRepository
    predictor: PredictorService

    def _ensure_dataset(self) -> None:
        if not self.repository.loaded:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Dataset de gestantes não carregado. "
                    "Configure DATASET_PATH ou coloque dataset_com_id_gestante.csv na raiz."
                ),
            )

    def listar(self, offset: int = 0, limit: int = 100) -> GestanteListResponse:
        self._ensure_dataset()
        offset = max(0, offset)
        limit = max(1, min(limit, 1000))
        rows = self.repository.list_rows(offset=offset, limit=limit)
        items = [GestanteListItem(**row_to_list_item(row)) for row in rows]
        return GestanteListResponse(
            total=self.repository.total,
            offset=offset,
            limit=limit,
            dataset=self.repository.path,
            gestantes=items,
        )

    def sincronizar(self, page: int = 1, per_page: int = 500) -> GestanteSyncResponse:
        """Paginação 1-based para sync Laravel (PythonApiService::syncGestantesPage)."""
        self._ensure_dataset()
        page = max(1, page)
        per_page = max(1, min(per_page, 1000))
        total = self.repository.total
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, total_pages)
        offset = (page - 1) * per_page
        rows = self.repository.list_rows(offset=offset, limit=per_page)
        items = [GestanteListItem(**row_to_list_item(row)) for row in rows]
        return GestanteSyncResponse(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            dataset=self.repository.path,
            gestantes=items,
        )

    def listar_ids(self) -> GestanteIdsResponse:
        self._ensure_dataset()
        ids = [str(r.get("id_gestante") or "") for r in self.repository.iter_all_rows()]
        ids = [i for i in ids if i]
        return GestanteIdsResponse(total=len(ids), ids=ids)

    def obter_por_id(self, id_gestante: str) -> GestanteDetailResponse:
        self._ensure_dataset()
        row = self.repository.get_by_id(id_gestante)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Gestante '{id_gestante}' não encontrada no dataset.",
            )
        return GestanteDetailResponse(
            id_gestante=str(row.get("id_gestante") or id_gestante),
            dados_clinicos=sanitize_clinical_row(row),
            features_modelo=map_row_to_model_features(row),
        )

    def _inferir_registro(
        self,
        row: dict[str, Any],
        *,
        quiet: bool = False,
    ) -> UnifiedInferenceResponse:
        gid = str(row.get("id_gestante") or "")
        # passa o registro clínico completo (36 features + id)
        payload = {**sanitize_clinical_row(row), **map_row_to_model_features(row)}
        return self.predictor.predict(payload, quiet=quiet, id_gestante=gid)

    def prever_por_id(self, id_gestante: str) -> UnifiedInferenceResponse:
        self._ensure_dataset()
        row = self.repository.get_by_id(id_gestante)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Gestante '{id_gestante}' não encontrada no dataset.",
            )
        try:
            result = self._inferir_registro(row, quiet=False)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("gestante_predict_failed id=%s", id_gestante)
            raise HTTPException(
                status_code=500,
                detail="Falha ao executar inferência para a gestante.",
            ) from exc
        return result

    def prever_todas(self) -> BatchPredictResponse:
        self._ensure_dataset()
        started = time.perf_counter()
        resultados: list[UnifiedInferenceResponse] = []
        falhas = 0

        for row in self.repository.iter_all_rows():
            try:
                resultados.append(self._inferir_registro(row, quiet=True))
            except Exception:  # noqa: BLE001
                falhas += 1
                logger.exception(
                    "batch_predict_item_failed id=%s",
                    row.get("id_gestante"),
                )

        tempo_total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "batch_predict_done total=%s ok=%s falhas=%s tempo_ms=%s",
            self.repository.total,
            len(resultados),
            falhas,
            tempo_total_ms,
        )
        return BatchPredictResponse(
            total=self.repository.total,
            processadas=len(resultados),
            falhas=falhas,
            tempo_total_ms=tempo_total_ms,
            resultados=resultados,
        )
