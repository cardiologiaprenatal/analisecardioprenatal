"""Persistência auditável do histórico de análises — somente inserção."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from api.database import SessionLocal, init_db
from api.db_models import AnaliseHistorico


def ensure_history_db() -> None:
    init_db()


def salvar_analise(
    resultado: dict[str, Any],
    versao_api: str,
    versao_modelo: str,
) -> str:
    """Persiste análise completa. Nunca sobrescreve registros anteriores."""
    ensure_history_db()
    analise_id = resultado["metadados"]["analise_id"]
    id_gestante = resultado.get("id_gestante") or "DESCONHECIDA"

    shap_payload = resultado.get("shap", {})

    with SessionLocal() as db:
        registro = AnaliseHistorico(
            id=analise_id,
            id_gestante=str(id_gestante),
            versao_api=versao_api,
            versao_modelo=versao_modelo,
            resultado_json=json.dumps(resultado, ensure_ascii=False, default=str),
            shap_json=json.dumps(shap_payload, ensure_ascii=False, default=str),
        )
        db.add(registro)
        db.commit()

    return analise_id


def listar_historico_gestante(
    id_gestante: str,
    limite: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    ensure_history_db()
    with SessionLocal() as db:
        query = (
            db.query(AnaliseHistorico)
            .filter(AnaliseHistorico.id_gestante == id_gestante)
            .order_by(AnaliseHistorico.data_analise.desc())
        )
        total = query.count()
        registros = query.offset(offset).limit(limite).all()

        return {
            "id_gestante": id_gestante,
            "total": total,
            "offset": offset,
            "limite": limite,
            "historico": [_serialize_resumo(r) for r in registros],
        }


def obter_analise_por_id(analise_id: str) -> dict[str, Any] | None:
    ensure_history_db()
    with SessionLocal() as db:
        registro = db.query(AnaliseHistorico).filter(AnaliseHistorico.id == analise_id).first()
        if not registro:
            return None
        return {
            "analise_id": registro.id,
            "id_gestante": registro.id_gestante,
            "data_analise": registro.data_analise.isoformat(),
            "versao_api": registro.versao_api,
            "versao_modelo": registro.versao_modelo,
            "resultado": json.loads(registro.resultado_json),
            "shap": json.loads(registro.shap_json),
        }


def _serialize_resumo(registro: AnaliseHistorico) -> dict[str, Any]:
    resultado = json.loads(registro.resultado_json)
    analise = resultado.get("analise", {})
    return {
        "analise_id": registro.id,
        "data_analise": registro.data_analise.isoformat(),
        "versao_api": registro.versao_api,
        "versao_modelo": registro.versao_modelo,
        "classificacao_risco": analise.get("classificacao_risco"),
        "probabilidade_ccf": analise.get("probabilidade_ccf"),
        "prioridade_ecocardiograma": analise.get("prioridade_ecocardiograma"),
        "score_prioridade": analise.get("score_prioridade"),
    }
