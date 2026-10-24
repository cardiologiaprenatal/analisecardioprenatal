"""Smoke Fase 2 — mock longitudinal sem HTTP."""
from __future__ import annotations

from app.schemas_interpretacao import InterpretacaoRequest
from app.services.interpretacao_service import interpretar_gestante


def main() -> None:
    one = InterpretacaoRequest(
        schema_version="1.1",
        gestante={"id_publico": "G1"},
        consulta_atual={"idade_gestacional": 20, "idade_materna": 25},
        historico_consultas=[{"consulta_id": 1}],
        deltas_numericos=[],
        predicao_catboost={
            "probabilidade_risco": 0.42,
            "classificacao_risco": "moderado",
            "shap": [{"feature": "idade", "impact": 0.1}],
        },
    )
    r1 = interpretar_gestante(one)
    assert r1.prompt_version == "system_v2"
    assert r1.model_id == "mock-fase2"
    assert "Sem base longitudinal" in r1.deltas[0]
    assert r1.predicao_ecoada["classificacao_risco"] == "moderado"

    multi = InterpretacaoRequest(
        schema_version="1.1",
        gestante={"id_publico": "G2"},
        consulta_atual={"idade_gestacional": 23, "idade_materna": 25},
        historico_consultas=[{}, {}],
        deltas_numericos=[
            {
                "campo": "idade_gestacional",
                "anterior": 19,
                "atual": 23,
                "delta": 4,
                "unidade": "semanas",
            },
            {
                "campo": "pressao_sistolica",
                "anterior": 118,
                "atual": 126,
                "delta": 8,
                "unidade": "mmHg",
            },
        ],
        predicao_catboost={
            "probabilidade_risco": 0.41,
            "classificacao_risco": "moderado",
            "analise_id": 2,
            "shap": [{"feature": "pa", "impact": 0.2}],
        },
        predicao_anterior={"classificacao_risco": "baixo", "probabilidade_risco": 0.2},
    )
    r2 = interpretar_gestante(multi)
    assert any("Idade gestacional" in d for d in r2.deltas)
    assert r2.predicao_ecoada["probabilidade_risco"] == 0.41
    assert "pré-natal" in (r2.resumo_gestante or "").lower() or "prenatal" in (
        r2.resumo_gestante or ""
    ).lower() or "equipe" in (r2.resumo_gestante or "").lower()
    print("ONE_OK", (r1.deltas[0][:50]).encode("ascii", "replace").decode("ascii"))
    print("MULTI_OK", len(r2.deltas), (r2.deltas[0]).encode("ascii", "replace").decode("ascii"))
    print("GESTANTE", ((r2.resumo_gestante or "")[:90]).encode("ascii", "replace").decode("ascii"))
    print("OK")


if __name__ == "__main__":
    main()
