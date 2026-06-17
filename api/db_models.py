"""Modelos ORM — histórico imutável de análises clínicas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class AnaliseHistorico(Base):
    """Registro auditável. Nunca atualizado — apenas inserções."""

    __tablename__ = "analises_historico"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_gestante: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    data_analise: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    versao_api: Mapped[str] = mapped_column(String(20), nullable=False)
    versao_modelo: Mapped[str] = mapped_column(String(100), nullable=False)
    resultado_json: Mapped[str] = mapped_column(Text, nullable=False)
    shap_json: Mapped[str] = mapped_column(Text, nullable=False)
