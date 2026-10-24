"""Cliente Gemini (Vertex AI ou API key) para interpretação clínica."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GeminiNarrativa(BaseModel):
    """Schema JSON forçado na saída do modelo (sem metadados de transporte)."""

    resumo_medico: str
    resumo_gestante: str | None = None
    fatores_explicados: list[dict[str, Any]] = Field(default_factory=list)
    recomendacoes: list[str] = Field(default_factory=list)
    deltas: list[str] = Field(default_factory=list)
    predicao_ecoada: dict[str, Any] = Field(default_factory=dict)


class GeminiUnavailableError(RuntimeError):
    """Gemini não configurado ou falhou na chamada."""


def gemini_disponivel(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    if not s.interpretacao_vertex_enabled:
        return False
    if s.google_cloud_project:
        return True
    if s.google_api_key:
        return True
    return False


def _build_client(settings: Settings):
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiUnavailableError(
            "Pacote google-genai não instalado. pip install google-genai"
        ) from exc

    if settings.google_cloud_project:
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.vertex_location,
        ), f"vertex:{settings.gemini_model}"

    if settings.google_api_key:
        return genai.Client(api_key=settings.google_api_key), f"api:{settings.gemini_model}"

    raise GeminiUnavailableError(
        "Defina GOOGLE_CLOUD_PROJECT (Vertex) ou GOOGLE_API_KEY, "
        "e INTERPRETACAO_VERTEX_ENABLED=true."
    )


def gerar_narrativa(
    system_prompt: str,
    user_prompt: str,
    settings: Settings | None = None,
) -> tuple[GeminiNarrativa, dict[str, Any]]:
    """
    Chama Gemini e devolve (narrativa validada, meta tokens/model).
    Raises GeminiUnavailableError em falha de config/SDK/API.
    """
    s = settings or get_settings()
    if not s.interpretacao_vertex_enabled:
        raise GeminiUnavailableError("INTERPRETACAO_VERTEX_ENABLED=false")

    client, model_id = _build_client(s)

    try:
        from google.genai import types
    except ImportError as exc:
        raise GeminiUnavailableError("google-genai types indisponível") from exc

    try:
        response = client.models.generate_content(
            model=s.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=s.gemini_temperature,
                response_mime_type="application/json",
                response_schema=GeminiNarrativa,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("gemini_generate_content_failed: %s", exc)
        raise GeminiUnavailableError(f"Falha na chamada Gemini: {exc}") from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, GeminiNarrativa):
        narrativa = parsed
    elif isinstance(parsed, dict):
        narrativa = GeminiNarrativa.model_validate(parsed)
    else:
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiUnavailableError("Resposta Gemini vazia")
        try:
            narrativa = GeminiNarrativa.model_validate(json.loads(text))
        except Exception as exc:  # noqa: BLE001
            raise GeminiUnavailableError(f"JSON Gemini inválido: {exc}") from exc

    usage = getattr(response, "usage_metadata", None)
    tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

    meta = {
        "model_id": model_id,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    return narrativa, meta
