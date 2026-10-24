"""Autenticação por API Key (Laravel ↔ microsserviço ML)."""

from __future__ import annotations

import logging
import secrets
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)

# Endpoints públicos (ops / docs) — sem dados clínicos
PUBLIC_PATHS = frozenset({
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
})


def _extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key and header_key.strip():
        return header_key.strip()

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


def api_keys_match(provided: str | None, expected: str) -> bool:
    """Comparação em tempo constante (quando os tamanhos coincidem)."""
    if not provided or not expected:
        return False
    a = provided.encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Se API_KEY estiver definida no ambiente, exige X-API-Key ou Authorization: Bearer.
    Se API_KEY estiver vazia, autenticação fica desligada (apenas para desenvolvimento).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        expected = (settings.api_key or "").strip()

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        # /docs e /redoc têm subpaths estáticos
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        if not expected:
            # Produção deve SEMPRE definir API_KEY
            return await call_next(request)

        provided = _extract_api_key(request)
        if not api_keys_match(provided, expected):
            logger.warning(
                "api_key_rejected path=%s client=%s",
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "API key inválida ou ausente. Envie o header X-API-Key.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
