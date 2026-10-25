"""Error handling: every error is returned as JSON in the brief's format ``{"error": ...}``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mykrml_25740287.data.crypto import SUPPORTED_TOKENS
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("mri_api")


class APIError(Exception):
    """An error with an HTTP status code and a JSON body."""

    def __init__(self, status_code: int, error: str, **extra):
        super().__init__(error)
        self.status_code = status_code
        self.body = {"error": error, **extra}


def unsupported_token(token: str) -> APIError:
    return APIError(
        400, "Unsupported cryptocurrency", token=token, supported_tokens=list(SUPPORTED_TOKENS)
    )


def unsupported_horizon(value: int) -> APIError:
    return APIError(
        422, "Unsupported forecast_horizon", forecast_horizon=value, supported_horizons=[7]
    )


def market_data_unavailable() -> APIError:
    return APIError(
        503,
        "Market data source unavailable",
        detail="Kraken could not be reached after retries; please try again shortly.",
        retry_after_seconds=60,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach handlers so no request ever returns a stack trace or a non-JSON body."""

    @app.exception_handler(APIError)
    async def handle_api_error(_: Request, exc: APIError):
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in err.get("loc", []) if p != "body"),
             "message": err.get("msg", "invalid value")}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"error": "Invalid request", "details": details})

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return JSONResponse(status_code=exc.status_code, content={"error": message})

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception):  # pragma: no cover - safety net
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
