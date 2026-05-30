from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import Settings, get_settings
from controllers.health import create_health_router
from controllers.search import create_search_router
from helpers.metrics import Metrics
from models import ErrorCode, ErrorResponse
from services.admission import AdmissionController
from services.cache import VersionedCache
from services.health import HealthService
from services.repository import InMemoryMagazineRepository
from services.search import HybridSearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""

    resolved_settings = settings or get_settings()
    metrics = Metrics()
    repository = InMemoryMagazineRepository(resolved_settings)
    cache = VersionedCache(resolved_settings)
    admission = AdmissionController(resolved_settings)
    search_service = HybridSearchService(resolved_settings, repository, cache, metrics)
    health_service = HealthService(resolved_settings, repository, cache)

    app = FastAPI(title=resolved_settings.app_name, version=resolved_settings.api_version)
    app.state.settings = resolved_settings
    app.state.metrics = metrics
    app.state.repository = repository
    app.state.cache = cache
    app.state.admission = admission
    app.state.search_service = search_service
    app.state.health_service = health_service
    app.include_router(create_search_router(admission, search_service, metrics))
    app.include_router(create_health_router(health_service, metrics))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return deterministic validation errors."""

        error = ErrorResponse(
            request_id=request.headers.get("x-request-id", "unknown"),
            error=ErrorCode.BAD_REQUEST,
            message="request validation failed",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Return deterministic HTTP errors."""

        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        error = ErrorResponse(
            request_id=request.headers.get("x-request-id", "unknown"),
            error=ErrorCode.SEARCH_FAILED,
            message=str(exc.detail),
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump(mode="json"))

    return app
