from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import Lifespan

from config import Settings, get_settings
from controllers.health import create_health_router
from controllers.search import create_search_router
from helpers.body_limit import BodySizeLimitMiddleware
from helpers.metrics import Metrics
from helpers.tracing import configure_tracing
from models import ErrorCode, ErrorResponse
from services.admission import AdmissionController
from services.cache import VersionedCache
from services.cache_base import CacheAdapter
from services.elasticsearch_repository import ElasticsearchMagazineRepository
from services.health import HealthService
from services.redis_cache import RedisCache
from services.repository import InMemoryMagazineRepository
from services.repository_base import MagazineRepository
from services.search import HybridSearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_tracing(resolved_settings)
    metrics = Metrics()
    repository = _create_repository(resolved_settings)
    cache = _create_cache(resolved_settings)
    admission = AdmissionController(resolved_settings)
    search_service = HybridSearchService(resolved_settings, repository, cache, metrics)
    health_service = HealthService(resolved_settings, repository, cache)

    async def lifespan(_: FastAPI) -> Lifespan:
        """Manage application lifespan."""

        yield
        await cache.close()
        await repository.close()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.api_version,
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, settings=resolved_settings)
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
            details={"errors": _sanitize_validation_errors(exc.errors())},
        )
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Return deterministic HTTP errors."""

        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
                headers=exc.headers,
            )
        error = ErrorResponse(
            request_id=request.headers.get("x-request-id", "unknown"),
            error=ErrorCode.SEARCH_FAILED,
            message=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.model_dump(mode="json"),
            headers=exc.headers,
        )

    return app


def _create_repository(settings: Settings) -> MagazineRepository:
    """Create the configured repository adapter."""

    if settings.search_backend == "elasticsearch":
        return ElasticsearchMagazineRepository(settings)
    return InMemoryMagazineRepository(settings)


def _create_cache(settings: Settings) -> CacheAdapter:
    """Create the configured cache adapter."""

    if settings.cache_backend == "redis":
        return RedisCache(settings)
    return VersionedCache(settings)


def _sanitize_validation_errors(errors: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove raw input values from validation errors."""

    sanitized: list[dict[str, object]] = []
    for error in errors:
        clean_error = dict(error)
        clean_error.pop("input", None)
        sanitized.append(clean_error)
    return sanitized
