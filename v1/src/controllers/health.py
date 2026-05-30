from fastapi import APIRouter, Response

from helpers.metrics import Metrics
from models import HealthResponse
from services.health import HealthService


def create_health_router(health: HealthService, metrics: Metrics) -> APIRouter:
    """Create operational routes."""

    router = APIRouter()

    @router.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        """Return process liveness."""

        return await health.live()

    @router.get("/health/ready", response_model=HealthResponse)
    async def ready(response: Response) -> HealthResponse:
        """Return strict readiness."""

        result = await health.ready()
        if result.status != "ready":
            response.status_code = 503
        return result

    @router.get("/metrics")
    async def render_metrics() -> Response:
        """Return Prometheus metrics."""

        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return router
