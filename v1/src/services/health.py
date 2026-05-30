from config import Settings
from models import HealthResponse
from services.cache_base import CacheAdapter
from services.repository_base import MagazineRepository


class HealthService:
    """Evaluate operational health and readiness."""

    def __init__(
        self,
        settings: Settings,
        repository: MagazineRepository,
        cache: CacheAdapter,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = cache

    async def live(self) -> HealthResponse:
        """Return process liveness."""

        return HealthResponse(
            status="live",
            checks={"process": True},
            index_version=self.settings.index_version,
            schema_version=self.settings.schema_version,
            model_version=self.settings.model_version,
        )

    async def ready(self) -> HealthResponse:
        """Return strict readiness."""

        repository_ready = await self.repository.validate()
        cache_available = await self.cache.is_available()
        checks = {
            "repository": repository_ready,
            "index": repository_ready,
            "cache": cache_available or not self.settings.cache_required_for_readiness,
            "model": self.settings.embedding_dimension > 0,
        }
        status = "ready" if all(checks.values()) else "not_ready"
        return HealthResponse(
            status=status,
            checks=checks,
            index_version=self.settings.index_version,
            schema_version=self.settings.schema_version,
            model_version=self.settings.model_version,
        )
