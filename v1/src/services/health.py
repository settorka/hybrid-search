from config import Settings
from models import HealthResponse
from services.cache import VersionedCache
from services.repository import InMemoryMagazineRepository


class HealthService:
    """Evaluate operational health and readiness."""

    def __init__(
        self,
        settings: Settings,
        repository: InMemoryMagazineRepository,
        cache: VersionedCache,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = cache

    def live(self) -> HealthResponse:
        """Return process liveness."""

        return HealthResponse(
            status="live",
            checks={"process": True},
            index_version=self.settings.index_version,
            schema_version=self.settings.schema_version,
            model_version=self.settings.model_version,
        )

    def ready(self) -> HealthResponse:
        """Return strict readiness."""

        checks = {
            "repository": bool(self.repository.all()),
            "index": self.repository.validate(),
            "cache": self.cache.available,
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
