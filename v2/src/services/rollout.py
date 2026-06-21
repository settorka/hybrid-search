from dataclasses import dataclass

from config import Settings
from services.cache_base import CacheAdapter
from services.index_lifecycle import IndexLifecycleService
from services.repository_base import MagazineRepository


@dataclass(frozen=True)
class RolloutGate:
    """One rollout gate result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RolloutReport:
    """Current rollout gate report."""

    active_version: str
    previous_version: str | None
    gates: tuple[RolloutGate, ...]

    @property
    def passed(self) -> bool:
        """Return whether all gates passed."""

        return all(gate.passed for gate in self.gates)


class RolloutService:
    """Evaluate v2 rollout hooks without adding product endpoints."""

    def __init__(
        self,
        settings: Settings,
        repository: MagazineRepository,
        cache: CacheAdapter,
        lifecycle: IndexLifecycleService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = cache
        self.lifecycle = lifecycle

    async def report(self) -> RolloutReport:
        """Return the current rollout gate state."""

        repository_ready = await self.repository.validate()
        cache_ready = await self.cache.is_available()
        active_version = self.lifecycle.active_version
        gates = (
            RolloutGate(
                name="active_index",
                passed=bool(active_version),
                detail=active_version,
            ),
            RolloutGate(
                name="repository_ready",
                passed=repository_ready,
                detail="ready" if repository_ready else "not_ready",
            ),
            RolloutGate(
                name="cache_ready",
                passed=cache_ready or not self.settings.cache_required_for_readiness,
                detail="ready" if cache_ready else "not_ready",
            ),
            RolloutGate(
                name="cutover_window",
                passed=self.settings.cutover_hour == 23 and self.settings.cutover_minute == 0,
                detail=f"{self.settings.cutover_hour:02d}:{self.settings.cutover_minute:02d}",
            ),
            RolloutGate(
                name="budget",
                passed=self.settings.monthly_budget_gbp <= 100
                and self.settings.observability_budget_gbp <= 20,
                detail=(
                    f"monthly={self.settings.monthly_budget_gbp},"
                    f"observability={self.settings.observability_budget_gbp}"
                ),
            ),
            RolloutGate(
                name="latency_budget",
                passed=self.settings.request_deadline_ms <= 1200
                and self.settings.redis_timeout_ms <= 100
                and self.settings.search_timeout_ms <= 800,
                detail=(
                    f"deadline={self.settings.request_deadline_ms},"
                    f"redis={self.settings.redis_timeout_ms},"
                    f"search={self.settings.search_timeout_ms}"
                ),
            ),
        )
        return RolloutReport(
            active_version=active_version,
            previous_version=self.lifecycle.previous_version,
            gates=gates,
        )
