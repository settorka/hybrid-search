from fastapi import APIRouter

from helpers.metrics import Metrics
from services.rollout import RolloutService


def create_rollout_router(rollout: RolloutService, metrics: Metrics) -> APIRouter:
    """Create read-only rollout hooks."""

    router = APIRouter()

    @router.get("/rollout/status")
    async def status() -> dict[str, object]:
        """Return rollout gate state for deployment automation."""

        report = await rollout.report()
        for gate in report.gates:
            metrics.rollout_gate.labels(gate=gate.name).set(1 if gate.passed else 0)
        return {
            "status": "pass" if report.passed else "fail",
            "active_version": report.active_version,
            "previous_version": report.previous_version,
            "gates": [
                {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
                for gate in report.gates
            ],
        }

    return router
