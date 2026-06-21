from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from config import Settings
from helpers.metrics import Metrics


class RequestState(StrEnum):
    """Finite request states required by the v2 contract."""

    RECEIVED = "received"
    REJECTED = "rejected"
    CACHE = "cache"
    RETRIEVE_KEYWORD = "retrieve_keyword"
    RETRIEVE_VECTOR = "retrieve_vector"
    FUSE = "fuse"
    RESPOND = "respond"
    DEGRADED = "degraded"
    FAILED = "failed"


class IndexLifecycleState(StrEnum):
    """Finite index lifecycle states required by the v2 contract."""

    ABSENT = "absent"
    CREATING = "creating"
    LOADING = "loading"
    VERIFYING = "verifying"
    READY = "ready"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETING = "deleting"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class LifecycleError(ValueError):
    """Raised when an index lifecycle transition violates the v2 contract."""


@dataclass(frozen=True)
class IndexVersionRecord:
    """Current state for one index version."""

    version: str
    state: IndexLifecycleState


@dataclass(frozen=True)
class LifecycleEvent:
    """Auditable index lifecycle event."""

    action: str
    version: str
    previous_version: str | None
    state: IndexLifecycleState
    reason: str


class IndexLifecycleService:
    """Manage active index version, daily cutover, and rollback state."""

    _ALLOWED: dict[IndexLifecycleState, set[IndexLifecycleState]] = {
        IndexLifecycleState.ABSENT: {IndexLifecycleState.CREATING},
        IndexLifecycleState.CREATING: {IndexLifecycleState.LOADING, IndexLifecycleState.FAILED},
        IndexLifecycleState.LOADING: {IndexLifecycleState.VERIFYING, IndexLifecycleState.FAILED},
        IndexLifecycleState.VERIFYING: {IndexLifecycleState.READY, IndexLifecycleState.FAILED},
        IndexLifecycleState.READY: {IndexLifecycleState.ACTIVATING, IndexLifecycleState.FAILED},
        IndexLifecycleState.ACTIVATING: {IndexLifecycleState.ACTIVE, IndexLifecycleState.FAILED},
        IndexLifecycleState.ACTIVE: {IndexLifecycleState.DEPRECATED},
        IndexLifecycleState.DEPRECATED: {IndexLifecycleState.DELETING, IndexLifecycleState.ACTIVE},
        IndexLifecycleState.DELETING: {IndexLifecycleState.ABSENT},
        IndexLifecycleState.FAILED: {IndexLifecycleState.ROLLED_BACK},
        IndexLifecycleState.ROLLED_BACK: {IndexLifecycleState.ACTIVE},
    }

    def __init__(self, settings: Settings, metrics: Metrics | None = None) -> None:
        self.settings = settings
        self.metrics = metrics
        self._lock = RLock()
        self._active_version = settings.index_version
        self._previous_version: str | None = None
        self._records: dict[str, IndexVersionRecord] = {
            settings.index_version: IndexVersionRecord(
                version=settings.index_version,
                state=IndexLifecycleState.ACTIVE,
            )
        }
        self._events: list[LifecycleEvent] = []

    @property
    def active_version(self) -> str:
        """Return the active read version."""

        with self._lock:
            return self._active_version

    @property
    def previous_version(self) -> str | None:
        """Return the last active version available for rollback."""

        with self._lock:
            return self._previous_version

    def records(self) -> tuple[IndexVersionRecord, ...]:
        """Return all lifecycle records."""

        with self._lock:
            return tuple(self._records.values())

    def events(self) -> tuple[LifecycleEvent, ...]:
        """Return lifecycle events."""

        with self._lock:
            return tuple(self._events)

    def transition(self, version: str, state: IndexLifecycleState, reason: str) -> None:
        """Record a bounded lifecycle transition."""

        with self._lock:
            current = self._records.get(
                version,
                IndexVersionRecord(version=version, state=IndexLifecycleState.ABSENT),
            )
            allowed = self._ALLOWED.get(current.state, set())
            if state not in allowed and state != current.state:
                raise LifecycleError(f"invalid transition {current.state.value}->{state.value}")
            self._records[version] = IndexVersionRecord(version=version, state=state)
            self._events.append(
                LifecycleEvent(
                    action="transition",
                    version=version,
                    previous_version=self._previous_version,
                    state=state,
                    reason=reason,
                )
            )
            if self.metrics is not None:
                self.metrics.index_lifecycle_transition_total.labels(state=state.value).inc()

    def mark_ready(self, version: str, reason: str = "verification_passed") -> None:
        """Move an index version through the pre-activation states to READY."""

        with self._lock:
            current = self._records.get(version)
        if current is None:
            self.transition(version, IndexLifecycleState.CREATING, reason)
            self.transition(version, IndexLifecycleState.LOADING, reason)
            self.transition(version, IndexLifecycleState.VERIFYING, reason)
            self.transition(version, IndexLifecycleState.READY, reason)
            return
        self.transition(version, IndexLifecycleState.READY, reason)

    def can_cutover_at(self, now: datetime) -> bool:
        """Return whether the current time is inside the configured cutover minute."""

        return now.hour == self.settings.cutover_hour and now.minute == self.settings.cutover_minute

    def cutover(self, version: str, now: datetime, *, force: bool = False) -> LifecycleEvent:
        """Atomically activate a READY version at the configured cutover time."""

        with self._lock:
            record = self._records.get(version)
            if record is None or record.state != IndexLifecycleState.READY:
                raise LifecycleError("cutover target must be ready")
            if not force and not self.can_cutover_at(now):
                raise LifecycleError("cutover outside configured window")

            old_active = self._active_version
            self._records[version] = IndexVersionRecord(
                version=version,
                state=IndexLifecycleState.ACTIVATING,
            )
            self._records[old_active] = IndexVersionRecord(
                version=old_active,
                state=IndexLifecycleState.DEPRECATED,
            )
            self._active_version = version
            self._previous_version = old_active
            self._records[version] = IndexVersionRecord(
                version=version,
                state=IndexLifecycleState.ACTIVE,
            )
            event = LifecycleEvent(
                action="cutover",
                version=version,
                previous_version=old_active,
                state=IndexLifecycleState.ACTIVE,
                reason="scheduled_cutover" if not force else "forced_cutover",
            )
            self._events.append(event)
            if self.metrics is not None:
                self.metrics.index_lifecycle_transition_total.labels(
                    state=IndexLifecycleState.ACTIVE.value
                ).inc()
                self.metrics.cutover_total.labels(outcome="ok").inc()
            return event

    def rollback(self, reason: str) -> LifecycleEvent:
        """Restore the previous active version in one state change."""

        with self._lock:
            if self._previous_version is None:
                raise LifecycleError("no previous active version available")
            failed_version = self._active_version
            restored_version = self._previous_version
            self._records[failed_version] = IndexVersionRecord(
                version=failed_version,
                state=IndexLifecycleState.FAILED,
            )
            self._records[restored_version] = IndexVersionRecord(
                version=restored_version,
                state=IndexLifecycleState.ACTIVE,
            )
            self._active_version = restored_version
            self._previous_version = failed_version
            event = LifecycleEvent(
                action="rollback",
                version=restored_version,
                previous_version=failed_version,
                state=IndexLifecycleState.ACTIVE,
                reason=reason,
            )
            self._events.append(event)
            if self.metrics is not None:
                self.metrics.index_lifecycle_transition_total.labels(
                    state=IndexLifecycleState.ACTIVE.value
                ).inc()
                self.metrics.rollback_total.labels(outcome="ok").inc()
            return event
