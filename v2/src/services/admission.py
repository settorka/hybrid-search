import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from fastapi import Request

from config import Settings
from models import ErrorCode, SearchRequest

T = TypeVar("T")


class AdmissionError(Exception):
    """Raised when admission control rejects a request."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class RequestContext:
    """Request-scoped operational context."""

    request_id: str
    client_id: str
    deadline_at: float

    def remaining_seconds(self) -> float:
        """Return remaining request budget in seconds."""

        # A request with no remaining time is already failed; keep a small positive
        # floor to avoid invalid zero/negative timeouts in downstream calls.
        return max(0.001, self.deadline_at - time.monotonic())

    def remaining_ms(self) -> float:
        """Return remaining request budget in milliseconds."""

        return self.remaining_seconds() * 1000


@dataclass
class _ClientWindow:
    """Client rate window state with bounded retention."""

    events: deque[float]
    last_seen: float


class AdmissionController:
    """Bound request rates, sizes, and concurrency."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._global_rate_lock = asyncio.Lock()
        self._client_rate_lock = asyncio.Lock()
        self._client_windows: dict[str, _ClientWindow] = {}
        self._global_window: deque[float] = deque()
        self._last_cleanup_at = 0.0

    def validate_body_size(self, request: Request) -> None:
        """Reject oversized request bodies."""

        content_length = request.headers.get("content-length")
        if content_length is None:
            return
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise AdmissionError(ErrorCode.BAD_REQUEST, "invalid content-length") from exc
        if declared_size > self.settings.max_body_size_bytes:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "request body too large",
                {"max_body_size_bytes": self.settings.max_body_size_bytes},
            )

    def validate_payload(self, payload: SearchRequest) -> None:
        """Reject payload values outside configured bounds."""

        token_count = len(payload.query.split())
        if token_count > self.settings.max_query_tokens:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "query has too many tokens",
                {"max_query_tokens": self.settings.max_query_tokens},
            )
        if len(payload.query) > self.settings.max_query_length:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "query too long",
                {"max_query_length": self.settings.max_query_length},
            )
        if payload.top_k > self.settings.max_top_k:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "top_k too large",
                {"max_top_k": self.settings.max_top_k},
            )
        if payload.offset > self.settings.max_offset:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "offset too large",
                {"max_offset": self.settings.max_offset},
            )

    def validate_client_id(self, client_id: str) -> str:
        """Reject or normalize client identifiers."""

        candidate = client_id.strip()
        if not candidate:
            raise AdmissionError(ErrorCode.BAD_REQUEST, "invalid client_id")
        if len(candidate) > self.settings.max_client_id_length:
            raise AdmissionError(
                ErrorCode.BAD_REQUEST,
                "client_id too long",
                {"max_client_id_length": self.settings.max_client_id_length},
            )
        # Bound memory risk from attacker-controlled keys: require printable ASCII.
        for ch in candidate:
            codepoint = ord(ch)
            if codepoint < 0x20 or codepoint > 0x7E:
                raise AdmissionError(ErrorCode.BAD_REQUEST, "client_id contains invalid characters")
        return candidate

    async def check_rate(self, client_id: str) -> None:
        """Reject requests exceeding rate limits."""

        now = time.monotonic()

        async with self._global_rate_lock:
            self._prune(self._global_window, now, self.settings.rate_window_seconds)
            if len(self._global_window) >= self.settings.global_rate_per_minute:
                raise AdmissionError(ErrorCode.RATE_LIMITED, "global rate limit exceeded")
            self._global_window.append(now)

        async with self._client_rate_lock:
            state = self._client_windows.get(client_id)
            if state is None:
                state = _ClientWindow(events=deque(), last_seen=now)
                self._client_windows[client_id] = state
            state.last_seen = now
            self._prune(state.events, now, self.settings.rate_window_seconds)
            if len(state.events) >= self.settings.per_client_rate_per_minute:
                raise AdmissionError(ErrorCode.RATE_LIMITED, "client rate limit exceeded")
            state.events.append(now)

            # Opportunistic cleanup to prevent unbounded growth in unique client_ids.
            if now - self._last_cleanup_at >= self.settings.rate_limiter_cleanup_interval_seconds:
                self._cleanup_clients(now)
                self._last_cleanup_at = now
            if len(self._client_windows) > self.settings.rate_limiter_max_clients:
                self._evict_oldest(now)

    async def run_bounded(
        self,
        context: RequestContext,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """Run an operation within concurrency bounds."""

        acquire_timeout = min(
            context.remaining_seconds(),
            self.settings.semaphore_acquire_timeout_ms / 1000,
        )
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=acquire_timeout)
        except TimeoutError as exc:
            raise AdmissionError(ErrorCode.OVERLOADED, "concurrency limit exceeded") from exc
        try:
            return await operation()
        finally:
            self._semaphore.release()

    @staticmethod
    def _prune(window: deque[float], now: float, window_seconds: int) -> None:
        """Remove entries outside the one-minute window."""

        while window and now - window[0] > window_seconds:
            window.popleft()

    def _cleanup_clients(self, now: float) -> None:
        """Remove idle clients whose windows are empty and expired."""

        expired: list[str] = []
        for key, state in self._client_windows.items():
            if state.events:
                continue
            if now - state.last_seen > self.settings.rate_window_seconds:
                expired.append(key)
        for key in expired:
            self._client_windows.pop(key, None)

    def _evict_oldest(self, now: float) -> None:
        """Evict oldest-seen clients until within max size."""

        # O(n) but only triggers when already beyond a configured hard cap.
        while len(self._client_windows) > self.settings.rate_limiter_max_clients:
            oldest_key = min(self._client_windows, key=lambda k: self._client_windows[k].last_seen)
            self._client_windows.pop(oldest_key, None)
