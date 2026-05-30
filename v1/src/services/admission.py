import asyncio
import time
from collections import defaultdict, deque
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

        return max(0.001, self.deadline_at - time.monotonic())


class AdmissionController:
    """Bound request rates, sizes, and concurrency."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._rate_lock = asyncio.Lock()
        self._client_windows: dict[str, deque[float]] = defaultdict(deque)
        self._global_window: deque[float] = deque()

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

    async def check_rate(self, client_id: str) -> None:
        """Reject requests exceeding rate limits."""

        async with self._rate_lock:
            now = time.monotonic()
            self._prune(self._global_window, now)
            self._prune(self._client_windows[client_id], now)
            if len(self._global_window) >= self.settings.global_rate_per_minute:
                raise AdmissionError(ErrorCode.RATE_LIMITED, "global rate limit exceeded")
            if len(self._client_windows[client_id]) >= self.settings.per_client_rate_per_minute:
                raise AdmissionError(ErrorCode.RATE_LIMITED, "client rate limit exceeded")
            self._global_window.append(now)
            self._client_windows[client_id].append(now)

    async def run_bounded(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run an operation within concurrency bounds."""

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.001)
        except TimeoutError as exc:
            raise AdmissionError(ErrorCode.OVERLOADED, "concurrency limit exceeded") from exc
        try:
            return await operation()
        finally:
            self._semaphore.release()

    @staticmethod
    def _prune(window: deque[float], now: float) -> None:
        """Remove entries outside the one-minute window."""

        while window and now - window[0] > 60:
            window.popleft()
