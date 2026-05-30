import json
from collections.abc import Awaitable, Callable
from typing import Any

from config import Settings
from models import ErrorCode

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
Scope = dict[str, Any]


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before parsing."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        settings: Settings,
    ) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Apply body limits to HTTP requests."""

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        request_id = headers.get(b"x-request-id", b"unknown").decode("utf-8", errors="replace")
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length.decode("ascii"))
            except ValueError:
                await self._send_error(send, request_id, "invalid content-length")
                return
            if declared_size > self.settings.max_body_size_bytes:
                await self._send_error(send, request_id, "request body too large", status=413)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.settings.max_body_size_bytes:
                    await self._drain(receive)
                    await self._send_error(send, request_id, "request body too large", status=413)
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _drain(receive: Receive) -> None:
        """Drain remaining request body messages."""

        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body", False):
                return

    async def _send_error(
        self,
        send: Send,
        request_id: str,
        message: str,
        *,
        status: int = 400,
    ) -> None:
        """Send deterministic body limit error."""

        payload = {
            "request_id": request_id,
            "error": ErrorCode.BAD_REQUEST.value,
            "message": message,
            "details": {"max_body_size_bytes": self.settings.max_body_size_bytes},
        }
        body = json.dumps(payload).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
