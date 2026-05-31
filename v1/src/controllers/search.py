import asyncio
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from helpers.metrics import Metrics
from models import ErrorCode, ErrorResponse, SearchRequest, SearchResponse
from services.admission import AdmissionController, AdmissionError, RequestContext
from services.search import HybridSearchService


def create_search_router(
    admission: AdmissionController,
    service: HybridSearchService,
    metrics: Metrics,
) -> APIRouter:
    """Create product search routes."""

    router = APIRouter()

    @router.post(
        "/search",
        response_model=SearchResponse,
        responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    async def search(request: Request, payload: SearchRequest) -> SearchResponse:
        """Run bounded hybrid search."""

        request_id = request.headers.get("x-request-id", str(uuid4()))
        if service.settings.trust_client_id_header:
            client_id = request.headers.get("x-client-id") or (
                request.client.host if request.client else "unknown"
            )
        else:
            client_id = request.client.host if request.client else "unknown"
        context = RequestContext(
            request_id=request_id,
            client_id=client_id,
            deadline_at=time.monotonic() + (service.settings.request_deadline_ms / 1000),
        )
        admitted = False
        try:
            admission.validate_body_size(request)
            admission.validate_payload(payload)
            validated_client_id = admission.validate_client_id(client_id)
            context.client_id = validated_client_id
            await admission.check_rate(validated_client_id)
            metrics.inflight_requests.inc()
            admitted = True
            started = time.perf_counter()

            async def operation() -> SearchResponse:
                return await service.search(payload, context)

            response = await asyncio.wait_for(
                admission.run_bounded(context, operation),
                timeout=context.remaining_seconds(),
            )
            metrics.requests_total.labels(status="ok").inc()
            metrics.request_latency.labels(path="/search", outcome="ok").observe(
                time.perf_counter() - started
            )
            return response
        except AdmissionError as exc:
            metrics.requests_total.labels(status=exc.code.value).inc()
            headers: dict[str, str] = {}
            if exc.code == ErrorCode.RATE_LIMITED:
                metrics.rate_limited_total.labels(scope="client_or_global").inc()
                status_code = 429
                if service.settings.retry_after_seconds > 0:
                    headers["retry-after"] = str(service.settings.retry_after_seconds)
            elif exc.code == ErrorCode.OVERLOADED:
                status_code = 429
                if service.settings.retry_after_seconds > 0:
                    headers["retry-after"] = str(service.settings.retry_after_seconds)
            else:
                status_code = 400
            raise _http_error(
                status_code,
                request_id,
                exc.code,
                exc.message,
                exc.details,
                headers=headers,
            ) from exc
        except TimeoutError as exc:
            metrics.timeouts_total.labels(component="request").inc()
            metrics.requests_total.labels(status=ErrorCode.TIMEOUT.value).inc()
            raise _http_error(
                504,
                request_id,
                ErrorCode.TIMEOUT,
                "request deadline exceeded",
            ) from exc
        finally:
            if admitted:
                metrics.inflight_requests.dec()

    return router


def _http_error(
    status_code: int,
    request_id: str,
    code: ErrorCode,
    message: str,
    details: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Build deterministic HTTP errors."""

    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            request_id=request_id,
            error=code,
            message=message,
            details=details or {},
        ).model_dump(mode="json"),
        headers=headers,
    )
