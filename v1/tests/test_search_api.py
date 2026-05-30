import asyncio

from fastapi.testclient import TestClient

from app import create_app
from config import Settings


def test_search_returns_request_id_and_results(client: TestClient) -> None:
    """Search returns deterministic metadata and results."""

    response = client.post(
        "/search",
        json={"query": "artificial intelligence clinical", "top_k": 3},
        headers={"x-request-id": "req-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["degraded"] is False
    assert body["index_version"] == "index-v1"
    assert body["model_version"] == "hash-embedding-v1"
    assert body["results"]
    assert body["results"][0]["score_metadata"]["fused_score"] >= 0


def test_category_filter_changes_candidates(client: TestClient) -> None:
    """Category filtering is applied."""

    response = client.post("/search", json={"query": "search databases", "category": "travel"})

    assert response.status_code == 200
    categories = {result["category"] for result in response.json()["results"]}
    assert categories <= {"travel"}


def test_no_second_product_endpoint(client: TestClient) -> None:
    """Only one product endpoint is exposed."""

    app = client.app
    product_routes = [
        route
        for route in app.routes
        if hasattr(route, "methods")
        and "POST" in route.methods
        and not route.path.startswith("/health")
        and route.path != "/metrics"
    ]

    assert [route.path for route in product_routes] == ["/search"]


def test_rejects_oversized_query(client: TestClient) -> None:
    """User-controlled query length is bounded."""

    response = client.post("/search", json={"query": "x" * 257})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_rejects_oversized_body_before_route() -> None:
    """Body limit is enforced before request handling."""

    settings = Settings(max_body_size_bytes=20)
    with TestClient(create_app(settings)) as local_client:
        response = local_client.post(
            "/search",
            content=b'{"query":"' + (b"x" * 100) + b'"}',
            headers={"content-type": "application/json", "x-request-id": "body-limit"},
        )

    assert response.status_code == 400
    assert response.json()["request_id"] == "body-limit"
    assert response.json()["error"] == "bad_request"


def test_validation_error_does_not_echo_raw_input(client: TestClient) -> None:
    """Validation errors do not leak raw user input."""

    response = client.post(
        "/search",
        json={"query": "ok", "top_k": "SECRET_RAW_INPUT"},
        headers={"x-request-id": "validation-redaction"},
    )

    assert response.status_code == 400
    assert response.json()["request_id"] == "validation-redaction"
    assert "SECRET_RAW_INPUT" not in response.text


def test_rejects_invalid_top_k(client: TestClient) -> None:
    """User-controlled result count is bounded."""

    response = client.post("/search", json={"query": "ai", "top_k": 21})

    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_rate_limit_returns_429() -> None:
    """Rate limits fail fast."""

    settings = Settings(per_client_rate_per_minute=1, global_rate_per_minute=100)
    with TestClient(create_app(settings)) as local_client:
        first = local_client.post("/search", json={"query": "ai"})
        second = local_client.post("/search", json={"query": "ai"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"] == "rate_limited"


def test_concurrent_rate_limit_is_enforced() -> None:
    """Rate limits hold under concurrent pressure."""

    async def run_requests() -> list[int]:
        settings = Settings(per_client_rate_per_minute=3, global_rate_per_minute=3)
        app = create_app(settings)

        async def send_request() -> int:
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/search", json={"query": "ai"})
                return response.status_code

        return await asyncio.gather(*(send_request() for _ in range(10)))

    statuses = asyncio.run(run_requests())

    assert statuses.count(200) <= 3
    assert statuses.count(429) >= 7


def test_cache_failure_is_visible_degradation(client: TestClient) -> None:
    """Cache failure is not silent."""

    client.app.state.cache.available = False

    response = client.post("/search", json={"query": "coffee roasters"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["degradation_reason"] == "cache_unavailable"


def test_request_deadline_returns_timeout() -> None:
    """Request deadlines are enforced."""

    settings = Settings(
        request_deadline_ms=20,
        redis_timeout_ms=1,
        search_timeout_ms=5,
        embedding_timeout_ms=5,
    )
    app = create_app(settings)

    async def slow_uncached(*_: object) -> object:
        await asyncio.sleep(0.1)

    app.state.search_service._execute_uncached = slow_uncached

    with TestClient(app) as local_client:
        response = local_client.post("/search", json={"query": "ai"})

    assert response.status_code == 504
    assert response.json()["error"] == "timeout"
