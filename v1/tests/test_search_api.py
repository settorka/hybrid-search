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

    settings = Settings(request_deadline_ms=20)
    app = create_app(settings)

    async def slow_uncached(*_: object) -> object:
        await asyncio.sleep(0.1)

    app.state.search_service._execute_uncached = slow_uncached

    with TestClient(app) as local_client:
        response = local_client.post("/search", json={"query": "ai"})

    assert response.status_code == 504
    assert response.json()["error"] == "timeout"
