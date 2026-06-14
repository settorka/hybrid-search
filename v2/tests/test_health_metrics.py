from fastapi.testclient import TestClient


def test_health_live_and_ready(client: TestClient) -> None:
    """Health endpoints expose liveness and strict readiness."""

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "live"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_invalid_index_blocks_readiness(client: TestClient) -> None:
    """Invalid vector state blocks readiness."""

    client.app.state.repository.documents[0].content.vector_representation = [0.1]

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["index"] is False


def test_cache_unavailable_blocks_v2_readiness_by_default(client: TestClient) -> None:
    """Cache is readiness-critical for v2 by default."""

    client.app.state.cache.available = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["cache"] is False


def test_metrics_expose_cache_and_no_raw_query(client: TestClient) -> None:
    """Metrics are observable without raw query labels."""

    raw_query = "sensitive private query text"
    client.post("/search", json={"query": raw_query})
    client.post("/search", json={"query": raw_query})

    response = client.get("/metrics")
    metrics = response.text

    assert response.status_code == 200
    assert "hybrid_search_cache_total" in metrics
    assert "hybrid_search_request_latency_seconds" in metrics
    assert raw_query not in metrics
