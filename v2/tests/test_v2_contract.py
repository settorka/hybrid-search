from datetime import datetime

from fastapi.testclient import TestClient

from services.index_lifecycle import IndexLifecycleService, LifecycleError


def test_cutover_requires_ready_version_and_window(client: TestClient) -> None:
    """Cutover only activates verified versions at the configured time."""

    lifecycle: IndexLifecycleService = client.app.state.lifecycle

    try:
        lifecycle.cutover("index-v3", datetime(2026, 1, 1, 23, 0))
    except LifecycleError as exc:
        assert "ready" in str(exc)
    else:
        raise AssertionError("expected unready cutover to fail")

    lifecycle.mark_ready("index-v3")

    try:
        lifecycle.cutover("index-v3", datetime(2026, 1, 1, 22, 59))
    except LifecycleError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("expected early cutover to fail")

    event = lifecycle.cutover("index-v3", datetime(2026, 1, 1, 23, 0))

    assert event.action == "cutover"
    assert event.previous_version == "index-v2"
    assert lifecycle.active_version == "index-v3"


def test_rollback_restores_previous_active_version(client: TestClient) -> None:
    """Rollback restores the previous active version."""

    lifecycle: IndexLifecycleService = client.app.state.lifecycle
    lifecycle.mark_ready("index-v3")
    lifecycle.cutover("index-v3", datetime(2026, 1, 1, 23, 0))

    event = lifecycle.rollback("verification_regression")

    assert event.action == "rollback"
    assert event.version == "index-v2"
    assert event.previous_version == "index-v3"
    assert lifecycle.active_version == "index-v2"


def test_search_reads_active_lifecycle_version(client: TestClient) -> None:
    """Search response and cache namespace follow the active lifecycle version."""

    first = client.post("/search", json={"query": "artificial intelligence clinical"})
    assert first.status_code == 200
    assert first.json()["index_version"] == "index-v2"

    lifecycle: IndexLifecycleService = client.app.state.lifecycle
    lifecycle.mark_ready("index-v3")
    lifecycle.cutover("index-v3", datetime(2026, 1, 1, 23, 0))

    second = client.post("/search", json={"query": "artificial intelligence clinical"})

    assert second.status_code == 200
    assert second.json()["index_version"] == "index-v3"
    assert all(result["index_version"] == "index-v3" for result in second.json()["results"])


def test_rollout_status_exposes_gates(client: TestClient) -> None:
    """Rollout hook exposes bounded gate state for deployment automation."""

    response = client.get("/rollout/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pass"
    assert body["active_version"] == "index-v2"
    gate_names = {gate["name"] for gate in body["gates"]}
    assert {
        "active_index",
        "repository_ready",
        "cache_ready",
        "cutover_window",
        "budget",
        "latency_budget",
    } <= gate_names


def test_contract_telemetry_is_exported(client: TestClient) -> None:
    """State machine, rollout, cutover, and rollback telemetry are exported."""

    lifecycle: IndexLifecycleService = client.app.state.lifecycle
    lifecycle.mark_ready("index-v3")
    lifecycle.cutover("index-v3", datetime(2026, 1, 1, 23, 0))
    lifecycle.rollback("telemetry_check")
    client.get("/rollout/status")
    client.post("/search", json={"query": "coffee roasters"})

    response = client.get("/metrics")
    metrics = response.text

    assert "hybrid_search_request_state_total" in metrics
    assert "hybrid_search_index_lifecycle_transition_total" in metrics
    assert "hybrid_search_cutover_total" in metrics
    assert "hybrid_search_rollback_total" in metrics
    assert "hybrid_search_rollout_gate" in metrics
