import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings


def make_settings(**overrides: object) -> Settings:
    """Return deterministic v2 settings isolated from local .env dependencies."""

    values: dict[str, object] = {
        "app_name": "hybrid-search-v2-test",
        "api_version": "v2",
        "schema_version": "schema-v2",
        "index_version": "index-v2",
        "model_version": "hash-embedding-v2",
        "content_version": "content-v2",
        "tracer_name": "hybrid_search_v2_test",
        "host": "127.0.0.1",
        "port": 8002,
        "seed_data_path": "data/seed_magazines.json",
        "search_backend": "memory",
        "cache_backend": "memory",
        "redis_url": "redis://localhost:6379/0",
        "elasticsearch_url": "http://localhost:9200",
        "magazine_info_index": "magazine_info_v2",
        "magazine_content_index": "magazine_content_v2",
        "elasticsearch_num_candidates": 100,
        "embedding_dimension": 32,
        "max_query_length": 256,
        "max_body_size_bytes": 4096,
        "max_top_k": 20,
        "max_offset": 1000,
        "max_keyword_candidates": 100,
        "max_vector_candidates": 100,
        "max_fusion_candidates": 200,
        "request_deadline_ms": 1200,
        "redis_timeout_ms": 100,
        "search_timeout_ms": 800,
        "embedding_timeout_ms": 500,
        "max_concurrent_requests": 16,
        "semaphore_acquire_timeout_ms": 25,
        "per_client_rate_per_minute": 60,
        "global_rate_per_minute": 120,
        "rate_window_seconds": 60,
        "rate_limiter_max_clients": 100_000,
        "rate_limiter_cleanup_interval_seconds": 30,
        "max_client_id_length": 128,
        "retry_after_seconds": 1,
        "trust_client_id_header": False,
        "max_query_tokens": 48,
        "cutover_hour": 23,
        "cutover_minute": 0,
        "monthly_budget_gbp": 100,
        "observability_budget_gbp": 20,
        "cache_ttl_seconds": 300,
        "cache_max_entries": 1024,
        "cache_required_for_readiness": True,
        "log_raw_queries": False,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    """Return tight test settings."""

    return make_settings()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    """Return a test client."""

    return TestClient(create_app(settings))
