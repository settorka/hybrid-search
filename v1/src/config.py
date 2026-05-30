from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for bounded v1 behavior."""

    app_name: str
    api_version: str
    schema_version: str
    index_version: str
    model_version: str
    content_version: str
    tracer_name: str
    host: str
    port: int
    seed_data_path: str
    embedding_dimension: int
    max_query_length: int
    max_body_size_bytes: int
    max_top_k: int
    max_offset: int
    max_keyword_candidates: int
    max_vector_candidates: int
    max_fusion_candidates: int
    request_deadline_ms: int
    redis_timeout_ms: int
    search_timeout_ms: int
    embedding_timeout_ms: int
    max_concurrent_requests: int
    per_client_rate_per_minute: int
    global_rate_per_minute: int
    cache_ttl_seconds: int
    log_raw_queries: bool

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRID_SEARCH_",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return default settings."""

    return Settings()
