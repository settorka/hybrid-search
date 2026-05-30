from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for bounded v1 behavior."""

    app_name: str = Field(min_length=1)
    api_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    tracer_name: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    seed_data_path: str = Field(min_length=1)
    search_backend: str = Field(pattern="^(memory|elasticsearch)$")
    cache_backend: str = Field(pattern="^(memory|redis)$")
    redis_url: str = Field(min_length=1)
    elasticsearch_url: str = Field(min_length=1)
    magazine_info_index: str = Field(min_length=1)
    magazine_content_index: str = Field(min_length=1)
    elasticsearch_num_candidates: int = Field(ge=1, le=10_000)
    embedding_dimension: int = Field(ge=1, le=4096)
    max_query_length: int = Field(ge=1, le=8192)
    max_body_size_bytes: int = Field(ge=1, le=1_048_576)
    max_top_k: int = Field(ge=1, le=1000)
    max_offset: int = Field(ge=0, le=1_000_000)
    max_keyword_candidates: int = Field(ge=1, le=10_000)
    max_vector_candidates: int = Field(ge=1, le=10_000)
    max_fusion_candidates: int = Field(ge=1, le=20_000)
    request_deadline_ms: int = Field(ge=1, le=60_000)
    redis_timeout_ms: int = Field(ge=1, le=60_000)
    search_timeout_ms: int = Field(ge=1, le=60_000)
    embedding_timeout_ms: int = Field(ge=1, le=60_000)
    max_concurrent_requests: int = Field(ge=1, le=10_000)
    per_client_rate_per_minute: int = Field(ge=1, le=1_000_000)
    global_rate_per_minute: int = Field(ge=1, le=10_000_000)
    cache_ttl_seconds: int = Field(ge=1, le=86_400)
    cache_max_entries: int = Field(ge=1, le=1_000_000)
    cache_required_for_readiness: bool
    log_raw_queries: bool

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRID_SEARCH_",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_cross_field_bounds(self) -> "Settings":
        """Validate relationships between runtime limits."""

        if self.redis_timeout_ms >= self.request_deadline_ms:
            raise ValueError("redis timeout must be below request deadline")
        if self.search_timeout_ms >= self.request_deadline_ms:
            raise ValueError("search timeout must be below request deadline")
        if self.embedding_timeout_ms >= self.request_deadline_ms:
            raise ValueError("embedding timeout must be below request deadline")
        if self.max_top_k > self.max_fusion_candidates:
            raise ValueError("top_k cannot exceed fusion candidate limit")
        if self.max_keyword_candidates + self.max_vector_candidates > self.max_fusion_candidates:
            raise ValueError("retrieval candidates cannot exceed fusion candidate limit")
        if self.per_client_rate_per_minute > self.global_rate_per_minute:
            raise ValueError("per-client rate cannot exceed global rate")
        if self.elasticsearch_num_candidates < self.max_vector_candidates:
            raise ValueError("elasticsearch num_candidates must cover vector candidate limit")
        return self


def get_settings() -> Settings:
    """Return default settings."""

    return Settings()
