from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Stable error categories for clients and metrics."""

    BAD_REQUEST = "bad_request"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    NOT_READY = "not_ready"
    SEARCH_FAILED = "search_failed"


class Magazine(BaseModel):
    """Magazine metadata document."""

    id: int
    title: str
    author: str
    publication_date: str
    category: str


class MagazineContent(BaseModel):
    """Magazine content document with embedding metadata."""

    id: int
    magazine_id: int
    content: str
    vector_representation: list[float]
    content_version: str
    embedding_model_version: str


class IndexedMagazine(BaseModel):
    """Joined searchable magazine document."""

    magazine: Magazine
    content: MagazineContent


class SearchRequest(BaseModel):
    """Validated product search request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1)
    offset: int = Field(default=0, ge=0)
    category: str | None = Field(default=None, max_length=80)


class ScoreBreakdown(BaseModel):
    """Debuggable score components."""

    keyword_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0


class SearchResult(BaseModel):
    """Single hybrid search result."""

    magazine_id: int
    title: str
    author: str
    category: str
    snippet: str
    score: float
    score_metadata: ScoreBreakdown
    index_version: str
    model_version: str


class SearchResponse(BaseModel):
    """Bounded hybrid search response."""

    request_id: str
    degraded: bool
    degradation_reason: str | None
    index_version: str
    schema_version: str
    model_version: str
    results: list[SearchResult]


class ErrorResponse(BaseModel):
    """Deterministic error response."""

    request_id: str
    error: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Operational health response."""

    status: str
    checks: dict[str, bool]
    index_version: str
    schema_version: str
    model_version: str
