import time
from dataclasses import dataclass

from config import Settings
from helpers.text import normalize_query
from models import SearchRequest, SearchResponse


@dataclass
class CacheRecord:
    """Stored cache payload with expiry."""

    response: SearchResponse
    expires_at: float


class VersionedCache:
    """Versioned in-memory cache with Redis-like failure semantics."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.records: dict[str, CacheRecord] = {}
        self.available = True

    def key_for(self, request: SearchRequest) -> str:
        """Build a version-safe cache key."""

        normalized = normalize_query(request.query)
        category = request.category or ""
        return (
            f"q={normalized}|top={request.top_k}|offset={request.offset}|category={category}"
            f"|schema={self.settings.schema_version}|index={self.settings.index_version}"
            f"|model={self.settings.model_version}"
        )

    def get(self, key: str) -> SearchResponse | None:
        """Return cached response when present and valid."""

        if not self.available:
            raise ConnectionError("cache unavailable")
        record = self.records.get(key)
        if record is None:
            return None
        if record.expires_at <= time.monotonic():
            self.records.pop(key, None)
            return None
        return record.response

    def set(self, key: str, response: SearchResponse) -> None:
        """Store cached response."""

        if not self.available:
            raise ConnectionError("cache unavailable")
        self.records[key] = CacheRecord(
            response=response,
            expires_at=time.monotonic() + self.settings.cache_ttl_seconds,
        )
