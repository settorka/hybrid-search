from abc import ABC, abstractmethod

from config import Settings
from helpers.text import normalize_query
from models import SearchRequest, SearchResponse


class CacheAdapter(ABC):
    """Base cache adapter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def key_for(self, request: SearchRequest, index_version: str | None = None) -> str:
        """Build a version-safe cache key."""

        normalized = normalize_query(request.query)
        category = request.category or ""
        active_index_version = index_version or self.settings.index_version
        return (
            f"q={normalized}|top={request.top_k}|offset={request.offset}|category={category}"
            f"|schema={self.settings.schema_version}|index={active_index_version}"
            f"|model={self.settings.model_version}"
        )

    @abstractmethod
    async def get(self, key: str) -> SearchResponse | None:
        """Return cached response when present."""

    @abstractmethod
    async def set(self, key: str, response: SearchResponse) -> bool:
        """Store cached response and return whether eviction occurred."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Return cache availability."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources."""
