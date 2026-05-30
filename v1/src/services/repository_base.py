from abc import ABC, abstractmethod

from models import IndexedMagazine, SearchRequest
from services.admission import RequestContext


class MagazineRepository(ABC):
    """Base searchable magazine repository."""

    load_error: str | None = None
    quarantined_records: list[dict[str, object]]

    @abstractmethod
    async def all(self) -> tuple[IndexedMagazine, ...]:
        """Return indexed documents when locally available."""

    @abstractmethod
    async def validate(self) -> bool:
        """Validate repository readiness."""

    @abstractmethod
    async def keyword_search(
        self,
        request: SearchRequest,
        context: RequestContext,
    ) -> dict[int, float]:
        """Return keyword candidates."""

    @abstractmethod
    async def vector_search(
        self,
        request: SearchRequest,
        query_vector: list[float],
        context: RequestContext,
    ) -> dict[int, float]:
        """Return vector candidates."""

    @abstractmethod
    async def get_documents(self, magazine_ids: list[int]) -> dict[int, IndexedMagazine]:
        """Return documents by magazine id."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources."""
