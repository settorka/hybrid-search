import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from config import Settings
from models import SearchResponse
from services.cache_base import CacheAdapter


@dataclass
class CacheRecord:
    """Stored cache payload with expiry."""

    response: SearchResponse
    expires_at: float


class VersionedCache(CacheAdapter):
    """Versioned in-memory cache with Redis-like failure semantics."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.records: OrderedDict[str, CacheRecord] = OrderedDict()
        self._available = True
        self._lock = RLock()

    @property
    def available(self) -> bool:
        """Return cache availability."""

        with self._lock:
            return self._available

    @available.setter
    def available(self, value: bool) -> None:
        """Set cache availability."""

        with self._lock:
            self._available = value

    async def get(self, key: str) -> SearchResponse | None:
        """Return cached response when present and valid."""

        with self._lock:
            if not self._available:
                raise ConnectionError("cache unavailable")
            record = self.records.get(key)
            if record is None:
                return None
            if record.expires_at <= time.monotonic():
                self.records.pop(key, None)
                return None
            self.records.move_to_end(key)
            return record.response

    async def set(self, key: str, response: SearchResponse) -> bool:
        """Store cached response."""

        with self._lock:
            if not self._available:
                raise ConnectionError("cache unavailable")
            evicted = False
            self.records[key] = CacheRecord(
                response=response,
                expires_at=time.monotonic() + self.settings.cache_ttl_seconds,
            )
            self.records.move_to_end(key)
            while len(self.records) > self.settings.cache_max_entries:
                self.records.popitem(last=False)
                evicted = True
            return evicted

    async def is_available(self) -> bool:
        """Return cache availability."""

        return self.available

    async def close(self) -> None:
        """Release cache resources."""

        return None
