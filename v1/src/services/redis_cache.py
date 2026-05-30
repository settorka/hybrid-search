import asyncio

from redis.asyncio import Redis

from config import Settings
from models import SearchResponse
from services.cache_base import CacheAdapter


class RedisCache(CacheAdapter):
    """Redis-backed cache adapter."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def get(self, key: str) -> SearchResponse | None:
        """Return cached response when present."""

        payload = await asyncio.wait_for(
            self.client.get(key),
            timeout=self.settings.redis_timeout_ms / 1000,
        )
        if payload is None:
            return None
        return SearchResponse.model_validate_json(payload)

    async def set(self, key: str, response: SearchResponse) -> bool:
        """Store cached response."""

        await asyncio.wait_for(
            self.client.set(key, response.model_dump_json(), ex=self.settings.cache_ttl_seconds),
            timeout=self.settings.redis_timeout_ms / 1000,
        )
        return False

    async def is_available(self) -> bool:
        """Return Redis availability."""

        try:
            return bool(
                await asyncio.wait_for(
                    self.client.ping(),
                    timeout=self.settings.redis_timeout_ms / 1000,
                )
            )
        except (TimeoutError, OSError, ConnectionError):
            return False
