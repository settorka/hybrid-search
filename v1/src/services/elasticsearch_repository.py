import asyncio

from elasticsearch import AsyncElasticsearch

from config import Settings
from models import IndexedMagazine, Magazine, MagazineContent, SearchRequest
from services.admission import RequestContext
from services.repository_base import MagazineRepository


class ElasticsearchMagazineRepository(MagazineRepository):
    """Elasticsearch-backed magazine repository."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncElasticsearch(settings.elasticsearch_url)
        self.load_error: str | None = None
        self.quarantined_records: list[dict[str, object]] = []

    async def all(self) -> tuple[IndexedMagazine, ...]:
        """Return no local documents in Elasticsearch mode."""

        return ()

    async def validate(self) -> bool:
        """Validate Elasticsearch readiness."""

        try:
            ping, content_exists = await asyncio.gather(
                self.client.ping(),
                self.client.indices.exists(index=self.settings.magazine_content_index),
            )
            return bool(ping and content_exists)
        except Exception:
            return False

    async def keyword_search(
        self,
        request: SearchRequest,
        context: RequestContext,
    ) -> dict[int, float]:
        """Return keyword candidates from Elasticsearch."""

        query: dict[str, object] = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": request.query,
                            "fields": ["title^3", "author^2", "content"],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": self._filters(request),
            }
        }
        response = await asyncio.wait_for(
            self.client.search(
                index=self.settings.magazine_content_index,
                query=query,
                size=self.settings.max_keyword_candidates,
                source=False,
            ),
            timeout=min(context.remaining_seconds(), self.settings.search_timeout_ms / 1000),
        )
        return {
            int(hit["_id"]): float(hit["_score"] or 0.0)
            for hit in response["hits"]["hits"]
        }

    async def vector_search(
        self,
        request: SearchRequest,
        query_vector: list[float],
        context: RequestContext,
    ) -> dict[int, float]:
        """Return vector candidates from Elasticsearch."""

        knn: dict[str, object] = {
            "field": "vector_representation",
            "query_vector": query_vector,
            "k": self.settings.max_vector_candidates,
            "num_candidates": self.settings.elasticsearch_num_candidates,
        }
        filters = self._filters(request)
        if filters:
            knn["filter"] = filters
        response = await asyncio.wait_for(
            self.client.search(
                index=self.settings.magazine_content_index,
                knn=knn,
                size=self.settings.max_vector_candidates,
                source=False,
            ),
            timeout=min(context.remaining_seconds(), self.settings.search_timeout_ms / 1000),
        )
        return {
            int(hit["_id"]): float(hit["_score"] or 0.0)
            for hit in response["hits"]["hits"]
        }

    async def get_documents(self, magazine_ids: list[int]) -> dict[int, IndexedMagazine]:
        """Return documents by magazine id."""

        if not magazine_ids:
            return {}
        response = await asyncio.wait_for(
            self.client.mget(
                index=self.settings.magazine_content_index,
                ids=[str(magazine_id) for magazine_id in magazine_ids],
            ),
            timeout=self.settings.search_timeout_ms / 1000,
        )
        documents: dict[int, IndexedMagazine] = {}
        for item in response["docs"]:
            if not item.get("found"):
                continue
            source = item["_source"]
            magazine = Magazine(
                id=int(source["magazine_id"]),
                title=source["title"],
                author=source["author"],
                publication_date=source["publication_date"],
                category=source["category"],
            )
            content = MagazineContent(
                id=int(source["id"]),
                magazine_id=int(source["magazine_id"]),
                content=source["content"],
                vector_representation=source.get("vector_representation", []),
                content_version=source["content_version"],
                embedding_model_version=source["embedding_model_version"],
            )
            documents[magazine.id] = IndexedMagazine(magazine=magazine, content=content)
        return documents

    async def close(self) -> None:
        """Release Elasticsearch client resources."""

        await self.client.close()

    @staticmethod
    def _filters(request: SearchRequest) -> list[dict[str, object]]:
        """Build Elasticsearch filters."""

        if request.category:
            return [{"term": {"category": request.category}}]
        return []
