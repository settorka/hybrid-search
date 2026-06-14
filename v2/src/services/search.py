import asyncio
import time

from config import Settings
from helpers.metrics import Metrics
from helpers.text import build_snippet, hash_embedding
from helpers.tracing import span
from models import ScoreBreakdown, SearchRequest, SearchResponse, SearchResult
from services.admission import RequestContext
from services.cache_base import CacheAdapter
from services.index_lifecycle import IndexLifecycleService, RequestState
from services.repository_base import MagazineRepository


class HybridSearchService:
    """Coordinate cache, retrieval, fusion, and response shaping."""

    def __init__(
        self,
        settings: Settings,
        repository: MagazineRepository,
        cache: CacheAdapter,
        metrics: Metrics,
        lifecycle: IndexLifecycleService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = cache
        self.metrics = metrics
        self.lifecycle = lifecycle

    async def search(self, request: SearchRequest, context: RequestContext) -> SearchResponse:
        """Run bounded hybrid search."""

        with span("search.request", top_k=request.top_k, offset=request.offset):
            active_index_version = self.lifecycle.active_version
            self.metrics.request_state_total.labels(state=RequestState.CACHE.value).inc()
            cache_key = self.cache.key_for(request, active_index_version)
            cached, degradation_reason = await self._get_cached(cache_key)
            if cached is not None:
                self.metrics.cache_total.labels(outcome="hit").inc()
                return cached.model_copy(update={"request_id": context.request_id})
            self.metrics.cache_total.labels(outcome="miss").inc()

            response = await asyncio.wait_for(
                self._execute_uncached(request, context, degradation_reason),
                timeout=context.remaining_seconds(),
            )
            await self._set_cached(cache_key, response)
            return response

    async def _get_cached(self, cache_key: str) -> tuple[SearchResponse | None, str | None]:
        """Read cache with degraded failure handling."""

        start = time.perf_counter()
        try:
            result = await self.cache.get(cache_key)
        except (ConnectionError, TimeoutError, OSError):
            self.metrics.degraded_total.labels(reason="cache_unavailable").inc()
            self.metrics.dependency_latency.labels(
                dependency="cache", outcome="error"
            ).observe(time.perf_counter() - start)
            return None, "cache_unavailable"
        self.metrics.dependency_latency.labels(dependency="cache", outcome="ok").observe(
            time.perf_counter() - start
        )
        return result, None

    async def _set_cached(self, cache_key: str, response: SearchResponse) -> None:
        """Write cache with degraded failure handling."""

        try:
            evicted = await self.cache.set(cache_key, response)
            if evicted:
                self.metrics.cache_evictions_total.labels(reason="max_entries").inc()
        except (ConnectionError, TimeoutError, OSError):
            self.metrics.degraded_total.labels(reason="cache_unavailable").inc()

    async def _execute_uncached(
        self,
        request: SearchRequest,
        context: RequestContext,
        degradation_reason: str | None,
    ) -> SearchResponse:
        """Run uncached keyword and vector retrieval."""

        with span("search.uncached"):
            query_vector = self._embed_query(request.query)
            self._ensure_budget(context)
            self.metrics.request_state_total.labels(state=RequestState.RETRIEVE_KEYWORD.value).inc()
            keyword_results = await self.repository.keyword_search(request, context)
            self._ensure_budget(context)
            self.metrics.request_state_total.labels(state=RequestState.RETRIEVE_VECTOR.value).inc()
            vector_results = await self.repository.vector_search(request, query_vector, context)
            self._ensure_budget(context)
            self.metrics.request_state_total.labels(state=RequestState.FUSE.value).inc()
            results = await self._fuse(request, keyword_results, vector_results)
            if not results:
                self.metrics.zero_results_total.inc()
            active_index_version = self.lifecycle.active_version
            self.metrics.request_state_total.labels(state=RequestState.RESPOND.value).inc()
            return SearchResponse(
                request_id=context.request_id,
                degraded=degradation_reason is not None,
                degradation_reason=degradation_reason,
                index_version=active_index_version,
                schema_version=self.settings.schema_version,
                model_version=self.settings.model_version,
                results=results[request.offset : request.offset + request.top_k],
            )

    def _embed_query(self, query: str) -> list[float]:
        """Generate deterministic query embedding."""

        with span("embedding.query"):
            return hash_embedding(query, self.settings.embedding_dimension)

    @staticmethod
    def _ensure_budget(context: RequestContext) -> None:
        """Raise timeout when request budget is exhausted."""

        if context.remaining_seconds() <= 0.001:
            raise TimeoutError("request deadline exceeded")

    async def _fuse(
        self,
        request: SearchRequest,
        keyword_scores: dict[int, float],
        vector_scores: dict[int, float],
    ) -> list[SearchResult]:
        """Fuse candidates into deterministic ranked results."""

        with span("search.fusion"):
            candidate_ids = list(dict.fromkeys([*keyword_scores.keys(), *vector_scores.keys()]))[
                : self.settings.max_fusion_candidates
            ]
            documents = await self.repository.get_documents(candidate_ids)
            max_keyword = max(keyword_scores.values(), default=1.0)
            max_vector = max(vector_scores.values(), default=1.0)
            fused: list[SearchResult] = []
            for magazine_id in candidate_ids:
                if magazine_id not in documents:
                    continue
                document = documents[magazine_id]
                keyword_score = keyword_scores.get(magazine_id, 0.0) / max_keyword
                vector_score = vector_scores.get(magazine_id, 0.0) / max_vector
                fused_score = (0.65 * keyword_score) + (0.35 * vector_score)
                fused.append(
                    SearchResult(
                        magazine_id=magazine_id,
                        title=document.magazine.title,
                        author=document.magazine.author,
                        category=document.magazine.category,
                        snippet=build_snippet(document.content.content),
                        score=round(fused_score, 6),
                        score_metadata=ScoreBreakdown(
                            keyword_score=round(keyword_score, 6),
                            vector_score=round(vector_score, 6),
                            fused_score=round(fused_score, 6),
                        ),
                        index_version=self.lifecycle.active_version,
                        model_version=self.settings.model_version,
                    )
                )
            return sorted(fused, key=lambda result: (-result.score, result.magazine_id))
