import asyncio
import time

from config import Settings
from helpers.metrics import Metrics
from helpers.text import build_snippet, cosine_similarity, hash_embedding, tokenize
from helpers.tracing import span
from models import ScoreBreakdown, SearchRequest, SearchResponse, SearchResult
from services.admission import RequestContext
from services.cache import VersionedCache
from services.repository import InMemoryMagazineRepository


class HybridSearchService:
    """Coordinate cache, retrieval, fusion, and response shaping."""

    def __init__(
        self,
        settings: Settings,
        repository: InMemoryMagazineRepository,
        cache: VersionedCache,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = cache
        self.metrics = metrics

    async def search(self, request: SearchRequest, context: RequestContext) -> SearchResponse:
        """Run bounded hybrid search."""

        with span("search.request", top_k=request.top_k, offset=request.offset):
            cache_key = self.cache.key_for(request)
            cached, degradation_reason = self._get_cached(cache_key)
            if cached is not None:
                self.metrics.cache_total.labels(outcome="hit").inc()
                return cached.model_copy(update={"request_id": context.request_id})
            self.metrics.cache_total.labels(outcome="miss").inc()

            response = await asyncio.wait_for(
                self._execute_uncached(request, context, degradation_reason),
                timeout=context.remaining_seconds(),
            )
            self._set_cached(cache_key, response)
            return response

    def _get_cached(self, cache_key: str) -> tuple[SearchResponse | None, str | None]:
        """Read cache with degraded failure handling."""

        start = time.perf_counter()
        try:
            result = self.cache.get(cache_key)
        except ConnectionError:
            self.metrics.degraded_total.labels(reason="cache_unavailable").inc()
            self.metrics.dependency_latency.labels(
                dependency="cache", outcome="error"
            ).observe(time.perf_counter() - start)
            return None, "cache_unavailable"
        self.metrics.dependency_latency.labels(dependency="cache", outcome="ok").observe(
            time.perf_counter() - start
        )
        return result, None

    def _set_cached(self, cache_key: str, response: SearchResponse) -> None:
        """Write cache with degraded failure handling."""

        try:
            self.cache.set(cache_key, response)
        except ConnectionError:
            self.metrics.degraded_total.labels(reason="cache_unavailable").inc()

    async def _execute_uncached(
        self,
        request: SearchRequest,
        context: RequestContext,
        degradation_reason: str | None,
    ) -> SearchResponse:
        """Run uncached keyword and vector retrieval."""

        with span("search.uncached"):
            query_vector = await asyncio.wait_for(
                asyncio.to_thread(self._embed_query, request.query),
                timeout=self.settings.embedding_timeout_ms / 1000,
            )
            keyword_results, vector_results = await asyncio.gather(
                asyncio.wait_for(
                    asyncio.to_thread(self._keyword_search, request),
                    timeout=self.settings.search_timeout_ms / 1000,
                ),
                asyncio.wait_for(
                    asyncio.to_thread(self._vector_search, request, query_vector),
                    timeout=self.settings.search_timeout_ms / 1000,
                ),
            )
            results = self._fuse(request, keyword_results, vector_results)
            if not results:
                self.metrics.zero_results_total.inc()
            return SearchResponse(
                request_id=context.request_id,
                degraded=degradation_reason is not None,
                degradation_reason=degradation_reason,
                index_version=self.settings.index_version,
                schema_version=self.settings.schema_version,
                model_version=self.settings.model_version,
                results=results[request.offset : request.offset + request.top_k],
            )

    def _embed_query(self, query: str) -> list[float]:
        """Generate deterministic query embedding."""

        with span("embedding.query"):
            return hash_embedding(query, self.settings.embedding_dimension)

    def _keyword_search(self, request: SearchRequest) -> dict[int, float]:
        """Return bounded keyword candidates."""

        with span("search.keyword"):
            query_terms = tokenize(request.query)
            scores: dict[int, float] = {}
            for document in self.repository.all():
                if request.category and document.magazine.category != request.category:
                    continue
                title_terms = tokenize(document.magazine.title)
                author_terms = tokenize(document.magazine.author)
                content_terms = tokenize(document.content.content)
                score = 0.0
                score += 3.0 * sum(term in title_terms for term in query_terms)
                score += 2.0 * sum(term in author_terms for term in query_terms)
                score += 1.0 * sum(term in content_terms for term in query_terms)
                if score > 0:
                    scores[document.magazine.id] = score
            return dict(
                sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                    : self.settings.max_keyword_candidates
                ]
            )

    def _vector_search(self, request: SearchRequest, query_vector: list[float]) -> dict[int, float]:
        """Return bounded vector candidates."""

        with span("search.vector"):
            scores: dict[int, float] = {}
            for document in self.repository.all():
                if request.category and document.magazine.category != request.category:
                    continue
                vector = document.content.vector_representation
                if len(vector) != self.settings.embedding_dimension:
                    continue
                score = cosine_similarity(query_vector, vector)
                if score > 0:
                    scores[document.magazine.id] = score
            return dict(
                sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                    : self.settings.max_vector_candidates
                ]
            )

    def _fuse(
        self,
        request: SearchRequest,
        keyword_scores: dict[int, float],
        vector_scores: dict[int, float],
    ) -> list[SearchResult]:
        """Fuse candidates into deterministic ranked results."""

        with span("search.fusion"):
            documents = {document.magazine.id: document for document in self.repository.all()}
            candidate_ids = list(dict.fromkeys([*keyword_scores.keys(), *vector_scores.keys()]))[
                : self.settings.max_fusion_candidates
            ]
            max_keyword = max(keyword_scores.values(), default=1.0)
            max_vector = max(vector_scores.values(), default=1.0)
            fused: list[SearchResult] = []
            for magazine_id in candidate_ids:
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
                        index_version=self.settings.index_version,
                        model_version=self.settings.model_version,
                    )
                )
            return sorted(fused, key=lambda result: (-result.score, result.magazine_id))
