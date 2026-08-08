import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration: every expensive dimension has a hard ceiling.
# -----------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "hybrid-search-api")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "magazine_content")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_FIELD = os.getenv("EMBEDDING_FIELD", "content_vector")

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "25"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "256"))
MAX_CATEGORY_CHARS = int(os.getenv("MAX_CATEGORY_CHARS", "64"))
MAX_SNIPPET_CHARS = int(os.getenv("MAX_SNIPPET_CHARS", "240"))

# Retrieval work is fixed/capped independently of caller-controlled top_k.
KEYWORD_CANDIDATES = int(os.getenv("KEYWORD_CANDIDATES", "40"))
VECTOR_CANDIDATES = int(os.getenv("VECTOR_CANDIDATES", "40"))
MAX_KEYWORD_CANDIDATES = int(os.getenv("MAX_KEYWORD_CANDIDATES", "64"))
MAX_VECTOR_CANDIDATES = int(os.getenv("MAX_VECTOR_CANDIDATES", "64"))
MAX_VECTOR_NUM_CANDIDATES = int(os.getenv("MAX_VECTOR_NUM_CANDIDATES", "160"))

SEARCH_TIMEOUT_SECONDS = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "1.25"))
EMBED_TIMEOUT_SECONDS = float(os.getenv("EMBED_TIMEOUT_SECONDS", "0.40"))
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "0.08"))

# Fail-fast admission. No unbounded waiter queue inside the application.
MAX_INFLIGHT_SEARCHES = int(os.getenv("MAX_INFLIGHT_SEARCHES", "32"))
MAX_INFLIGHT_EMBEDDINGS = int(os.getenv("MAX_INFLIGHT_EMBEDDINGS", "4"))
EMBEDDING_WORKERS = int(os.getenv("EMBEDDING_WORKERS", "4"))

# Bound client pools as well as request concurrency.
ES_CONNECTIONS_PER_NODE = int(os.getenv("ES_CONNECTIONS_PER_NODE", "16"))
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "32"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "180"))
MAX_CACHE_VALUE_BYTES = int(os.getenv("MAX_CACHE_VALUE_BYTES", "65536"))
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "search:v2")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
ENABLE_VECTOR_SEARCH = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"

# Uvicorn protects the process before application-level admission.
UVICORN_LIMIT_CONCURRENCY = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "64"))
UVICORN_BACKLOG = int(os.getenv("UVICORN_BACKLOG", "128"))

logger = logging.getLogger(APP_NAME)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    category: Optional[str] = Field(default=None, max_length=MAX_CATEGORY_CHARS)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = " ".join(value.strip().split())
        return value or None


class SearchResult(BaseModel):
    id: str
    title: str
    author: str = ""
    content: str = ""
    category: str = ""
    updated_at: str = ""
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    took_ms: float
    cached: bool
    vector_used: bool
    results: List[SearchResult]


# -----------------------------------------------------------------------------
# Fail-fast capacity limiter.
# asyncio.Semaphore bounds active work but allows an unbounded waiter population.
# This limiter rejects before a request joins an internal queue.
# -----------------------------------------------------------------------------
class CapacityLimiter:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.in_use = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self.in_use >= self.capacity:
                return False
            self.in_use += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self.in_use = max(0, self.in_use - 1)


search_capacity = CapacityLimiter(MAX_INFLIGHT_SEARCHES)
embedding_capacity = CapacityLimiter(MAX_INFLIGHT_EMBEDDINGS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting application")

    # Avoid an unbounded default asyncio executor queue for CPU-bound embedding work.
    app.state.embedding_executor = ThreadPoolExecutor(
        max_workers=EMBEDDING_WORKERS,
        thread_name_prefix="embedding",
    )

    app.state.es = AsyncElasticsearch(
        ES_URL,
        request_timeout=SEARCH_TIMEOUT_SECONDS,
        retry_on_timeout=False,
        max_retries=0,
        connections_per_node=ES_CONNECTIONS_PER_NODE,
    )

    app.state.redis = Redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=REDIS_TIMEOUT_SECONDS,
        socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        max_connections=REDIS_MAX_CONNECTIONS,
    )

    # One model per process. Keep process count intentional because this is resident memory.
    app.state.model = SentenceTransformer(EMBEDDING_MODEL)

    try:
        # Bound torch CPU parallelism if torch is present. This prevents one inference
        # from silently consuming every core while several searches are in flight.
        try:
            import torch
            torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
            torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_INTEROP_THREADS", "1"))))
        except Exception as exc:
            logger.warning("unable to set torch thread bounds: %s", exc)
        yield
    finally:
        logger.info("shutting down application")
        app.state.embedding_executor.shutdown(wait=False, cancel_futures=True)
        await app.state.es.close()
        await app.state.redis.aclose()


app = FastAPI(title=APP_NAME, lifespan=lifespan)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def canonical_cache_key(payload: SearchQuery) -> str:
    normalized = {
        "query": payload.query.casefold(),
        "top_k": payload.top_k,
        "category": (payload.category or "").casefold(),
    }
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return f"{CACHE_PREFIX}:{hashlib.sha256(raw.encode()).hexdigest()}"


def category_filter(category: Optional[str]) -> List[Dict[str, Any]]:
    return [] if not category else [{"term": {"category.keyword": category}}]


async def get_cached(app: FastAPI, key: str) -> Optional[List[SearchResult]]:
    if not ENABLE_CACHE:
        return None
    try:
        raw = await app.state.redis.get(key)
        if not raw or len(raw.encode("utf-8")) > MAX_CACHE_VALUE_BYTES:
            return None
        return [SearchResult(**item) for item in json.loads(raw)]
    except Exception as exc:
        logger.warning("cache read failed: %s", exc)
        return None


async def set_cached(app: FastAPI, key: str, results: List[SearchResult]) -> None:
    if not ENABLE_CACHE:
        return
    try:
        raw = json.dumps([item.model_dump() for item in results], separators=(",", ":"))
        if len(raw.encode("utf-8")) > MAX_CACHE_VALUE_BYTES:
            logger.info("cache write skipped: value exceeds byte ceiling")
            return
        await app.state.redis.set(key, raw, ex=CACHE_TTL_SECONDS)
    except Exception as exc:
        logger.warning("cache write failed: %s", exc)


async def embed_query(app: FastAPI, query: str) -> List[float]:
    if not await embedding_capacity.try_acquire():
        raise RuntimeError("embedding capacity exhausted")

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            app.state.embedding_executor,
            lambda: app.state.model.encode(query, normalize_embeddings=True),
        )
        vector = await asyncio.wait_for(future, timeout=EMBED_TIMEOUT_SECONDS)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("embedding timeout") from exc
    finally:
        await embedding_capacity.release()


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
async def keyword_search(app: FastAPI, query: str, category: Optional[str]) -> List[Dict[str, Any]]:
    candidate_count = min(max(KEYWORD_CANDIDATES, MAX_TOP_K), MAX_KEYWORD_CANDIDATES)
    body = {
        "size": candidate_count,
        "track_total_hits": False,
        "terminate_after": int(os.getenv("ES_TERMINATE_AFTER", "10000")),
        "_source": ["title", "author", "content", "category", "updated_at"],
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "author^1.5", "content"],
                        "type": "best_fields",
                        "operator": "or",
                    }
                }],
                "filter": category_filter(category),
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "content": {"fragment_size": MAX_SNIPPET_CHARS, "number_of_fragments": 1},
            }
        },
    }
    response = await app.state.es.search(index=ES_INDEX, body=body)
    return response.get("hits", {}).get("hits", [])[:MAX_KEYWORD_CANDIDATES]


async def vector_search(app: FastAPI, query: str, category: Optional[str]) -> List[Dict[str, Any]]:
    vector = await embed_query(app, query)
    k = min(max(VECTOR_CANDIDATES, MAX_TOP_K), MAX_VECTOR_CANDIDATES)
    num_candidates = min(max(k * 3, k), MAX_VECTOR_NUM_CANDIDATES)

    knn: Dict[str, Any] = {
        "field": EMBEDDING_FIELD,
        "query_vector": vector,
        "k": k,
        "num_candidates": num_candidates,
    }
    if category:
        knn["filter"] = {"term": {"category.keyword": category}}

    response = await app.state.es.search(
        index=ES_INDEX,
        body={
            "size": k,
            "track_total_hits": False,
            "_source": ["title", "author", "content", "category", "updated_at"],
            "knn": knn,
        },
    )
    return response.get("hits", {}).get("hits", [])[:MAX_VECTOR_CANDIDATES]


# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------
def rrf_fuse(keyword_hits: List[Dict[str, Any]], vector_hits: List[Dict[str, Any]], top_k: int) -> List[SearchResult]:
    combined: Dict[str, Dict[str, Any]] = {}

    def add_hits(hits: List[Dict[str, Any]]) -> None:
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.get("_id"))
            source = hit.get("_source", {})
            highlight = hit.get("highlight", {})
            entry = combined.setdefault(doc_id, {
                "id": doc_id,
                "title": str(source.get("title", ""))[:300],
                "author": str(source.get("author", ""))[:160],
                "content": str(source.get("content", ""))[:MAX_SNIPPET_CHARS],
                "category": str(source.get("category", ""))[:MAX_CATEGORY_CHARS],
                "updated_at": str(source.get("updated_at", ""))[:64],
                "score": 0.0,
            })
            if highlight.get("title"):
                entry["title"] = str(highlight["title"][0])[:300]
            if highlight.get("content"):
                entry["content"] = str(highlight["content"][0])[:MAX_SNIPPET_CHARS]
            entry["score"] += 1.0 / (60 + rank)

    add_hits(keyword_hits[:MAX_KEYWORD_CANDIDATES])
    add_hits(vector_hits[:MAX_VECTOR_CANDIDATES])
    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    return [SearchResult(**item) for item in ranked[:min(top_k, MAX_TOP_K)]]


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
async def execute_search(app: FastAPI, payload: SearchQuery) -> tuple[List[SearchResult], bool]:
    keyword_task = asyncio.create_task(keyword_search(app, payload.query, payload.category))
    vector_task = (
        asyncio.create_task(vector_search(app, payload.query, payload.category))
        if ENABLE_VECTOR_SEARCH else None
    )

    try:
        keyword_hits = await asyncio.wait_for(keyword_task, timeout=SEARCH_TIMEOUT_SECONDS)
    except Exception as exc:
        if vector_task:
            vector_task.cancel()
        raise HTTPException(status_code=503, detail="search backend unavailable") from exc

    vector_hits: List[Dict[str, Any]] = []
    vector_used = False
    if vector_task:
        try:
            vector_hits = await asyncio.wait_for(vector_task, timeout=SEARCH_TIMEOUT_SECONDS)
            vector_used = True
        except Exception as exc:
            vector_task.cancel()
            logger.warning("vector search degraded to keyword-only: %s", exc)

    return rrf_fuse(keyword_hits, vector_hits, payload.top_k), vector_used


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health/live")
async def liveness() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(request: Request) -> Dict[str, Any]:
    try:
        es_ok = bool(await asyncio.wait_for(request.app.state.es.ping(), timeout=0.20))
    except Exception:
        es_ok = False

    if not es_ok:
        raise HTTPException(status_code=503, detail={"elasticsearch": False})
    return {"status": "ready", "elasticsearch": True}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchQuery, request: Request) -> SearchResponse:
    started = time.perf_counter()
    key = canonical_cache_key(payload)

    cached = await get_cached(request.app, key)
    if cached is not None:
        return SearchResponse(
            query=payload.query,
            count=len(cached),
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            cached=True,
            vector_used=ENABLE_VECTOR_SEARCH,
            results=cached,
        )

    # Immediate rejection rather than waiting behind a semaphore.
    if not await search_capacity.try_acquire():
        raise HTTPException(
            status_code=503,
            detail="search capacity exhausted",
            headers={"Retry-After": "1"},
        )

    try:
        results, vector_used = await asyncio.wait_for(
            execute_search(request.app, payload),
            timeout=SEARCH_TIMEOUT_SECONDS * 1.5,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="search deadline exceeded") from exc
    finally:
        await search_capacity.release()

    await set_cached(request.app, key, results)
    took_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "search query_len=%d top_k=%d inflight=%d cached=false vector_used=%s results=%d took_ms=%.2f",
        len(payload.query), payload.top_k, search_capacity.in_use,
        vector_used, len(results), took_ms,
    )

    return SearchResponse(
        query=payload.query,
        count=len(results),
        took_ms=round(took_ms, 2),
        cached=False,
        vector_used=vector_used,
        results=results,
    )


if __name__ == "__main__":
    # limit_concurrency and backlog provide a second resource boundary at the server edge.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        limit_concurrency=UVICORN_LIMIT_CONCURRENCY,
        backlog=UVICORN_BACKLOG,
        timeout_keep_alive=int(os.getenv("KEEP_ALIVE_SECONDS", "5")),
    )
