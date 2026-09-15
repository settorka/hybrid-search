"""
Hybrid search API over a magazine corpus (info + content indices).

Design stance:
  bounded     - every external call has a timeout; every collection has a size bound
  scarce      - model and vector index are the dominant costs; both are bounded and cached
  adversarial - every input is validated; every failure has a defined behavior
  measurable  - every phase is timed, every decision is logged, every bound is justified

Run:
  uvicorn app:app --host 127.0.0.1 --port 8000 --workers 2

Environment:
  ES_HOST, ES_PORT, ES_SCHEME, ES_USER, ES_PASSWORD
  REDIS_URL
  EMBEDDING_MODEL, EMBEDDING_MAX_TOKENS, EMBEDDING_CONCURRENCY
  REQUEST_TIMEOUT_MS, ES_TIMEOUT_MS, REDIS_TIMEOUT_MS
  CACHE_TTL_SECONDS, CACHE_MAX_PAYLOAD_BYTES
  MAX_QUERY_LEN, MAX_FROM, MAX_TOP_K
  KNN_NUM_CANDIDATES_MIN, KNN_NUM_CANDIDATES_MAX, KNN_NUM_CANDIDATES_MULT
  RRF_K
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis_async
import uvicorn
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration (externalized, validated at startup)
# --------------------------------------------------------------------------- #

def _env_str(key: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(f"missing required env var: {key}")
    return val  # type: ignore[return-value]


def _env_int(key: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"env {key} is not an int: {raw!r}") from exc
    if not (lo <= val <= hi):
        raise RuntimeError(f"env {key}={val} outside [{lo}, {hi}]")
    return val


ES_HOST = _env_str("ES_HOST", required=True)
ES_PORT = _env_int("ES_PORT", 9200, 1, 65535)
ES_SCHEME = _env_str("ES_SCHEME", "https")
ES_USER = _env_str("ES_USER", required=True)
ES_PASSWORD = _env_str("ES_PASSWORD", required=True)

REDIS_URL = _env_str("REDIS_URL", required=True)

EMBEDDING_MODEL_NAME = _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_MAX_TOKENS = _env_int("EMBEDDING_MAX_TOKENS", 128, 16, 512)
EMBEDDING_CONCURRENCY = _env_int("EMBEDDING_CONCURRENCY", 2, 1, 32)

REQUEST_TIMEOUT_MS = _env_int("REQUEST_TIMEOUT_MS", 2000, 100, 30000)
ES_TIMEOUT_MS = _env_int("ES_TIMEOUT_MS", 800, 50, 10000)
REDIS_TIMEOUT_MS = _env_int("REDIS_TIMEOUT_MS", 100, 10, 5000)

CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 300, 1, 86400)
CACHE_MAX_PAYLOAD_BYTES = _env_int("CACHE_MAX_PAYLOAD_BYTES", 65536, 1024, 1048576)

MAX_QUERY_LEN = _env_int("MAX_QUERY_LEN", 512, 8, 4096)
MAX_FROM = _env_int("MAX_FROM", 1000, 0, 10000)
MAX_TOP_K = _env_int("MAX_TOP_K", 50, 1, 200)

KNN_NUM_CANDIDATES_MIN = _env_int("KNN_NUM_CANDIDATES_MIN", 100, 10, 1000)
KNN_NUM_CANDIDATES_MAX = _env_int("KNN_NUM_CANDIDATES_MAX", 500, 10, 5000)
KNN_NUM_CANDIDATES_MULT = _env_int("KNN_NUM_CANDIDATES_MULT", 10, 1, 100)

RRF_K = _env_int("RRF_K", 60, 1, 1000)

MAGAZINE_INFO_INDEX = "magazine_info"
MAGAZINE_CONTENT_INDEX = "magazine_content"

VECTOR_FIELD = "content_vector"
VECTOR_DIMS = 384  # all-MiniLM-L6-v2

# Fields returned from ES. Content is intentionally excluded from _source
# (it's large); it comes back via `highlight` as a bounded snippet.
RESULT_SOURCE_FIELDS = ["id", "title", "author", "category", "updated_at"]

log = logging.getLogger("hybrid_search")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)

# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #

es: Optional[AsyncElasticsearch] = None
redis: Optional[redis_async.Redis] = None
model: Optional[SentenceTransformer] = None
_embedding_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global es, redis, model, _embedding_semaphore

    t0 = time.perf_counter()

    log.info('"startup: connecting to elasticsearch"')
    es = AsyncElasticsearch(
        hosts=[f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"],
        basic_auth=(ES_USER, ES_PASSWORD),
        request_timeout=ES_TIMEOUT_MS / 1000,
        max_retries=1,
        retry_on_timeout=False,
        connections_per_node=10,
        http_compress=True,
    )
    try:
        await asyncio.wait_for(es.ping(), timeout=ES_TIMEOUT_MS / 1000)
    except Exception as exc:
        log.error(json.dumps({"event": "startup_es_unreachable", "error": str(exc)}))
        raise

    # Validate the vector mapping is present and correctly shaped.
    try:
        mapping = await asyncio.wait_for(
            es.indices.get_mapping(index=MAGAZINE_CONTENT_INDEX),
            timeout=ES_TIMEOUT_MS / 1000,
        )
        props = (
            mapping.get(MAGAZINE_CONTENT_INDEX, {})
            .get("mappings", {})
            .get("properties", {})
        )
        vec = props.get(VECTOR_FIELD, {})
        if vec.get("type") != "dense_vector":
            raise RuntimeError(f"{VECTOR_FIELD} is not dense_vector")
        if int(vec.get("dims", 0)) != VECTOR_DIMS:
            raise RuntimeError(
                f"{VECTOR_FIELD} dims={vec.get('dims')} expected {VECTOR_DIMS}"
            )
    except Exception as exc:
        log.error(json.dumps({"event": "startup_mapping_invalid", "error": str(exc)}))
        raise

    log.info('"startup: connecting to redis"')
    redis = redis_async.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=REDIS_TIMEOUT_MS / 1000,
        socket_connect_timeout=REDIS_TIMEOUT_MS / 1000,
        max_connections=32,
        health_check_interval=30,
    )
    try:
        await asyncio.wait_for(redis.ping(), timeout=REDIS_TIMEOUT_MS / 1000)
    except Exception as exc:
        log.error(json.dumps({"event": "startup_redis_unreachable", "error": str(exc)}))
        raise

    log.info('"startup: loading embedding model"')
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = EMBEDDING_MAX_TOKENS
    _embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    # Warmup: pay the cold-start cost once, here, not on the first request.
    try:
        _ = model.encode("warmup", normalize_embeddings=False)
    except Exception as exc:
        log.error(json.dumps({"event": "startup_warmup_failed", "error": str(exc)}))
        raise

    log.info(json.dumps({
        "event": "startup_ready",
        "took_ms": int((time.perf_counter() - t0) * 1000),
        "workers_model_dims": VECTOR_DIMS,
    }))
    yield

    log.info('"shutdown: closing connections"')
    if es is not None:
        await es.close()
    if redis is not None:
        await redis.aclose()


app = FastAPI(lifespan=lifespan, title="Hybrid Magazine Search")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    from_: int = Field(default=0, ge=0, le=MAX_FROM)
    category: Optional[str] = Field(default=None, max_length=64)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @field_validator("query")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if any(ord(c) < 0x20 and c not in "\t\n" for c in v):
            raise ValueError("query contains control characters")
        # collapse interior whitespace so "a   b" and "a b" share a cache key
        return " ".join(v.split())

    @field_validator("category")
    @classmethod
    def _cat_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if any(ord(c) < 0x20 for c in v):
            raise ValueError("category contains control characters")
        return v.strip()


class SearchResult(BaseModel):
    id: str
    title: str
    author: str
    content: str
    score: float
    category: str
    updated_at: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    cached: bool
    took_ms: int
    request_id: str


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #

def _make_cache_key(query: str, top_k: int, from_: int, category: Optional[str]) -> str:
    raw = json.dumps(
        {
            "q": query,
            "k": top_k,
            "f": from_,
            "c": category or "",
            "v": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"search:v1:{digest}"


def _embedding_cache_key(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"embed:v1:{digest}"


# --------------------------------------------------------------------------- #
# Embedding (off the loop, cached)
# --------------------------------------------------------------------------- #

async def _embed(query: str) -> list[float]:
    assert model is not None and _embedding_semaphore is not None

    # Embedding cache: same string -> same vector, avoid model call entirely.
    ekey = _embedding_cache_key(query)
    if redis is not None:
        try:
            raw = await asyncio.wait_for(
                redis.get(ekey), timeout=REDIS_TIMEOUT_MS / 1000
            )
            if raw:
                return json.loads(raw)
        except Exception as exc:
            log.debug(json.dumps({"event": "embed_cache_read_failed", "error": str(exc)}))

    async with _embedding_semaphore:
        loop = asyncio.get_running_loop()
        vector = await asyncio.wait_for(
            loop.run_in_executor(None, model.encode, query),
            timeout=REQUEST_TIMEOUT_MS / 1000,
        )
    vec_list = vector.tolist()

    if redis is not None:
        try:
            await asyncio.wait_for(
                redis.set(ekey, json.dumps(vec_list), ex=CACHE_TTL_SECONDS),
                timeout=REDIS_TIMEOUT_MS / 1000,
            )
        except Exception as exc:
            log.debug(json.dumps({"event": "embed_cache_write_failed", "error": str(exc)}))

    return vec_list


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

async def _cached_get(cache_key: str) -> Optional[list[SearchResult]]:
    if redis is None:
        return None
    try:
        raw = await asyncio.wait_for(
            redis.get(cache_key), timeout=REDIS_TIMEOUT_MS / 1000
        )
        if not raw:
            return None
        return [SearchResult(**item) for item in json.loads(raw)]
    except Exception as exc:
        log.warning(json.dumps({"event": "cache_read_failed", "error": str(exc)}))
        return None


async def _cached_set(cache_key: str, results: list[SearchResult]) -> None:
    if redis is None:
        return
    try:
        payload = json.dumps([r.model_dump() for r in results])
        if len(payload.encode("utf-8")) > CACHE_MAX_PAYLOAD_BYTES:
            log.info(json.dumps({"event": "cache_payload_too_large", "bytes": len(payload)}))
            return
        await asyncio.wait_for(
            redis.set(cache_key, payload, ex=CACHE_TTL_SECONDS),
            timeout=REDIS_TIMEOUT_MS / 1000,
        )
    except Exception as exc:
        log.warning(json.dumps({"event": "cache_write_failed", "error": str(exc)}))


# --------------------------------------------------------------------------- #
# Stats (bounded key, fire-and-forget, tracked)
# --------------------------------------------------------------------------- #

_bg_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _record_stat(query: str) -> None:
    if redis is None:
        return
    try:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        await asyncio.wait_for(
            redis.incr(f"stats:v1:{digest}"), timeout=REDIS_TIMEOUT_MS / 1000
        )
    except Exception as exc:
        log.debug(json.dumps({"event": "stat_write_failed", "error": str(exc)}))


# --------------------------------------------------------------------------- #
# Search primitives
# --------------------------------------------------------------------------- #

def _category_clause(category: Optional[str]) -> list[dict]:
    if not category:
        return []
    return [{"term": {"category.keyword": category}}]


def _to_results(resp: dict) -> list[SearchResult]:
    out: list[SearchResult] = []
    for hit in resp.get("hits", {}).get("hits", []):
        src = hit.get("_source", {}) or {}
        hl = hit.get("highlight", {}) or {}

        def pick(field: str) -> str:
            v = hl.get(field)
            if isinstance(v, list) and v:
                return str(v[0])
            raw = src.get(field, "")
            return "" if raw is None else str(raw)

        content = ""
        if isinstance(hl.get("content"), list) and hl["content"]:
            content = str(hl["content"][0])

        out.append(
            SearchResult(
                id=str(src.get("id") or hit.get("_id") or ""),
                title=pick("title"),
                author=pick("author"),
                content=content,
                score=float(hit.get("_score") or 0.0),
                category=str(src.get("category") or ""),
                updated_at=str(src.get("updated_at") or ""),
            )
        )
    return out


async def keyword_search(
    query: str, top_k: int, from_: int, category: Optional[str]
) -> list[SearchResult]:
    assert es is not None
    body = {
        "size": top_k,
        "from": from_,
        "_source": RESULT_SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "author", "content"],
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                            "prefix_length": 2,
                            "minimum_should_match": "75%",
                        }
                    }
                ],
                "filter": _category_clause(category),
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "author": {},
                "content": {"fragment_size": 150, "number_of_fragments": 1},
            }
        },
        "track_total_hits": False,
    }
    resp = await asyncio.wait_for(
        es.search(index=MAGAZINE_INFO_INDEX, body=body),
        timeout=ES_TIMEOUT_MS / 1000,
    )
    return _to_results(resp)


async def vector_search(
    query: str, top_k: int, from_: int, category: Optional[str]
) -> list[SearchResult]:
    assert es is not None
    query_vector = await _embed(query)
    num_candidates = min(
        max(top_k * KNN_NUM_CANDIDATES_MULT, KNN_NUM_CANDIDATES_MIN),
        KNN_NUM_CANDIDATES_MAX,
    )
    # kNN does not support efficient offset pagination; we fetch top_k+from_ and
    # slice. This is bounded by MAX_FROM + MAX_TOP_K.
    knn_k = top_k + from_
    body = {
        "knn": {
            "field": VECTOR_FIELD,
            "query_vector": query_vector,
            "k": knn_k,
            "num_candidates": num_candidates,
            "filter": _category_clause(category),
        },
        "size": knn_k,
        "_source": RESULT_SOURCE_FIELDS,
    }
    resp = await asyncio.wait_for(
        es.search(index=MAGAZINE_CONTENT_INDEX, body=body),
        timeout=ES_TIMEOUT_MS / 1000,
    )
    return _to_results(resp)[from_:from_ + top_k]


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def _rrf_merge(
    keyword: list[SearchResult],
    vector: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}

    for rank, r in enumerate(keyword, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(r.id, r)
    for rank, r in enumerate(vector, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(r.id, r)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    merged: list[SearchResult] = []
    for doc_id, rrf_score in ordered:
        r = by_id[doc_id]
        r.score = rrf_score
        merged.append(r)
    return merged


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #

@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery, request: Request) -> SearchResponse:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    phase: dict[str, int] = {}

    query = search_query.query
    top_k = search_query.top_k
    from_ = search_query.from_
    category = search_query.category

    cache_key = _make_cache_key(query, top_k, from_, category)

    t = time.perf_counter()
    cached = await _cached_get(cache_key)
    phase["cache_read_ms"] = int((time.perf_counter() - t) * 1000)

    if cached is not None:
        _fire_and_forget(_record_stat(query))
        took_ms = int((time.perf_counter() - started) * 1000)
        log.info(json.dumps({
            "event": "search",
            "request_id": request_id,
            "cached": True,
            "took_ms": took_ms,
            "phase": phase,
            "top_k": top_k,
            "from": from_,
            "has_category": bool(category),
        }))
        return SearchResponse(
            results=cached, cached=True, took_ms=took_ms, request_id=request_id
        )

    t = time.perf_counter()
    results = await asyncio.gather(
        keyword_search(query, top_k, from_, category),
        vector_search(query, top_k, from_, category),
        return_exceptions=True,
    )
    phase["search_ms"] = int((time.perf_counter() - t) * 1000)

    kw = results[0] if not isinstance(results[0], Exception) else []
    vec = results[1] if not isinstance(results[1], Exception) else []
    errors = [str(r) for r in results if isinstance(r, Exception)]

    if not kw and not vec:
        log.error(json.dumps({
            "event": "search_failed",
            "request_id": request_id,
            "errors": errors,
        }))
        raise HTTPException(status_code=503, detail="search backend unavailable")

    t = time.perf_counter()
    merged = _rrf_merge(kw, vec, top_k)
    phase["merge_ms"] = int((time.perf_counter() - t) * 1000)

    _fire_and_forget(_cached_set(cache_key, merged))
    _fire_and_forget(_record_stat(query))

    took_ms = int((time.perf_counter() - started) * 1000)
    log.info(json.dumps({
        "event": "search",
        "request_id": request_id,
        "cached": False,
        "took_ms": took_ms,
        "phase": phase,
        "kw_hits": len(kw),
        "vec_hits": len(vec),
        "merged": len(merged),
        "errors": errors,
        "top_k": top_k,
        "from": from_,
        "has_category": bool(category),
    }))
    return SearchResponse(
        results=merged, cached=False, took_ms=took_ms, request_id=request_id
    )


# --------------------------------------------------------------------------- #
# Health / readiness
# --------------------------------------------------------------------------- #

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


_readyz_cache: tuple[float, bool, dict] = (0.0, False, {})


@app.get("/readyz")
async def readyz() -> JSONResponse:
    global _readyz_cache
    now = time.monotonic()
    ts, ready, checks = _readyz_cache
    # cache readiness for 2s to avoid hammering dependencies from probes
    if now - ts < 2.0:
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready},
        )

    checks = {"es": False, "redis": False, "model": model is not None}
    if es is not None:
        try:
            checks["es"] = bool(
                await asyncio.wait_for(es.ping(), timeout=ES_TIMEOUT_MS / 1000)
            )
        except Exception:
            checks["es"] = False
    if redis is not None:
        try:
            checks["redis"] = bool(
                await asyncio.wait_for(redis.ping(), timeout=REDIS_TIMEOUT_MS / 1000)
            )
        except Exception:
            checks["redis"] = False

    # redis is optional; model and es are required
    ready = checks["es"] and checks["model"]
    _readyz_cache = (now, ready, checks)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready},
    )


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("x-request-id") or "-"
    log.exception(json.dumps({
        "event": "unhandled_exception",
        "request_id": request_id,
        "path": request.url.path,
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request_id},
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    workers = _env_int("WORKERS", 2, 1, 32)
    limit_concurrency = _env_int("LIMIT_CONCURRENCY", 200, 1, 10000)
    limit_max_requests = _env_int("LIMIT_MAX_REQUESTS", 10000, 100, 1000000)

    log.info(json.dumps({
        "event": "server_start",
        "workers": workers,
        "limit_concurrency": limit_concurrency,
        "limit_max_requests": limit_max_requests,
    }))

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        workers=workers,
        limit_concurrency=limit_concurrency,
        timeout_keep_alive=5,
        limit_max_requests=limit_max_requests,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
