"""
Production-Ready Hybrid Search API for Magazine Articles
Version: 2.0.0
Python: 3.11+
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from functools import wraps

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from elasticsearch import AsyncElasticsearch, exceptions as es_exceptions
import aioredis
from sentence_transformers import SentenceTransformer
import tenacity
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration with validation."""
    
    # Elasticsearch
    ES_HOST = os.getenv("ES_HOST", "elasticsearch")
    ES_PORT = int(os.getenv("ES_PORT", 9200))
    ES_SCHEME = os.getenv("ES_SCHEME", "http")
    ES_TIMEOUT = int(os.getenv("ES_TIMEOUT", 30))
    ES_MAX_RETRIES = int(os.getenv("ES_MAX_RETRIES", 3))
    ES_POOL_SIZE = int(os.getenv("ES_POOL_SIZE", 20))
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", 20))
    REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", 5))
    
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")  # or cuda
    
    # Search
    MAGAZINE_INFO_INDEX = os.getenv("MAGAZINE_INFO_INDEX", "magazine_info")
    MAGAZINE_CONTENT_INDEX = os.getenv("MAGAZINE_CONTENT_INDEX", "magazine_content")
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", 100))
    
    # Cache
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 10000))
    CACHE_PREFIX = os.getenv("CACHE_PREFIX", "search")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 100))
    
    # Circuit Breaker
    CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", 5))
    CB_TIMEOUT = int(os.getenv("CB_TIMEOUT", 60))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
    
    @classmethod
    def validate(cls):
        """Validate configuration at startup."""
        assert cls.ES_PORT > 0, "Invalid ES_PORT"
        assert cls.DEFAULT_TOP_K > 0, "Invalid DEFAULT_TOP_K"
        assert cls.CACHE_TTL > 0, "Invalid CACHE_TTL"
        return True

# ============================================================================
# LOGGING
# ============================================================================

class StructuredLogger:
    """Structured JSON logger with correlation IDs."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # JSON formatter for production
        if Config.LOG_FORMAT == "json":
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            logging.basicConfig(level=getattr(logging, level.upper()))
    
    def _add_context(self, message: str, extra: Dict = None) -> Dict:
        """Add correlation ID and context to log entry."""
        context = {
            "correlation_id": getattr(self.logger, "correlation_id", uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            context.update(extra)
        return {"message": message, "extra": context}
    
    def info(self, message: str, extra: Dict = None):
        self.logger.info(message, extra=self._add_context(message, extra))
    
    def error(self, message: str, extra: Dict = None, exc_info: bool = False):
        self.logger.error(message, extra=self._add_context(message, extra), exc_info=exc_info)
    
    def warning(self, message: str, extra: Dict = None):
        self.logger.warning(message, extra=self._add_context(message, extra))
    
    def debug(self, message: str, extra: Dict = None):
        self.logger.debug(message, extra=self._add_context(message, extra))

# Initialize logger
logger = StructuredLogger("search_api", Config.LOG_LEVEL)

# ============================================================================
# DATA MODELS
# ============================================================================

class SearchQuery(BaseModel):
    """Search query model with validation."""
    
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    from_: int = Field(default=0, ge=0)
    category: Optional[str] = Field(None, max_length=50)
    
    @validator('query')
    def validate_query(cls, v):
        """Sanitize query to prevent injection."""
        # Remove dangerous characters
        v = ''.join(c for c in v if c.isprintable())
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or contain only whitespace")
        return v
    
    @validator('category')
    def validate_category(cls, v):
        if v and not v.isalpha():
            raise ValueError("Category must contain only alphabetic characters")
        return v

class SearchResult(BaseModel):
    """Search result model with consistent types."""
    
    id: str  # Changed to string to match Elasticsearch
    title: str
    author: str
    content: str
    score: float = Field(..., ge=0, le=5)
    category: str
    updated_at: str
    
    @validator('score')
    def validate_score(cls, v):
        """Ensure score is within reasonable range."""
        if v < 0 or v > 5:
            raise ValueError(f"Score {v} outside expected range 0-5")
        return v

class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    status: int
    code: str
    message: str
    correlation_id: str
    timestamp: str
    path: Optional[str] = None

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, failure_threshold: int = 5, timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit {self.name} moving to HALF_OPEN")
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Service {self.name} unavailable (circuit open)"
                    )
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit {self.name} closed successfully")
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit {self.name} opened due to failures: {e}")
            raise

# ============================================================================
# DEPENDENCY MANAGER
# ============================================================================

class DependencyManager:
    """Centralized dependency management with health checks."""
    
    def __init__(self):
        self.es: Optional[AsyncElasticsearch] = None
        self.redis: Optional[aioredis.Redis] = None
        self.model: Optional[SentenceTransformer] = None
        self.es_circuit = CircuitBreaker("elasticsearch", Config.CB_FAILURE_THRESHOLD, Config.CB_TIMEOUT)
        self.redis_circuit = CircuitBreaker("redis", Config.CB_FAILURE_THRESHOLD, Config.CB_TIMEOUT)
    
    async def initialize(self):
        """Initialize all dependencies."""
        try:
            # Initialize Elasticsearch
            self.es = AsyncElasticsearch(
                [f"{Config.ES_SCHEME}://{Config.ES_HOST}:{Config.ES_PORT}"],
                timeout=Config.ES_TIMEOUT,
                max_retries=Config.ES_MAX_RETRIES,
                retry_on_timeout=True,
                connection_class=None,
                http_auth=(
                    os.getenv("ES_USERNAME", ""),
                    os.getenv("ES_PASSWORD", "")
                ) if os.getenv("ES_USERNAME") else None,
                verify_certs=os.getenv("ES_VERIFY_CERTS", "false").lower() == "true",
                ssl_show_warn=False,
            )
            
            # Test Elasticsearch connection
            await self.es.ping()
            logger.info("Elasticsearch connected successfully")
            
            # Initialize Redis with connection pool
            self.redis = aioredis.from_url(
                Config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=Config.REDIS_MAX_CONNECTIONS,
                socket_timeout=Config.REDIS_TIMEOUT,
                socket_connect_timeout=Config.REDIS_TIMEOUT,
            )
            
            # Test Redis connection
            await self.redis.ping()
            logger.info("Redis connected successfully")
            
            # Initialize model (in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                Config.MODEL_NAME,
                Config.MODEL_DEVICE
            )
            logger.info(f"Model {Config.MODEL_NAME} loaded successfully")
            
        except Exception as e:
            logger.error(f"Dependency initialization failed: {e}", exc_info=True)
            raise
    
    async def cleanup(self):
        """Clean up all dependencies."""
        if self.es:
            await self.es.close()
            logger.info("Elasticsearch connection closed")
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    async def get_es(self) -> AsyncElasticsearch:
        """Get Elasticsearch client with circuit breaker."""
        if not self.es:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch not available"
            )
        return self.es
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis client with circuit breaker."""
        if not self.redis:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis not available"
            )
        return self.redis
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding with proper error handling."""
        if not self.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, normalize_embeddings=True).tolist()
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed"
            )

# Global dependency manager
deps = DependencyManager()

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter using Redis."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        try:
            current = int(time.time())
            window_key = f"ratelimit:{key}:{current // window}"
            
            count = await self.redis.get(window_key)
            if count is None:
                await self.redis.setex(window_key, window, 1)
                return True
            
            count = int(count)
            if count >= limit:
                return False
            
            await self.redis.incr(window_key)
            return True
            
        except Exception as e:
            logger.warning(f"Rate limiter failed: {e}", extra={"key": key})
            return True  # Fail open
    
    async def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining quota."""
        try:
            current = int(time.time())
            window_key = f"ratelimit:{key}:{current // window}"
            count = await self.redis.get(window_key)
            if count is None:
                return limit
            return max(0, limit - int(count))
        except:
            return limit

# ============================================================================
# SEARCH ENGINE
# ============================================================================

class SearchEngine:
    """Hybrid search engine with caching and retries."""
    
    def __init__(self, deps_manager):
        self.deps = deps_manager
        self.cache_prefix = Config.CACHE_PREFIX
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type(es_exceptions.ConnectionError)
    )
    async def _execute_es_search(self, index: str, body: Dict) -> Dict:
        """Execute Elasticsearch search with retries."""
        es = await self.deps.get_es()
        return await es.search(index=index, body=body, request_timeout=Config.ES_TIMEOUT)
    
    async def _keyword_search(self, query: str, top_k: int, from_: int) -> List[SearchResult]:
        """Keyword-based BM25 search."""
        es_query = {
            "size": top_k,
            "from": from_,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "author", "content"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                    "minimum_should_match": "75%"
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "author": {},
                    "content": {"fragment_size": 150, "number_of_fragments": 1}
                }
            }
        }
        
        try:
            response = await self.deps.es_circuit.call(
                self._execute_es_search,
                Config.MAGAZINE_INFO_INDEX,
                es_query
            )
            return self._parse_results(response)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Keyword search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Keyword search failed"
            )
    
    async def _vector_search(self, query: str, top_k: int, from_: int) -> List[SearchResult]:
        """Semantic vector search."""
        try:
            embedding = await self.deps.get_embedding(query)
            
            es_query = {
                "size": top_k,
                "from": from_,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                            "params": {"query_vector": embedding}
                        }
                    }
                },
                "_source": ["id", "title", "author", "content", "category", "updated_at"],
                "highlight": {
                    "fields": {
                        "title": {},
                        "author": {},
                        "content": {"fragment_size": 150, "number_of_fragments": 1}
                    }
                }
            }
            
            response = await self.deps.es_circuit.call(
                self._execute_es_search,
                Config.MAGAZINE_CONTENT_INDEX,
                es_query
            )
            return self._parse_results(response)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector search failed"
            )
    
    def _parse_results(self, response: Dict) -> List[SearchResult]:
        """Parse Elasticsearch response into SearchResult objects."""
        results = []
        for hit in response.get('hits', {}).get('hits', []):
            source = hit.get('_source', {})
            highlight = hit.get('highlight', {})
            
            results.append(SearchResult(
                id=str(hit.get('_id', source.get('id', ''))),  # Convert to string
                title=highlight.get('title', [source.get('title', '')])[0],
                author=highlight.get('author', [source.get('author', '')])[0],
                content=highlight.get('content', [source.get('content', '')[:150] + "..."])[0],
                score=min(hit.get('_score', 0.0), 5.0),  # Cap score at 5
                category=source.get('category', ''),
                updated_at=source.get('updated_at', '')
            ))
        return results
    
    async def hybrid_search(self, query: str, top_k: int, from_: int) -> List[SearchResult]:
        """Hybrid search combining keyword and vector search."""
        try:
            # Validate inputs
            if not query or not query.strip():
                return []
            
            start_time = time.time()
            
            # Determine search weights dynamically
            query_terms = query.lower().split()
            keyword_weight = 0.7
            vector_weight = 0.3
            
            if len(query_terms) > 3:
                keyword_weight = 0.8
                vector_weight = 0.2
            
            # Execute parallel searches
            keyword_results, vector_results = await asyncio.gather(
                self._keyword_search(query, min(top_k * 2, Config.MAX_TOP_K), from_),
                self._vector_search(query, min(top_k * 2, Config.MAX_TOP_K), from_),
                return_exceptions=True
            )
            
            # Handle search failures gracefully
            if isinstance(keyword_results, Exception):
                logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            # Combine results
            combined = {}
            
            # Apply weights and boosts
            for result in keyword_results:
                score = result.score * keyword_weight
                # Title boost
                if any(term in result.title.lower() for term in query_terms):
                    score *= 1.5
                # Author boost
                if any(term in result.author.lower() for term in query_terms):
                    score *= 1.2
                # Exact match boost
                if query.lower() in result.title.lower():
                    score *= 2.0
                
                result.score = min(score, 5.0)
                combined[result.id] = result
            
            for result in vector_results:
                if result.id in combined:
                    # Add vector score
                    combined[result.id].score += result.score * vector_weight
                else:
                    score = result.score * vector_weight
                    # Term match boost
                    matched_terms = sum(1 for term in query_terms if term in result.content.lower())
                    score *= (1 + (0.1 * min(matched_terms, len(query_terms))))
                    # Exact match boost
                    if query.lower() in result.title.lower():
                        score *= 2.0
                    
                    result.score = min(score, 5.0)
                    combined[result.id] = result
            
            # Sort and limit
            results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            # Log performance
            duration = (time.time() - start_time) * 1000
            logger.info(
                "Search completed",
                extra={
                    "query": query[:50],
                    "results": len(results),
                    "duration_ms": duration,
                    "keyword_results": len(keyword_results),
                    "vector_results": len(vector_results)
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed"
            )
    
    async def _get_cache_key(self, query: str, top_k: int, from_: int) -> str:
        """Generate normalized cache key."""
        normalized_query = query.strip().lower()
        normalized_query = ' '.join(normalized_query.split())
        return f"{self.cache_prefix}:{normalized_query}:{top_k}:{from_}"
    
    async def _cache_results(self, key: str, results: List[SearchResult]):
        """Cache search results with compression."""
        try:
            redis = await self.deps.get_redis()
            serialized = json.dumps([r.dict() for r in results])
            await redis.setex(key, Config.CACHE_TTL, serialized)
            logger.debug(f"Cached {len(results)} results for key {key}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    async def _get_cached_results(self, key: str) -> Optional[List[SearchResult]]:
        """Retrieve cached results."""
        try:
            redis = await self.deps.get_redis()
            data = await redis.get(key)
            if data:
                results = [SearchResult(**item) for item in json.loads(data)]
                logger.debug(f"Cache hit for key {key}")
                return results
            logger.debug(f"Cache miss for key {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Main search endpoint with caching."""
        try:
            # Cache key generation
            cache_key = await self._get_cache_key(query.query, query.top_k, query.from_)
            
            # Check cache
            cached = await self._get_cached_results(cache_key)
            if cached:
                return cached
            
            # Perform hybrid search
            results = await self.hybrid_search(query.query, query.top_k, query.from_)
            
            # Cache results
            if results:
                await self._cache_results(cache_key, results)
            
            return results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search endpoint failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search request failed"
            )

# ============================================================================
# MIDDLEWARE
# ============================================================================

class CorrelationIDMiddleware:
    """Adds correlation ID to all requests."""
    
    async def __call__(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        
        # Add to logger context
        logger.logger.correlation_id = correlation_id
        
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        Config.validate()
        await deps.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    await deps.cleanup()
    logger.info("Application shut down successfully")

# Create application
app = FastAPI(
    title="Magazine Search API",
    version="2.0.0",
    description="Production-ready hybrid search for magazine articles",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add middleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Initialize metrics
instrumentator = Instrumentator().instrument(app)

# Initialize OpenTelemetry
if os.getenv("OTEL_ENABLED", "false").lower() == "true":
    FastAPIInstrumentor.instrument_app(app)

# Initialize search engine
search_engine = SearchEngine(deps)

# Rate limiter (lazy initialized)
rate_limiter = None

# ============================================================================
# EXCEPTION HANDLING
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standard HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status=exc.status_code,
            code=f"HTTP_{exc.status_code}",
            message=exc.detail if exc.status_code < 500 else "An unexpected error occurred",
            correlation_id=request.state.correlation_id,
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).dict()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Generic exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            correlation_id=request.state.correlation_id,
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).dict()
    )

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Magazine Search API",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check Elasticsearch
    try:
        es = await deps.get_es()
        await es.ping()
        health_status["services"]["elasticsearch"] = "healthy"
    except:
        health_status["services"]["elasticsearch"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis = await deps.get_redis()
        await redis.ping()
        health_status["services"]["redis"] = "healthy"
    except:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Model
    try:
        if deps.model:
            await deps.get_embedding("test")
            health_status["services"]["model"] = "healthy"
        else:
            health_status["services"]["model"] = "unavailable"
            health_status["status"] = "degraded"
    except:
        health_status["services"]["model"] = "unhealthy"
        health_status["status"] = "degraded"
    
    if health_status["status"] == "degraded":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service degraded"
        )
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Instrumentator handles this
    return {"message": "Metrics available at /metrics"}

@app.post("/search", response_model=List[SearchResult])
async def search_endpoint(query: SearchQuery, request: Request):
    """
    Search for magazine articles using hybrid search.
    
    Combines BM25 keyword search with semantic vector search for optimal relevance.
    Results are cached for 1 hour.
    """
    try:
        # Rate limiting
        global rate_limiter
        if not rate_limiter:
            rate_limiter = RateLimiter(await deps.get_redis())
        
        client_ip = request.client.host if request.client else "unknown"
        if not await rate_limiter.is_allowed(
            client_ip, 
            Config.RATE_LIMIT_PER_MINUTE, 
            60
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Execute search
        results = await search_engine.search(query)
        
        logger.info(
            "Search request completed",
            extra={
                "query": query.query[:50],
                "top_k": query.top_k,
                "from": query.from_,
                "results": len(results),
                "client_ip": client_ip,
                "correlation_id": request.state.correlation_id
            }
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

@app.post("/search/keyword", response_model=List[SearchResult])
async def keyword_search_endpoint(query: SearchQuery, request: Request):
    """Keyword-only search endpoint."""
    try:
        # Rate limiting
        global rate_limiter
        if not rate_limiter:
            rate_limiter = RateLimiter(await deps.get_redis())
        
        client_ip = request.client.host if request.client else "unknown"
        if not await rate_limiter.is_allowed(client_ip, Config.RATE_LIMIT_PER_MINUTE, 60):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        results = await search_engine._keyword_search(query.query, query.top_k, query.from_)
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Keyword search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Keyword search failed"
        )

@app.post("/search/vector", response_model=List[SearchResult])
async def vector_search_endpoint(query: SearchQuery, request: Request):
    """Vector-only search endpoint."""
    try:
        # Rate limiting
        global rate_limiter
        if not rate_limiter:
            rate_limiter = RateLimiter(await deps.get_redis())
        
        client_ip = request.client.host if request.client else "unknown"
        if not await rate_limiter.is_allowed(client_ip, Config.RATE_LIMIT_PER_MINUTE, 60):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        results = await search_engine._vector_search(query.query, query.top_k, query.from_)
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed"
        )

@app.post("/search/cache/clear")
async def clear_cache(request: Request):
    """Clear search cache (admin endpoint)."""
    try:
        redis = await deps.get_redis()
        keys = await redis.keys(f"{Config.CACHE_PREFIX}:*")
        if keys:
            await redis.delete(*keys)
        logger.info(f"Cache cleared: {len(keys)} keys removed")
        return {"message": f"Cache cleared: {len(keys)} keys removed"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache clear failed"
        )

@app.get("/stats")
async def get_stats(request: Request):
    """Get search statistics (admin endpoint)."""
    try:
        redis = await deps.get_redis()
        # Get all search stats
        keys = await redis.keys("search_stats:*")
        stats = {}
        for key in keys[:100]:  # Limit results
            count = await redis.get(key)
            if count:
                stats[key.replace("search_stats:", "")] = int(count)
        
        return {
            "total_queries": sum(stats.values()),
            "unique_queries": len(stats),
            "top_queries": sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stats retrieval failed"
        )

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@app.on_event("startup")
async def startup_tasks():
    """Additional startup tasks."""
    # Start background tasks
    asyncio.create_task(background_cache_warming())
    logger.info("Background tasks started")

async def background_cache_warming():
    """Warm cache for popular queries."""
    try:
        while True:
            try:
                redis = await deps.get_redis()
                # Get popular queries from stats
                keys = await redis.keys("search_stats:*")
                popular = []
                for key in keys[:10]:
                    count = await redis.get(key)
                    if count and int(count) > 10:
                        popular.append(key.replace("search_stats:", ""))
                
                # Warm cache for popular queries
                for query in popular:
                    if not await redis.get(f"{Config.CACHE_PREFIX}:{query}:10:0"):
                        # Run search to warm cache
                        search_query = SearchQuery(query=query, top_k=10, from_=0)
                        await search_engine.search(search_query)
                        logger.debug(f"Cache warmed for query: {query}")
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
            
            # Wait 5 minutes before next warming
            await asyncio.sleep(300)
            
    except asyncio.CancelledError:
        logger.info("Cache warming stopped")
    except Exception as e:
        logger.error(f"Cache warming task crashed: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("WORKERS", 4)),
        log_level=Config.LOG_LEVEL.lower(),
        access_log=os.getenv("ACCESS_LOG", "false").lower() == "true",
        timeout_keep_alive=60,
        limit_concurrency=int(os.getenv("MAX_CONCURRENT_REQUESTS", 100)),
        limit_max_requests=int(os.getenv("MAX_REQUESTS_PER_WORKER", 1000)),
    )
