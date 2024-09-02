# Hybrid Search API - Magazine Articles 

## Architecture Overview

The Magazine Search API is built using FastAPI and integrates with Elasticsearch for search functionality and Redis for result caching. It employs a hybrid search approach, combining traditional keyword search with vector-based semantic search.

## Deployment

1. Ensure Docker and Docker Compose are installed; Docker daemon should be active.
2. Clone the repository.
3. Navigate to the project directory and run:

```bash
docker-compose up --build -d
docker-compose exec api python create_magazine_indices.py
docker-compose exec api python create_mock_magazine_data.py
docker-compose exec api python insert_magazine_data.py
```

This sets up a Docker environment with FastAPI, Elasticsearch, and Redis containers, creates necessary indices, and populates them with mock data.

## Core Components

### FastAPI Application
- Defined in `app = FastAPI()`
- Handles HTTP requests and responses

### Elasticsearch Client
- Asynchronous client: `AsyncElasticsearch`
- Connection: `es = AsyncElasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"])`

### Redis Client
- Asynchronous client: `aioredis`
- Connection: `redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)`

### Sentence Transformer Model
- Used for generating embeddings: `model = SentenceTransformer("all-MiniLM-L6-v2")`

## API Endpoint

### Search Endpoint
- **Route**: `@app.post("/search", response_model=List[SearchResult])`
- **Function**: `async def search(search_query: SearchQuery)`

#### Request Model
```python
class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    from_: int = Field(default=0, ge=0)
    category: Optional[str] = None
```

#### Response Model
```python
class SearchResult(BaseModel):
    id: int
    title: str
    author: str
    content: str
    score: float
    category: str
    updated_at: str
```

## Key Functions

### `search(search_query: SearchQuery) -> List[SearchResult]`
- Entry point for search requests
- Implements Redis caching
- Calls `hybrid_search` if cache miss
- Caching logic:
  ```python
  cache_key = f"search:{query}:{top_k}:{from_}"
  cached_results = await get_cached_results(cache_key)
  if cached_results:
      return cached_results
  # ... perform search if cache miss
  await cache_search_results(cache_key, results)
  ```

### `hybrid_search(query: str, top_k: int = 10, from_: int = 0, keyword_weight: float = 0.7, vector_weight: float = 0.3, exact_match_boost: float = 2.0) -> List[SearchResult]`
- Combines keyword and vector search results
- Concurrent execution:
  ```python
  keyword_results, vector_results = await asyncio.gather(
      keyword_search(query, top_k * 2, from_),
      vector_search(query, top_k * 2, from_)
  )
  ```
- Implements custom scoring logic, including term matching and exact match boosting

### `keyword_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- Elasticsearch query:
  ```python
  es_query = {
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
      "highlight": { ... }
  }
  ```
- Executes search on `MAGAZINE_INFO_INDEX`

### `vector_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- Generates query embedding: `query_vector = model.encode(query).tolist()`
- Elasticsearch query:
  ```python
  es_query = {
      "query": {
          "script_score": {
              "query": {"match_all": {}},
              "script": {
                  "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                  "params": {"query_vector": query_vector}
              }
          }
      },
      ...
  }
  ```
- Executes search on `MAGAZINE_CONTENT_INDEX`

## Caching Implementation

- Uses Redis for caching search results
- TTL defined by `CACHE_TTL` (default: 3600 seconds)
- Caching functions:
  - `cache_search_results(cache_key: str, results: List[SearchResult]) -> None`
    - Serializes results to JSON and stores in Redis with TTL
  - `get_cached_results(cache_key: str) -> Optional[List[SearchResult]]`
    - Retrieves and deserializes cached results if they exist

## Advanced Features (GPU/multi-node architecture)

### Indexing Functions (commented out in current implementation)

#### `chunk_vector_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- Performs vector search on document chunks
- Uses nested query for searching within chunk vectors
- Elasticsearch query structure:
  ```python
  es_query = {
      "query": {
          "nested": {
              "path": "chunks",
              "query": {
                  "script_score": {
                      "query": {"match_all": {}},
                      "script": {
                          "source": "cosineSimilarity(params.query_vector, doc['chunks.chunk_vector']) + 1.0",
                          "params": {"query_vector": query_vector}
                      }
                  }
              }
          }
      },
      ...
  }
  ```

#### `sentence_vector_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- Performs vector search on individual sentences
- Similar to `chunk_vector_search`, but uses `sentences` path in nested query

### `indexed_hybrid_search_rrf(query: str, top_k: int = 10, from_: int = 0, k: int = 60) -> List[SearchResult]`
- Implements Reciprocal Rank Fusion for result combination
- Combines results from multiple search methods:
  ```python
  keyword_results, vector_results, chunk_results, sentence_results = await asyncio.gather(
      keyword_search(query, top_k, from_),
      vector_search(query, top_k, from_),
      chunk_vector_search(query, top_k, from_),
      sentence_vector_search(query, top_k, from_)
  )
  ```
- RRF scoring: `score = 1.0 / (k + rank)`
- Sorts final results based on combined RRF scores

To utilize GPU/multi-node features:
1. Configure Elasticsearch for GPU acceleration or distributed setup
2. Implement and optimize indexing for chunk and sentence vectors
3. Uncomment and adjust `indexed_hybrid_search_rrf` and related functions
4. Replace `hybrid_search` call in `search` function with `indexed_hybrid_search_rrf`
5. Fine-tune `k` parameter for RRF scoring based on performance requirements

