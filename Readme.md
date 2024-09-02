# Hybrid Search API - Magazine Articles 

##  Overview
The Magazine Search API is built using FastAPI and integrates with Elasticsearch for search functionality and Redis for result caching. It employs a hybrid search approach, combining traditional keyword search with vector-based semantic search.

### Project Structure
After unzipping the project folder, you should see the following structure:
```sh
hybrid-search/
- Dockerfile
- Readme.md
- docker-compose.yml
- app/
    - create_magazine_indices.py
    - create_mock_magazine_data.py
    - elasticsearch_orm.py
    - insert_magazine_data.py
    - main.py
    - requirements.txt
```

### Key Files
```yaml
- Dockerfile: Defines the container for the FastAPI application
- docker-compose.yml: Orchestrates the services (FastAPI, Elasticsearch, Redis)
- app/main.py: Contains the main FastAPI application and search logic
- app/create_magazine_indices.py: Creates necessary Elasticsearch indices
- app/create_mock_magazine_data.py: Generates mock magazine data for n records
- app/insert_magazine_data.py: Inserts mock data into Elasticsearch
- app/elasticsearch_orm.py: Defines Elasticsearch mappings and models
- app/requirements.txt: Lists Python dependencies
```

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

## API Endpoint
- Search Endpoint for hybrid search
- URL: /search
- Method: POST
- Headers: Content-Type: application/json

### Request Body
```yaml
{
  "query": "string",
  "top_k": integer,
  "from_": integer,
  "category": "string (optional)"
}
```
### Response Body

Array of search results:


```yaml
[
  {
    "id": integer,
    "title": "string",
    "author": "string",
    "content": "string",
    "score": float,
    "category": "string",
    "updated_at": "string"
  },
  ... # top_k results
]
```

## Example Usage (Postman)
- Set URL to http://localhost:8000/search
- Set method to POST
- Add header: Content-Type: application/json
- In Body (raw, JSON), enter:

```yaml
{
  "query": "artificial intelligence",
  "top_k": 5,
  "from_": 0
}
```
- Send request

## Production Testing

Under `api-tests` and `distributed`, there are four types of tests designed to evaluate the magazine search API's performance and reliability in a production-like environment. 
These tests should be run when the application is load balanced across multiple nodes (e.g., using NGINX) to simulate real-world traffic distribution.

### Test Types and Commands

1. **Spike Test**: Simulates sudden, extreme load.
   ```bash
   locust -f locustfile.py --headless -u 1000 -r 1000 --run-time 1m
   ```

2. **Stress Test**: Gradually increases load to find breaking point.
   ```bash
   locust -f locustfile.py --headless -u 5000 -r 100 --run-time 30m
   ```

3. **Soak Test**: Evaluates stability over extended periods.
   ```bash
   locust -f locustfile.py --headless -u 200 -r 10 --run-time 48h
   ```

4. **Load Test**: Assesses performance under expected peak load.
   ```bash
   locust -f locustfile.py --headless -u 500 -r 50 --run-time 1h
   ```

### Running the Tests

1. Ensure Locust is installed:
   ```bash
   pip install locust
   ```

2. Prepare your `locustfile.py` with appropriate user behavior definitions.

3. Execute each test using the provided commands in a terminal.

### Key Parameters

- `-u`: Total number of users to simulate
- `-r`: Rate at which new users are spawned
- `--run-time`: Duration of the test
- `--headless`: Runs Locust without the web UI

Adjust user numbers and test durations based on production capacity and expected traffic patterns.

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
- Implements Redis caching using lazy loading (cache-aside)
    - Write Operation
        - When data is written or updated, the cache is updated first.
        - Data Store Update: After updating the cache, the change is written to the data store.
    - Read Operation: 
        - When data is requested, it is first fetched from the cache. 
        - If it’s not in the cache (a cache miss), the data is fetched from the data store and then stored in the cache for future requests.
- Cache TTL defined to minimize space usage (3600s default) 
- Calls `hybrid_search` if cache miss (request not stored in cache or made already)
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
- Purpose: Combines keyword search and vector search results to deliver a comprehensive set of search results based on both exact matches and semantic relevance.
- Inputs:
    - query (str): The search query string.
    - top_k (int): Number of top results to return (default is 10).
    - from_ (int): The starting index for the results (default is 0).
    - keyword_weight (float): Weighting factor for keyword search results (default is 0.7).
    - vector_weight (float): Weighting factor for vector search results (default is 0.3).
    - exact_match_boost (float): Boost factor for exact matches (default is 2.0).
- Functionality:
    - Executes keyword and vector searches concurrently.
    - Merges and scores results from both searches using weighted scoring.
    - Applies additional boosts for exact matches in titles and content.
- Returns: A list of SearchResult objects, sorted by the combined score of keyword and vector relevance.

### `keyword_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- **Purpose:** Executes a keyword-based search for the magazine articles on Elasticsearch.
- **Inputs:**
  - `query` (str): The search query string.
  - `top_k` (int): Number of top results to return (default is 10).
  - `from_` (int): The starting index for the results (default is 0).
- **Functionality:**
  - Constructs an Elasticsearch query that matches the query string against multiple fields (`title`, `author`, `content`).
  - Applies fuzziness and prefix length for more flexible matching.
- **Elasticsearch Query:**
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
- Executes search on `MAGAZINE_INFO_INDEX` macro for data model

### `vector_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
- Purpose: Performs a vector-based search using embeddings.
- Inputs:
    - query (str): The search query string.
    - top_k (int): Number of top results to return (default is 10).
    - from_ (int): The starting index for the results (default is 0).
- Functionality:
    - Generates an embedding for the query using a sentence transformer model.
    - Constructs an Elasticsearch query to score documents based on cosine similarity between the query vector and stored document vectors.
Generate Query Embedding: query_vector = model.encode(query).tolist()

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

## Advanced Features (need to be activated for GPUs / multi-node architecture/ cloud)
### Indexing Functions 
#### `chunk_vector_search(query: str, top_k: int = 10, from_: int = 0) -> List[SearchResult]`
-Purpose: Searches within document chunks using vector embeddings.
- Inputs:
    - query (str): The search query string.
    - top_k (int): Number of top results to return (default is 10).
    - from_ (int): The starting index for the results (default is 0).
- Functionality:Performs vector search on nested document chunks.
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
- Purpose: Searches within individual sentences using vector embeddings.
- Inputs:
    - query (str): The search query string.
    - top_k (int): Number of top results to return (default is 10).
    - from_ (int): The starting index for the results (default is 0).
- Functionality: Similar to chunk_vector_search, but operates on sentence-level vectors.


### `indexed_hybrid_search_rrf(query: str, top_k: int = 10, from_: int = 0, k: int = 60) -> List[SearchResult]`
- Purpose: 
    - Combines multiple search methods using Reciprocal Rank Fusion (RRF) and indexed strategies.  
    - Evaluates the search scores from multiple, previously ranked results to produce a unified result set. 
- Inputs:
    - query (str): The search query string.
    - top_k (int): Number of top results to return (default is 10).
    - from_ (int): The starting index for the results (default is 0).
    - k (int): Parameter for RRF scoring (default is 60).
- Functionality: 
    - Executes keyword, vector, chunk, and sentence searches concurrently.
    - Combines results using RRF to balance contributions from different search methods.
-RRF Scoring: score = 1.0 / (k + rank)

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
#### Local Setup
```yaml
PUT /my_index
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 1
  }
}
```

#### Distributed (Multi-node Setup)
```yaml
PUT /my_index
{
  "settings": {
    "number_of_shards": 3,  # Increase shards for better distribution and parallelism across nodes
    "number_of_replicas": 2 # Increase of replicas for high availability and fault tolerance of nodes
  }
}
``` 
2. Implement and optimize indexing for chunk and sentence vectors
3. Uncomment and adjust `indexed_hybrid_search_rrf` and related functions
4. Replace `hybrid_search` call in `search` function with `indexed_hybrid_search_rrf`


