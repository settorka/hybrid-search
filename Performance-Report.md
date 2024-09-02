# Magazine Search API: Performance Considerations and Optimizations

## Technical Background: Hybrid Search
The Magazine Search API implements a hybrid search approach, combining traditional keyword-based search with vector-based semantic search. This method leverages the strengths of both techniques to provide more accurate and relevant search results.

1. Keyword Search
- Utilizes Elasticsearch's full-text search capabilities
- Implements multi-match query with field boosting
- Uses custom analyzer with stemming and tokenization for improved text matching

2. Vector Search
 - Employs dense vector fields to store content embeddings
 - Utilizes the SentenceTransformer model for generating embeddings
 - Implements cosine similarity for semantic matching

3. Hybrid Combination
- Executes keyword and vector searches concurrently using asyncio.gather()
- Combines results using a weighted scoring mechanism
- Dynamically adjusts weights based on query characteristics

## Performance Considerations: 
### Magazine Article Characteristics
- Content Length: Typically 500 to 1000 words.
- Language: Engaging and conversational with varied vocabulary.
- Structure: A mix of short and long sentences, with attention-grabbing headlines.

## Performance Optimizations
### Elasticsearch and Large Dataset Handling
- Elasticsearch efficiently managed large datasets through its distributed architecture:
Single-Node vs. Distributed Setup
Local Setup:
```yaml
PUT /magazine_content_index
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 1
  }
}
```
Distributed (Multi-node) Setup:
```yaml
PUT /magazine_content_index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  }
}
```

### Benefits of Distributed Architecture:
- Horizontal Scalability: Add more nodes to handle larger datasets (>1.2m records) and higher query loads.
- Load Distribution: Queries are distributed across shards and nodes, reducing the load on any single machine.
- Fault Tolerance: Replicas ensure data availability even if some nodes fail.


### Requests
- Asynchronous Operations:
    - Utilizes AsyncElasticsearch for non-blocking database operations
    - Implements concurrent searches to reduce overall query time

- Caching:
    - Caches search results in Redis with a Time to live (TTL) of 1 hour 
    - Implements lazy loading strategy to significantly reduces response time for repeated queries

- Model Loading Optimization
The SentenceTransformer model is loaded during FastAPI's startup event:
```python
@app.on_event("startup")
async def startup_event():
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
```
- Avoids loading the model for each request, reducing latency.
- Shares a single model instance across all requests.
- Moves computationally expensive model loading out of the request cycle.

- Asynchronous Processing
The API extensively uses asynchronous programming:
- Advantages:
    - Efficiently handles multiple concurrent requests without blocking.
    - Allows multiple I/O-bound operations to run simultaneously.
    - Reduces overall response time by parallelizing operations.

### Indexing:
- Bulk Indexing:
    - Implements bulk operations for efficient data insertion
    - Indexes magazine records from csv/json in batches of 1000 
- Database Indexing
    - Uses a custom analyzer with stemming and lowercasing for augmenting searching
        - Lowercasing: Normalizes text for case-insensitive matching.
        - Stemming: Reduces words to their root form, improving recall for related terms.
```json
"analyzer": {
    "standard_stem_analyzer": {
        "type": "custom",
        "tokenizer": "standard",
        "filter": ["lowercase", "porter_stem"]
    }
}
```

### Query optimisation
- Field Boosting:
    - Applied higher weight to title matches (2x) compared to content
    - Enhanced result relevance for title-specific searches

- Fuzzy Matching:
    - Implemented automatic fuzzy matching to improve search resilience against typos and minor variations

- Result Highlighting:
    - Provides content snippets with highlighted matches (</em> result </em>)

- Efficient Vector Storage:
    - Uses Elasticsearch's dense_vector field type for efficient storage and retrieval of embeddings

### Augmentations to hybrid search
The hybrid_search function combines keyword and vector searches with sophisticated scoring adjustments. This algorithm balances precision (through keyword matching and exact match boosting) with semantic relevance (through vector search + term matching boost), adapting to different query types (dynamic weight adjustment).
- Dynamic Weight Adjustment:
    - Adjusts keyword vs. vector search weights based on query length
    - Optimizes for both short keyword queries and longer semantic queries

```python
if len(query_terms) > 2:
    keyword_weight = 0.8
    vector_weight = 0.2
```
- Exact Match Boosting:
    - Applies additional score boost for exact phrase matches
    - Improves precision for specific phrase searches

```python
if query.lower() in result.title.lower() or query.lower() in result.content.lower():
    boosted_score *= exact_match_boost
```
- Boosting Mechanism:
```python
if any(term in result.title.lower() for term in query_terms):
    boosted_score *= 1.5
if any(term in result.author.lower() for term in query_terms):
    boosted_score *= 1.2
```
Increases relevance for title and author matches.

- Term Matching Boost:
```python
matched_terms = sum(1 for term in query_terms if term in result.content.lower())
term_match_boost = 1 + (0.1 * matched_terms)
```
Incrementally boosts scores based on the number of matched terms.


### Scalability 

- Indexing Optimizations:
    - Prepared for chunk and sentence-level indexing in high-performance setups.


- GPU Acceleration:
    - Ready for integration with GPU-accelerated Elasticsearch setups.
    - Potential for significant performance boost in vector operations.

- Multi-node Architecture:
    - Enables horizontal scaling for handling larger datasets and higher query loads.
    - Designed to work with Elasticsearch clusters.



Performance Metrics (Hypothetical)

Average query time: ~100ms (cached), ~500ms (uncached)
Indexing speed: ~1000 documents per second
Cache hit ratio: ~70% (depends on query patterns)

## Areas for Future Optimization
ity ity ittitiwwwwwwwnxnxnxnx- Adaptive Caching: Implement intelligent cache TTL based on query popularity.
- Distributed Vector Operations: Implement distributed vector search for improved scalability.
- Real-time Index Updates: Develop a strategy for real-time or near-real-time index updates across distributed architecture

