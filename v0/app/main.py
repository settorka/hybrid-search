import uvicorn
import os
import json, asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from elasticsearch import AsyncElasticsearch
from typing import List, Optional
import aioredis
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Elasticsearch connection
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_SCHEME = os.getenv("ES_SCHEME", "http")
es = AsyncElasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"])

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# Sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Macros used in search 
MAGAZINE_INFO_INDEX = "magazine_info"
MAGAZINE_CONTENT_INDEX = "magazine_content"

# TTL for caching search results in seconds (this case is 1 hour)
CACHE_TTL = 3600

# Data model for search query
class SearchQuery(BaseModel):
    query: str  # The search query string
    top_k: int = Field(default=10, ge=1, le=100)  # Number of top results to return
    from_: int = Field(default=0, ge=0)  # Offset for pagination
    category: Optional[str] = None  # Optional category filter

# Data model for search result
class SearchResult(BaseModel):
    id: int  # Unique identifier for the magazine article in elasticsearch
    title: str  # Title of the magazine article
    author: str  # Author of the magazine article
    content: str  # Content of the magazine article (will be truncated)
    score: float  # Relevancy score of the result (0-4)
    category: str  # Category of the magazine article
    updated_at: str  # Last updated timestamp 

async def update_search_stats(query: str):
    """
    Background task to update search statistics in Redis.
    
    Parameter:
    - query: The search query string.
    
    Returns:
    - None
    """
    await redis.incr(f"search_stats:{query}")  # Increments the query count in Redis


def extract_filters(query: str, category: Optional[str]):
    """
    Extracts filters for the Elasticsearch query to refine magazine article search.
    
    Parameters:
    - query: The magazine article search query string.
    - category: Optional magazine category filter (e.g., "Technology", "Fashion", "Travel").
    
    Returns:
    - filters: A dictionary of filters for the Elasticsearch query on magazine articles.
    - query: The modified magazine search query string.
    """
    filters = {}
    if category:
        # Add a filter for the specified magazine category
        filters["filter"] = [
            {"term": {"category.keyword": {"value": category}}}
        ]
    return filters, query

async def get_embedding(text: str):
    """
    Generates an embedding for the provided magazine article text.
    
    Parameter:
    - text: The magazine article text to generate an embedding for.
    
    Returns:
    - A list representing the embedding vector of the magazine article text.
    """
    return model.encode(text).tolist()  # Encode the magazine article text using the model

async def keyword_search(query: str, top_k: int = 10, from_: int = 0):
    """
    Performs a keyword-based search on title, content or author using Elasticsearch (BM25).

    Parameters:
    - query: The search query string.
    - top_k: Number of top results to return.
    - from_: Offset for pagination.

    Returns:
    - A list of SearchResult objects containing the top (top_k) results.
    """
    
    try:
        es_query = { #elastisearch query specification
            "size": top_k,  # Number of results to retrieve
            "from": from_,  # Pagination offset
            "query": {
                "multi_match": {
                    "query": query,  # The search query string
                    "fields": ["title^2", "author", "content"],  # Fields to search in with boost for title
                    "type": "best_fields",  # Search type
                    "fuzziness": "AUTO",  # Automatic handling of typos
                    "prefix_length": 2,  # Minimum prefix length for fuzzy matching
                    "minimum_should_match": "75%"  # Minimum match threshold
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "author": {},
                    "content": {"fragment_size": 150, "number_of_fragments": 1}  # Snippet character length for content
                }
            }
        }

        # Execute the search query on the Elasticsearch index using ORM search function
        response = await es.search(index=MAGAZINE_INFO_INDEX, body=es_query)
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']  # Retrieve source document
            highlight = hit.get('highlight', {})  # Retrieve highlight information
            # Create SearchResult object with highlighted snippets
            results.append(SearchResult(
                id=hit['_id'],
                title=highlight.get('title', [source['title']])[0],
                author=highlight.get('author', [source['author']])[0],
                content=highlight.get('content', [source['content'][:150] + "..."])[0],  # Snippet content
                score=hit['_score'],  # Relevance score from Elasticsearch
                category=source.get('category', ''),
                updated_at=source.get('updated_at', '')
            ))
        return results
    except Exception as e:
        # Handle any exceptions and raise HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {str(e)}")
    
async def vector_search(query: str, top_k: int = 10, from_: int = 0):
    """
    Performs a vector-based semantic search on magazine articles using Elasticsearch.
    
    This function converts the query into a vector representation and uses cosine similarity
    to find semantically similar magazine articles, regardless of exact keyword matches.

    Parameters:
    - query: The search query string for finding relevant magazine articles.
    - top_k: Number of top magazine article results to return.
    - from_: Offset for pagination of magazine article results.

    Returns:
    - A list of SearchResult objects containing the top semantically similar magazine articles.
    """
    try:
        # Generate an embedding vector for the search query
        query_vector = model.encode(query).tolist()
        es_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},  # Match all magazine articles initially
                    "script": {
                        # Script for calculating cosine similarity between query and article vectors
                        # Cosine similarity measures the cosine of the angle between two vectors,
                        # providing a similarity score between -1 and 1. Adding 1.0 shifts the range to 0-2.
                        # This ensures all scores are positive, with higher values indicating greater similarity.
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": top_k,  # Number of magazine articles to retrieve
            "from": from_,  # Pagination offset for magazine articles
            "_source": ["id", "title", "author", "content", "category", "updated_at"],  # Magazine article fields to return
            "highlight": {
                "fields": {
                    "title": {},
                    "author": {},
                    "content": {"fragment_size": 150, "number_of_fragments": 1}  # Snippet settings for article content
                }
            }
        }

        # Execute the search query on the Elasticsearch index containing magazine articles
        response = await es.search(index=MAGAZINE_CONTENT_INDEX, body=es_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']  # Retrieve source magazine article
            highlight = hit.get('highlight', {})  # Retrieve highlighted snippets from the article
            # Create SearchResult object with highlighted snippets from the magazine article
            results.append(SearchResult(
                id=source['id'],
                title=highlight.get('title', [source['title']])[0],
                author=highlight.get('author', [source['author']])[0],
                content=highlight.get('content', [source['content'][:150] + "..."])[0],  # Article content snippet
                score=hit['_score'],  # Semantic similarity score from Elasticsearch
                category=source.get('category', ''),
                updated_at=source.get('updated_at', '')
            ))
        
        return results

    except Exception as e:
        # Handle any exceptions and raise HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Elasticsearch error in vector search: {str(e)}")

async def hybrid_search(query: str, top_k: int = 10, from_: int = 0, keyword_weight: float = 0.7, vector_weight: float = 0.3, exact_match_boost: float = 2.0):
    """
    Perform a hybrid search on magazine articles by combining full-text (keyword) and vector search 
    results with dynamic weighting and exact match boosting.
    This function employs several advanced  techniques to provide highly relevant results 
    tailored to the data model of magazine articles in Elasticsearch. 
    
    Sequence flow:
    1. Asynchronous Dual Search:
       - Uses asyncio.gather to concurrently execute both keyword and vector searches.
       - This approach significantly reduces total search time by running searches in parallel.

    2. Dynamic Weighting:
       - Assigns different weights to keyword (0.7) and vector (0.3) search results by default.
       - Adjusts these weights based on query length, favoring keyword search for longer queries.
       - This adapts the search strategy to different types of user queries.

    3. Exact Match Boosting:
       - Applies a substantial boost (2.0x) to results that exactly match the query in title or content.
       - This ensures that highly relevant, exact matches are prioritized in the results.

    4. Term Matching and Boosting:
       - keyword results: Boosts scores for matches in title (1.5x) and author (1.2x).
       - vector results: Applies a graduated boost based on the number of query terms found in the content.
       - This nuanced approach balances semantic similarity with keyword relevance.

    5. Result Combination and De-duplication:
       - Merges results from both search types, combining scores for articles found by both methods.
       - Ensures each unique article appears only once in the final results.

    6. Final Ranking:
       - Sorts the combined results based on the calculated relevance scores.
       - Returns the top_k most relevant magazine articles.

    Parameters:
    - query: The search query string for finding relevant magazine articles.
    - top_k: Number of top magazine article results to return.
    - from_: Offset for pagination of magazine article results.
    - keyword_weight: Initial weight for keyword search results (default 0.7).
    - vector_weight: Initial weight for vector search results (default 0.3).
    - exact_match_boost: Boost factor for exact query matches (default 2.0).

    Returns:
    - A sorted list of SearchResult objects containing the top hybrid-ranked magazine articles.
    """
    try:
        
        # Performs both keyword and vector searches concurrently
        keyword_results, vector_results = await asyncio.gather(
            keyword_search(query, top_k * 2, from_),
            vector_search(query, top_k * 2, from_)
        )
        
        query_terms = query.lower().split() # Splits query into terms
        
        # Dynamic Weighting
        # Adjusts search weights based on query complexity
        if len(query_terms) > 2:
            keyword_weight = 0.8  # Increase keyword weight for longer queries
            vector_weight = 0.2   # Decrease vector weight for longer queries
        # dictionary to handle result combination
        combined_results = {}

        # Helper function for applying weights, boosts, and term matching
        def apply_weights_and_boost(result, weight, is_keyword):
            if is_keyword:
                # Keyword search scoring
                boosted_score = result.score * weight
                # Title and author match boosting
                if any(term in result.title.lower() for term in query_terms):
                    boosted_score *= 1.5  # 50% boost for title matches
                if any(term in result.author.lower() for term in query_terms):
                    boosted_score *= 1.2  # 20% boost for author matches
            else:
                # Vector search scoring
                boosted_score = result.score * weight
                # Term matching boost for vector results
                matched_terms = sum(1 for term in query_terms if term in result.content.lower())
                term_match_boost = 1 + (0.1 * matched_terms)  # 10% boost per matched term
                boosted_score *= term_match_boost
            # Exact match boosting for both search types
            if query.lower() in result.title.lower() or query.lower() in result.content.lower():
                boosted_score *= exact_match_boost  # Double the score for exact matches

            return boosted_score
        
        # Process and combine keyword search results
        for result in keyword_results:
            if result.id not in combined_results:
                result.score = apply_weights_and_boost(result, keyword_weight, is_keyword=True)
                combined_results[result.id] = result
            else:
                # Add scores if the article was found by both search methods
                combined_results[result.id].score += apply_weights_and_boost(result, keyword_weight, is_keyword=True)
        # Process and combine vector search results
        for result in vector_results:
            if result.id not in combined_results:
                result.score = apply_weights_and_boost(result, vector_weight, is_keyword=False)
                combined_results[result.id] = result
            else:
                # Add scores if the article was found by both search methods
                combined_results[result.id].score += apply_weights_and_boost(result, vector_weight, is_keyword=False)
        # Final ranking
        # Sort the combined results based on the calculated relevance scores
        sorted_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        # Return the top_k most relevant magazine articles
        return sorted_results[:top_k]
    except Exception as e:
        # Error handling
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")
    
async def cache_search_results(cache_key: str, results: List[SearchResult]):
    """Cache search results in Redis."""
    # Serialize the search results to JSON and set them in Redis with a TTL
    await redis.set(cache_key, json.dumps([result.dict() for result in results]), ex=CACHE_TTL)

async def get_cached_results(cache_key: str):
    """Retrieve cached search results from Redis."""
    cached_results = await redis.get(cache_key)
    if cached_results:
        return [SearchResult(**item) for item in json.loads(cached_results)]
    return None

@app.post("/search", response_model=List[SearchResult])
async def search(search_query: SearchQuery):
    """
    Executes an asynchronous, cached hybrid search for magazine articles.

    Sequence flow:
    1. Extracts search parameters from the SearchQuery object.
    2. Generates a unique cache key based on the query and pagination parameters.
    3. Attempts to retrieve cached results using a lazy loading strategy.
    4. If cache miss occurs, performs an asynchronous hybrid search on the magazine index.
    5. Caches the fresh search results for future queries.
    6. Returns the search results as a list of SearchResult objects.

    Parameters:
    - search_query (SearchQuery): Contains:
      - query (str): The search terms for finding relevant magazine articles.
      - top_k (int): Number of results to return (default: 10, range: 1-100).
      - from_ (int): Starting offset for pagination (default: 0).

    Returns:
    - List[SearchResult]: A list of magazine article search results, each containing:
      id, title, author, content snippet, relevance score, category, and last updated timestamp.

    Asynchronous Operations:
    - get_cached_results(): Asynchronously checks for cached results in memory and Redis.
    - hybrid_search(): Asynchronously performs a combined keyword and vector search on the
      magazine article index, leveraging Elasticsearch's async parallel processing capabilities.
    - cache_search_results(): Asynchronously stores results in the caching system for future use.

    Note: The hybrid_search can be replaced with keyword_search or vector_search by
    uncommenting the respective lines, allowing for flexible search strategy selection.
    """
    # Extracts search parameters from the SearchQuery object
    query, top_k, from_ = search_query.query, search_query.top_k, search_query.from_
    
    # Generates a unique cache key for the query and pagination parameters
    cache_key = f"search:{query}:{top_k}:{from_}"

    # Checks if the results for this query are already cached to 
    # avoid redundant searches and speed up response times
    cached_results = await get_cached_results(cache_key)
    
    # If cache hit, return the cached results immediately
    if cached_results:
        return cached_results
    
    # If cache miss, perform a hybrid search to get fresh results
    # Hybrid search combines keyword and vector searches 
    # as specified in its function definition to improve result relevance
    results = await hybrid_search(query, top_k, from_)

    # Cache the fresh results for future queries to 
    # improve performance and reduce load
    await cache_search_results(cache_key, results)
    
    # Return the search results as 
    # a list of SearchResult objects
    return results

@app.on_event("startup")
async def startup_event():
    """
    Function to execute tasks during the startup phase of the application.

    Purpose:
    - Initializes resources needed throughout the application's lifecycle.

    Commented Code:
    - `app.state.model = SentenceTransformer("all-MiniLM-L6-v2")`
      This would load the "all-MiniLM-L6-v2" model from the SentenceTransformers library 
      into the application's state, making it available across different endpoints.

    Note:
    - Uncomment the line if you want the model to be loaded at startup.
    Ensure you have sufficient memory and processing power; 
    loading large models can be resource-intensive.
    """
    # Load resources or initialize components needed by the app
    # Uncomment the following line if the model is required to be loaded during startup
    # app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """
    Function to execute tasks during the shutdown phase of the application.

    Purpose:
    - Gracefully closes any open connections or resources to prevent memory leaks
      and ensure a clean shutdown of the application.

    Asynchronous Operations:
    - `es.close()`: Closes the Elasticsearch client connection.
    - `redis.close()`: Closes the Redis client connection.

    """
    # Closes any open connections or resources
    await es.close()  # Close the Elasticsearch client
    await redis.close()  # Close the Redis client

if __name__ == "__main__":
    """
    The main block to run the Hybrid Search API using Uvicorn.

    Configurations:
    - `host="0.0.0.0"`: The server will listen on all available IP addresses.
    - `port=8000`: The server will be accessible on port 8000.
    
    - Running this script will start the server, making it accessible at 
      http://localhost:8000 or the server's IP address.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)