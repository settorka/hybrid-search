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
# model = SentenceTransformer("all-MiniLM-L6-v2")

MAGAZINE_INFO_INDEX = "magazine_info"
MAGAZINE_CONTENT_INDEX = "magazine_content"
# TTL for caching search results in seconds (e.g., cache results for 1 hour)
CACHE_TTL = 3600

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    from_: int = Field(default=0, ge=0)
    category: Optional[str] = None


class SearchResult(BaseModel):
    id: int
    title: str
    author: str
    content: str
    score: float
    category: str
    updated_at: str

async def update_search_stats(query: str):
    """Background task to update search statistics in Redis."""
    await redis.incr(f"search_stats:{query}")


def extract_filters(query: str, category: Optional[str]):
    filters = {}
    if category:
        filters["filter"] = [
            {"term": {"category.keyword": {"value": category}}}]
    return filters, query

async def get_embedding(text: str):
    model = Request.app.state.model
    return model.encode(text).tolist()

async def keyword_search(query: str, top_k: int = 10, from_: int = 0):
    try:
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

        response = await es.search(index=MAGAZINE_INFO_INDEX, body=es_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            highlight = hit.get('highlight', {})
            results.append(SearchResult(
                id=hit['_id'],
                title=highlight.get('title', [source['title']])[0],
                author=highlight.get('author', [source['author']])[0],
                content=highlight.get('content', [source['content'][:150] + "..."])[0],
                score=hit['_score'],
                category=source.get('category', ''),
                updated_at=source.get('updated_at', '')
            ))
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {str(e)}")

async def vector_search(query: str, top_k: int = 10, from_: int = 0):
    try:
        model = Request.app.state.model
        # Generate embedding for the query
        query_vector = model.encode(query).tolist()

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
            "size": top_k,
            "from": from_,
            "_source": ["id", "title", "author", "content", "category", "updated_at"],
            "highlight": {
                "fields": {
                    "title": {},
                    "author": {},
                    "content": {"fragment_size": 150, "number_of_fragments": 1}
                }
            }
        }

        response = await es.search(index=MAGAZINE_CONTENT_INDEX, body=es_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            highlight = hit.get('highlight', {})
            results.append(SearchResult(
                id=source['id'],
                title=highlight.get('title', [source['title']])[0],
                author=highlight.get('author', [source['author']])[0],
                content=highlight.get('content', [source['content'][:150] + "..."])[0],
                score=hit['_score'],
                category=source.get('category', ''),
                updated_at=source.get('updated_at', '')
            ))
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {str(e)}")

async def hybrid_search(query: str, top_k: int = 10, from_: int = 0, keyword_weight: float = 0.7, vector_weight: float = 0.3, exact_match_boost: float = 2.0):
    """Perform a hybrid search by combining full-text and vector search 
    results with dynamic weighting and exact match boosting."""
    try:
        # Perform both searches concurrently
        keyword_results, vector_results = await asyncio.gather(
            keyword_search(query, top_k * 2, from_),
            vector_search(query, top_k * 2, from_)
        )
        
        query_terms = query.lower().split()
        
        # Adjust weights based on query length
        if len(query_terms) > 2:
            keyword_weight = 0.8
            vector_weight = 0.2

        # Create a dictionary to hold merged results
        combined_results = {}

        # Helper function to apply weights, boost, and term matching
        def apply_weights_and_boost(result, weight, is_keyword):
            if is_keyword:
                # Preserve the original keyword search scoring
                boosted_score = result.score * weight
                # Additional boost for title and author matches
                if any(term in result.title.lower() for term in query_terms):
                    boosted_score *= 1.5
                if any(term in result.author.lower() for term in query_terms):
                    boosted_score *= 1.2
            else:
                # For vector search, use the new scoring method
                boosted_score = result.score * weight
                matched_terms = sum(1 for term in query_terms if term in result.content.lower())
                term_match_boost = 1 + (0.1 * matched_terms)
                boosted_score *= term_match_boost

            # Apply exact match boost for both keyword and vector results
            if query.lower() in result.title.lower() or query.lower() in result.content.lower():
                boosted_score *= exact_match_boost
            
            return boosted_score

        # Process keyword search results
        for result in keyword_results:
            if result.id not in combined_results:
                result.score = apply_weights_and_boost(result, keyword_weight, is_keyword=True)
                combined_results[result.id] = result
            else:
                combined_results[result.id].score += apply_weights_and_boost(result, keyword_weight, is_keyword=True)

        # Process vector search results
        for result in vector_results:
            if result.id not in combined_results:
                result.score = apply_weights_and_boost(result, vector_weight, is_keyword=False)
                combined_results[result.id] = result
            else:
                combined_results[result.id].score += apply_weights_and_boost(result, vector_weight, is_keyword=False)
        # Convert combined results to a sorted list
        sorted_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        # Return top_k results
        return sorted_results[:top_k]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")
# production - multinode cluster with GPUs 
#indexes
# async def chunk_vector_search(query: str, top_k: int = 10, from_: int = 0):
#     query_vector = model.encode(query).tolist()
#     es_query = {
#         "query": {
#             "nested": {
#                 "path": "chunks",
#                 "query": {
#                     "script_score": {
#                         "query": {"match_all": {}},
#                         "script": {
#                             "source": "cosineSimilarity(params.query_vector, doc['chunks.chunk_vector']) + 1.0",
#                             "params": {"query_vector": query_vector}
#                         }
#                     }
#                 }
#             }
#         },
#         "size": top_k,
#         "from": from_
#     }
#     response = await es.search(index=MAGAZINE_CONTENT_INDEX, body=es_query)
#     return [SearchResult(
#         id=hit['_source']['id'],
#         title=hit['_source']['title'],
#         author=hit['_source']['author'],
#         content=hit['_source']['chunks'][0]['chunk_content'][:200] + "...",
#         score=hit['_score'],
#         category=hit['_source']['category'],
#         updated_at=hit['_source']['updated_at']
#     ) for hit in response['hits']['hits']]

# async def sentence_vector_search(query: str, top_k: int = 10, from_: int = 0):
#     query_vector = model.encode(query).tolist()
#     es_query = {
#         "query": {
#             "nested": {
#                 "path": "sentences",
#                 "query": {
#                     "script_score": {
#                         "query": {"match_all": {}},
#                         "script": {
#                             "source": "cosineSimilarity(params.query_vector, doc['sentences.sentence_vector']) + 1.0",
#                             "params": {"query_vector": query_vector}
#                         }
#                     }
#                 }
#             }
#         },
#         "size": top_k,
#         "from": from_
#     }
#     response = await es.search(index=MAGAZINE_CONTENT_INDEX, body=es_query)
#     return [SearchResult(
#         id=hit['_source']['id'],
#         title=hit['_source']['title'],
#         author=hit['_source']['author'],
#         content=hit['_source']['sentences'][0]['sentence'],
#         score=hit['_score'],
#         category=hit['_source']['category'],
#         updated_at=hit['_source']['updated_at']
#     ) for hit in response['hits']['hits']]

# async def indexed_hybrid_search_rrf(query: str, top_k: int = 10, from_: int = 0, k: int = 60):
#     """Perform an enhanced hybrid search using all available search methods."""
#     try:
#         keyword_results, vector_results, chunk_results, sentence_results, tfidf_results = await asyncio.gather(
#             keyword_search(query, top_k, from_),
#             vector_search(query, top_k, from_),
#             chunk_vector_search(query, top_k, from_),
#             sentence_vector_search(query, top_k, from_)
#         )

#         # Create a dictionary to hold RRF scores
#         rrf_scores = {}

#         # Helper function to add or update RRF score
#         def update_rrf_score(doc_id, rank):
#             if doc_id not in rrf_scores:
#                 rrf_scores[doc_id] = 0.0
#             rrf_scores[doc_id] += 1.0 / (k + rank)

#         # Update scores for all search results
#         for result_set in [keyword_results, vector_results, chunk_results, sentence_results, tfidf_results]:
#             for rank, result in enumerate(result_set, start=1):
#                 update_rrf_score(result.id, rank)

#         # Combine results with their final RRF scores
#         combined_results = {}
#         for result_set in [keyword_results, vector_results, chunk_results, sentence_results, tfidf_results]:
#             for result in result_set:
#                 if result.id not in combined_results:
#                     combined_results[result.id] = result
#                 combined_results[result.id].score = rrf_scores[result.id]

#         # Sort combined results by the new RRF scores in descending order
#         sorted_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)

#         # Return top_k results
#         return sorted_results[:top_k]

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Enhanced hybrid search RRF error: {str(e)}")

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
    query = search_query.query
    top_k = search_query.top_k
    from_ = search_query.from_
    
    # Generate a unique cache key for the search query and parameters
    cache_key = f"search:{query}:{top_k}:{from_}"

    # Check if the search results are already cached
    cached_results = await get_cached_results(cache_key)
    if cached_results:
        return cached_results
    
    # results = await keyword_search(query, top_k, from_)
    # results = await vector_search(query, top_k, from_)
    results = await hybrid_search(query, top_k, from_)
    # results = await indexed_hybrid_search_rrf(query, top_k, from_)   #multinode cluster
    # Cache the search results
    await cache_search_results(cache_key, results)
    return results


@app.on_event("startup")
async def startup_event():
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
   
@app.on_event("shutdown")
async def shutdown_event():
    await es.close()
    await redis.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
