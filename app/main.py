import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from elasticsearch import AsyncElasticsearch
from typing import List, Optional
import aioredis
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Elasticsearch connection
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = int(os.getenv("ES_PORT", 9200))
es = AsyncElasticsearch([{'host': ES_HOST, 'port': ES_PORT}])

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# Sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    from_: int = Field(default=0, ge=0)
    category: Optional[str] = None


class SearchResult(BaseModel):
    id: str
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
    return model.encode(text).tolist()


@app.post("/search", response_model=List[SearchResult])
async def hybrid_search(search_query: SearchQuery, background_tasks: BackgroundTasks):
    query = search_query.query
    top_k = search_query.top_k
    from_ = search_query.from_
    category = search_query.category

    # Check cache
    cache_key = f"search:{query}:{top_k}:{from_}:{category}"
    cached_results = await redis.get(cache_key)
    if cached_results:
        return json.loads(cached_results)

    filters, parsed_query = extract_filters(query, category)

    # Generate query vector
    query_vector = await get_embedding(parsed_query)

    # Construct the hybrid search query
    es_query = {
        "size": top_k,
        "from": from_,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": parsed_query,
                            "fields": ["title^2", "author", "content", "summary"],
                            "type": "best_fields",
                            "tie_breaker": 0.3
                        }
                    }
                ],
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                ],
                "filter": filters.get("filter", [])
            }
        }
    }

    try:
        response = await es.search(index="magazine_content", body=es_query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Elasticsearch error: {str(e)}")

    results = []
    for hit in response['hits']['hits']:
        results.append(SearchResult(
            id=hit['_id'],
            title=hit['_source']['title'],
            author=hit['_source']['author'],
            content=hit['_source']['content'][:200] +
            "...",  # Truncate content for preview
            score=hit['_score'],
            category=hit['_source']['category'],
            updated_at=hit['_source']['updated_at']
        ))

    # Cache results
    await redis.setex(cache_key, 3600, json.dumps([result.dict() for result in results]))

    # Update search stats in the background
    background_tasks.add_task(update_search_stats, query)

    return results


@app.on_event("startup")
async def startup_event():
    # Any startup events (if needed)
    pass


@app.on_event("shutdown")
async def shutdown_event():
    await es.close()
    await redis.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
