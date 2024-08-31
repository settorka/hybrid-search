import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from elasticsearch import AsyncElasticsearch
from typing import List
import aioredis

app = FastAPI()

# Elasticsearch connection
ES_HOST = os.getenv("ES_HOST", "elasticsearch")
ES_PORT = int(os.getenv("ES_PORT", 9200))
es = AsyncElasticsearch([{'host': ES_HOST, 'port': ES_PORT}])

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    id: str
    title: str
    author: str
    content: str
    score: float


async def update_search_stats(query: str):
    """Background task to update search statistics in Redis."""
    await redis.incr(f"search_stats:{query}")


@app.post("/search", response_model=List[SearchResult])
async def hybrid_search(search_query: SearchQuery, background_tasks: BackgroundTasks):
    query = search_query.query
    top_k = search_query.top_k

    # Check cache
    cache_key = f"search:{query}:{top_k}"
    cached_results = await redis.get(cache_key)
    if cached_results:
        return json.loads(cached_results)

    # RRF query combining BM25 and ELSER
    es_query = {
        "size": top_k,
        "query": {
            "rrf": {
                "window_size": 50,
                "rank_constant": 20,
                "queries": [
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 1
                            }
                        }
                    },
                    {
                        "text_expansion": {
                            "content_vector": {
                                "model_id": ".elser_model_1",
                                "model_text": query
                            }
                        }
                    }
                ]
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
        # Fetch magazine info for each content hit
        magazine_info = await es.get(index="magazine_info", id=hit['_source']['magazine_id'])
        results.append(SearchResult(
            id=hit['_id'],
            title=magazine_info['_source']['title'],
            author=magazine_info['_source']['author'],
            content=hit['_source']['content'],
            score=hit['_score']
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
