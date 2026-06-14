import argparse
import asyncio
from collections.abc import AsyncIterator

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from faker import Faker

from config import get_settings
from helpers.text import hash_embedding


async def main() -> None:
    """Create Elasticsearch indices and ingest Faker magazine records."""

    args = parse_args()
    settings = get_settings()
    fake = Faker()
    Faker.seed(args.seed)
    client = AsyncElasticsearch(settings.elasticsearch_url)
    refresh_interval = None if args.no_refresh_interval else args.refresh_interval
    if args.reset:
        await client.indices.delete(index=settings.magazine_info_index, ignore_unavailable=True)
        await client.indices.delete(index=settings.magazine_content_index, ignore_unavailable=True)
    await create_indices(client)
    if refresh_interval is not None:
        await client.indices.put_settings(
            index=settings.magazine_info_index,
            settings={"index": {"refresh_interval": refresh_interval}},
        )
        await client.indices.put_settings(
            index=settings.magazine_content_index,
            settings={"index": {"refresh_interval": refresh_interval}},
        )
    indexed, errors = await async_bulk(
        client,
        actions(settings, fake, args.count, args.paragraphs),
        chunk_size=args.batch_size,
        raise_on_error=False,
    )
    if refresh_interval is not None:
        await client.indices.put_settings(
            index=settings.magazine_info_index,
            settings={"index": {"refresh_interval": "1s"}},
        )
        await client.indices.put_settings(
            index=settings.magazine_content_index,
            settings={"index": {"refresh_interval": "1s"}},
        )
    await client.indices.refresh(index=settings.magazine_info_index)
    await client.indices.refresh(index=settings.magazine_content_index)
    await client.close()
    print(f"indexed={indexed} errors={len(errors)} records={args.count}")


def parse_args() -> argparse.Namespace:
    """Parse ingest arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--paragraphs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--refresh-interval",
        default="-1",
        help="Elasticsearch refresh_interval during ingest; use '-1' for none, or '1s'.",
    )
    parser.add_argument(
        "--no-refresh-interval",
        action="store_true",
        help="Do not change refresh_interval during ingest.",
    )
    return parser.parse_args()


async def create_indices(client: AsyncElasticsearch) -> None:
    """Create magazine indices when absent."""

    settings = get_settings()
    await client.options(ignore_status=400).indices.create(
        index=settings.magazine_info_index,
        mappings={
            "properties": {
                "id": {"type": "integer"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "publication_date": {"type": "date"},
                "category": {"type": "keyword"},
            }
        },
    )
    await client.options(ignore_status=400).indices.create(
        index=settings.magazine_content_index,
        mappings={
            "properties": {
                "id": {"type": "integer"},
                "magazine_id": {"type": "integer"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "publication_date": {"type": "date"},
                "category": {"type": "keyword"},
                "content": {"type": "text"},
                "content_version": {"type": "keyword"},
                "embedding_model_version": {"type": "keyword"},
                "vector_representation": {
                    "type": "dense_vector",
                    "dims": settings.embedding_dimension,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )


async def actions(
    settings, fake: Faker, count: int, paragraphs: int
) -> AsyncIterator[dict[str, object]]:
    """Yield bulk index actions."""

    categories = ["technology", "travel", "fashion", "food", "environment", "business"]
    for item_id in range(1, count + 1):
        category = fake.random_element(categories)
        title = fake.sentence(nb_words=6).rstrip(".")
        author = fake.name()
        publication_date = fake.date_between(start_date="-5y", end_date="today").isoformat()
        content = " ".join(fake.paragraphs(nb=paragraphs))
        vector = hash_embedding(f"{title} {category} {content}", settings.embedding_dimension)
        info = {
            "id": item_id,
            "title": title,
            "author": author,
            "publication_date": publication_date,
            "category": category,
        }
        content_doc = {
            **info,
            "magazine_id": item_id,
            "content": content,
            "content_version": settings.content_version,
            "embedding_model_version": settings.model_version,
            "vector_representation": vector,
        }
        yield {"_index": settings.magazine_info_index, "_id": str(item_id), "_source": info}
        yield {
            "_index": settings.magazine_content_index,
            "_id": str(item_id),
            "_source": content_doc,
        }


if __name__ == "__main__":
    asyncio.run(main())
