import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from elasticsearch import AsyncElasticsearch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings


async def main() -> None:
    """Run Elasticsearch alias rollout hooks for v2."""

    args = parse_args()
    settings = get_settings()
    client = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        if args.command == "verify":
            await verify_index(client, args.index, settings.embedding_dimension)
        elif args.command == "cutover":
            await cutover(client, settings.magazine_content_index, args.index, args.force)
        elif args.command == "rollback":
            await cutover(client, settings.magazine_content_index, args.index, True)
        else:
            raise ValueError(f"unknown command {args.command}")
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    """Parse rollout hook arguments."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--index", required=True)

    cutover_parser = subparsers.add_parser("cutover")
    cutover_parser.add_argument("--index", required=True)
    cutover_parser.add_argument("--force", action="store_true")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--index", required=True)

    return parser.parse_args()


async def verify_index(client: AsyncElasticsearch, index: str, embedding_dimension: int) -> None:
    """Verify document count and vector mapping before cutover."""

    exists = await client.indices.exists(index=index)
    if not exists:
        raise RuntimeError(f"index does not exist: {index}")
    count = await client.count(index=index)
    if int(count["count"]) <= 0:
        raise RuntimeError(f"index has no documents: {index}")
    mapping = await client.indices.get_mapping(index=index)
    properties = mapping[index]["mappings"]["properties"]
    vector = properties.get("vector_representation", {})
    if vector.get("dims") != embedding_dimension:
        raise RuntimeError(f"vector dims mismatch: {vector.get('dims')} != {embedding_dimension}")
    print({"status": "ready", "index": index, "docs": int(count["count"])})


async def cutover(
    client: AsyncElasticsearch,
    alias: str,
    target_index: str,
    force: bool,
) -> None:
    """Atomically move the active read alias to a verified index."""

    now = datetime.now()
    if not force and (now.hour != 23 or now.minute != 0):
        raise RuntimeError("cutover outside configured 23:00 window; use --force for drills")
    alias_exists = await client.indices.exists_alias(name=alias)
    actions: list[dict[str, object]] = []
    if alias_exists:
        current = await client.indices.get_alias(name=alias)
        for index in current:
            actions.append({"remove": {"index": index, "alias": alias}})
    actions.append({"add": {"index": target_index, "alias": alias}})
    await client.indices.update_aliases(actions=actions)
    print({"status": "active", "alias": alias, "index": target_index})


if __name__ == "__main__":
    asyncio.run(main())
