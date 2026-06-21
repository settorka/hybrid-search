import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def main() -> None:
    """Run a fixed-rate HTTP load test and write a compact JSON report."""

    args = parse_args()
    total_requests = int(args.rate_per_minute * args.duration_seconds / 60)
    interval = 60 / args.rate_per_minute
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
    )
    semaphore = asyncio.Semaphore(args.max_in_flight)
    results: list[dict[str, object]] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout, limits=limits) as client:
        tasks = []
        for index in range(total_requests):
            scheduled_at = started + (index * interval)
            sleep_for = scheduled_at - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            tasks.append(
                asyncio.create_task(
                    send_request(client, semaphore, index, args.query, args.top_k)
                )
            )
        results = await asyncio.gather(*tasks)

    finished = time.perf_counter()
    report = summarize(
        results=results,
        target_rate_per_minute=args.rate_per_minute,
        duration_seconds=args.duration_seconds,
        actual_elapsed_seconds=finished - started,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    index: int,
    query: str,
    top_k: int,
) -> dict[str, object]:
    """Send one request and capture bounded result metadata."""

    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/search",
                json={"query": f"{query} {index}", "top_k": top_k},
                headers={"x-request-id": f"load-{index}"},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            return {
                "status": response.status_code,
                "latency_ms": elapsed_ms,
                "error": response.json().get("error") if response.status_code >= 400 else None,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return {
                "status": 0,
                "latency_ms": elapsed_ms,
                "error": type(exc).__name__,
            }


def summarize(
    results: list[dict[str, object]],
    target_rate_per_minute: int,
    duration_seconds: int,
    actual_elapsed_seconds: float,
) -> dict[str, object]:
    """Summarize load results."""

    latencies = [float(result["latency_ms"]) for result in results]
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        error = result.get("error")
        if error:
            error_name = str(error)
            error_counts[error_name] = error_counts.get(error_name, 0) + 1
    return {
        "target_rate_per_minute": target_rate_per_minute,
        "duration_seconds": duration_seconds,
        "actual_elapsed_seconds": round(actual_elapsed_seconds, 3),
        "sent": len(results),
        "actual_rate_per_minute": round(len(results) / actual_elapsed_seconds * 60, 3),
        "status_counts": status_counts,
        "error_counts": error_counts,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else None,
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "p95": round(percentile(latencies, 95), 3) if latencies else None,
            "p99": round(percentile(latencies, 99), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
    }


def percentile(values: list[float], target: int) -> float:
    """Return nearest-rank percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((target / 100) * (len(ordered) - 1))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    """Parse load test arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--query", default="technology search databases vectors")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rate-per-minute", type=int, default=10_000)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-in-flight", type=int, default=1000)
    parser.add_argument("--max-connections", type=int, default=1000)
    parser.add_argument("--max-keepalive-connections", type=int, default=200)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
