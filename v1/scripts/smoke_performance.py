import argparse
import random
import statistics
import time

import httpx


def main() -> None:
    """Run a bounded local smoke performance check."""

    args = parse_args()
    latencies: list[float] = []
    rate_limited = 0
    failed = 0
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for index in range(args.requests):
            query = f"{args.query} {index}" if args.unique else args.query
            attempt = 0
            while True:
                started = time.perf_counter()
                response = client.post(
                    "/search",
                    json={"query": query, "top_k": args.top_k},
                    headers={"x-request-id": f"perf-{index}"},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000

                if response.status_code == 429:
                    rate_limited += 1
                    attempt += 1
                    if attempt > args.max_retries:
                        failed += 1
                        break
                    # Backoff with jitter; intended to respect bounded admission limits.
                    time.sleep(args.backoff_ms / 1000 + random.random() * (args.backoff_ms / 1000))
                    continue

                response.raise_for_status()
                latencies.append(elapsed_ms)
                break

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000)
    print(
        {
            "requests": args.requests,
            "ok": len(latencies),
            "rate_limited": rate_limited,
            "failed": failed,
            "min_ms": round(min(latencies), 3) if latencies else None,
            "p50_ms": round(statistics.median(latencies), 3) if latencies else None,
            "p95_ms": round(percentile(latencies, 95), 3) if latencies else None,
            "max_ms": round(max(latencies), 3) if latencies else None,
            "req_per_second": (
                round(len(latencies) / (sum(latencies) / 1000), 3) if latencies else 0.0
            ),
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse performance arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--query", default="technology search databases vectors")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-ms", type=float, default=50.0)
    return parser.parse_args()


def percentile(values: list[float], target: int) -> float:
    """Return nearest-rank percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((target / 100) * (len(ordered) - 1))))
    return ordered[index]


if __name__ == "__main__":
    main()
