import argparse
import statistics
import time

import httpx


def main() -> None:
    """Run a bounded local smoke performance check."""

    args = parse_args()
    latencies: list[float] = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for index in range(args.requests):
            started = time.perf_counter()
            query = f"{args.query} {index}" if args.unique else args.query
            response = client.post(
                "/search",
                json={"query": query, "top_k": args.top_k},
                headers={"x-request-id": f"perf-{index}"},
            )
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000)
    print(
        {
            "requests": args.requests,
            "min_ms": round(min(latencies), 3),
            "p50_ms": round(statistics.median(latencies), 3),
            "p95_ms": round(percentile(latencies, 95), 3),
            "max_ms": round(max(latencies), 3),
            "req_per_second": round(args.requests / (sum(latencies) / 1000), 3),
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
    return parser.parse_args()


def percentile(values: list[float], target: int) -> float:
    """Return nearest-rank percentile."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((target / 100) * (len(ordered) - 1))))
    return ordered[index]


if __name__ == "__main__":
    main()
