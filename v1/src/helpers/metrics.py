from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    """Prometheus metrics with bounded labels."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "hybrid_search_requests_total",
            "Total product requests.",
            ["status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "hybrid_search_request_latency_seconds",
            "Product request latency.",
            ["path", "outcome"],
            registry=self.registry,
        )
        self.cache_total = Counter(
            "hybrid_search_cache_total",
            "Cache outcomes.",
            ["outcome"],
            registry=self.registry,
        )
        self.dependency_latency = Histogram(
            "hybrid_search_dependency_latency_seconds",
            "Dependency latency.",
            ["dependency", "outcome"],
            registry=self.registry,
        )
        self.degraded_total = Counter(
            "hybrid_search_degraded_total",
            "Degraded responses.",
            ["reason"],
            registry=self.registry,
        )
        self.zero_results_total = Counter(
            "hybrid_search_zero_results_total",
            "Searches returning no results.",
            registry=self.registry,
        )
        self.rate_limited_total = Counter(
            "hybrid_search_rate_limited_total",
            "Rate limited requests.",
            ["scope"],
            registry=self.registry,
        )
        self.timeouts_total = Counter(
            "hybrid_search_timeouts_total",
            "Timeouts by component.",
            ["component"],
            registry=self.registry,
        )
        self.inflight_requests = Gauge(
            "hybrid_search_inflight_requests",
            "Current in-flight product requests.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render metrics in Prometheus format."""

        return generate_latest(self.registry)
