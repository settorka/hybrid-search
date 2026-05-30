from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace

from config import Settings

_tracer_name = "hybrid_search_v1"


def configure_tracing(settings: Settings) -> None:
    """Configure tracing from runtime settings."""

    global _tracer_name
    _tracer_name = settings.tracer_name


def get_tracer() -> trace.Tracer:
    """Return the package tracer."""

    return trace.get_tracer(_tracer_name)


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Iterator[None]:
    """Create an OpenTelemetry span without making export required."""

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as active_span:
        for key, value in attributes.items():
            active_span.set_attribute(key, value)
        yield
