from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace

from config import get_settings


def get_tracer() -> trace.Tracer:
    """Return the package tracer."""

    return trace.get_tracer(get_settings().tracer_name)


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Iterator[None]:
    """Create an OpenTelemetry span without making export required."""

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as active_span:
        for key, value in attributes.items():
            active_span.set_attribute(key, value)
        yield
