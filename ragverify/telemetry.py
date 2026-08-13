"""OpenTelemetry spans, optional and no-op by default.

The in-process ``Tracer`` in ``trace.py`` answers "what is happening right
now" for a UI. It cannot answer the questions that matter once this runs
somewhere you are not watching: which stage consumed the latency, how deep the
hop tree went, which model call retried, how one slow run differs from a
thousand fast ones. Events are a flat list with no nesting and no duration;
spans are a tree with both.

The two are complements rather than alternatives, so this bridges rather than
replaces -- every ``Tracer`` event can become a span event on the active span,
and the UI keeps working unchanged.

Everything degrades to nothing when ``opentelemetry`` is not installed. That
is the default, and the no-op path costs a null check: a project that never
enables tracing should not pay for the ability to.

Attribute names follow the OpenTelemetry GenAI semantic conventions where they
exist (``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.*``), so
the spans are readable by tooling that already understands LLM traces instead
of only by something written for this project.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("ragverify.telemetry")

_TRACER = None
_ENABLED = False


def available() -> bool:
    try:
        import opentelemetry.trace  # noqa: F401

        return True
    except ImportError:
        return False


def configure(
    service_name: str = "ragverify",
    endpoint: str | None = None,
    console: bool = False,
) -> bool:
    """Set up a tracer provider. Returns whether tracing is active.

    ``endpoint`` defaults to ``OTEL_EXPORTER_OTLP_ENDPOINT``, so a deployment
    that already exports traces needs no code change here.
    """
    global _TRACER, _ENABLED

    if not available():
        log.info("opentelemetry not installed; tracing disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        from . import __version__

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name, "service.version": __version__})
        )

        endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        if console or not endpoint:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("ragverify")
        _ENABLED = True
        log.info("tracing enabled (endpoint=%s)", endpoint or "console")
        return True
    except Exception as exc:  # noqa: BLE001 - observability must not break the thing observed
        log.warning("could not configure tracing (%s); continuing without", exc)
        _TRACER, _ENABLED = None, False
        return False


def enabled() -> bool:
    return _ENABLED and _TRACER is not None


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, or do nothing when tracing is off.

    Yields the span (or ``None``) so callers can attach attributes learned
    during the block -- token counts and verdicts are only known at the end,
    and those are the attributes worth querying on.
    """
    if not enabled():
        yield None
        return

    from opentelemetry import trace

    with _TRACER.start_as_current_span(name) as current:  # type: ignore[union-attr]
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, _coerce(value))
        try:
            yield current
        except Exception as exc:
            # Record on the span, then re-raise: telemetry observes failures,
            # it does not swallow them.
            current.record_exception(exc)
            current.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise


def set_attributes(current: Any, **attributes: Any) -> None:
    if current is None:
        return
    for key, value in attributes.items():
        if value is not None:
            with contextlib.suppress(Exception):
                current.set_attribute(key, _coerce(value))


def add_event(name: str, **attributes: Any) -> None:
    """Attach an event to whatever span is currently active."""
    if not enabled():
        return
    from opentelemetry import trace

    current = trace.get_current_span()
    if current is not None:
        with contextlib.suppress(Exception):
            current.add_event(name, {k: _coerce(v) for k, v in attributes.items() if v is not None})


def _coerce(value: Any) -> Any:
    """OTel attributes accept only scalars and homogeneous sequences."""
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    return str(value)


def bridge(tracer) -> None:
    """Forward an in-process ``Tracer``'s events onto the active span.

    Keeps one source of truth: the orchestrator emits events for the UI, and
    those same events become span events for the backend, rather than every
    call site having to remember to do both.
    """
    if not enabled():
        return

    original = tracer._on_event

    def forward(event) -> None:
        add_event(
            f"ragverify.{event.kind.value}",
            message=event.message,
            round=event.round_index,
            **{k: v for k, v in event.data.items() if isinstance(v, str | bool | int | float)},
        )
        if original is not None:
            original(event)

    tracer._on_event = forward
