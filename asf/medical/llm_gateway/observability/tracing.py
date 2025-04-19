"""
Distributed tracing service for MCP Provider.

This module provides distributed tracing capabilities using OpenTelemetry,
allowing end-to-end visibility of requests across services.

NOTE: This version has been modified to disable tracing functionality.
"""

import contextlib
import os
from typing import Any, Dict, Iterator, Optional

import structlog

# Check if tracing is disabled
TRACING_DISABLED = os.environ.get("DISABLE_TRACING", "0").lower() in ("1", "true", "yes")
OBSERVABILITY_DISABLED = os.environ.get("DISABLE_OBSERVABILITY", "0").lower() in ("1", "true", "yes")

# Import dummy implementations if tracing is disabled
if TRACING_DISABLED or OBSERVABILITY_DISABLED:
    from .disable_patch import DummyTracingService as TracingService
    from .disable_patch import DummySpan as Span
    from .disable_patch import DummyTracer

    # Create dummy enum classes
    class SpanKind:
        INTERNAL = "internal"
        CLIENT = "client"
        SERVER = "server"
        PRODUCER = "producer"
        CONSUMER = "consumer"

    class StatusCode:
        OK = "ok"
        ERROR = "error"
        UNSET = "unset"

    # Create dummy trace module
    class DummyTrace:
        @staticmethod
        def get_current_span():
            return Span()

        @staticmethod
        def get_tracer(*args, **kwargs):
            return DummyTracer()

        @staticmethod
        def set_tracer_provider(*args, **kwargs):
            pass

    trace = DummyTrace()

    logger = structlog.get_logger("mcp_observability.tracing")
    logger.info("Tracing disabled via environment variable")
else:
    # Import real implementations if tracing is enabled
    import structlog
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, StatusCode

    logger = structlog.get_logger("mcp_observability.tracing")


class TracingService:
    """
    Distributed tracing service for MCP Provider.

    Features:
    - OpenTelemetry tracing with OTLP export
    - Automatic context propagation
    - Integration with structured logging
    - Custom span attributes for MCP-specific info
    - Support for manual spans and events
    """

    def __init__(
        self,
        service_name: str = "mcp_provider",
        enable_otlp: bool = True,
        otlp_endpoint: str = "localhost:4317",
        additional_attributes: Optional[Dict[str, str]] = None
    ):
        """
        Initialize tracing service.

        Args:
            service_name: Name of the service for tracing context
            enable_otlp: Whether to enable OTLP export
            otlp_endpoint: OTLP exporter endpoint
            additional_attributes: Static attributes to add to all spans
        """
        self.service_name = service_name
        self.enable_otlp = enable_otlp
        self.otlp_endpoint = otlp_endpoint
        self.additional_attributes = additional_attributes or {}

        # Initialize tracing infrastructure
        self._init_tracing_infrastructure()

        # Get a tracer
        self.tracer = trace.get_tracer(
            f"{self.service_name}_tracer",
            schema_url="https://opentelemetry.io/schemas/1.9.0"
        )

        self.logger = logger.bind(service=service_name)

        self.logger.info(
            "Initialized tracing service",
            otlp_enabled=enable_otlp,
            otlp_endpoint=otlp_endpoint if enable_otlp else None
        )

    def _init_tracing_infrastructure(self) -> None:
        """Initialize OpenTelemetry tracing infrastructure."""
        # Skip initialization if tracing is disabled
        if os.environ.get("DISABLE_TRACING", "0").lower() in ("1", "true", "yes") or \
           os.environ.get("DISABLE_OBSERVABILITY", "0").lower() in ("1", "true", "yes"):
            logger.info("Tracing infrastructure initialization skipped due to environment variables")
            return

        try:
            # Create resource with service info
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                **self.additional_attributes
            })
        except NameError:
            logger.warning("Resource not defined. Using dummy implementation.")
            # Create a dummy resource
            resource = None

        # Create tracer provider
        try:
            if resource is None:
                tracer_provider = TracerProvider()
            else:
                tracer_provider = TracerProvider(resource=resource)
        except Exception as e:
            logger.warning(f"Failed to create tracer provider: {str(e)}")
            # Create a dummy tracer provider
            class DummyTracerProvider:
                def get_tracer(self, *args, **kwargs):
                    return self.tracer

                def __init__(self):
                    self.tracer = DummyTracer()

            tracer_provider = DummyTracerProvider()

        # Add OTLP exporter if enabled
        if self.enable_otlp:
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            logger.info(f"Added OTLP exporter with endpoint {self.otlp_endpoint}")

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ) -> Iterator[Span]:
        """
        Start a new span as context manager.

        Args:
            name: Name of the span
            kind: Span kind (internal, client, server, etc.)
            attributes: Additional span attributes
            record_exception: Whether to record exceptions

        Yields:
            Active span
        """
        # Merge with default attributes
        all_attrs = {**self.additional_attributes}
        if attributes:
            all_attrs.update(attributes)

        # Start new span
        with self.tracer.start_as_current_span(name, kind=kind, attributes=all_attrs) as span:
            try:
                yield span
            except Exception as e:
                if record_exception:
                    # Record exception in span
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, str(e))
                raise

    def create_span_decorator(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """
        Create a decorator for tracing functions.

        Args:
            name: Name of the span (defaults to function name)
            kind: Span kind
            attributes: Additional span attributes
            record_exception: Whether to record exceptions

        Returns:
            Decorator function
        """
        def decorator(func):
            span_name = name or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.start_span(span_name, kind, attributes, record_exception) as span:
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.start_span(span_name, kind, attributes, record_exception) as span:
                    return func(*args, **kwargs)

            # Choose appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def add_span_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an event to the current active span.

        Args:
            name: Name of the event
            attributes: Event attributes
        """
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes=attributes)

    def set_span_status(
        self,
        status: StatusCode,
        description: Optional[str] = None
    ) -> None:
        """
        Set status on the current active span.

        Args:
            status: Status code (OK, ERROR, UNSET)
            description: Status description
        """
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_status(status, description)

    def set_span_attribute(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Set attribute on the current active span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)

    def end_span(self) -> None:
        """End the current span early."""
        current_span = trace.get_current_span()
        if current_span:
            current_span.end()

    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record exception in the current span.

        Args:
            exception: Exception to record
            attributes: Additional attributes about the exception
        """
        current_span = trace.get_current_span()
        if current_span:
            current_span.record_exception(exception, attributes=attributes)
            current_span.set_status(StatusCode.ERROR, str(exception))


import asyncio
import functools