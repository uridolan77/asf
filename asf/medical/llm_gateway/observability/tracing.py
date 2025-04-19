"""
Distributed tracing service for MCP Provider.

This module provides distributed tracing capabilities using OpenTelemetry,
allowing end-to-end visibility of requests across services.

NOTE: This version has been completely disabled - no tracing functionality is active.
All imports and initializations are bypassed to prevent server hanging.
"""

import contextlib
from typing import Any, Dict, Iterator, Optional

import structlog

# Create minimal no-op implementations without any imports or initializations
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

class Span:
    """Minimal no-op implementation of Span."""
    def __enter__(self): return self
    def __exit__(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass
    def record_exception(self, *args, **kwargs): pass

# No logger output to prevent confusing logs
logger = structlog.get_logger("mcp_observability.tracing")

class TracingService:
    """
    Distributed tracing service for MCP Provider - completely disabled.
    No initialization code is executed to prevent server hanging.
    """

    def __init__(self, service_name: str = "", **kwargs):
        """Initialize tracing service with absolute minimal implementation."""
        self.tracer = DummyTracer()
        # No logging during initialization

    def _init_tracing_infrastructure(self) -> None:
        """Empty implementation that does nothing."""
        pass

    @contextlib.contextmanager
    def start_span(self, name: str, **kwargs) -> Iterator[Span]:
        """Return a dummy span that does nothing."""
        yield Span()

    def start_as_current_span(self, *args, **kwargs) -> Span:
        """Return a dummy span that does nothing."""
        return Span()
        
    # All other methods replaced with no-ops
    def create_span_decorator(self, *args, **kwargs):
        """Return a decorator that does nothing."""
        def decorator(func): return func
        return decorator

    def add_span_event(self, *args, **kwargs): pass
    def set_span_status(self, *args, **kwargs): pass
    def set_span_attribute(self, *args, **kwargs): pass
    def end_span(self): pass
    def record_exception(self, *args, **kwargs): pass

class DummyTracer:
    """Minimal no-op implementation of a tracer."""
    def start_span(self, *args, **kwargs): return Span()
    def start_as_current_span(self, *args, **kwargs): return Span()

# Create a dummy trace module
class DummyTrace:
    @staticmethod
    def get_current_span(): return Span()
    @staticmethod
    def get_tracer(*args, **kwargs): return DummyTracer()
    @staticmethod
    def set_tracer_provider(*args, **kwargs): pass

# Replace trace with dummy implementation
trace = DummyTrace()