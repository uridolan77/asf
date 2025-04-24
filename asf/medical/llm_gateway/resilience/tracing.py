"""
Distributed tracing integration for resilience patterns.

This module provides distributed tracing integration for resilience patterns,
including circuit breakers, retries, and timeouts.

NOTE: This version has been completely disabled - no tracing functionality is active.
All imports and initializations are bypassed to prevent server hanging.
"""

import functools
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Callable, TypeVar, Union

import structlog

# Create type variables for better type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Create a minimal Span class
class Span:
    """Minimal no-op implementation of Span."""
    def __enter__(self): return self
    def __exit__(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass
    def record_exception(self, *args, **kwargs): pass

# Create a minimal dummy tracer
class DummyTracer:
    """Minimal no-op implementation of a tracer."""
    def start_span(self, *args, **kwargs): return Span()
    def start_as_current_span(self, *args, **kwargs): return Span()

# Create silent logger
logger = structlog.get_logger("llm_gateway.resilience.tracing")

class ResilienceTracing:
    """
    Distributed tracing integration for resilience patterns - completely disabled.
    No initialization code is executed to prevent server hanging.
    """
    
    def __init__(self, service_name: str = ""):
        """Initialize resilience tracing with absolute minimal implementation."""
        self.tracer = DummyTracer()
        # No logging during initialization
    
    @contextmanager
    def circuit_breaker_span(self, *args, **kwargs) -> Iterator[Span]:
        """Return a dummy span that does nothing."""
        yield Span()
    
    # Empty implementations for all methods
    def record_state_change(self, *args, **kwargs): pass
    def record_failure(self, *args, **kwargs): pass
    def record_success(self, *args, **kwargs): pass
    def record_retry(self, *args, **kwargs): pass
    def record_timeout(self, *args, **kwargs): pass


def with_circuit_breaker_tracing(*args, **kwargs):
    """Return a decorator that does nothing."""
    def decorator(func):
        return func
    return decorator


# Singleton instance - initialize immediately to avoid later calls
_resilience_tracing = ResilienceTracing()


def get_resilience_tracing() -> ResilienceTracing:
    """Get the singleton instance of the ResilienceTracing."""
    global _resilience_tracing
    return _resilience_tracing
