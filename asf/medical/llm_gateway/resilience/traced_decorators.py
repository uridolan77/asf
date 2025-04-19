"""
Decorators for resilience patterns with tracing integration.

This module has been completely replaced with empty implementations to prevent server hanging.
"""

import functools
from typing import Any, Callable, TypeVar

# Define type variables for better type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Import the necessary exception class
class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass

def with_traced_circuit_breaker(*args, **kwargs):
    """No-op implementation of circuit breaker with tracing."""
    def decorator(func):
        return func
    return decorator

def with_traced_provider_circuit_breaker(*args, **kwargs):
    """No-op implementation of provider circuit breaker with tracing."""
    def decorator(func):
        return func
    return decorator

def with_traced_retry(*args, **kwargs):
    """No-op implementation of retry with tracing."""
    def decorator(func):
        return func
    return decorator

def with_traced_timeout(*args, **kwargs):
    """No-op implementation of timeout with tracing."""
    def decorator(func):
        return func
    return decorator
