"""
Resilience patterns for LLM Gateway.

This package provides resilience patterns for the LLM Gateway,
including circuit breakers, retries, and timeouts.

NOTE: Tracing and metrics functionality has been disabled.
"""

import sys
import logging
import functools
from typing import Any, Callable, TypeVar, cast

# Import the non-observability components directly
from .circuit_breaker import CircuitBreaker, CircuitState
from .enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    RecoveryStrategy,
    FailureCategory
)
from .factory import ResilienceFactory, get_resilience_factory
from .decorators import (
    with_circuit_breaker,
    with_provider_circuit_breaker,
    with_retry,
    with_timeout,
    CircuitOpenError
)

# Create no-op implementations for tracing and metrics
F = TypeVar('F', bound=Callable[..., Any])

class DummyClass:
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Create dummy metrics implementation
class ResilienceMetrics(DummyClass):
    pass

def get_resilience_metrics():
    return ResilienceMetrics()

# Create dummy tracing implementation
class ResilienceTracing(DummyClass):
    def __init__(self, *args, **kwargs):
        self.tracer = DummyClass()

def get_resilience_tracing():
    return ResilienceTracing()

# Define working decorator functions that pass through the original function
def with_circuit_breaker_tracing(*args, **kwargs):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

# Create working traced decorators that preserve function signatures
def with_traced_circuit_breaker(*args, **kwargs):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def with_traced_provider_circuit_breaker(*args, **kwargs):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def with_traced_retry(*args, **kwargs):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def with_traced_timeout(*args, **kwargs):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

# Mark tracing as available to avoid conditional imports elsewhere
TRACING_AVAILABLE = True

# Base class for module overrides
class ResilienceModuleOverride:
    def __init__(self):
        self.ResilienceTracing = ResilienceTracing
        self.ResilienceMetrics = ResilienceMetrics
        self.get_resilience_tracing = get_resilience_tracing
        self.get_resilience_metrics = get_resilience_metrics
        self.with_circuit_breaker_tracing = with_circuit_breaker_tracing
        
    def __getattr__(self, name):
        # For any other attribute, return a function that returns a dummy
        def default_function(*args, **kwargs):
            return None
        return default_function

# Override the module imports with proper decorator functions
class TracedDecoratorsModuleOverride:
    def __init__(self):
        self.with_traced_circuit_breaker = with_traced_circuit_breaker
        self.with_traced_provider_circuit_breaker = with_traced_provider_circuit_breaker
        self.with_traced_retry = with_traced_retry
        self.with_traced_timeout = with_traced_timeout
        
    def __getattr__(self, name):
        # For any other attribute, return a function that returns the identity function
        def default_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return default_decorator

# Inject our overrides into sys.modules
sys.modules['asf.medical.llm_gateway.resilience.tracing'] = ResilienceModuleOverride()
sys.modules['asf.medical.llm_gateway.resilience.metrics'] = ResilienceModuleOverride()
sys.modules['asf.medical.llm_gateway.resilience.traced_decorators'] = TracedDecoratorsModuleOverride()

__all__ = [
    # Circuit breaker
    'CircuitBreaker',
    'CircuitState',

    # Enhanced circuit breaker
    'EnhancedCircuitBreaker',
    'CircuitBreakerRegistry',
    'get_circuit_breaker_registry',
    'RecoveryStrategy',
    'FailureCategory',

    # Factory
    'ResilienceFactory',
    'get_resilience_factory',

    # Metrics
    'ResilienceMetrics',
    'get_resilience_metrics',

    # Decorators
    'with_circuit_breaker',
    'with_provider_circuit_breaker',
    'with_retry',
    'with_timeout',
    'CircuitOpenError',
    
    # Tracing
    'ResilienceTracing',
    'get_resilience_tracing',
    'with_circuit_breaker_tracing',

    # Traced decorators
    'with_traced_circuit_breaker',
    'with_traced_provider_circuit_breaker',
    'with_traced_retry',
    'with_traced_timeout'
]
