"""
Resilience patterns for LLM Gateway.

This package provides resilience patterns for the LLM Gateway,
including circuit breakers, retries, and timeouts.
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    RecoveryStrategy,
    FailureCategory
)
from .factory import ResilienceFactory, get_resilience_factory
from .metrics import ResilienceMetrics, get_resilience_metrics
from .decorators import (
    with_circuit_breaker,
    with_provider_circuit_breaker,
    with_retry,
    with_timeout,
    CircuitOpenError
)

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
    'CircuitOpenError'
]
