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

# Import tracing components
try:
    from .tracing import (
        ResilienceTracing,
        get_resilience_tracing,
        with_circuit_breaker_tracing
    )
    from .traced_decorators import (
        with_traced_circuit_breaker,
        with_traced_provider_circuit_breaker,
        with_traced_retry,
        with_traced_timeout
    )
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

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

# Add tracing components if available
if TRACING_AVAILABLE:
    __all__.extend([
        # Tracing
        'ResilienceTracing',
        'get_resilience_tracing',
        'with_circuit_breaker_tracing',

        # Traced decorators
        'with_traced_circuit_breaker',
        'with_traced_provider_circuit_breaker',
        'with_traced_retry',
        'with_traced_timeout'
    ])
