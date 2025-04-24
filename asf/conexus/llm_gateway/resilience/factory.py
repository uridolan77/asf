"""
Factory for creating resilience components.

This module provides factories for creating resilience components,
such as circuit breakers, retries, and timeouts.
"""

import logging
from typing import Dict, Any, Optional, Type, Union

from asf.conexus.llm_gateway.resilience.circuit_breaker import CircuitBreaker, CircuitState
from asf.conexus.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.conexus.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger("conexus.llm_gateway.resilience.factory")

# Singleton circuit breaker registry
_circuit_breaker_registry: Dict[str, CircuitBreaker] = {}


class ResilienceFactory:
    """
    Factory for creating resilience components.
    
    This class provides methods for creating resilience components,
    such as circuit breakers, retries, and rate limiters.
    """
    
    def __init__(self, metrics_service: Optional[Any] = None):
        """
        Initialize the resilience factory.
        
        Args:
            metrics_service: Optional metrics service for recording metrics
        """
        self.metrics_service = metrics_service
        logger.info("Initialized resilience factory")
    
    def create_circuit_breaker(
        self,
        name: str,
        provider_id: Optional[str] = None,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
        reset_timeout: int = 600,  # 10 minutes
        jitter_factor: float = 0.2,  # 20% jitter
        on_state_change: Optional[callable] = None
    ) -> CircuitBreaker:
        """
        Create a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider (for metrics)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Base seconds to wait before testing recovery
            half_open_max_calls: Max calls to allow in HALF_OPEN before deciding
            reset_timeout: Seconds after which to reset failure count in CLOSED state
            jitter_factor: Factor for jitter in recovery timeout (0-1)
            on_state_change: Callback for state change notifications
            
        Returns:
            Circuit breaker instance
        """
        # Check if circuit breaker already exists in registry
        if name in _circuit_breaker_registry:
            logger.info(f"Using existing circuit breaker from registry: {name}")
            return _circuit_breaker_registry[name]
        
        # Create a state change callback that updates metrics
        original_callback = on_state_change
        
        def state_change_callback(old_state: CircuitState, new_state: CircuitState):
            # Call original callback if provided
            if original_callback:
                original_callback(old_state, new_state)
            
            # Update metrics
            if self.metrics_service:
                is_open = new_state == CircuitState.OPEN
                self.metrics_service.record_gauge(
                    f"circuit_breaker.{name}.is_open", 
                    1 if is_open else 0
                )
        
        # Create the circuit breaker
        logger.info(f"Creating circuit breaker: {name}")
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            reset_timeout=reset_timeout,
            on_state_change=state_change_callback
        )
        
        # Store in registry
        _circuit_breaker_registry[name] = circuit_breaker
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name from the registry.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Circuit breaker or None if not found
        """
        return _circuit_breaker_registry.get(name)
    
    def get_or_create_circuit_breaker(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """
        Get a circuit breaker by name or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            Circuit breaker
        """
        circuit_breaker = self.get_circuit_breaker(name)
        if circuit_breaker:
            return circuit_breaker
        
        return self.create_circuit_breaker(name=name, **kwargs)
    
    def create_retry_policy(
        self,
        name: str = "default",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        jitter_factor: float = 0.2,
        retry_codes: Optional[set] = None,
        retry_exceptions: Optional[list] = None,
        retry_predicate: Optional[callable] = None
    ) -> RetryPolicy:
        """
        Create a retry policy.
        
        Args:
            name: Name for this policy
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter_factor: Randomness factor (0.0-1.0) to apply to delays
            retry_codes: Set of error codes to retry
            retry_exceptions: List of exception types to retry
            retry_predicate: Custom function to determine if an error is retryable
            
        Returns:
            RetryPolicy instance
        """
        logger.info(f"Creating retry policy: {name}")
        
        return RetryPolicy(
            name=name,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_factor=jitter_factor,
            retry_codes=retry_codes,
            retry_exceptions=retry_exceptions,
            retry_predicate=retry_predicate
        )
    
    def create_rate_limiter(
        self,
        name: str = "default",
        requests_per_minute: int = 60,
        strategy: str = "token_bucket",
        burst_size: int = 10,
        window_size_seconds: int = 60,
        adaptive_factor: float = 0.5,
        adaptive_min_rate: float = 0.1,
        adaptive_max_rate: float = 2.0
    ) -> RateLimiter:
        """
        Create a rate limiter.
        
        Args:
            name: Name for this rate limiter
            requests_per_minute: Number of requests allowed per minute
            strategy: Rate limiting strategy (token_bucket, sliding_window, adaptive)
            burst_size: Burst size for token bucket strategy
            window_size_seconds: Window size in seconds for sliding window strategy
            adaptive_factor: Adaptive factor for adaptive strategy
            adaptive_min_rate: Minimum rate factor for adaptive strategy
            adaptive_max_rate: Maximum rate factor for adaptive strategy
            
        Returns:
            RateLimiter instance
        """
        logger.info(f"Creating rate limiter: {name}")
        
        # Create configuration
        config = RateLimitConfig(
            strategy=strategy,
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            window_size_seconds=window_size_seconds,
            adaptive_factor=adaptive_factor,
            adaptive_min_rate=adaptive_min_rate,
            adaptive_max_rate=adaptive_max_rate
        )
        
        return RateLimiter(config=config, name=name)


# Singleton instance
_resilience_factory = None


def get_resilience_factory(metrics_service: Optional[Any] = None) -> ResilienceFactory:
    """
    Get the singleton instance of the ResilienceFactory.
    
    Args:
        metrics_service: Optional metrics service for recording metrics
        
    Returns:
        ResilienceFactory instance
    """
    global _resilience_factory
    if _resilience_factory is None:
        _resilience_factory = ResilienceFactory(metrics_service)
    return _resilience_factory