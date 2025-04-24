"""
Resilient provider wrapper for LLM Gateway.

This module provides a wrapper for LLM providers that adds resilience 
patterns such as circuit breaking, retry, and rate limiting.
"""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type

from asf.conexus.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.conexus.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.conexus.llm_gateway.resilience.rate_limiter import RateLimiter
from asf.conexus.llm_gateway.resilience.decorators import with_resilience
from asf.conexus.llm_gateway.resilience.factory import get_resilience_factory

logger = logging.getLogger("conexus.llm_gateway.providers.resilient_provider")


class ResilientProviderWrapper:
    """
    A wrapper for an LLM provider that adds resilience patterns.
    
    This wrapper can be applied to any provider implementation to add:
    - Circuit breaking to prevent cascading failures
    - Retries with exponential backoff for transient errors
    - Rate limiting to respect provider API limits
    """
    
    def __init__(
        self,
        provider: Any,
        provider_id: str,
        circuit_breaker_name: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        rate_limiter: Optional[RateLimiter] = None,
        metrics_service: Optional[Any] = None
    ):
        """
        Initialize the resilient provider wrapper.
        
        Args:
            provider: The provider instance to wrap
            provider_id: Unique identifier for this provider
            circuit_breaker_name: Name for the circuit breaker (default is provider_id)
            retry_policy: Retry policy to use (default is DEFAULT_RETRY_POLICY)
            rate_limiter: Rate limiter to use (default is None)
            metrics_service: Optional metrics service for recording metrics
        """
        self.provider = provider
        self.provider_id = provider_id
        self.metrics_service = metrics_service
        
        # Create resilience factory
        self.resilience_factory = get_resilience_factory(metrics_service)
        
        # Set up circuit breaker
        cb_name = circuit_breaker_name or f"provider_{provider_id}"
        self.circuit_breaker = self.resilience_factory.get_or_create_circuit_breaker(
            name=cb_name,
            provider_id=provider_id
        )
        
        # Set up retry policy
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        
        # Set up rate limiter
        self.rate_limiter = rate_limiter
        
        logger.info(
            f"Initialized resilient provider wrapper for {provider_id} "
            f"with circuit_breaker={cb_name}, retry_policy={self.retry_policy.name}"
        )
        
        # Apply resilience to methods
        self._wrap_methods()
    
    def _wrap_methods(self):
        """
        Wrap provider methods with resilience patterns.
        """
        # Save original methods
        self._original_generate = self.provider.generate
        self._original_generate_stream = self.provider.generate_stream
        self._original_health_check = getattr(self.provider, "health_check", None)
        
        # Wrap generate method
        self.provider.generate = with_resilience(
            circuit_breaker=self.circuit_breaker,
            retry_policy=self.retry_policy,
            rate_limiter=self.rate_limiter
        )(self.provider.generate)
        
        # Wrap generate_stream method (without retries which are hard with streams)
        self.provider.generate_stream = with_resilience(
            circuit_breaker=self.circuit_breaker,
            retry_policy=None,  # Don't retry streaming calls
            rate_limiter=self.rate_limiter
        )(self.provider.generate_stream)
        
        # Wrap health_check method if it exists
        if self._original_health_check:
            self.provider.health_check = with_resilience(
                circuit_breaker=None,  # Don't use circuit breaker for health checks
                retry_policy=self.retry_policy,
                rate_limiter=None  # Don't rate limit health checks
            )(self.provider.health_check)
    
    def unwrap(self):
        """
        Unwrap provider methods, removing resilience patterns.
        """
        # Restore original methods
        self.provider.generate = self._original_generate
        self.provider.generate_stream = self._original_generate_stream
        
        if self._original_health_check:
            self.provider.health_check = self._original_health_check
        
        logger.info(f"Unwrapped resilience from provider {self.provider_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this resilient provider.
        
        Returns:
            Dict with metrics
        """
        metrics = {
            "provider_id": self.provider_id,
            "circuit_breaker": self.circuit_breaker.name,
            "retry_policy": self.retry_policy.name,
            "has_rate_limiter": self.rate_limiter is not None
        }
        
        # Add circuit breaker metrics
        metrics["circuit_breaker_metrics"] = self.circuit_breaker.get_metrics()
        
        # Add retry policy metrics
        metrics["retry_policy_metrics"] = self.retry_policy.get_metrics()
        
        # Add rate limiter metrics if available
        if self.rate_limiter:
            metrics["rate_limiter_metrics"] = self.rate_limiter.get_stats()
        
        return metrics