"""Circuit Breaker Registry

This module provides a registry for managing multiple circuit breakers.
It allows for centralized creation, retrieval, and management of circuit breakers.
"""

import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List

from .circuit_breaker import CircuitBreaker, AsyncCircuitBreaker, CircuitState

# Set up logging
logger = logging.getLogger(__name__)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.
    
    This class provides a centralized way to create, retrieve, and manage
    circuit breakers for different resources.
    
    Attributes:
        _circuit_breakers: Dictionary of circuit breakers
        _lock: Lock for thread safety
    """
    
    def __init__(self):
        """Initialize the circuit breaker registry."""
        self._circuit_breakers = {}
        self._lock = threading.RLock()
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 1,
        excluded_exceptions: Optional[List[type]] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Name of the protected resource
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in half-open state to close
            half_open_max_calls: Maximum concurrent calls in half-open state
            excluded_exceptions: Exceptions that should not count as failures
            
        Returns:
            CircuitBreaker: The circuit breaker instance
        """
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    success_threshold=success_threshold,
                    half_open_max_calls=half_open_max_calls,
                    excluded_exceptions=excluded_exceptions
                )
            return self._circuit_breakers[name]
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of all circuit breakers.
        
        Returns:
            Dict[str, Any]: Status of all circuit breakers
        """
        with self._lock:
            return {
                name: cb.get_metrics()
                for name, cb in self._circuit_breakers.items()
            }
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset one or all circuit breakers.
        
        Args:
            name: Name of the circuit breaker to reset, or None to reset all
        """
        with self._lock:
            if name is not None:
                if name in self._circuit_breakers:
                    cb = self._circuit_breakers[name]
                    cb.reset()
                    logger.info(f"Circuit breaker '{name}' reset")
            else:
                for name, cb in self._circuit_breakers.items():
                    cb.reset()
                logger.info("All circuit breakers reset")


class AsyncCircuitBreakerRegistry:
    """Registry for managing multiple asynchronous circuit breakers.
    
    This class provides a centralized way to create, retrieve, and manage
    asynchronous circuit breakers for different resources.
    
    Attributes:
        _circuit_breakers: Dictionary of circuit breakers
        _lock: Lock for thread safety
    """
    
    def __init__(self):
        """Initialize the async circuit breaker registry."""
        self._circuit_breakers = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 1,
        excluded_exceptions: Optional[List[type]] = None
    ) -> AsyncCircuitBreaker:
        """Get or create an async circuit breaker.
        
        Args:
            name: Name of the protected resource
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in half-open state to close
            half_open_max_calls: Maximum concurrent calls in half-open state
            excluded_exceptions: Exceptions that should not count as failures
            
        Returns:
            AsyncCircuitBreaker: The async circuit breaker instance
        """
        async with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = AsyncCircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    success_threshold=success_threshold,
                    half_open_max_calls=half_open_max_calls,
                    excluded_exceptions=excluded_exceptions
                )
            return self._circuit_breakers[name]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the status of all async circuit breakers.
        
        Returns:
            Dict[str, Any]: Status of all async circuit breakers
        """
        async with self._lock:
            status = {}
            for name, cb in self._circuit_breakers.items():
                status[name] = await cb.get_metrics()
            return status
    
    async def reset(self, name: Optional[str] = None) -> None:
        """Reset one or all async circuit breakers.
        
        Args:
            name: Name of the circuit breaker to reset, or None to reset all
        """
        async with self._lock:
            if name is not None:
                if name in self._circuit_breakers:
                    cb = self._circuit_breakers[name]
                    await cb.reset()
                    logger.info(f"Async circuit breaker '{name}' reset")
            else:
                for name, cb in self._circuit_breakers.items():
                    await cb.reset()
                logger.info("All async circuit breakers reset")


# Global registry instances
_circuit_breaker_registry = None
_async_circuit_breaker_registry = None
_registry_lock = threading.RLock()
_async_registry_lock = asyncio.Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.
    
    Returns:
        CircuitBreakerRegistry: The global circuit breaker registry
    """
    global _circuit_breaker_registry
    with _registry_lock:
        if _circuit_breaker_registry is None:
            _circuit_breaker_registry = CircuitBreakerRegistry()
        return _circuit_breaker_registry


async def get_async_circuit_breaker_registry() -> AsyncCircuitBreakerRegistry:
    """Get the global async circuit breaker registry.
    
    Returns:
        AsyncCircuitBreakerRegistry: The global async circuit breaker registry
    """
    global _async_circuit_breaker_registry
    async with _async_registry_lock:
        if _async_circuit_breaker_registry is None:
            _async_circuit_breaker_registry = AsyncCircuitBreakerRegistry()
        return _async_circuit_breaker_registry


# Export all classes and functions
__all__ = [
    'CircuitBreakerRegistry',
    'AsyncCircuitBreakerRegistry',
    'get_circuit_breaker_registry',
    'get_async_circuit_breaker_registry'
]
