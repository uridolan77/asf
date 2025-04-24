"""
Circuit Breaker Registry

This module provides registry classes for managing circuit breakers.
"""

import logging
import functools
from typing import Dict, Optional, Any, Union

from .circuit_breaker import CircuitBreaker, AsyncCircuitBreaker

# Set up logging
logger = logging.getLogger(__name__)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize the registry."""
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register(
        self,
        name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        **kwargs
    ) -> CircuitBreaker:
        """
        Register a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            circuit_breaker: Circuit breaker instance (optional)
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            CircuitBreaker: The registered circuit breaker
        """
        if name in self._circuit_breakers:
            logger.debug(f"Circuit breaker {name} already registered")
            return self._circuit_breakers[name]
        
        if circuit_breaker is None:
            circuit_breaker = CircuitBreaker(name=name, **kwargs)
        
        logger.debug(f"Registering circuit breaker {name}")
        self._circuit_breakers[name] = circuit_breaker
        
        return circuit_breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a registered circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Optional[CircuitBreaker]: The circuit breaker if found, None otherwise
        """
        return self._circuit_breakers.get(name)
    
    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """
        Get a registered circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            CircuitBreaker: The circuit breaker
        """
        circuit_breaker = self.get(name)
        if circuit_breaker is None:
            circuit_breaker = self.register(name, **kwargs)
        
        return circuit_breaker
    
    def unregister(self, name: str) -> None:
        """
        Unregister a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
        """
        if name in self._circuit_breakers:
            logger.debug(f"Unregistering circuit breaker {name}")
            del self._circuit_breakers[name]
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        logger.info("Resetting all circuit breakers")
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.reset()


class AsyncCircuitBreakerRegistry:
    """Registry for managing multiple async circuit breakers."""
    
    def __init__(self):
        """Initialize the registry."""
        self._circuit_breakers: Dict[str, AsyncCircuitBreaker] = {}
    
    def register(
        self,
        name: str,
        circuit_breaker: Optional[AsyncCircuitBreaker] = None,
        **kwargs
    ) -> AsyncCircuitBreaker:
        """
        Register an async circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            circuit_breaker: Circuit breaker instance (optional)
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            AsyncCircuitBreaker: The registered circuit breaker
        """
        if name in self._circuit_breakers:
            logger.debug(f"Async circuit breaker {name} already registered")
            return self._circuit_breakers[name]
        
        if circuit_breaker is None:
            circuit_breaker = AsyncCircuitBreaker(name=name, **kwargs)
        
        logger.debug(f"Registering async circuit breaker {name}")
        self._circuit_breakers[name] = circuit_breaker
        
        return circuit_breaker
    
    def get(self, name: str) -> Optional[AsyncCircuitBreaker]:
        """
        Get a registered async circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Optional[AsyncCircuitBreaker]: The circuit breaker if found, None otherwise
        """
        return self._circuit_breakers.get(name)
    
    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> AsyncCircuitBreaker:
        """
        Get a registered async circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            AsyncCircuitBreaker: The circuit breaker
        """
        circuit_breaker = self.get(name)
        if circuit_breaker is None:
            circuit_breaker = self.register(name, **kwargs)
        
        return circuit_breaker
    
    def unregister(self, name: str) -> None:
        """
        Unregister an async circuit breaker.
        
        Args:
            name: Name of the circuit breaker
        """
        if name in self._circuit_breakers:
            logger.debug(f"Unregistering async circuit breaker {name}")
            del self._circuit_breakers[name]
    
    async def reset_all(self) -> None:
        """Reset all async circuit breakers."""
        logger.info("Resetting all async circuit breakers")
        for circuit_breaker in self._circuit_breakers.values():
            await circuit_breaker.reset()


# Singleton instances
_circuit_breaker_registry: Optional[CircuitBreakerRegistry] = None
_async_circuit_breaker_registry: Optional[AsyncCircuitBreakerRegistry] = None


@functools.lru_cache()
def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Get the circuit breaker registry singleton.
    
    Returns:
        CircuitBreakerRegistry: The circuit breaker registry
    """
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    
    return _circuit_breaker_registry


@functools.lru_cache()
def get_async_circuit_breaker_registry() -> AsyncCircuitBreakerRegistry:
    """
    Get the async circuit breaker registry singleton.
    
    Returns:
        AsyncCircuitBreakerRegistry: The async circuit breaker registry
    """
    global _async_circuit_breaker_registry
    if _async_circuit_breaker_registry is None:
        _async_circuit_breaker_registry = AsyncCircuitBreakerRegistry()
    
    return _async_circuit_breaker_registry


# Export
__all__ = [
    "CircuitBreakerRegistry",
    "AsyncCircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "get_async_circuit_breaker_registry",
]
