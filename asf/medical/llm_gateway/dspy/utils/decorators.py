"""
Circuit Breaker Decorators

This module provides decorators for applying circuit breakers to functions.
"""

import logging
import asyncio
import functools
from typing import Callable, Any, Optional, Dict, Union, TypeVar

from .circuit_breaker import CircuitBreaker, AsyncCircuitBreaker
from .registry import get_circuit_breaker_registry, get_async_circuit_breaker_registry

# Set up logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Any])


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 3,
    recovery_timeout: float = 60.0,
) -> Callable[[F], F]:
    """
    Decorator for applying a circuit breaker to a function.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before trying to recover
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        # Generate name based on function if not provided
        cb_name = name or f"cb_{func.__module__}.{func.__name__}"
        
        # Get registry
        registry = get_circuit_breaker_registry()
        
        # Get or create circuit breaker
        circuit_breaker = registry.get_or_create(
            cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Call function with circuit breaker
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def async_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 3,
    recovery_timeout: float = 60.0,
) -> Callable[[AF], AF]:
    """
    Decorator for applying an async circuit breaker to an async function.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before trying to recover
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: AF) -> AF:
        # Generate name based on function if not provided
        cb_name = name or f"async_cb_{func.__module__}.{func.__name__}"
        
        # Get registry
        registry = get_async_circuit_breaker_registry()
        
        # Get or create circuit breaker
        circuit_breaker = registry.get_or_create(
            cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Call function with circuit breaker
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def with_fallback(fallback_func: Callable[..., T]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for providing a fallback function when the main function fails.
    
    Args:
        fallback_func: Fallback function to call when the main function fails
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {str(e)}")
                return fallback_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def async_with_fallback(fallback_func: Callable[..., T]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for providing a fallback function when the main async function fails.
    
    Args:
        fallback_func: Fallback function to call when the main function fails
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {str(e)}")
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Export
__all__ = [
    "circuit_breaker",
    "async_circuit_breaker",
    "with_fallback",
    "async_with_fallback",
]
