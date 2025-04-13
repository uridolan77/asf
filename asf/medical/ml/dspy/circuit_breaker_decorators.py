"""Circuit Breaker Decorators

This module provides decorators for easily applying circuit breakers to functions.
It allows for simple integration of circuit breaker pattern into existing code.
"""

import logging
import asyncio
import functools
from typing import Callable, Any, Optional, List, TypeVar, cast

from .circuit_breaker import CircuitBreaker, AsyncCircuitBreaker, CircuitOpenError
from .circuit_breaker_registry import (
    get_circuit_breaker_registry,
    get_async_circuit_breaker_registry
)

# Set up logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AF = TypeVar('AF', bound=Callable[..., Any])


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 30.0,
    success_threshold: int = 2,
    half_open_max_calls: int = 1,
    excluded_exceptions: Optional[List[type]] = None
) -> Callable[[F], F]:
    """Decorator for applying circuit breaker pattern to a function.
    
    This decorator wraps a function with a circuit breaker, which will prevent
    calls to the function if it has been failing consistently.
    
    Args:
        name: Name of the protected resource
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before attempting recovery
        success_threshold: Successes needed in half-open state to close
        half_open_max_calls: Maximum concurrent calls in half-open state
        excluded_exceptions: Exceptions that should not count as failures
        
    Returns:
        Callable: Decorated function
    
    Example:
        ```python
        @circuit_breaker("api_call", failure_threshold=3)
        def call_external_api(param1, param2):
            # Function implementation
            pass
        ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = get_circuit_breaker_registry()
            cb = registry.get_or_create(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                success_threshold=success_threshold,
                half_open_max_calls=half_open_max_calls,
                excluded_exceptions=excluded_exceptions
            )
            
            with cb:
                return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def async_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 30.0,
    success_threshold: int = 2,
    half_open_max_calls: int = 1,
    excluded_exceptions: Optional[List[type]] = None
) -> Callable[[AF], AF]:
    """Decorator for applying circuit breaker pattern to an async function.
    
    This decorator wraps an async function with a circuit breaker, which will
    prevent calls to the function if it has been failing consistently.
    
    Args:
        name: Name of the protected resource
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before attempting recovery
        success_threshold: Successes needed in half-open state to close
        half_open_max_calls: Maximum concurrent calls in half-open state
        excluded_exceptions: Exceptions that should not count as failures
        
    Returns:
        Callable: Decorated async function
    
    Example:
        ```python
        @async_circuit_breaker("async_api_call", failure_threshold=3)
        async def call_external_api_async(param1, param2):
            # Async function implementation
            pass
        ```
    """
    def decorator(func: AF) -> AF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = await get_async_circuit_breaker_registry()
            cb = await registry.get_or_create(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                success_threshold=success_threshold,
                half_open_max_calls=half_open_max_calls,
                excluded_exceptions=excluded_exceptions
            )
            
            async with cb:
                return await func(*args, **kwargs)
        
        return cast(AF, wrapper)
    
    return decorator


def with_fallback(
    fallback_function: Callable[..., T],
    handle_exceptions: List[type] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for providing a fallback function when a circuit is open.
    
    This decorator should be applied after a circuit_breaker decorator to
    provide a fallback function when the circuit is open.
    
    Args:
        fallback_function: Function to call when the circuit is open
        handle_exceptions: List of exception types to handle with the fallback
        
    Returns:
        Callable: Decorated function
    
    Example:
        ```python
        def fallback_api_call(param1, param2):
            return "Fallback response"
            
        @with_fallback(fallback_api_call, handle_exceptions=[CircuitOpenError])
        @circuit_breaker("api_call", failure_threshold=3)
        def call_external_api(param1, param2):
            # Function implementation
            pass
        ```
    """
    if handle_exceptions is None:
        handle_exceptions = [CircuitOpenError]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except tuple(handle_exceptions) as e:
                logger.warning(f"Circuit open or error, using fallback for {func.__name__}: {str(e)}")
                return fallback_function(*args, **kwargs)
        
        return wrapper
    
    return decorator


def async_with_fallback(
    fallback_function: Callable[..., Any],
    handle_exceptions: List[type] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for providing a fallback function when a circuit is open for async functions.
    
    This decorator should be applied after an async_circuit_breaker decorator to
    provide a fallback function when the circuit is open.
    
    Args:
        fallback_function: Async function to call when the circuit is open
        handle_exceptions: List of exception types to handle with the fallback
        
    Returns:
        Callable: Decorated async function
    
    Example:
        ```python
        async def fallback_api_call_async(param1, param2):
            return "Fallback response"
            
        @async_with_fallback(fallback_api_call_async, handle_exceptions=[CircuitOpenError])
        @async_circuit_breaker("async_api_call", failure_threshold=3)
        async def call_external_api_async(param1, param2):
            # Async function implementation
            pass
        ```
    """
    if handle_exceptions is None:
        handle_exceptions = [CircuitOpenError]
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except tuple(handle_exceptions) as e:
                logger.warning(f"Circuit open or error, using fallback for {func.__name__}: {str(e)}")
                if asyncio.iscoroutinefunction(fallback_function):
                    return await fallback_function(*args, **kwargs)
                else:
                    return fallback_function(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Export all decorators
__all__ = [
    'circuit_breaker',
    'async_circuit_breaker',
    'with_fallback',
    'async_with_fallback'
]
