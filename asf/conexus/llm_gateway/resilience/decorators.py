"""
Resilience decorators for LLM Gateway.

This module provides decorators for adding resilience patterns such as
circuit breaking, retry, and rate limiting to LLM provider functions.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Dict, Union

from asf.conexus.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.conexus.llm_gateway.resilience.retry import RetryPolicy, retry_async, DEFAULT_RETRY_POLICY
from asf.conexus.llm_gateway.resilience.rate_limiter import RateLimiter
from asf.conexus.llm_gateway.resilience.factory import get_resilience_factory

logger = logging.getLogger("conexus.llm_gateway.resilience.decorators")


def with_circuit_breaker(
    circuit_breaker: Union[CircuitBreaker, str],
    fallback_fn: Optional[Callable] = None
):
    """
    Decorator to add circuit breaker to a function.
    
    If the circuit breaker is open, the fallback function is called or a failure is raised.
    
    Args:
        circuit_breaker: Circuit breaker instance or name (to be looked up in the registry)
        fallback_fn: Function to call if circuit breaker is open
        
    Returns:
        Decorated function
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Get circuit breaker if a name was provided
            cb = circuit_breaker
            if isinstance(circuit_breaker, str):
                cb = get_resilience_factory().get_or_create_circuit_breaker(circuit_breaker)
                
            # Check if circuit is open
            if cb.is_open():
                logger.warning(
                    f"Circuit breaker '{cb.name}' is open, failing fast for {fn.__name__}"
                )
                
                # Call fallback function if provided
                if fallback_fn:
                    return await fallback_fn(*args, **kwargs)
                
                raise RuntimeError(f"Circuit breaker '{cb.name}' is open")
            
            # Circuit is closed or half-open, proceed with the call
            try:
                result = await fn(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                logger.warning(
                    f"Call to {fn.__name__} failed, recording failure to circuit breaker '{cb.name}': {str(e)}"
                )
                raise
                
        return wrapper
    return decorator


def with_retry(
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator to add retry behavior to a function.
    
    Args:
        retry_policy: RetryPolicy to use
        on_retry: Callback function called before each retry
        
    Returns:
        Decorated function
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                fn=fn,
                retry_policy=retry_policy,
                args=args,
                kwargs=kwargs,
                on_retry=on_retry
            )
        return wrapper
    return decorator


def with_rate_limit(
    rate_limiter: RateLimiter
):
    """
    Decorator to add rate limiting to a function.
    
    Args:
        rate_limiter: RateLimiter instance
        
    Returns:
        Decorated function
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Wait for rate limit
            await rate_limiter.wait()
            
            try:
                # Call the function
                start_time = time.time()
                result = await fn(*args, **kwargs)
                
                # Record success
                await rate_limiter.record_success()
                
                return result
            except Exception as e:
                # Record failure
                await rate_limiter.record_failure()
                raise
                
        return wrapper
    return decorator


def with_resilience(
    circuit_breaker: Union[CircuitBreaker, str] = None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    rate_limiter: RateLimiter = None,
    fallback_fn: Optional[Callable] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Combined decorator to add circuit breaking, retry, and rate limiting to a function.
    
    Args:
        circuit_breaker: Circuit breaker instance or name
        retry_policy: RetryPolicy instance
        rate_limiter: RateLimiter instance
        fallback_fn: Function to call if circuit breaker is open
        on_retry: Callback function called before each retry
        
    Returns:
        Decorated function
    """
    def decorator(fn):
        # Apply decorators in reverse order
        decorated_fn = fn
        
        # Apply retry
        if retry_policy:
            decorated_fn = with_retry(
                retry_policy=retry_policy,
                on_retry=on_retry
            )(decorated_fn)
        
        # Apply circuit breaker
        if circuit_breaker:
            decorated_fn = with_circuit_breaker(
                circuit_breaker=circuit_breaker,
                fallback_fn=fallback_fn
            )(decorated_fn)
        
        # Apply rate limiting
        if rate_limiter:
            decorated_fn = with_rate_limit(
                rate_limiter=rate_limiter
            )(decorated_fn)
        
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await decorated_fn(*args, **kwargs)
        
        return wrapper
    return decorator