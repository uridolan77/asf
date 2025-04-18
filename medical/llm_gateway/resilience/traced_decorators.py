"""
Decorators for resilience patterns with tracing integration.

This module provides decorators for applying resilience patterns
to functions and methods, with integrated distributed tracing.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import structlog

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    FailureCategory
)
from asf.medical.llm_gateway.resilience.factory import get_resilience_factory
from asf.medical.llm_gateway.resilience.metrics import get_resilience_metrics
from asf.medical.llm_gateway.resilience.tracing import get_resilience_tracing
from asf.medical.llm_gateway.resilience.decorators import (
    CircuitOpenError,
    _get_or_create_circuit_breaker
)

logger = structlog.get_logger("llm_gateway.resilience.traced_decorators")

# Type variables for better type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def with_traced_circuit_breaker(
    circuit_breaker: Optional[Union[CircuitBreaker, EnhancedCircuitBreaker, str]] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    failure_exceptions: Optional[List[Type[Exception]]] = None,
    failure_category_mapping: Optional[Dict[Type[Exception], FailureCategory]] = None,
    enhanced: bool = True,
    provider_id: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for applying circuit breaker pattern with tracing to a function or method.
    
    Args:
        circuit_breaker: Circuit breaker instance or name (if string, will be looked up in registry)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        failure_exceptions: List of exception types to count as failures
        failure_category_mapping: Mapping of exception types to failure categories
        enhanced: Whether to use enhanced circuit breaker if creating a new one
        provider_id: ID of the provider (for metrics)
        
    Returns:
        Decorated function
    """
    # Default failure exceptions if not provided
    if failure_exceptions is None:
        failure_exceptions = [Exception]
    
    # Default failure category mapping if not provided
    if failure_category_mapping is None:
        failure_category_mapping = {}
    
    def decorator(func: F) -> F:
        # Get function name for circuit breaker name if not provided
        func_name = func.__qualname__
        cb_name = func_name if circuit_breaker is None else circuit_breaker
        
        # Get or create circuit breaker
        cb = _get_or_create_circuit_breaker(
            cb_name,
            failure_threshold,
            recovery_timeout,
            enhanced,
            provider_id
        )
        
        # Get tracing service
        tracing = get_resilience_tracing()
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Check if circuit is open
                if cb.is_open():
                    # Create circuit open span
                    with tracing.circuit_breaker_span(
                        circuit_breaker=cb,
                        operation="open",
                        provider_id=provider_id,
                        attributes={"function": func_name}
                    ) as span:
                        span.set_attribute("circuit_breaker.open", True)
                        raise CircuitOpenError(f"Circuit '{cb.name}' is open")
                
                # Create circuit breaker span
                with tracing.circuit_breaker_span(
                    circuit_breaker=cb,
                    operation="call",
                    provider_id=provider_id,
                    attributes={"function": func_name}
                ) as span:
                    try:
                        # Call the function
                        result = await func(*args, **kwargs)
                        
                        # Record success
                        cb.record_success()
                        
                        # Record success in tracing
                        tracing.record_success(
                            circuit_breaker=cb,
                            provider_id=provider_id
                        )
                        
                        return result
                    except Exception as e:
                        # Check if this exception counts as a failure
                        if any(isinstance(e, exc_type) for exc_type in failure_exceptions):
                            # Determine failure category for enhanced circuit breaker
                            category = FailureCategory.UNKNOWN
                            if isinstance(cb, EnhancedCircuitBreaker):
                                for exc_type, cat in failure_category_mapping.items():
                                    if isinstance(e, exc_type):
                                        category = cat
                                        break
                                
                                # Record failure with category
                                cb.record_failure(category, exception=e)
                            else:
                                # Record failure without category
                                cb.record_failure()
                            
                            # Record failure in tracing
                            tracing.record_failure(
                                circuit_breaker=cb,
                                failure_count=getattr(cb, "_failure_count", 1),
                                failure_category=category if isinstance(cb, EnhancedCircuitBreaker) else None,
                                provider_id=provider_id,
                                exception=e
                            )
                        
                        # Re-raise the exception
                        raise
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Check if circuit is open
                if cb.is_open():
                    # Create circuit open span
                    with tracing.circuit_breaker_span(
                        circuit_breaker=cb,
                        operation="open",
                        provider_id=provider_id,
                        attributes={"function": func_name}
                    ) as span:
                        span.set_attribute("circuit_breaker.open", True)
                        raise CircuitOpenError(f"Circuit '{cb.name}' is open")
                
                # Create circuit breaker span
                with tracing.circuit_breaker_span(
                    circuit_breaker=cb,
                    operation="call",
                    provider_id=provider_id,
                    attributes={"function": func_name}
                ) as span:
                    try:
                        # Call the function
                        result = func(*args, **kwargs)
                        
                        # Record success
                        cb.record_success()
                        
                        # Record success in tracing
                        tracing.record_success(
                            circuit_breaker=cb,
                            provider_id=provider_id
                        )
                        
                        return result
                    except Exception as e:
                        # Check if this exception counts as a failure
                        if any(isinstance(e, exc_type) for exc_type in failure_exceptions):
                            # Determine failure category for enhanced circuit breaker
                            category = FailureCategory.UNKNOWN
                            if isinstance(cb, EnhancedCircuitBreaker):
                                for exc_type, cat in failure_category_mapping.items():
                                    if isinstance(e, exc_type):
                                        category = cat
                                        break
                                
                                # Record failure with category
                                cb.record_failure(category, exception=e)
                            else:
                                # Record failure without category
                                cb.record_failure()
                            
                            # Record failure in tracing
                            tracing.record_failure(
                                circuit_breaker=cb,
                                failure_count=getattr(cb, "_failure_count", 1),
                                failure_category=category if isinstance(cb, EnhancedCircuitBreaker) else None,
                                provider_id=provider_id,
                                exception=e
                            )
                        
                        # Re-raise the exception
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def with_traced_provider_circuit_breaker(
    provider_method: bool = True,
    method_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    failure_exceptions: Optional[List[Type[Exception]]] = None,
    failure_category_mapping: Optional[Dict[Type[Exception], FailureCategory]] = None
) -> Callable[[F], F]:
    """
    Decorator for applying circuit breaker pattern with tracing to a provider method.
    
    This decorator is specifically designed for LLM Gateway providers,
    automatically using the provider ID and method name for the circuit breaker.
    
    Args:
        provider_method: Whether this is a provider method (requires self.provider_id)
        method_name: Override method name for circuit breaker name
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        failure_exceptions: List of exception types to count as failures
        failure_category_mapping: Mapping of exception types to failure categories
        
    Returns:
        Decorated function
    """
    # Default failure exceptions if not provided
    if failure_exceptions is None:
        failure_exceptions = [Exception]
    
    # Default failure category mapping if not provided
    if failure_category_mapping is None:
        failure_category_mapping = {}
    
    def decorator(func: F) -> F:
        # Get method name
        func_name = method_name or func.__name__
        
        # Get tracing service
        tracing = get_resilience_tracing()
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Get provider ID
                provider_id = getattr(self, "provider_id", None)
                if provider_method and provider_id is None:
                    logger.warning(
                        "Provider ID not found on self, using function name as circuit breaker name",
                        function=func_name
                    )
                
                # Create circuit breaker name
                cb_name = f"{provider_id}_{func_name}" if provider_id else func_name
                
                # Get or create circuit breaker
                cb = _get_or_create_circuit_breaker(
                    cb_name,
                    failure_threshold,
                    recovery_timeout,
                    True,  # Always use enhanced circuit breaker
                    provider_id
                )
                
                # Check if circuit is open
                if cb.is_open():
                    # Create circuit open span
                    with tracing.circuit_breaker_span(
                        circuit_breaker=cb,
                        operation="open",
                        provider_id=provider_id,
                        attributes={"function": func_name}
                    ) as span:
                        span.set_attribute("circuit_breaker.open", True)
                        raise CircuitOpenError(f"Circuit '{cb.name}' is open")
                
                # Create circuit breaker span
                with tracing.circuit_breaker_span(
                    circuit_breaker=cb,
                    operation="call",
                    provider_id=provider_id,
                    attributes={"function": func_name}
                ) as span:
                    try:
                        # Call the function
                        result = await func(self, *args, **kwargs)
                        
                        # Record success
                        cb.record_success()
                        
                        # Record success in tracing
                        tracing.record_success(
                            circuit_breaker=cb,
                            provider_id=provider_id
                        )
                        
                        return result
                    except Exception as e:
                        # Check if this exception counts as a failure
                        if any(isinstance(e, exc_type) for exc_type in failure_exceptions):
                            # Determine failure category
                            category = FailureCategory.UNKNOWN
                            for exc_type, cat in failure_category_mapping.items():
                                if isinstance(e, exc_type):
                                    category = cat
                                    break
                            
                            # Record failure with category
                            cb.record_failure(category, exception=e)
                            
                            # Record failure in tracing
                            tracing.record_failure(
                                circuit_breaker=cb,
                                failure_count=getattr(cb, "_failure_count", 1),
                                failure_category=category,
                                provider_id=provider_id,
                                exception=e
                            )
                        
                        # Re-raise the exception
                        raise
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Get provider ID
                provider_id = getattr(self, "provider_id", None)
                if provider_method and provider_id is None:
                    logger.warning(
                        "Provider ID not found on self, using function name as circuit breaker name",
                        function=func_name
                    )
                
                # Create circuit breaker name
                cb_name = f"{provider_id}_{func_name}" if provider_id else func_name
                
                # Get or create circuit breaker
                cb = _get_or_create_circuit_breaker(
                    cb_name,
                    failure_threshold,
                    recovery_timeout,
                    True,  # Always use enhanced circuit breaker
                    provider_id
                )
                
                # Check if circuit is open
                if cb.is_open():
                    # Create circuit open span
                    with tracing.circuit_breaker_span(
                        circuit_breaker=cb,
                        operation="open",
                        provider_id=provider_id,
                        attributes={"function": func_name}
                    ) as span:
                        span.set_attribute("circuit_breaker.open", True)
                        raise CircuitOpenError(f"Circuit '{cb.name}' is open")
                
                # Create circuit breaker span
                with tracing.circuit_breaker_span(
                    circuit_breaker=cb,
                    operation="call",
                    provider_id=provider_id,
                    attributes={"function": func_name}
                ) as span:
                    try:
                        # Call the function
                        result = func(self, *args, **kwargs)
                        
                        # Record success
                        cb.record_success()
                        
                        # Record success in tracing
                        tracing.record_success(
                            circuit_breaker=cb,
                            provider_id=provider_id
                        )
                        
                        return result
                    except Exception as e:
                        # Check if this exception counts as a failure
                        if any(isinstance(e, exc_type) for exc_type in failure_exceptions):
                            # Determine failure category
                            category = FailureCategory.UNKNOWN
                            for exc_type, cat in failure_category_mapping.items():
                                if isinstance(e, exc_type):
                                    category = cat
                                    break
                            
                            # Record failure with category
                            cb.record_failure(category, exception=e)
                            
                            # Record failure in tracing
                            tracing.record_failure(
                                circuit_breaker=cb,
                                failure_count=getattr(cb, "_failure_count", 1),
                                failure_category=category,
                                provider_id=provider_id,
                                exception=e
                            )
                        
                        # Re-raise the exception
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def with_traced_retry(
    max_attempts: int = 3,
    retry_exceptions: Optional[List[Type[Exception]]] = None,
    backoff_factor: float = 0.5,
    jitter: bool = True,
    max_backoff: float = 60.0,
    retry_on_circuit_open: bool = False
) -> Callable[[F], F]:
    """
    Decorator for applying retry pattern with tracing to a function or method.
    
    Args:
        max_attempts: Maximum number of attempts
        retry_exceptions: List of exception types to retry on
        backoff_factor: Factor for exponential backoff
        jitter: Whether to add jitter to backoff
        max_backoff: Maximum backoff in seconds
        retry_on_circuit_open: Whether to retry on CircuitOpenError
        
    Returns:
        Decorated function
    """
    # Default retry exceptions if not provided
    if retry_exceptions is None:
        retry_exceptions = [Exception]
    
    # Add CircuitOpenError to retry exceptions if requested
    if retry_on_circuit_open and CircuitOpenError not in retry_exceptions:
        retry_exceptions.append(CircuitOpenError)
    
    def decorator(func: F) -> F:
        # Get function name for logging
        func_name = func.__qualname__
        
        # Get resilience metrics
        metrics = get_resilience_metrics()
        
        # Get tracing service
        tracing = get_resilience_tracing()
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get provider ID if available (for metrics)
                provider_id = "unknown"
                if args and hasattr(args[0], "provider_id"):
                    provider_id = getattr(args[0], "provider_id")
                
                # Try the function with retries
                attempt = 1
                last_exception = None
                
                while attempt <= max_attempts:
                    try:
                        # Record retry attempt
                        if attempt > 1:
                            metrics.record_retry_attempt(
                                name=func_name,
                                provider_id=provider_id,
                                attempt=attempt,
                                max_attempts=max_attempts
                            )
                            
                            # Record retry in tracing
                            tracing.record_retry(
                                operation=func_name,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                provider_id=provider_id,
                                exception=last_exception
                            )
                            
                            logger.info(
                                f"Retry attempt {attempt}/{max_attempts}",
                                function=func_name,
                                provider_id=provider_id
                            )
                        
                        # Call the function
                        return await func(*args, **kwargs)
                    except Exception as e:
                        # Check if this exception should be retried
                        if not any(isinstance(e, exc_type) for exc_type in retry_exceptions):
                            # Don't retry this exception
                            raise
                        
                        # Save the exception
                        last_exception = e
                        
                        # Check if we've reached max attempts
                        if attempt >= max_attempts:
                            logger.warning(
                                f"Max retry attempts reached",
                                function=func_name,
                                provider_id=provider_id,
                                max_attempts=max_attempts,
                                exception=str(e)
                            )
                            raise
                        
                        # Calculate backoff
                        backoff = min(
                            backoff_factor * (2 ** (attempt - 1)),
                            max_backoff
                        )
                        
                        # Add jitter if requested
                        if jitter:
                            import random
                            backoff = backoff * (0.5 + random.random())
                        
                        logger.info(
                            f"Retrying after exception",
                            function=func_name,
                            provider_id=provider_id,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            backoff=backoff,
                            exception=str(e)
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(backoff)
                        
                        # Increment attempt counter
                        attempt += 1
                
                # Should never get here, but just in case
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry loop exited without returning or raising")
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get provider ID if available (for metrics)
                provider_id = "unknown"
                if args and hasattr(args[0], "provider_id"):
                    provider_id = getattr(args[0], "provider_id")
                
                # Try the function with retries
                attempt = 1
                last_exception = None
                
                while attempt <= max_attempts:
                    try:
                        # Record retry attempt
                        if attempt > 1:
                            metrics.record_retry_attempt(
                                name=func_name,
                                provider_id=provider_id,
                                attempt=attempt,
                                max_attempts=max_attempts
                            )
                            
                            # Record retry in tracing
                            tracing.record_retry(
                                operation=func_name,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                provider_id=provider_id,
                                exception=last_exception
                            )
                            
                            logger.info(
                                f"Retry attempt {attempt}/{max_attempts}",
                                function=func_name,
                                provider_id=provider_id
                            )
                        
                        # Call the function
                        return func(*args, **kwargs)
                    except Exception as e:
                        # Check if this exception should be retried
                        if not any(isinstance(e, exc_type) for exc_type in retry_exceptions):
                            # Don't retry this exception
                            raise
                        
                        # Save the exception
                        last_exception = e
                        
                        # Check if we've reached max attempts
                        if attempt >= max_attempts:
                            logger.warning(
                                f"Max retry attempts reached",
                                function=func_name,
                                provider_id=provider_id,
                                max_attempts=max_attempts,
                                exception=str(e)
                            )
                            raise
                        
                        # Calculate backoff
                        backoff = min(
                            backoff_factor * (2 ** (attempt - 1)),
                            max_backoff
                        )
                        
                        # Add jitter if requested
                        if jitter:
                            import random
                            backoff = backoff * (0.5 + random.random())
                        
                        logger.info(
                            f"Retrying after exception",
                            function=func_name,
                            provider_id=provider_id,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            backoff=backoff,
                            exception=str(e)
                        )
                        
                        # Wait before retrying
                        time.sleep(backoff)
                        
                        # Increment attempt counter
                        attempt += 1
                
                # Should never get here, but just in case
                if last_exception:
                    raise last_exception
                raise RuntimeError("Retry loop exited without returning or raising")
            
            return cast(F, sync_wrapper)
    
    return decorator


def with_traced_timeout(
    timeout_seconds: float,
    timeout_exception: Type[Exception] = asyncio.TimeoutError
) -> Callable[[F], F]:
    """
    Decorator for applying timeout pattern with tracing to an async function or method.
    
    Args:
        timeout_seconds: Timeout in seconds
        timeout_exception: Exception to raise on timeout
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Get function name for logging
        func_name = func.__qualname__
        
        # Get resilience metrics
        metrics = get_resilience_metrics()
        
        # Get tracing service
        tracing = get_resilience_tracing()
        
        # Check if function is async
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"Function {func_name} must be async to use with_traced_timeout")
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get provider ID if available (for metrics)
            provider_id = "unknown"
            if args and hasattr(args[0], "provider_id"):
                provider_id = getattr(args[0], "provider_id")
            
            # Start timer
            start_time = time.time()
            
            try:
                # Call the function with timeout
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                # Calculate actual duration
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Record timeout
                metrics.record_timeout(
                    name=func_name,
                    provider_id=provider_id,
                    timeout_ms=int(timeout_seconds * 1000),
                    actual_duration_ms=duration_ms
                )
                
                # Record timeout in tracing
                tracing.record_timeout(
                    operation=func_name,
                    timeout_seconds=timeout_seconds,
                    actual_duration_ms=duration_ms,
                    provider_id=provider_id
                )
                
                logger.warning(
                    f"Function timed out",
                    function=func_name,
                    provider_id=provider_id,
                    timeout_seconds=timeout_seconds,
                    actual_duration_ms=duration_ms
                )
                
                # Raise the specified exception
                if timeout_exception is asyncio.TimeoutError:
                    raise
                else:
                    raise timeout_exception(
                        f"Function {func_name} timed out after {timeout_seconds} seconds"
                    )
            finally:
                # Calculate actual duration
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Log duration
                logger.debug(
                    f"Function duration",
                    function=func_name,
                    provider_id=provider_id,
                    duration_ms=duration_ms,
                    timeout_seconds=timeout_seconds
                )
        
        return cast(F, async_wrapper)
    
    return decorator
