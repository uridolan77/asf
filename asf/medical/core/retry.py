"""
Retry module for the Medical Research Synthesizer.

This module provides utilities for implementing retry logic with exponential backoff.
"""

import logging
import random
import time
from typing import Callable, TypeVar, Optional, List, Type, Dict, Any, Union
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log,
)

from asf.medical.core.exceptions import ExternalServiceError, DatabaseError

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
ExceptionTypes = Union[Type[Exception], List[Type[Exception]]]

def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exception_types: Optional[ExceptionTypes] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        min_wait: Minimum wait time between attempts (in seconds)
        max_wait: Maximum wait time between attempts (in seconds)
        exception_types: Exception types to retry on (default: all exceptions)
        on_retry: Callback function to call before retrying
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if exception_types is not None:
                        if isinstance(exception_types, list):
                            if not any(isinstance(e, exc_type) for exc_type in exception_types):
                                raise
                        elif not isinstance(e, exception_types):
                            raise
                    
                    # Check if we've reached the maximum number of attempts
                    if attempt >= max_attempts:
                        break
                    
                    # Calculate wait time with exponential backoff and jitter
                    wait_time = min(max_wait, min_wait * (2 ** (attempt - 1)))
                    wait_time = wait_time * (1 + random.uniform(-0.1, 0.1))  # Add jitter
                    
                    # Call on_retry callback if provided
                    if on_retry is not None:
                        on_retry(e, attempt)
                    
                    # Log the retry
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e}. "
                        f"Attempt {attempt}/{max_attempts}. "
                        f"Waiting {wait_time:.2f} seconds."
                    )
                    
                    # Wait before retrying
                    time.sleep(wait_time)
            
            # If we've exhausted all attempts, raise the last exception
            if last_exception is not None:
                raise last_exception
            
            # This should never happen, but just in case
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")
        
        return wrapper
    
    return decorator

def with_tenacity_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exception_types: Optional[ExceptionTypes] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying a function with tenacity.
    
    Args:
        max_attempts: Maximum number of attempts
        min_wait: Minimum wait time between attempts (in seconds)
        max_wait: Maximum wait time between attempts (in seconds)
        exception_types: Exception types to retry on (default: all exceptions)
        
    Returns:
        Decorated function
    """
    # Set up retry conditions
    if exception_types is None:
        retry_condition = None
    elif isinstance(exception_types, list):
        retry_condition = retry_if_exception_type(tuple(exception_types))
    else:
        retry_condition = retry_if_exception_type(exception_types)
    
    # Create the decorator
    tenacity_decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    
    return tenacity_decorator

# Common retry decorators
retry_external_service = with_retry(
    max_attempts=3,
    min_wait=1.0,
    max_wait=10.0,
    exception_types=ExternalServiceError,
)

retry_database = with_retry(
    max_attempts=5,
    min_wait=0.5,
    max_wait=5.0,
    exception_types=DatabaseError,
)

# Export decorators
__all__ = [
    "with_retry",
    "with_tenacity_retry",
    "retry_external_service",
    "retry_database",
]
