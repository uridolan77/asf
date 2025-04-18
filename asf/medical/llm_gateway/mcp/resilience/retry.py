"""
Advanced retry policy implementation for resilient service communication.

This module provides configurable retry policies with exponential backoff,
jitter, and retry classification based on error types and codes.
"""

import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import structlog

# Custom imports for exception types
try:
    from mcp.shared.exceptions import McpError
except ImportError:
    # Define placeholder if MCP SDK is not available
    class McpError(Exception):
        pass

logger = structlog.get_logger("mcp_resilience.retry")


class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.
    
    Features:
    - Exponential backoff with configurable base/max delays
    - Random jitter to prevent retry storms
    - Error classification based on type or code
    - Customizable retry predicates
    - Retry budget management
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.2,
        retry_codes: Optional[Set[str]] = None,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
        retry_predicate: Optional[Callable[[Exception], bool]] = None,
        name: str = "default"
    ):
        """
        Initialize retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter_factor: Randomness factor (0.0-1.0) to apply to delays
            retry_codes: Set of error codes to retry (for McpError)
            retry_exceptions: List of exception types to retry
            retry_predicate: Custom function to determine if an error is retryable
            name: Name for this policy (for logging/metrics)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.retry_codes = retry_codes or set()
        self.retry_exceptions = retry_exceptions or [
            TimeoutError,
            ConnectionError
        ]
        self.retry_predicate = retry_predicate
        self.name = name
        
        # Statistics
        self.retry_attempts = 0
        self.retry_successes = 0
        self.retry_failures = 0
        self.last_retry_time = None
        
        self.logger = logger.bind(policy=name)
        
        self.logger.debug(
            "Initialized retry policy",
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_factor=jitter_factor,
        )
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a specific retry attempt with exponential backoff and jitter.
        
        Args:
            attempt: Current retry attempt (1-based)
            
        Returns:
            Delay in seconds for this attempt
        """
        # Base exponential delay: base_delay * 2^(attempt-1)
        delay = self.base_delay * (2 ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter_factor > 0:
            # Calculate jitter range
            jitter_range = delay * self.jitter_factor
            
            # Apply random jitter within range [-jitter_range/2, +jitter_range/2]
            jitter = random.uniform(-jitter_range/2, jitter_range/2)
            
            # Apply jitter, ensuring delay is positive
            delay = max(0.1, delay + jitter)
        
        return delay
    
    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error should be retried based on this policy.
        
        Args:
            error: The exception to check
            
        Returns:
            True if error should be retried
        """
        # Check custom predicate first if provided
        if self.retry_predicate is not None:
            try:
                return self.retry_predicate(error)
            except Exception as e:
                self.logger.warning(
                    "Error in custom retry predicate",
                    error=str(e)
                )
        
        # Check for MCP-specific errors with codes
        if isinstance(error, McpError) and hasattr(error, 'error') and hasattr(error.error, 'code'):
            return error.error.code in self.retry_codes
        
        # Check HTTP-like errors with status codes
        if hasattr(error, 'status_code'):
            status = error.status_code
            return status >= 500 or status == 429
        
        # Check response-like errors
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status = error.response.status_code
            return status >= 500 or status == 429
        
        # Check exception type
        for exc_type in self.retry_exceptions:
            if isinstance(error, exc_type):
                return True
        
        # Not retryable by our criteria
        return False

    def record_retry_attempt(self, successful: bool) -> None:
        """
        Record statistics about retry attempts.
        
        Args:
            successful: Whether the retry ultimately succeeded
        """
        self.retry_attempts += 1
        self.last_retry_time = datetime.utcnow()
        
        if successful:
            self.retry_successes += 1
        else:
            self.retry_failures += 1
        
        self.logger.debug(
            "Recorded retry attempt",
            successful=successful,
            total_attempts=self.retry_attempts,
            success_rate=f"{(self.retry_successes / self.retry_attempts * 100):.1f}%" if self.retry_attempts > 0 else "N/A"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this retry policy.
        
        Returns:
            Dict with retry metrics
        """
        return {
            "name": self.name,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter_factor": self.jitter_factor,
            "retry_attempts": self.retry_attempts,
            "retry_successes": self.retry_successes,
            "retry_failures": self.retry_failures,
            "success_rate": (self.retry_successes / self.retry_attempts * 100) if self.retry_attempts > 0 else 0,
            "last_retry": self.last_retry_time.isoformat() if self.last_retry_time else None,
        }


# Common retry policies
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    jitter_factor=0.2,
    name="default"
)

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    jitter_factor=0.1,
    name="aggressive"
)

CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
    jitter_factor=0.3,
    name="conservative"
)


async def retry_async(
    fn: Callable,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Any:
    """
    Retry an async function based on the provided retry policy.
    
    Args:
        fn: Async function to retry
        retry_policy: RetryPolicy to use
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        on_retry: Callback function called before each retry
        
    Returns:
        Result of the function
        
    Raises:
        Exception: The last exception if all retries fail
    """
    kwargs = kwargs or {}
    last_exception = None
    
    for attempt in range(retry_policy.max_retries + 1):
        try:
            # First attempt (attempt=0) or retry attempts
            if attempt > 0:
                # Calculate delay for this retry
                delay = retry_policy.calculate_delay(attempt)
                
                # Call retry callback if provided
                if on_retry:
                    try:
                        on_retry(attempt, last_exception, delay)
                    except Exception as e:
                        logger.warning(
                            "Error in retry callback",
                            error=str(e)
                        )
                
                # Log retry attempt
                logger.info(
                    "Retrying function call",
                    attempt=attempt,
                    max_attempts=retry_policy.max_retries + 1,
                    delay=delay,
                    error=str(last_exception)
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
            
            # Call the function
            result = await fn(*args, **kwargs)
            
            # Record successful retry if this wasn't the first attempt
            if attempt > 0:
                retry_policy.record_retry_attempt(successful=True)
            
            return result
        
        except Exception as e:
            last_exception = e
            
            # Check if the exception is retryable
            is_retryable = retry_policy.is_retryable_error(e)
            
            # Log the error
            log_method = logger.info if is_retryable else logger.warning
            log_method(
                "Function call failed",
                attempt=attempt + 1,
                max_attempts=retry_policy.max_retries + 1,
                error=str(e),
                error_type=type(e).__name__,
                retryable=is_retryable
            )
            
            # If not retryable or last attempt, don't retry
            if not is_retryable or attempt >= retry_policy.max_retries:
                if attempt > 0:
                    retry_policy.record_retry_attempt(successful=False)
                raise
    
    # This should never be reached due to the raise in the loop
    raise last_exception


import asyncio
from functools import wraps


def retry_async_decorator(retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY):
    """
    Decorator for retrying async functions based on a retry policy.
    
    Args:
        retry_policy: RetryPolicy to use
        
    Returns:
        Decorated function
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                fn=fn,
                retry_policy=retry_policy,
                args=args,
                kwargs=kwargs
            )
        return wrapper
    return decorator