"""
Retry module for the Medical Research Synthesizer.

This module provides functionality for retrying operations that may fail
transiently, with configurable backoff strategies and retry conditions.

Classes:
    RetryStrategy: Base class for retry strategies.
    ExponentialBackoff: Retry strategy with exponential backoff.
    FixedBackoff: Retry strategy with fixed delay between retries.
    LinearBackoff: Retry strategy with linearly increasing delays.

Functions:
    retry: Decorator for retrying functions with a specific strategy.
    async_retry: Decorator for retrying async functions with a specific strategy.
"""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log,
)

from .exceptions import ExternalServiceError, DatabaseError

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExceptionTypes = Union[Type[Exception], List[Type[Exception]]]

class RetryStrategy:
    """
    Base class for retry strategies.

    This class defines the interface for retry strategies and provides
    common functionality for determining whether to retry an operation.

    Attributes:
        max_retries (int): Maximum number of retries.
        retry_on (Tuple[Type[Exception], ...]): Exception types to retry on.
        timeout (float): Maximum total time to spend on retries.
    """

    def __init__(self, max_retries: int = 3, retry_on: Tuple[Type[Exception], ...] = (Exception,), timeout: float = None):
        """
        Initialize the RetryStrategy.

        Args:
            max_retries (int, optional): Maximum number of retries. Defaults to 3.
            retry_on (Tuple[Type[Exception], ...], optional): Exception types to retry on. Defaults to (Exception,).
            timeout (float, optional): Maximum total time to spend on retries. Defaults to None.
        """
        pass

    def should_retry(self, attempt: int, elapsed: float, exception: Exception) -> bool:
        """
        Determine if an operation should be retried.

        Args:
            attempt (int): Current attempt number (1-based).
            elapsed (float): Time elapsed since first attempt.
            exception (Exception): Exception that caused the retry.

        Returns:
            bool: True if the operation should be retried, False otherwise.
        """
        pass

    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.

        Args:
            attempt (int): Current attempt number (0-based).

        Returns:
            float: Delay in seconds.
        """
        pass

class ExponentialBackoff(RetryStrategy):
    """
    Retry strategy with exponential backoff.

    This strategy increases the delay between retries exponentially.

    Attributes:
        base_delay (float): Base delay in seconds.
        max_delay (float): Maximum delay in seconds.
        factor (float): Exponential factor.
        jitter (bool): Whether to add randomness to delays.
    """

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, factor: float = 2.0, jitter: bool = True, **kwargs):
        """
        Initialize the ExponentialBackoff strategy.

        Args:
            base_delay (float, optional): Base delay in seconds. Defaults to 1.0.
            max_delay (float, optional): Maximum delay in seconds. Defaults to 60.0.
            factor (float, optional): Exponential factor. Defaults to 2.0.
            jitter (bool, optional): Whether to add randomness to delays. Defaults to True.
            **kwargs: Additional arguments passed to RetryStrategy.
        """
        pass

    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.

        Args:
            attempt (int): Current attempt number (0-based).

        Returns:
            float: Delay in seconds.
        """
        pass

class FixedBackoff(RetryStrategy):
    """
    Retry strategy with fixed delay between retries.

    This strategy uses the same delay for all retry attempts.

    Attributes:
        delay (float): Delay in seconds.
        jitter (bool): Whether to add randomness to delays.
    """

    def __init__(self, delay: float = 1.0, jitter: bool = True, **kwargs):
        """
        Initialize the FixedBackoff strategy.

        Args:
            delay (float, optional): Delay in seconds. Defaults to 1.0.
            jitter (bool, optional): Whether to add randomness to delays. Defaults to True.
            **kwargs: Additional arguments passed to RetryStrategy.
        """
        pass

    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.

        Args:
            attempt (int): Current attempt number (0-based).

        Returns:
            float: Delay in seconds.
        """
        pass

class LinearBackoff(RetryStrategy):
    """
    Retry strategy with linearly increasing delays.

    This strategy increases the delay between retries linearly.

    Attributes:
        base_delay (float): Base delay in seconds.
        factor (float): Linear factor.
        jitter (bool): Whether to add randomness to delays.
    """

    def __init__(self, base_delay: float = 1.0, factor: float = 1.0, jitter: bool = True, **kwargs):
        """
        Initialize the LinearBackoff strategy.

        Args:
            base_delay (float, optional): Base delay in seconds. Defaults to 1.0.
            factor (float, optional): Linear factor. Defaults to 1.0.
            jitter (bool, optional): Whether to add randomness to delays. Defaults to True.
            **kwargs: Additional arguments passed to RetryStrategy.
        """
        pass

    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.

        Args:
            attempt (int): Current attempt number (0-based).

        Returns:
            float: Delay in seconds.
        """
        pass

def retry(strategy: RetryStrategy = None, **kwargs):
    """
    Decorator for retrying functions with a specific strategy.

    Args:
        strategy (RetryStrategy, optional): Retry strategy to use. Defaults to None.
        **kwargs: Arguments passed to the default strategy if no strategy is provided.

    Returns:
        Callable: Decorator function.
    """
    pass

def async_retry(strategy: RetryStrategy = None, **kwargs):
    """
    Decorator for retrying async functions with a specific strategy.

    Args:
        strategy (RetryStrategy, optional): Retry strategy to use. Defaults to None.
        **kwargs: Arguments passed to the default strategy if no strategy is provided.

    Returns:
        Callable: Decorator function.
    """
    pass

def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exception_types: Optional[ExceptionTypes] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with custom retry logic.

    Args:
        max_attempts (int): Maximum number of retry attempts.
        min_wait (float): Minimum wait time between retries.
        max_wait (float): Maximum wait time between retries.
        exception_types (Optional[ExceptionTypes]): Exception types to retry on.
        on_retry (Optional[Callable[[Exception, int], None]]): Callback function on retry.

    Returns:
        Callable[[Callable[..., T]], Callable[..., T]]: Decorated function with retry logic.
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
                    
                    if exception_types is not None:
                        if isinstance(exception_types, list):
                            if not any(isinstance(e, exc_type) for exc_type in exception_types):
                                raise
                        elif not isinstance(e, exception_types):
                            raise
                    
                    if attempt >= max_attempts:
                        break
                    
                    wait_time = min(max_wait, min_wait * (2 ** (attempt - 1)))
                    wait_time = wait_time * (1 + random.uniform(-0.1, 0.1))  # Add jitter
                    
                    if on_retry is not None:
                        on_retry(e, attempt)
                    
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e}. "
                        f"Attempt {attempt}/{max_attempts}. "
                        f"Waiting {wait_time:.2f} seconds."
                    )
                    
                    time.sleep(wait_time)
            
            if last_exception is not None:
                raise last_exception
            
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
    Retry decorator using the Tenacity library.

    Args:
        max_attempts (int): Maximum number of retry attempts.
        min_wait (float): Minimum wait time between retries.
        max_wait (float): Maximum wait time between retries.
        exception_types (Optional[ExceptionTypes]): Exception types to retry on.

    Returns:
        Callable[[Callable[..., T]], Callable[..., T]]: Decorated function with retry logic.
    """
    if exception_types is None:
        retry_condition = None
    elif isinstance(exception_types, list):
        retry_condition = retry_if_exception_type(tuple(exception_types))
    else:
        retry_condition = retry_if_exception_type(exception_types)
    
    tenacity_decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    
    return tenacity_decorator

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

__all__ = [
    "RetryStrategy",
    "ExponentialBackoff",
    "FixedBackoff",
    "LinearBackoff",
    "retry",
    "async_retry",
    "with_retry",
    "with_tenacity_retry",
    "retry_external_service",
    "retry_database",
]
