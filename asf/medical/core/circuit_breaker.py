"""
Circuit Breaker module for the Medical Research Synthesizer.
This module provides a circuit breaker pattern implementation to prevent
cascading failures when interacting with external services.
"""
import logging
import time
from enum import Enum
from typing import TypeVar, List, Type, Union
logger = logging.getLogger(__name__)
T = TypeVar("T")
ExceptionTypes = Union[Type[Exception], List[Type[Exception]]]
class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests are allowed
    OPEN = "open"  # Circuit is open, requests are not allowed
    HALF_OPEN = "half_open"  # Testing if the service is back online
class CircuitBreaker:
    """
    Circuit breaker implementation.
    This class implements the circuit breaker pattern to prevent cascading failures
    when interacting with external services.
        Get a circuit breaker instance by name.
        Args:
            name: Name of the circuit breaker
        Returns:
            CircuitBreaker instance
        Initialize the circuit breaker.
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time to wait before trying to recover (in seconds)
            half_open_max_calls: Maximum number of calls allowed in half-open state
            exception_types: Exception types to count as failures
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned from OPEN to HALF_OPEN")
            return self._state
    def success(self) -> None:
        """Record a successful call.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    def failure(self, exception: Exception) -> None:
        """
        Record a failed call.
        Args:
            exception: The exception that caused the failure
        Check if a request is allowed.
        Returns:
            True if the request is allowed, False otherwise
        Decorator for applying the circuit breaker to a function.
        Args:
            func: The function to decorate
        Returns:
            Decorated function
    Decorator for applying a circuit breaker to a function.
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time to wait before trying to recover (in seconds)
        half_open_max_calls: Maximum number of calls allowed in half-open state
        exception_types: Exception types to count as failures
    Returns:
        Decorated function