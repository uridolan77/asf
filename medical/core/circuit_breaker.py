"""
Circuit Breaker module for the Medical Research Synthesizer.

This module provides a circuit breaker implementation to handle failures gracefully
and prevent cascading failures in the system.

Classes:
    CircuitBreaker: Implements the circuit breaker pattern.

Exceptions:
    CircuitBreakerOpenException: Raised when the circuit breaker is open.
"""

import logging
import time
from enum import Enum
from typing import TypeVar, List, Type, Union
from threading import Lock

logger = logging.getLogger(__name__)
T = TypeVar("T")
ExceptionTypes = Union[Type[Exception], List[Type[Exception]]]


class CircuitState(Enum):
    """
    Circuit breaker states.
    """
    CLOSED = "closed"  # Normal operation, requests are allowed
    OPEN = "open"  # Circuit is open, requests are not allowed
    HALF_OPEN = "half_open"  # Testing if the service is back online


class CircuitBreakerOpenException(Exception):
    """
    Exception raised when the circuit breaker is open.

    Attributes:
        message (str): Explanation of the exception.
    """

    def __init__(self, message: str = "Circuit breaker is open."):
        """
        Initialize the CircuitBreakerOpenException.

        Args:
            message (str): Explanation of the exception. Defaults to "Circuit breaker is open.".
        """
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker implementation.

    This class implements the circuit breaker pattern to prevent cascading failures
    when interacting with external services.

    Attributes:
        failure_threshold (int): Number of failures before opening the circuit.
        recovery_timeout (float): Time in seconds before attempting recovery.
        half_open_max_calls (int): Maximum number of calls allowed in half-open state.
        exception_types (ExceptionTypes): Exception types to count as failures.
        state (CircuitState): Current state of the circuit.
        failure_count (int): Current count of failures.
        last_failure_time (float): Timestamp of the last failure.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        exception_types: ExceptionTypes = Exception
    ):
        """
        Initialize the circuit breaker.

        Args:
            name (str): Name of the circuit breaker.
            failure_threshold (int): Number of failures before opening the circuit.
            recovery_timeout (float): Time to wait before trying to recover (in seconds).
            half_open_max_calls (int): Maximum number of calls allowed in half-open state.
            exception_types (ExceptionTypes): Exception types to count as failures.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exception_types = exception_types
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._last_failure_time = 0.0
        self._lock = Lock()

    def get_state(self) -> CircuitState:
        """
        Get the current circuit breaker state.

        Returns:
            CircuitState: The current circuit breaker state.
        """
        return self._state

    def is_open(self) -> bool:
        """
        Check if the circuit is open.

        Returns:
            bool: True if the circuit is open, False otherwise.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned from OPEN to HALF_OPEN")
            return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """
        Check if the circuit breaker is in the half-open state.

        Returns:
            bool: True if the circuit breaker is half-open, False otherwise.
        """
        with self._lock:
            return self._state == CircuitState.HALF_OPEN

    def success(self) -> None:
        """
        Record a successful call.

        Transitions the circuit breaker to CLOSED state if it was in HALF_OPEN state.
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
            exception (Exception): The exception that caused the failure.
        """
        with self._lock:
            if not isinstance(exception, self.exception_types):
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN state")