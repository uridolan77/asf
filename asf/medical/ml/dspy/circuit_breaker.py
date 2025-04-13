"""Circuit Breaker Pattern for API Reliability

This module implements the circuit breaker pattern for external API calls,
preventing cascading failures by temporarily disabling endpoints that are experiencing high failure rates.
"""

import logging
import time
import asyncio
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, TypeVar, Generic

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation - requests go through
    OPEN = "open"          # Circuit is open - requests fail fast
    HALF_OPEN = "half-open"  # Testing if service is healthy again


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open."""

    def __init__(self, message: str):
        """Initialize the exception.

        Args:
            message: Error message
        """
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """Circuit breaker for external API calls.

    This class implements the circuit breaker pattern to prevent cascading failures
    by temporarily disabling endpoints that are experiencing high failure rates.

    Attributes:
        name: Name of the protected resource
        state: Current state of the circuit breaker
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before attempting recovery
        success_threshold: Successes needed in half-open state to close
        half_open_max_calls: Maximum concurrent calls in half-open state
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 1,
        excluded_exceptions: Optional[List[type]] = None
    ):
        """Initialize the circuit breaker.

        Args:
            name: Name of the protected resource
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in half-open state to close
            half_open_max_calls: Maximum concurrent calls in half-open state
            excluded_exceptions: Exceptions that should not count as failures
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or []

        self.failure_counter = 0
        self.last_failure_time = 0
        self.success_counter = 0

        self._lock = threading.RLock()
        self._half_open_semaphore = threading.BoundedSemaphore(half_open_max_calls)

        # Metrics
        self.total_failures = 0
        self.total_successes = 0
        self.open_circuits = 0
        self.last_state_change = time.time()

        logger.info(f"Circuit breaker '{name}' initialized")

    def __enter__(self):
        """Enter the circuit breaker context."""
        self.before_request()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the circuit breaker context."""
        if exc_type is not None:
            # Check if this exception should be excluded
            if not any(isinstance(exc_val, exc_type) for exc_type in self.excluded_exceptions):
                # An exception occurred that should count as a failure
                self.on_failure()
            return False  # Propagate the exception
        else:
            # No exception - success
            self.on_success()
            return False

    def before_request(self):
        """Check if the request can proceed.

        Raises:
            CircuitOpenError: If the circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if reset timeout has expired
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF-OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_counter = 0
                    self.last_state_change = time.time()
                else:
                    # Still open
                    remaining = self.reset_timeout - (time.time() - self.last_failure_time)
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {remaining:.1f}s"
                    )

            if self.state == CircuitState.HALF_OPEN:
                # Only allow limited calls in half-open state
                if not self._half_open_semaphore.acquire(blocking=False):
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is HALF-OPEN and at capacity"
                    )

    def on_success(self):
        """Record a successful operation."""
        with self._lock:
            self.total_successes += 1

            if self.state == CircuitState.CLOSED:
                # Reset failure counter on success in closed state
                self.failure_counter = 0
            elif self.state == CircuitState.HALF_OPEN:
                # Release semaphore
                self._half_open_semaphore.release()

                # Increment success counter and check if we should close the circuit
                self.success_counter += 1
                if self.success_counter >= self.success_threshold:
                    logger.info(f"Circuit '{self.name}' closing after {self.success_counter} successful calls")
                    self.state = CircuitState.CLOSED
                    self.failure_counter = 0
                    self.success_counter = 0
                    self.last_state_change = time.time()

    def on_failure(self):
        """Record a failed operation."""
        with self._lock:
            self.total_failures += 1

            if self.state == CircuitState.HALF_OPEN:
                # Release semaphore
                self._half_open_semaphore.release()

                # Any failure in half-open state opens the circuit again
                logger.warning(f"Circuit '{self.name}' reopening after failure in HALF-OPEN state")
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.open_circuits += 1
                self.last_state_change = time.time()
            elif self.state == CircuitState.CLOSED:
                # Increment failure counter and check threshold
                self.failure_counter += 1
                if self.failure_counter >= self.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}' opening after {self.failure_counter} "
                        f"consecutive failures"
                    )
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
                    self.open_circuits += 1
                    self.last_state_change = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics.

        Returns:
            Dict[str, Any]: Circuit breaker metrics
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_counter": self.failure_counter,
                "success_counter": self.success_counter,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "open_circuits": self.open_circuits,
                "time_in_current_state": time.time() - self.last_state_change,
                "last_failure_time": self.last_failure_time,
                "reset_timeout": self.reset_timeout,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold
            }

    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_counter = 0
            self.success_counter = 0
            self.last_state_change = time.time()


class AsyncCircuitBreaker:
    """Asynchronous circuit breaker for external API calls.

    This class implements the circuit breaker pattern for asynchronous operations
    to prevent cascading failures by temporarily disabling endpoints that are
    experiencing high failure rates.

    Attributes:
        name: Name of the protected resource
        state: Current state of the circuit breaker
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before attempting recovery
        success_threshold: Successes needed in half-open state to close
        half_open_max_calls: Maximum concurrent calls in half-open state
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 1,
        excluded_exceptions: Optional[List[type]] = None
    ):
        """Initialize the circuit breaker.

        Args:
            name: Name of the protected resource
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in half-open state to close
            half_open_max_calls: Maximum concurrent calls in half-open state
            excluded_exceptions: Exceptions that should not count as failures
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or []

        self.failure_counter = 0
        self.last_failure_time = 0
        self.success_counter = 0

        self._lock = asyncio.Lock()
        self._half_open_semaphore = asyncio.Semaphore(half_open_max_calls)

        # Metrics
        self.total_failures = 0
        self.total_successes = 0
        self.open_circuits = 0
        self.last_state_change = time.time()

        logger.info(f"Async circuit breaker '{name}' initialized")

    async def __aenter__(self):
        """Enter the circuit breaker context."""
        await self.before_request()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the circuit breaker context."""
        if exc_type is not None:
            # Check if this exception should be excluded
            if not any(isinstance(exc_val, exc_type) for exc_type in self.excluded_exceptions):
                # An exception occurred that should count as a failure
                await self.on_failure()
            return False  # Propagate the exception
        else:
            # No exception - success
            await self.on_success()
            return False

    async def before_request(self):
        """Check if the request can proceed.

        Raises:
            CircuitOpenError: If the circuit is open
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if reset timeout has expired
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF-OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_counter = 0
                    self.last_state_change = time.time()
                else:
                    # Still open
                    remaining = self.reset_timeout - (time.time() - self.last_failure_time)
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {remaining:.1f}s"
                    )

            if self.state == CircuitState.HALF_OPEN:
                # Only allow limited calls in half-open state
                try:
                    # Acquire semaphore but don't wait if not available
                    await asyncio.wait_for(
                        self._half_open_semaphore.acquire(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is HALF-OPEN and at capacity"
                    )

    async def on_success(self):
        """Record a successful operation."""
        async with self._lock:
            self.total_successes += 1

            if self.state == CircuitState.CLOSED:
                # Reset failure counter on success in closed state
                self.failure_counter = 0
            elif self.state == CircuitState.HALF_OPEN:
                # Release semaphore
                self._half_open_semaphore.release()

                # Increment success counter and check if we should close the circuit
                self.success_counter += 1
                if self.success_counter >= self.success_threshold:
                    logger.info(f"Circuit '{self.name}' closing after {self.success_counter} successful calls")
                    self.state = CircuitState.CLOSED
                    self.failure_counter = 0
                    self.success_counter = 0
                    self.last_state_change = time.time()

    async def on_failure(self):
        """Record a failed operation."""
        async with self._lock:
            self.total_failures += 1

            if self.state == CircuitState.HALF_OPEN:
                # Release semaphore
                self._half_open_semaphore.release()

                # Any failure in half-open state opens the circuit again
                logger.warning(f"Circuit '{self.name}' reopening after failure in HALF-OPEN state")
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.open_circuits += 1
                self.last_state_change = time.time()
            elif self.state == CircuitState.CLOSED:
                # Increment failure counter and check threshold
                self.failure_counter += 1
                if self.failure_counter >= self.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}' opening after {self.failure_counter} "
                        f"consecutive failures"
                    )
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
                    self.open_circuits += 1
                    self.last_state_change = time.time()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics.

        Returns:
            Dict[str, Any]: Circuit breaker metrics
        """
        async with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_counter": self.failure_counter,
                "success_counter": self.success_counter,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "open_circuits": self.open_circuits,
                "time_in_current_state": time.time() - self.last_state_change,
                "last_failure_time": self.last_failure_time,
                "reset_timeout": self.reset_timeout,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold
            }

    async def reset(self):
        """Reset the circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_counter = 0
            self.success_counter = 0
            self.last_state_change = time.time()


# Export all classes
__all__ = [
    'CircuitState',
    'CircuitOpenError',
    'CircuitBreaker',
    'AsyncCircuitBreaker'
]
