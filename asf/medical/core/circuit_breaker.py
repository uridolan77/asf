"""
Circuit Breaker module for the Medical Research Synthesizer.

This module provides a circuit breaker pattern implementation to prevent
cascading failures when interacting with external services.
"""

import logging
import time
import threading
from enum import Enum
from typing import Callable, TypeVar, Dict, Any, Optional, List, Type, Union
from functools import wraps

from asf.medical.core.exceptions import ExternalServiceError

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
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
    """
    
    _instances: Dict[str, "CircuitBreaker"] = {}
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, name: str) -> "CircuitBreaker":
        """
        Get a circuit breaker instance by name.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            CircuitBreaker instance
        """
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = CircuitBreaker(name)
            return cls._instances[name]
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        exception_types: Optional[ExceptionTypes] = None,
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time to wait before trying to recover (in seconds)
            half_open_max_calls: Maximum number of calls allowed in half-open state
            exception_types: Exception types to count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exception_types = exception_types
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned from OPEN to HALF_OPEN")
            
            return self._state
    
    def success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # If we're in HALF_OPEN state and a call succeeds, close the circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to CLOSED")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def failure(self, exception: Exception) -> None:
        """
        Record a failed call.
        
        Args:
            exception: The exception that caused the failure
        """
        # Check if we should count this exception as a failure
        if self.exception_types is not None:
            if isinstance(self.exception_types, list):
                if not any(isinstance(exception, exc_type) for exc_type in self.exception_types):
                    return
            elif not isinstance(exception, self.exception_types):
                return
        
        with self._lock:
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # If we're in HALF_OPEN state and a call fails, open the circuit again
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to OPEN")
            elif self._state == CircuitState.CLOSED:
                # Increment failure count
                self._failure_count += 1
                
                # Check if we should open the circuit
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' transitioned from CLOSED to OPEN")
    
    def allow_request(self) -> bool:
        """
        Check if a request is allowed.
        
        Returns:
            True if the request is allowed, False otherwise
        """
        current_state = self.state
        
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        elif current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        
        # This should never happen, but just in case
        return False
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for applying the circuit breaker to a function.
        
        Args:
            func: The function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.allow_request():
                raise ExternalServiceError(
                    self.name,
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            try:
                result = func(*args, **kwargs)
                self.success()
                return result
            except Exception as e:
                self.failure(e)
                raise
        
        return wrapper


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    exception_types: Optional[ExceptionTypes] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for applying a circuit breaker to a function.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time to wait before trying to recover (in seconds)
        half_open_max_calls: Maximum number of calls allowed in half-open state
        exception_types: Exception types to count as failures
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        circuit_breaker = CircuitBreaker.get_instance(name)
        circuit_breaker.failure_threshold = failure_threshold
        circuit_breaker.recovery_timeout = recovery_timeout
        circuit_breaker.half_open_max_calls = half_open_max_calls
        circuit_breaker.exception_types = exception_types
        
        return circuit_breaker(func)
    
    return decorator

# Export decorators
__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "with_circuit_breaker",
]
