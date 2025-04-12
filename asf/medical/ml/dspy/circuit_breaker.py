Circuit Breaker Pattern for API Reliability

This module implements the circuit breaker pattern for external API calls,
preventing cascading failures by temporarily disabling endpoints that are experiencing high failure rates.

import logging
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    Circuit breaker states.
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_counter = 0
        self.last_failure_time = 0
        self.success_counter = 0
        
        self._lock = asyncio.Lock()
        self._half_open_semaphore = asyncio.Semaphore(half_open_max_calls)
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def __aenter__(self):
        """Enter the circuit breaker context."""
        await self.before_request()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the circuit breaker context."""
        if exc_type is not None:
            # An exception occurred
            await self.on_failure()
            return False  # Propagate the exception
        else:
            # No exception - success
            await self.on_success()
            return False
    
    async def before_request(self):
        """
        Check if the request can proceed.
        
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
                else:
                    # Still open
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {self.reset_timeout - (time.time() - self.last_failure_time):.1f}s"
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                # Only allow limited calls in half-open state
                if not self._half_open_semaphore.locked():
                    # Acquire semaphore but don't wait if not available
                    try:
                        await asyncio.wait_for(
                            self._half_open_semaphore.acquire(),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        raise CircuitOpenError(
                            f"Circuit '{self.name}' is HALF-OPEN and at capacity"
                        )
                else:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is HALF-OPEN and at capacity"
                    )
    
    async def on_success(self):
        """Record a successful operation."""
        async with self._lock:
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
    
    async def on_failure(self):
        """Record a failed operation."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Release semaphore
                self._half_open_semaphore.release()
                
                # Any failure in half-open state opens the circuit again
                logger.warning(f"Circuit '{self.name}' reopening after failure in HALF-OPEN state")
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
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


class CircuitBreakerRegistry:
    Registry for managing multiple circuit breakers.
    
    This class provides a centralized way to create, retrieve, and manage
    circuit breakers for different resources.
    
    def __init__(self):
        Initialize the circuit breaker registry.
        
        Args:
        
        Get or create a circuit breaker.
        
        Args:
            name: Name of the protected resource
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            success_threshold: Successes needed in half-open state to close
            half_open_max_calls: Maximum concurrent calls in half-open state
            
        Returns:
            CircuitBreaker: The circuit breaker instance
        """
        async with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    success_threshold=success_threshold,
                    half_open_max_calls=half_open_max_calls
                )
            return self._circuit_breakers[name]
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all circuit breakers.
        
        Returns:
            Dict[str, Any]: Status of all circuit breakers
        """
        async with self._lock:
            return {
                name: {
                    "state": cb.state.value,
                    "failure_counter": cb.failure_counter,
                    "success_counter": cb.success_counter,
                    "last_failure_time": cb.last_failure_time
                }
                for name, cb in self._circuit_breakers.items()
            }
    
    async def reset(self, name: Optional[str] = None) -> None:
        """
        Reset one or all circuit breakers.
        
        Args:
            name: Name of the circuit breaker to reset, or None to reset all
        """
        async with self._lock:
            if name is not None:
                if name in self._circuit_breakers:
                    cb = self._circuit_breakers[name]
                    cb.state = CircuitState.CLOSED
                    cb.failure_counter = 0
                    cb.success_counter = 0
                    logger.info(f"Circuit breaker '{name}' reset")
            else:
                for name, cb in self._circuit_breakers.items():
                    cb.state = CircuitState.CLOSED
                    cb.failure_counter = 0
                    cb.success_counter = 0
                logger.info("All circuit breakers reset")


# Global circuit breaker registry
_circuit_breaker_registry = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Get the global circuit breaker registry.
    
    Returns:
        CircuitBreakerRegistry: The global circuit breaker registry
    """
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry


# Export all classes and functions
__all__ = [
    'CircuitState',
    'CircuitOpenError',
    'CircuitBreaker',
    'CircuitBreakerRegistry',
    'get_circuit_breaker_registry'
]
