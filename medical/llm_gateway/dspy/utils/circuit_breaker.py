"""
Circuit Breaker

This module provides a circuit breaker implementation for handling failures.
The circuit breaker pattern is a design pattern used to detect failures and
prevent cascading failures in a distributed system.
"""

import enum
import time
import logging
import asyncio
from typing import Callable, Any, Optional, Dict, TypeVar, Generic, Union

# Set up logging
logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    
    def __init__(self, message: str = "Circuit breaker is open"):
        """Initialize the exception."""
        self.message = message
        super().__init__(self.message)


T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for handling failures in synchronous code."""
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        name: Optional[str] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before trying to recover
            name: Name of the circuit breaker for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name or "Circuit"
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        # Check if it's time to try recovery
        if (
            self._state == CircuitState.OPEN and
            time.time() - self._last_failure_time >= self.recovery_timeout
        ):
            logger.info(f"{self.name}: Trying recovery, moving to half-open state")
            self._state = CircuitState.HALF_OPEN
        
        return self._state
    
    def success(self) -> None:
        """
        Register a successful operation.
        
        If the circuit breaker is half-open, this will close it.
        """
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"{self.name}: Recovery successful, closing circuit")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
    
    def failure(self) -> None:
        """
        Register a failed operation.
        
        If the failure count exceeds the threshold, this will open the circuit.
        """
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            self._failure_count += 1
            
            if self._failure_count >= self.failure_threshold:
                logger.warning(f"{self.name}: Failure threshold reached, opening circuit")
                self._state = CircuitState.OPEN
                
        elif self._state == CircuitState.HALF_OPEN:
            logger.warning(f"{self.name}: Recovery failed, opening circuit")
            self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Reset the circuit breaker to its closed state."""
        logger.info(f"{self.name}: Manually resetting circuit to closed state")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            T: Function result
            
        Raises:
            CircuitOpenError: If the circuit breaker is open
        """
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(f"{self.name}: Circuit is open")
        
        try:
            result = func(*args, **kwargs)
            self.success()
            return result
        except Exception as e:
            self.failure()
            raise


class AsyncCircuitBreaker:
    """Circuit breaker for handling failures in asynchronous code."""
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        name: Optional[str] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before trying to recover
            name: Name of the circuit breaker for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name or "AsyncCircuit"
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
    
    async def _get_state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        async with self._lock:
            # Check if it's time to try recovery
            if (
                self._state == CircuitState.OPEN and
                time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                logger.info(f"{self.name}: Trying recovery, moving to half-open state")
                self._state = CircuitState.HALF_OPEN
            
            return self._state
    
    async def success(self) -> None:
        """
        Register a successful operation.
        
        If the circuit breaker is half-open, this will close it.
        """
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"{self.name}: Recovery successful, closing circuit")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
    
    async def failure(self) -> None:
        """
        Register a failed operation.
        
        If the failure count exceeds the threshold, this will open the circuit.
        """
        async with self._lock:
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                
                if self._failure_count >= self.failure_threshold:
                    logger.warning(f"{self.name}: Failure threshold reached, opening circuit")
                    self._state = CircuitState.OPEN
                    
            elif self._state == CircuitState.HALF_OPEN:
                logger.warning(f"{self.name}: Recovery failed, opening circuit")
                self._state = CircuitState.OPEN
    
    async def reset(self) -> None:
        """Reset the circuit breaker to its closed state."""
        async with self._lock:
            logger.info(f"{self.name}: Manually resetting circuit to closed state")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
    
    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Call an async function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            CircuitOpenError: If the circuit breaker is open
        """
        state = await self._get_state()
        if state == CircuitState.OPEN:
            raise CircuitOpenError(f"{self.name}: Circuit is open")
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            await self.success()
            return result
        except Exception as e:
            await self.failure()
            raise


# Export
__all__ = [
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreaker",
    "AsyncCircuitBreaker",
]