import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Dict, Optional, TypeVar, Awaitable
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """The state of a circuit breaker."""
    CLOSED = 'closed'  # Normal operation, requests are allowed
    OPEN = 'open'      # Circuit is open, requests are blocked
    HALF_OPEN = 'half_open'  # Testing if the service is back


class CircuitBreaker:
    """Circuit breaker for protecting against service failures."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: The name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before trying to recover
            half_open_max_calls: Maximum number of calls to allow in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.lock = asyncio.Lock()
    
    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
            
        Raises:
            CircuitBreakerOpenError: If the circuit is open
            Exception: Any exception raised by the function
        """
        async with self.lock:
            # Check if the circuit is open
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name} is OPEN until {self.last_failure_time + self.recovery_timeout}"
                    )
            
            # Check if we've reached the maximum number of calls in half-open state
            if self.state == CircuitState.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit {self.name} is HALF_OPEN and has reached maximum calls"
                )
            
            # Increment the number of calls in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            
            # Reset the circuit if it was successful
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(f"Circuit {self.name} transitioning from HALF_OPEN to CLOSED")
                    self.state = CircuitState.CLOSED
                
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            # Record the failure
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Open the circuit if we've reached the failure threshold
                if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit {self.name} transitioning from CLOSED to OPEN")
                    self.state = CircuitState.OPEN
                
                # Keep the circuit open if we're in half-open state
                if self.state == CircuitState.HALF_OPEN:
                    logger.warning(f"Circuit {self.name} transitioning from HALF_OPEN to OPEN")
                    self.state = CircuitState.OPEN
            
            # Re-raise the exception
            raise


class CircuitBreakerOpenError(Exception):
    """Error raised when a circuit breaker is open."""
    pass


class LLMCircuitBreaker:
    """Circuit breaker for LLM providers."""
    
    def __init__(self):
        """Initialize the LLM circuit breaker."""
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get a circuit breaker for a provider.
        
        Args:
            provider: The name of the provider
            
        Returns:
            A circuit breaker for the provider
        """
        if provider not in self.breakers:
            self.breakers[provider] = CircuitBreaker(
                name=f"llm-{provider}",
                failure_threshold=5,
                recovery_timeout=30,
                half_open_max_calls=1
            )
        
        return self.breakers[provider]
    
    async def call_with_breaker(
        self,
        provider: str,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Call a function with circuit breaker protection.
        
        Args:
            provider: The name of the provider
            func: The function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
        """
        breaker = self.get_breaker(provider)
        return await breaker.execute(func, *args, **kwargs)
