"""
Resilience component for the Enhanced LLM Service.

This module provides resilience functionality for the Enhanced LLM Service,
including retry logic, circuit breakers, and timeouts.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Awaitable, Type

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.interfaces.exceptions import TimeoutException

logger = logging.getLogger(__name__)
T = TypeVar('T')

class ResilienceComponent:
    """
    Resilience component for the Enhanced LLM Service.
    
    This class provides resilience functionality for the Enhanced LLM Service,
    including retry logic, circuit breakers, and timeouts.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the resilience component.
        
        Args:
            enabled: Whether resilience is enabled
        """
        self.enabled = enabled
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    async def with_retry(self, 
                        operation: Callable[[], Awaitable[T]], 
                        max_retries: int = 3,
                        retry_delay: float = 1.0,
                        backoff_factor: float = 2.0,
                        retryable_errors: Optional[List[Type[Exception]]] = None) -> T:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async operation to execute
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            retryable_errors: List of exception types that should trigger a retry
            
        Returns:
            Result of the operation
        """
        if not self.enabled:
            return await operation()
        
        # Default retryable errors if not provided
        if retryable_errors is None:
            from asf.medical.llm_gateway.interfaces.exceptions import (
                TransportException, ServiceUnavailableException
            )
            retryable_errors = [TransportException, ServiceUnavailableException]
        
        # Initialize retry counter
        retry_count = 0
        current_delay = retry_delay
        
        # Try the operation with retries
        while True:
            try:
                return await operation()
            except Exception as e:
                # Check if we should retry
                should_retry = False
                for error_type in retryable_errors:
                    if isinstance(e, error_type):
                        should_retry = True
                        break
                
                # If not retryable or max retries reached, re-raise
                if not should_retry or retry_count >= max_retries:
                    logger.warning(f"Operation failed after {retry_count} retries: {str(e)}")
                    raise
                
                # Increment retry counter
                retry_count += 1
                
                # Log retry attempt
                logger.info(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                
                # Wait before retrying
                await asyncio.sleep(current_delay)
                
                # Increase delay for next retry
                current_delay *= backoff_factor
    
    async def with_circuit_breaker(self,
                                  operation: Callable[[], Awaitable[T]],
                                  circuit_name: str,
                                  fallback: Optional[Callable[[], Awaitable[T]]] = None) -> T:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            operation: Async operation to execute
            circuit_name: Name of the circuit breaker
            fallback: Optional fallback operation if circuit is open
            
        Returns:
            Result of the operation or fallback
        """
        if not self.enabled:
            return await operation()
        
        # Get or create circuit breaker
        circuit_breaker = self._get_or_create_circuit_breaker(circuit_name)
        
        # Check if circuit is open
        if circuit_breaker.is_open():
            logger.warning(f"Circuit {circuit_name} is open, using fallback")
            if fallback:
                return await fallback()
            else:
                from asf.medical.llm_gateway.interfaces.exceptions import ServiceUnavailableException
                raise ServiceUnavailableException(f"Circuit {circuit_name} is open and no fallback provided")
        
        # Execute operation with circuit breaker
        try:
            result = await operation()
            circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            raise
    
    async def with_timeout(self,
                          operation: Callable[[], Awaitable[T]],
                          timeout_seconds: float) -> T:
        """
        Execute an operation with a timeout.
        
        Args:
            operation: Async operation to execute
            timeout_seconds: Timeout in seconds
            
        Returns:
            Result of the operation
        """
        if not self.enabled:
            return await operation()
        
        try:
            return await asyncio.wait_for(operation(), timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
    
    def _get_or_create_circuit_breaker(self, circuit_name: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker.
        
        Args:
            circuit_name: Name of the circuit breaker
            
        Returns:
            Circuit breaker instance
        """
        if circuit_name not in self._circuit_breakers:
            # Create new circuit breaker with default settings
            self._circuit_breakers[circuit_name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                name=circuit_name
            )
        
        return self._circuit_breakers[circuit_name]
    
    def get_circuit_breaker_status(self, circuit_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a circuit breaker.
        
        Args:
            circuit_name: Name of the circuit breaker
            
        Returns:
            Dictionary containing circuit breaker status or None if not found
        """
        if not self.enabled or circuit_name not in self._circuit_breakers:
            return None
        
        circuit_breaker = self._circuit_breakers[circuit_name]
        return {
            "name": circuit_name,
            "state": "open" if circuit_breaker.is_open() else "closed",
            "failure_count": circuit_breaker.failure_count,
            "last_failure_time": circuit_breaker.last_failure_time,
            "recovery_timeout": circuit_breaker.recovery_timeout
        }
    
    def get_all_circuit_breaker_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all circuit breakers.
        
        Returns:
            Dictionary mapping circuit breaker names to their status
        """
        if not self.enabled:
            return {}
        
        return {
            name: self.get_circuit_breaker_status(name)
            for name in self._circuit_breakers
        }
    
    def reset_circuit_breaker(self, circuit_name: str) -> bool:
        """
        Reset a circuit breaker.
        
        Args:
            circuit_name: Name of the circuit breaker
            
        Returns:
            True if the circuit breaker was reset, False otherwise
        """
        if not self.enabled or circuit_name not in self._circuit_breakers:
            return False
        
        self._circuit_breakers[circuit_name].reset()
        return True
    
    def reset_all_circuit_breakers(self) -> None:
        """
        Reset all circuit breakers.
        """
        if not self.enabled:
            return
        
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.reset()
