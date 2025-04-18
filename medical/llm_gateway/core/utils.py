"""
Utility functions and classes for the LLM Gateway.
"""

import time
import logging
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """Possible states of a circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failing fast, no requests allowed
    HALF_OPEN = "half_open"  # Testing recovery with limited requests


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    
    The circuit breaker maintains state based on failure patterns:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service deemed unhealthy, requests fail fast
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(self, threshold: int = 5, reset_timeout: int = 60):
        """
        Initialize a circuit breaker.
        
        Args:
            threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before testing recovery
        """
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        
        # State
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._recovery_time = None
        
        # Thread safety
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get the current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get the current failure count."""
        with self._lock:
            return self._failure_count
    
    @property
    def recovery_time(self) -> Optional[datetime]:
        """Get the time when the circuit will attempt recovery."""
        with self._lock:
            return self._recovery_time
    
    def is_open(self) -> bool:
        """
        Check if the circuit is open (failing fast).
        
        Returns:
            True if the circuit is OPEN, False otherwise
        """
        with self._lock:
            # Check if we're in the OPEN state
            if self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has elapsed
                if self._recovery_time and datetime.now() >= self._recovery_time:
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
                    return False  # Allow this call as first test
                return True  # Still open
            
            # CLOSED or HALF_OPEN state - circuit is not open
            return False
    
    def record_success(self) -> None:
        """
        Record a successful operation.
        
        This can trigger state transition from HALF_OPEN → CLOSED.
        """
        with self._lock:
            # Handle success based on current state
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Transition to CLOSED on success in HALF_OPEN
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._recovery_time = None
                logger.info("Circuit breaker transitioning from HALF_OPEN to CLOSED")
    
    def record_failure(self) -> None:
        """
        Record a failed operation.
        
        This can trigger state transitions:
        - CLOSED → OPEN if failure threshold is reached
        - HALF_OPEN → OPEN on any failure
        """
        with self._lock:
            now = datetime.now()
            self._last_failure_time = now
            
            # Handle failure based on current state
            if self._state == CircuitBreakerState.CLOSED:
                # Increment failure counter
                self._failure_count += 1
                
                logger.info(f"Circuit breaker failure recorded: {self._failure_count}/{self.threshold}")
                
                # Check if we've reached the threshold to open the circuit
                if self._failure_count >= self.threshold:
                    self._state = CircuitBreakerState.OPEN
                    self._recovery_time = now + timedelta(seconds=self.reset_timeout)
                    logger.warning(f"Circuit breaker transitioning to OPEN until {self._recovery_time}")
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state returns to open
                self._state = CircuitBreakerState.OPEN
                self._recovery_time = now + timedelta(seconds=self.reset_timeout)
                logger.warning(f"Circuit breaker transitioning from HALF_OPEN back to OPEN until {self._recovery_time}")
    
    def reset(self) -> None:
        """
        Reset the circuit breaker to its initial state.
        
        This forces the circuit to CLOSED state and resets all counters.
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._recovery_time = None
            
            logger.info(f"Circuit breaker manually reset from {old_state} to CLOSED")
