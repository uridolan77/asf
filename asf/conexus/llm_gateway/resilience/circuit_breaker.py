"""
Circuit breaker implementation for the Conexus LLM Gateway.

This module provides a circuit breaker pattern implementation to prevent
cascading failures when a downstream service is experiencing problems.
"""

import logging
import time
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Possible states of a circuit breaker."""
    CLOSED = "closed"     # Normal operation, requests pass through
    OPEN = "open"         # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back to normal


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.
    
    The circuit breaker monitors failures in calls to external services and
    temporarily blocks further calls when the service appears to be down or
    experiencing issues. This prevents cascading failures and allows the
    external service time to recover.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5, 
                 reset_timeout_seconds: int = 30,
                 half_open_success_threshold: int = 2):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            reset_timeout_seconds: Seconds to wait before trying to close circuit again
            half_open_success_threshold: Number of successful calls in half-open state 
                                         before closing circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change_time = datetime.utcnow()
        
        self._lock = Lock()
        
        logger.info(f"Circuit breaker initialized with failure threshold {failure_threshold}, "
                   f"reset timeout {reset_timeout_seconds}s")
    
    def is_open(self) -> bool:
        """
        Check if the circuit breaker is open.
        
        Returns:
            True if the circuit is open, False otherwise
        """
        now = datetime.utcnow()
        
        with self._lock:
            # If circuit is open, check if it's time to try half-open state
            if self.state == CircuitState.OPEN:
                if (now - self.last_state_change_time).total_seconds() >= self.reset_timeout_seconds:
                    logger.info("Circuit breaker transitioning from OPEN to HALF-OPEN state")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.last_state_change_time = now
            
            return self.state == CircuitState.OPEN
    
    def record_failure(self) -> None:
        """
        Record a failed call.
        
        This may cause the circuit to open if the failure threshold is reached.
        """
        now = datetime.utcnow()
        
        with self._lock:
            self.last_failure_time = now
            
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                
                if self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit breaker threshold reached ({self.failure_count} failures). "
                                  f"Opening circuit for {self.reset_timeout_seconds}s")
                    self.state = CircuitState.OPEN
                    self.last_state_change_time = now
            
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning("Failure in half-open state. Reopening circuit.")
                self.state = CircuitState.OPEN
                self.last_state_change_time = now
    
    def record_success(self) -> None:
        """
        Record a successful call.
        
        This may cause the circuit to close if enough successful calls are made
        in the half-open state.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                # Reset failure count after a successful call
                self.failure_count = 0
            
            elif self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.half_open_success_threshold:
                    logger.info(f"Circuit breaker success threshold reached in half-open state "
                               f"({self.success_count} successes). Closing circuit.")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_state_change_time = datetime.utcnow()
    
    def retry_after_seconds(self) -> int:
        """
        Get the number of seconds until the next retry is allowed.
        
        Returns:
            Seconds until retry is allowed (0 if retries are currently allowed)
        """
        if self.state != CircuitState.OPEN:
            return 0
            
        now = datetime.utcnow()
        seconds_since_opened = (now - self.last_state_change_time).total_seconds()
        
        if seconds_since_opened >= self.reset_timeout_seconds:
            return 0
        
        return max(0, int(self.reset_timeout_seconds - seconds_since_opened))
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state regardless of current state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_state_change_time = datetime.utcnow()
            logger.info("Circuit breaker manually reset to CLOSED state")