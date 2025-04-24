"""
Circuit breaker implementation for resilient service communication.

This module provides a circuit breaker pattern implementation that
prevents cascading failures by failing fast when a service is unhealthy.
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Callable, Optional

import structlog

logger = structlog.get_logger("mcp_resilience.circuit_breaker")


class CircuitState(str, Enum):
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
    
    Features:
    - Configurable thresholds and recovery timeouts
    - Thread-safe state management
    - Metrics for monitoring
    - Event hooks for state changes
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
        reset_timeout: int = 60 * 10,  # 10 minutes
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        """
        Initialize a circuit breaker.
        
        Args:
            name: Name of this circuit breaker for logging/metrics
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery (OPEN → HALF_OPEN)
            half_open_max_calls: Max calls to allow in HALF_OPEN before deciding
            reset_timeout: Seconds after which to reset failure count in CLOSED state
            on_state_change: Callback for state change notifications
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_timeout = reset_timeout
        self.on_state_change = on_state_change
        
        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._recovery_time: Optional[datetime] = None
        self._last_state_change_time = datetime.utcnow()
        
        # Thread safety
        self._lock = Lock()
        
        # Logging context
        self.logger = logger.bind(circuit_breaker=name)
        
        self.logger.info(
            "Initialized circuit breaker",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    
    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get the current failure count."""
        with self._lock:
            return self._failure_count
    
    @property
    def last_failure_time(self) -> Optional[datetime]:
        """Get the time of the last failure."""
        with self._lock:
            return self._last_failure_time
    
    @property
    def recovery_time(self) -> Optional[datetime]:
        """Get the scheduled recovery time."""
        with self._lock:
            return self._recovery_time
    
    def is_open(self) -> bool:
        """
        Check if the circuit is open (failing fast).
        
        This also handles state transitions from OPEN → HALF_OPEN based on timeout.
        
        Returns:
            True if the circuit is OPEN or HALF_OPEN but max calls reached
        """
        with self._lock:
            # Check if we're in the OPEN state
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._recovery_time and datetime.utcnow() >= self._recovery_time:
                    self._to_half_open()
                    return False  # Allow this call as first test
                return True  # Still open
            
            # Check if we're in HALF_OPEN state
            elif self._state == CircuitState.HALF_OPEN:
                # Allow up to half_open_max_calls
                return self._success_count >= self.half_open_max_calls
            
            # CLOSED state - circuit is not open
            return False
    
    def record_success(self) -> None:
        """
        Record a successful operation.
        
        This can trigger state transition from HALF_OPEN → CLOSED.
        """
        with self._lock:
            self._last_success_time = datetime.utcnow()
            
            # Handle success based on current state
            if self._state == CircuitState.CLOSED:
                # Reset failure count if enough time has passed
                if (self._last_failure_time is None or 
                    (datetime.utcnow() - self._last_failure_time).total_seconds() >= self.reset_timeout):
                    if self._failure_count > 0:
                        self._failure_count = 0
                        self.logger.debug("Reset failure count after timeout")
            
            elif self._state == CircuitState.HALF_OPEN:
                # Increment success counter
                self._success_count += 1
                
                self.logger.info(
                    "Successful call in HALF_OPEN state",
                    success_count=self._success_count,
                    half_open_max_calls=self.half_open_max_calls
                )
                
                # Check if we've reached the threshold to close the circuit
                if self._success_count >= self.half_open_max_calls:
                    self._to_closed()
    
    def record_failure(self) -> None:
        """
        Record a failed operation.
        
        This can trigger state transitions:
        - CLOSED → OPEN if failure threshold is reached
        - HALF_OPEN → OPEN on any failure
        """
        with self._lock:
            now = datetime.utcnow()
            self._last_failure_time = now
            
            # Handle failure based on current state
            if self._state == CircuitState.CLOSED:
                # Increment failure counter
                self._failure_count += 1
                
                self.logger.info(
                    "Failure recorded",
                    failure_count=self._failure_count,
                    threshold=self.failure_threshold
                )
                
                # Check if we've reached the threshold to open the circuit
                if self._failure_count >= self.failure_threshold:
                    self._to_open()
            
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state returns to open
                self.logger.info("Failure in HALF_OPEN state, reopening circuit")
                self._to_open()
    
    def reset(self) -> None:
        """
        Reset the circuit breaker to its initial state.
        
        This forces the circuit to CLOSED state and resets all counters.
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._recovery_time = None
            self._last_state_change_time = datetime.utcnow()
            
            self.logger.info(
                "Circuit breaker manually reset",
                old_state=old_state
            )
            
            # Call state change callback if provided
            if old_state != CircuitState.CLOSED and self.on_state_change:
                try:
                    self.on_state_change(old_state, CircuitState.CLOSED)
                except Exception as e:
                    self.logger.error(
                        "Error in state change callback",
                        error=str(e),
                        exc_info=True
                    )
    
    def _to_open(self) -> None:
        """Transition to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._recovery_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)
        self._success_count = 0
        self._last_state_change_time = datetime.utcnow()
        
        self.logger.warning(
            "Circuit OPENED",
            failure_count=self._failure_count,
            recovery_at=self._recovery_time.isoformat()
        )
        
        # Call state change callback if provided
        if old_state != CircuitState.OPEN and self.on_state_change:
            try:
                self.on_state_change(old_state, CircuitState.OPEN)
            except Exception as e:
                self.logger.error(
                    "Error in state change callback",
                    error=str(e),
                    exc_info=True
                )
    
    def _to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._last_state_change_time = datetime.utcnow()
        
        self.logger.info(
            "Circuit HALF_OPEN",
            max_test_calls=self.half_open_max_calls
        )
        
        # Call state change callback if provided
        if old_state != CircuitState.HALF_OPEN and self.on_state_change:
            try:
                self.on_state_change(old_state, CircuitState.HALF_OPEN)
            except Exception as e:
                self.logger.error(
                    "Error in state change callback",
                    error=str(e),
                    exc_info=True
                )
    
    def _to_closed(self) -> None:
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._recovery_time = None
        self._last_state_change_time = datetime.utcnow()
        
        self.logger.info("Circuit CLOSED")
        
        # Call state change callback if provided
        if old_state != CircuitState.CLOSED and self.on_state_change:
            try:
                self.on_state_change(old_state, CircuitState.CLOSED)
            except Exception as e:
                self.logger.error(
                    "Error in state change callback",
                    error=str(e),
                    exc_info=True
                )
    
    def get_metrics(self) -> dict:
        """
        Get current metrics for this circuit breaker.
        
        Returns:
            Dict with current metrics
        """
        with self._lock:
            metrics = {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "success_count": self._success_count,
                "last_state_change": self._last_state_change_time.isoformat(),
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_success": self._last_success_time.isoformat() if self._last_success_time else None,
                "recovery_timeout": self.recovery_timeout,
                "recovery_time": self._recovery_time.isoformat() if self._recovery_time else None,
            }
            
            # Add state-specific metrics
            if self._state == CircuitState.OPEN and self._recovery_time:
                metrics["seconds_until_recovery"] = max(
                    0, 
                    (self._recovery_time - datetime.utcnow()).total_seconds()
                )
            elif self._state == CircuitState.HALF_OPEN:
                metrics["remaining_test_calls"] = max(
                    0,
                    self.half_open_max_calls - self._success_count
                )
            
            return metrics