"""
Enhanced circuit breaker implementation with advanced resilience patterns.

This module provides an enhanced circuit breaker implementation with:
- Metrics integration for monitoring circuit breaker state
- Adaptive recovery timeouts based on failure patterns
- Integration with a centralized circuit breaker registry
"""

import time
import math
import random
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Callable, Dict, List, Optional, Any, Union, Tuple

import structlog

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitState, CircuitBreaker

# Try to import tracing
try:
    from asf.medical.llm_gateway.resilience.tracing import get_resilience_tracing
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

logger = structlog.get_logger("llm_gateway.resilience.enhanced_circuit_breaker")


class FailureCategory(str, Enum):
    """Categories of failures for adaptive recovery."""

    TIMEOUT = "timeout"  # Request timed out
    RATE_LIMIT = "rate_limit"  # Rate limit exceeded
    AUTH = "auth"  # Authentication/authorization error
    SERVER = "server"  # Server error (5xx)
    CONNECTION = "connection"  # Connection error
    VALIDATION = "validation"  # Validation error
    UNKNOWN = "unknown"  # Unknown error


class RecoveryStrategy(str, Enum):
    """Recovery strategies for circuit breaker."""

    FIXED = "fixed"  # Fixed recovery timeout
    EXPONENTIAL = "exponential"  # Exponential backoff
    ADAPTIVE = "adaptive"  # Adaptive based on failure patterns
    JITTERED = "jittered"  # Jittered exponential backoff


class EnhancedCircuitBreaker(CircuitBreaker):
    """
    Enhanced circuit breaker with advanced resilience patterns.

    Features beyond the base CircuitBreaker:
    - Metrics integration for monitoring circuit breaker state
    - Adaptive recovery timeouts based on failure patterns
    - Failure categorization for targeted recovery strategies
    - Jittered exponential backoff for recovery
    - Detailed health metrics and state tracking
    - Integration with centralized circuit breaker registry
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
        reset_timeout: int = 60 * 10,  # 10 minutes
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
        metrics_service: Optional[Any] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL,
        max_recovery_timeout: int = 60 * 30,  # 30 minutes
        min_recovery_timeout: int = 1,  # 1 second
        jitter_factor: float = 0.2,  # 20% jitter
        failure_timeout_multipliers: Optional[Dict[FailureCategory, float]] = None,
        registry: Optional["CircuitBreakerRegistry"] = None
    ):
        """
        Initialize an enhanced circuit breaker.

        Args:
            name: Name of this circuit breaker for logging/metrics
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Base seconds to wait before testing recovery
            half_open_max_calls: Max calls to allow in HALF_OPEN before deciding
            reset_timeout: Seconds after which to reset failure count in CLOSED state
            on_state_change: Callback for state change notifications
            metrics_service: Optional metrics service for recording metrics
            recovery_strategy: Strategy for determining recovery timeout
            max_recovery_timeout: Maximum recovery timeout in seconds
            min_recovery_timeout: Minimum recovery timeout in seconds
            jitter_factor: Factor for jitter in recovery timeout (0-1)
            failure_timeout_multipliers: Multipliers for recovery timeout by failure category
            registry: Optional circuit breaker registry for centralized management
        """
        # Initialize base circuit breaker
        super().__init__(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            reset_timeout=reset_timeout,
            on_state_change=on_state_change
        )

        # Enhanced features
        self.metrics_service = metrics_service
        self.recovery_strategy = recovery_strategy
        self.max_recovery_timeout = max_recovery_timeout
        self.min_recovery_timeout = min_recovery_timeout
        self.jitter_factor = jitter_factor
        self.base_recovery_timeout = recovery_timeout

        # Default multipliers if not provided
        self.failure_timeout_multipliers = failure_timeout_multipliers or {
            FailureCategory.TIMEOUT: 2.0,
            FailureCategory.RATE_LIMIT: 5.0,
            FailureCategory.AUTH: 1.0,
            FailureCategory.SERVER: 2.0,
            FailureCategory.CONNECTION: 3.0,
            FailureCategory.VALIDATION: 1.0,
            FailureCategory.UNKNOWN: 2.0
        }

        # Additional state tracking
        self._consecutive_failures = 0
        self._failure_categories: Dict[FailureCategory, int] = {
            category: 0 for category in FailureCategory
        }
        self._failure_timestamps: List[datetime] = []
        self._recovery_attempts = 0
        self._current_recovery_timeout = recovery_timeout
        self._last_failure_category: Optional[FailureCategory] = None

        # Register with registry if provided
        self.registry = registry
        if registry:
            registry.register(self)

        # Update metrics on initialization
        self._update_metrics()

        self.logger.info(
            "Initialized enhanced circuit breaker",
            recovery_strategy=recovery_strategy.value,
            base_recovery_timeout=recovery_timeout,
            max_recovery_timeout=max_recovery_timeout
        )

    def record_failure(self, category: FailureCategory = FailureCategory.UNKNOWN, exception: Optional[Exception] = None) -> None:
        """
        Record a failed operation with category.

        This can trigger state transitions:
        - CLOSED → OPEN if failure threshold is reached
        - HALF_OPEN → OPEN on any failure

        Args:
            category: Category of the failure
            exception: Exception that caused the failure
        """
        with self._lock:
            now = datetime.utcnow()
            self._last_failure_time = now
            self._last_failure_category = category
            self._failure_timestamps.append(now)
            self._failure_categories[category] += 1
            self._consecutive_failures += 1

            # Trim old timestamps (older than reset_timeout)
            cutoff = now - timedelta(seconds=self.reset_timeout)
            self._failure_timestamps = [ts for ts in self._failure_timestamps if ts >= cutoff]

            # Handle failure based on current state
            if self._state == CircuitState.CLOSED:
                # Increment failure counter
                self._failure_count += 1

                self.logger.info(
                    "Failure recorded",
                    failure_count=self._failure_count,
                    threshold=self.failure_threshold,
                    category=category.value,
                    consecutive_failures=self._consecutive_failures
                )

                # Check if we've reached the threshold to open the circuit
                if self._failure_count >= self.failure_threshold:
                    self._to_open(category)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state returns to open
                self.logger.info(
                    "Failure in HALF_OPEN state, reopening circuit",
                    category=category.value
                )
                self._to_open(category)

            # Update metrics
            self._update_metrics()

            # Record in tracing if available
            if TRACING_AVAILABLE:
                try:
                    tracing = get_resilience_tracing()
                    tracing.record_failure(
                        circuit_breaker=self,
                        failure_count=self._failure_count,
                        failure_category=category,
                        provider_id=getattr(self, "provider_id", None),
                        exception=exception
                    )
                except Exception as e:
                    self.logger.error(
                        "Error recording failure in tracing",
                        error=str(e),
                        exc_info=True
                    )

    def record_success(self) -> None:
        """
        Record a successful operation.

        This can trigger state transition from HALF_OPEN → CLOSED.
        """
        with self._lock:
            self._last_success_time = datetime.utcnow()
            self._consecutive_failures = 0

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

            # Update metrics
            self._update_metrics()

            # Record in tracing if available
            if TRACING_AVAILABLE:
                try:
                    tracing = get_resilience_tracing()
                    tracing.record_success(
                        circuit_breaker=self,
                        provider_id=getattr(self, "provider_id", None)
                    )
                except Exception as e:
                    self.logger.error(
                        "Error recording success in tracing",
                        error=str(e),
                        exc_info=True
                    )

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
            self._consecutive_failures = 0
            self._recovery_attempts = 0
            self._current_recovery_timeout = self.base_recovery_timeout

            # Reset failure categories
            for category in FailureCategory:
                self._failure_categories[category] = 0

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

            # Update metrics
            self._update_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for this circuit breaker.

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            metrics = {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "consecutive_failures": self._consecutive_failures,
                "recovery_attempts": self._recovery_attempts,
                "current_recovery_timeout": self._current_recovery_timeout,
                "failure_categories": {k.value: v for k, v in self._failure_categories.items()},
                "last_failure_category": self._last_failure_category.value if self._last_failure_category else None,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
                "last_state_change_time": self._last_state_change_time.isoformat(),
                "recovery_time": self._recovery_time.isoformat() if self._recovery_time else None,
                "failure_rate": self._calculate_failure_rate(),
                "time_since_last_failure": self._calculate_time_since_last_failure(),
                "time_since_last_success": self._calculate_time_since_last_success(),
                "time_in_current_state": self._calculate_time_in_current_state()
            }
            return metrics

    def get_health(self) -> Dict[str, Any]:
        """
        Get health information for this circuit breaker.

        Returns:
            Dictionary with health information
        """
        metrics = self.get_metrics()

        # Calculate health score (0-100)
        health_score = self._calculate_health_score()

        return {
            "name": self.name,
            "state": metrics["state"],
            "health_score": health_score,
            "is_healthy": health_score >= 50,
            "failure_rate": metrics["failure_rate"],
            "time_since_last_failure": metrics["time_since_last_failure"],
            "consecutive_failures": metrics["consecutive_failures"],
            "recovery_attempts": metrics["recovery_attempts"],
            "current_recovery_timeout": metrics["current_recovery_timeout"],
            "most_common_failure": self._get_most_common_failure()
        }

    def _to_open(self, category: FailureCategory = FailureCategory.UNKNOWN) -> None:
        """
        Transition to OPEN state with adaptive recovery timeout.

        Args:
            category: Category of the failure that triggered the transition
        """
        old_state = self._state
        self._state = CircuitState.OPEN
        self._recovery_attempts += 1

        # Calculate recovery timeout based on strategy
        self._current_recovery_timeout = self._calculate_recovery_timeout(category)

        # Set recovery time
        self._recovery_time = datetime.utcnow() + timedelta(seconds=self._current_recovery_timeout)

        self._success_count = 0
        self._last_state_change_time = datetime.utcnow()

        self.logger.warning(
            "Circuit OPENED",
            failure_count=self._failure_count,
            recovery_at=self._recovery_time.isoformat(),
            recovery_timeout=self._current_recovery_timeout,
            recovery_strategy=self.recovery_strategy.value,
            failure_category=category.value,
            recovery_attempts=self._recovery_attempts
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

        # Record state change in tracing if available
        if TRACING_AVAILABLE:
            try:
                tracing = get_resilience_tracing()
                tracing.record_state_change(
                    circuit_breaker=self,
                    old_state=old_state,
                    new_state=CircuitState.OPEN,
                    provider_id=getattr(self, "provider_id", None),
                    failure_category=category
                )
            except Exception as e:
                self.logger.error(
                    "Error recording state change in tracing",
                    error=str(e),
                    exc_info=True
                )

        # Update metrics
        self._update_metrics()

    def _to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._last_state_change_time = datetime.utcnow()

        self.logger.info(
            "Circuit HALF_OPEN",
            max_test_calls=self.half_open_max_calls,
            recovery_attempts=self._recovery_attempts
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

        # Record state change in tracing if available
        if TRACING_AVAILABLE:
            try:
                tracing = get_resilience_tracing()
                tracing.record_state_change(
                    circuit_breaker=self,
                    old_state=old_state,
                    new_state=CircuitState.HALF_OPEN,
                    provider_id=getattr(self, "provider_id", None)
                )
            except Exception as e:
                self.logger.error(
                    "Error recording state change in tracing",
                    error=str(e),
                    exc_info=True
                )

        # Update metrics
        self._update_metrics()

    def _to_closed(self) -> None:
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._recovery_time = None
        self._last_state_change_time = datetime.utcnow()
        self._consecutive_failures = 0

        # Reset recovery timeout to base value on successful recovery
        self._current_recovery_timeout = self.base_recovery_timeout

        self.logger.info(
            "Circuit CLOSED",
            recovery_attempts=self._recovery_attempts
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

        # Record state change in tracing if available
        if TRACING_AVAILABLE:
            try:
                tracing = get_resilience_tracing()
                tracing.record_state_change(
                    circuit_breaker=self,
                    old_state=old_state,
                    new_state=CircuitState.CLOSED,
                    provider_id=getattr(self, "provider_id", None)
                )
            except Exception as e:
                self.logger.error(
                    "Error recording state change in tracing",
                    error=str(e),
                    exc_info=True
                )

        # Update metrics
        self._update_metrics()

    def _calculate_recovery_timeout(self, category: FailureCategory) -> int:
        """
        Calculate recovery timeout based on strategy and failure category.

        Args:
            category: Category of the failure

        Returns:
            Recovery timeout in seconds
        """
        # Get base timeout
        base_timeout = self.base_recovery_timeout

        # Apply strategy
        if self.recovery_strategy == RecoveryStrategy.FIXED:
            timeout = base_timeout

        elif self.recovery_strategy == RecoveryStrategy.EXPONENTIAL:
            # Exponential backoff: base * 2^attempts
            timeout = base_timeout * (2 ** (self._recovery_attempts - 1))

        elif self.recovery_strategy == RecoveryStrategy.ADAPTIVE:
            # Adaptive based on failure category and pattern
            category_multiplier = self.failure_timeout_multipliers.get(category, 1.0)

            # Calculate failure rate in the last minute
            recent_failures = len([
                ts for ts in self._failure_timestamps
                if (datetime.utcnow() - ts).total_seconds() <= 60
            ])

            # Adjust multiplier based on recent failures
            rate_multiplier = max(1.0, min(5.0, recent_failures / 5.0))

            # Apply both multipliers
            timeout = base_timeout * category_multiplier * rate_multiplier

        elif self.recovery_strategy == RecoveryStrategy.JITTERED:
            # Jittered exponential backoff
            exp_timeout = base_timeout * (2 ** (self._recovery_attempts - 1))
            jitter = random.uniform(1 - self.jitter_factor, 1 + self.jitter_factor)
            timeout = exp_timeout * jitter

        else:
            # Default to base timeout
            timeout = base_timeout

        # Ensure timeout is within bounds
        timeout = max(self.min_recovery_timeout, min(self.max_recovery_timeout, timeout))

        return int(timeout)

    def _update_metrics(self) -> None:
        """Update metrics if metrics service is available."""
        if not self.metrics_service:
            return

        try:
            # Update circuit breaker state metric
            self.metrics_service.record_circuit_breaker_state(
                name=self.name,
                provider_id=getattr(self, "provider_id", "unknown"),
                is_open=self._state == CircuitState.OPEN
            )

            # Update failure count metric
            self.metrics_service.record_circuit_breaker_failures(
                name=self.name,
                provider_id=getattr(self, "provider_id", "unknown"),
                failure_count=self._failure_count
            )

            # Update recovery timeout metric
            self.metrics_service.record_circuit_breaker_recovery_timeout(
                name=self.name,
                provider_id=getattr(self, "provider_id", "unknown"),
                timeout=self._current_recovery_timeout
            )
        except Exception as e:
            self.logger.error(
                "Error updating metrics",
                error=str(e),
                exc_info=True
            )

    def _calculate_failure_rate(self) -> float:
        """
        Calculate the current failure rate.

        Returns:
            Failure rate as a percentage (0-100)
        """
        total = self._failure_count + self._success_count
        if total == 0:
            return 0.0
        return (self._failure_count / total) * 100.0

    def _calculate_time_since_last_failure(self) -> Optional[float]:
        """
        Calculate time since last failure in seconds.

        Returns:
            Time in seconds or None if no failures
        """
        if not self._last_failure_time:
            return None
        return (datetime.utcnow() - self._last_failure_time).total_seconds()

    def _calculate_time_since_last_success(self) -> Optional[float]:
        """
        Calculate time since last success in seconds.

        Returns:
            Time in seconds or None if no successes
        """
        if not self._last_success_time:
            return None
        return (datetime.utcnow() - self._last_success_time).total_seconds()

    def _calculate_time_in_current_state(self) -> float:
        """
        Calculate time in current state in seconds.

        Returns:
            Time in seconds
        """
        return (datetime.utcnow() - self._last_state_change_time).total_seconds()

    def _calculate_health_score(self) -> int:
        """
        Calculate a health score for this circuit breaker.

        Returns:
            Health score (0-100)
        """
        # Start with base score
        score = 100

        # Reduce score based on state
        if self._state == CircuitState.OPEN:
            score -= 50
        elif self._state == CircuitState.HALF_OPEN:
            score -= 25

        # Reduce score based on failure count
        failure_penalty = min(30, self._failure_count * 5)
        score -= failure_penalty

        # Reduce score based on consecutive failures
        consecutive_penalty = min(20, self._consecutive_failures * 5)
        score -= consecutive_penalty

        # Reduce score based on recovery attempts
        recovery_penalty = min(20, self._recovery_attempts * 2)
        score -= recovery_penalty

        # Ensure score is within bounds
        return max(0, min(100, score))

    def _get_most_common_failure(self) -> Optional[str]:
        """
        Get the most common failure category.

        Returns:
            Most common failure category or None if no failures
        """
        if not any(self._failure_categories.values()):
            return None

        most_common = max(self._failure_categories.items(), key=lambda x: x[1])
        return most_common[0].value


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    This class provides a centralized registry for circuit breakers,
    allowing for easy access to circuit breaker state and metrics.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CircuitBreakerRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, metrics_service: Optional[Any] = None):
        """
        Initialize the circuit breaker registry.

        Args:
            metrics_service: Optional metrics service for recording metrics
        """
        # Only initialize once (singleton pattern)
        with self._lock:
            if self._initialized:
                return

            self._circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
            self.metrics_service = metrics_service
            self._initialized = True

            self.logger = logger.bind(component="circuit_breaker_registry")
            self.logger.info("Initialized circuit breaker registry")

    def register(self, circuit_breaker: EnhancedCircuitBreaker) -> None:
        """
        Register a circuit breaker.

        Args:
            circuit_breaker: Circuit breaker to register
        """
        with self._lock:
            name = circuit_breaker.name
            if name in self._circuit_breakers:
                self.logger.warning(f"Circuit breaker {name} already registered")
                return

            self._circuit_breakers[name] = circuit_breaker
            self.logger.info(f"Registered circuit breaker: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a circuit breaker.

        Args:
            name: Name of the circuit breaker to unregister
        """
        with self._lock:
            if name not in self._circuit_breakers:
                self.logger.warning(f"Circuit breaker {name} not found")
                return

            del self._circuit_breakers[name]
            self.logger.info(f"Unregistered circuit breaker: {name}")

    def get(self, name: str) -> Optional[EnhancedCircuitBreaker]:
        """
        Get a circuit breaker by name.

        Args:
            name: Name of the circuit breaker

        Returns:
            Circuit breaker or None if not found
        """
        with self._lock:
            return self._circuit_breakers.get(name)

    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> EnhancedCircuitBreaker:
        """
        Get a circuit breaker by name or create a new one.

        Args:
            name: Name of the circuit breaker
            **kwargs: Arguments for creating a new circuit breaker

        Returns:
            Circuit breaker
        """
        with self._lock:
            circuit_breaker = self.get(name)
            if circuit_breaker:
                return circuit_breaker

            # Create new circuit breaker
            circuit_breaker = EnhancedCircuitBreaker(
                name=name,
                metrics_service=self.metrics_service,
                registry=self,
                **kwargs
            )

            # Register it
            self._circuit_breakers[name] = circuit_breaker
            self.logger.info(f"Created and registered circuit breaker: {name}")

            return circuit_breaker

    def get_all(self) -> Dict[str, EnhancedCircuitBreaker]:
        """
        Get all registered circuit breakers.

        Returns:
            Dictionary of circuit breakers by name
        """
        with self._lock:
            return self._circuit_breakers.copy()

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for name, circuit_breaker in self._circuit_breakers.items():
                self.logger.info(f"Resetting circuit breaker: {name}")
                circuit_breaker.reset()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all circuit breakers.

        Returns:
            Dictionary of metrics by circuit breaker name
        """
        with self._lock:
            metrics = {}
            for name, circuit_breaker in self._circuit_breakers.items():
                metrics[name] = circuit_breaker.get_metrics()
            return metrics

    def get_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all circuit breakers.

        Returns:
            Dictionary of health information by circuit breaker name
        """
        with self._lock:
            health = {}
            for name, circuit_breaker in self._circuit_breakers.items():
                health[name] = circuit_breaker.get_health()
            return health

    def get_open_circuits(self) -> List[str]:
        """
        Get names of all open circuit breakers.

        Returns:
            List of circuit breaker names
        """
        with self._lock:
            return [
                name for name, cb in self._circuit_breakers.items()
                if cb.state == CircuitState.OPEN
            ]

    def get_half_open_circuits(self) -> List[str]:
        """
        Get names of all half-open circuit breakers.

        Returns:
            List of circuit breaker names
        """
        with self._lock:
            return [
                name for name, cb in self._circuit_breakers.items()
                if cb.state == CircuitState.HALF_OPEN
            ]

    def get_closed_circuits(self) -> List[str]:
        """
        Get names of all closed circuit breakers.

        Returns:
            List of circuit breaker names
        """
        with self._lock:
            return [
                name for name, cb in self._circuit_breakers.items()
                if cb.state == CircuitState.CLOSED
            ]

    def get_unhealthy_circuits(self) -> List[str]:
        """
        Get names of all unhealthy circuit breakers.

        Returns:
            List of circuit breaker names
        """
        with self._lock:
            return [
                name for name, cb in self._circuit_breakers.items()
                if cb.get_health()["health_score"] < 50
            ]


# Singleton instance
_circuit_breaker_registry = None


def get_circuit_breaker_registry(metrics_service: Optional[Any] = None) -> CircuitBreakerRegistry:
    """
    Get the singleton instance of the CircuitBreakerRegistry.

    Args:
        metrics_service: Optional metrics service for recording metrics

    Returns:
        CircuitBreakerRegistry instance
    """
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        with CircuitBreakerRegistry._lock:
            if _circuit_breaker_registry is None:
                _circuit_breaker_registry = CircuitBreakerRegistry(metrics_service)
    return _circuit_breaker_registry
