"""
Metrics integration for resilience patterns.

This module provides metrics integration for resilience patterns,
including circuit breakers, retries, and timeouts.
"""

import time
from typing import Dict, Any, Optional, List, Set

import structlog

# Try to import metrics service
try:
    from asf.medical.llm_gateway.observability.metrics import MetricsService
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Try to import prometheus exporter
try:
    from asf.medical.llm_gateway.observability.prometheus import (
        get_prometheus_exporter,
        circuit_breaker_state,
        circuit_breaker_failures
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger("llm_gateway.resilience.metrics")


class ResilienceMetrics:
    """
    Metrics integration for resilience patterns.
    
    This class provides methods for recording metrics related to
    resilience patterns, including circuit breakers, retries, and timeouts.
    
    It supports both the MetricsService and Prometheus exporter.
    """
    
    def __init__(self):
        """Initialize the resilience metrics."""
        self.logger = logger.bind(component="resilience_metrics")
        
        # Try to get metrics service
        self.metrics_service = None
        if METRICS_AVAILABLE:
            try:
                self.metrics_service = MetricsService()
                self.logger.info("Initialized metrics service for resilience patterns")
            except Exception as e:
                self.logger.error(
                    "Failed to initialize metrics service",
                    error=str(e),
                    exc_info=True
                )
        
        # Try to get prometheus exporter
        self.prometheus_exporter = None
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_exporter = get_prometheus_exporter()
                self.logger.info("Initialized Prometheus exporter for resilience patterns")
            except Exception as e:
                self.logger.error(
                    "Failed to initialize Prometheus exporter",
                    error=str(e),
                    exc_info=True
                )
    
    def record_circuit_breaker_state(
        self,
        name: str,
        provider_id: str,
        is_open: bool
    ) -> None:
        """
        Record circuit breaker state.
        
        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider
            is_open: Whether the circuit is open
        """
        # Log the state change
        self.logger.debug(
            "Recording circuit breaker state",
            name=name,
            provider_id=provider_id,
            is_open=is_open
        )
        
        # Record with metrics service
        if self.metrics_service:
            try:
                self.metrics_service.record_circuit_breaker_state(
                    name=name,
                    provider_id=provider_id,
                    is_open=is_open
                )
            except Exception as e:
                self.logger.error(
                    "Failed to record circuit breaker state with metrics service",
                    error=str(e),
                    exc_info=True
                )
        
        # Record with prometheus exporter
        if self.prometheus_exporter:
            try:
                self.prometheus_exporter.update_circuit_breaker(
                    provider_id=provider_id,
                    is_open=is_open,
                    failure_count=0  # We'll update this separately
                )
            except Exception as e:
                self.logger.error(
                    "Failed to record circuit breaker state with Prometheus exporter",
                    error=str(e),
                    exc_info=True
                )
    
    def record_circuit_breaker_failures(
        self,
        name: str,
        provider_id: str,
        failure_count: int
    ) -> None:
        """
        Record circuit breaker failures.
        
        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider
            failure_count: Number of failures
        """
        # Log the failures
        self.logger.debug(
            "Recording circuit breaker failures",
            name=name,
            provider_id=provider_id,
            failure_count=failure_count
        )
        
        # Record with metrics service
        if self.metrics_service:
            try:
                # Check if the metrics service has a method for recording failures
                if hasattr(self.metrics_service, "record_circuit_breaker_failures"):
                    self.metrics_service.record_circuit_breaker_failures(
                        name=name,
                        provider_id=provider_id,
                        failure_count=failure_count
                    )
                # Fall back to generic counter
                elif hasattr(self.metrics_service, "increment"):
                    self.metrics_service.increment(
                        "circuit_breaker_failures",
                        value=failure_count,
                        labels={
                            "name": name,
                            "provider": provider_id
                        }
                    )
            except Exception as e:
                self.logger.error(
                    "Failed to record circuit breaker failures with metrics service",
                    error=str(e),
                    exc_info=True
                )
        
        # Record with prometheus exporter
        if self.prometheus_exporter:
            try:
                # Update the circuit breaker failures counter
                circuit_breaker_failures.labels(provider=provider_id).inc(failure_count)
            except Exception as e:
                self.logger.error(
                    "Failed to record circuit breaker failures with Prometheus exporter",
                    error=str(e),
                    exc_info=True
                )
    
    def record_circuit_breaker_recovery_timeout(
        self,
        name: str,
        provider_id: str,
        timeout: int
    ) -> None:
        """
        Record circuit breaker recovery timeout.
        
        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider
            timeout: Recovery timeout in seconds
        """
        # Log the timeout
        self.logger.debug(
            "Recording circuit breaker recovery timeout",
            name=name,
            provider_id=provider_id,
            timeout=timeout
        )
        
        # Record with metrics service
        if self.metrics_service:
            try:
                # Check if the metrics service has a method for recording timeout
                if hasattr(self.metrics_service, "record_circuit_breaker_recovery_timeout"):
                    self.metrics_service.record_circuit_breaker_recovery_timeout(
                        name=name,
                        provider_id=provider_id,
                        timeout=timeout
                    )
                # Fall back to generic gauge
                elif hasattr(self.metrics_service, "gauge"):
                    self.metrics_service.gauge(
                        "circuit_breaker_recovery_timeout",
                        value=timeout,
                        labels={
                            "name": name,
                            "provider": provider_id
                        }
                    )
            except Exception as e:
                self.logger.error(
                    "Failed to record circuit breaker recovery timeout with metrics service",
                    error=str(e),
                    exc_info=True
                )
    
    def record_retry_attempt(
        self,
        name: str,
        provider_id: str,
        attempt: int,
        max_attempts: int
    ) -> None:
        """
        Record retry attempt.
        
        Args:
            name: Name of the retry
            provider_id: ID of the provider
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
        """
        # Log the retry attempt
        self.logger.debug(
            "Recording retry attempt",
            name=name,
            provider_id=provider_id,
            attempt=attempt,
            max_attempts=max_attempts
        )
        
        # Record with metrics service
        if self.metrics_service:
            try:
                # Check if the metrics service has a method for recording retry
                if hasattr(self.metrics_service, "record_retry_attempt"):
                    self.metrics_service.record_retry_attempt(
                        name=name,
                        provider_id=provider_id,
                        attempt=attempt,
                        max_attempts=max_attempts
                    )
                # Fall back to generic counter
                elif hasattr(self.metrics_service, "increment"):
                    self.metrics_service.increment(
                        "retry_attempts",
                        value=1,
                        labels={
                            "name": name,
                            "provider": provider_id,
                            "attempt": str(attempt),
                            "max_attempts": str(max_attempts)
                        }
                    )
            except Exception as e:
                self.logger.error(
                    "Failed to record retry attempt with metrics service",
                    error=str(e),
                    exc_info=True
                )
    
    def record_timeout(
        self,
        name: str,
        provider_id: str,
        timeout_ms: int,
        actual_duration_ms: Optional[int] = None
    ) -> None:
        """
        Record timeout.
        
        Args:
            name: Name of the timeout
            provider_id: ID of the provider
            timeout_ms: Timeout in milliseconds
            actual_duration_ms: Actual duration in milliseconds (if available)
        """
        # Log the timeout
        self.logger.debug(
            "Recording timeout",
            name=name,
            provider_id=provider_id,
            timeout_ms=timeout_ms,
            actual_duration_ms=actual_duration_ms
        )
        
        # Record with metrics service
        if self.metrics_service:
            try:
                # Check if the metrics service has a method for recording timeout
                if hasattr(self.metrics_service, "record_timeout"):
                    self.metrics_service.record_timeout(
                        name=name,
                        provider_id=provider_id,
                        timeout_ms=timeout_ms,
                        actual_duration_ms=actual_duration_ms
                    )
                # Fall back to generic counter and histogram
                elif hasattr(self.metrics_service, "increment") and hasattr(self.metrics_service, "observe"):
                    # Increment timeout counter
                    self.metrics_service.increment(
                        "timeouts",
                        value=1,
                        labels={
                            "name": name,
                            "provider": provider_id
                        }
                    )
                    
                    # Record actual duration if available
                    if actual_duration_ms is not None:
                        self.metrics_service.observe(
                            "timeout_duration_ms",
                            value=actual_duration_ms,
                            labels={
                                "name": name,
                                "provider": provider_id,
                                "timeout_ms": str(timeout_ms)
                            }
                        )
            except Exception as e:
                self.logger.error(
                    "Failed to record timeout with metrics service",
                    error=str(e),
                    exc_info=True
                )


# Singleton instance
_resilience_metrics = None


def get_resilience_metrics() -> ResilienceMetrics:
    """
    Get the singleton instance of the ResilienceMetrics.
    
    Returns:
        ResilienceMetrics instance
    """
    global _resilience_metrics
    if _resilience_metrics is None:
        _resilience_metrics = ResilienceMetrics()
    return _resilience_metrics
