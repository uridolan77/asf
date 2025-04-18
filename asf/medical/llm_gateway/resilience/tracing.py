"""
Distributed tracing integration for resilience patterns.

This module provides distributed tracing integration for resilience patterns,
including circuit breakers, retries, and timeouts.
"""

import asyncio
import functools
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Callable, TypeVar, cast, Union

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, StatusCode

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitState
from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    FailureCategory
)
from asf.medical.llm_gateway.resilience.decorators import CircuitOpenError

# Try to import tracing service
try:
    from asf.medical.llm_gateway.observability.tracing import TracingService
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

logger = structlog.get_logger("llm_gateway.resilience.tracing")

# Type variables for better type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ResilienceTracing:
    """
    Distributed tracing integration for resilience patterns.
    
    This class provides methods for tracing resilience patterns,
    including circuit breakers, retries, and timeouts.
    """
    
    def __init__(self, service_name: str = "llm_gateway_resilience"):
        """
        Initialize resilience tracing.
        
        Args:
            service_name: Name of the service for tracing context
        """
        self.service_name = service_name
        self.logger = logger.bind(component="resilience_tracing")
        
        # Try to get tracing service
        self.tracing_service = None
        if TRACING_AVAILABLE:
            try:
                self.tracing_service = TracingService(
                    service_name=service_name,
                    additional_attributes={"component": "resilience"}
                )
                self.logger.info("Initialized tracing service for resilience patterns")
            except Exception as e:
                self.logger.error(
                    "Failed to initialize tracing service",
                    error=str(e),
                    exc_info=True
                )
        
        # Get tracer directly if tracing service is not available
        if not self.tracing_service:
            self.tracer = trace.get_tracer(
                f"{service_name}_tracer",
                schema_url="https://opentelemetry.io/schemas/1.9.0"
            )
        else:
            self.tracer = self.tracing_service.tracer
    
    @contextmanager
    def circuit_breaker_span(
        self,
        circuit_breaker: Union[EnhancedCircuitBreaker, str],
        operation: str,
        provider_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Iterator[Span]:
        """
        Create a span for circuit breaker operations.
        
        Args:
            circuit_breaker: Circuit breaker instance or name
            operation: Operation being performed (call, success, failure)
            provider_id: Provider ID (for metrics)
            attributes: Additional span attributes
            
        Yields:
            Active span
        """
        # Get circuit breaker name
        cb_name = circuit_breaker.name if hasattr(circuit_breaker, "name") else str(circuit_breaker)
        
        # Create span name
        span_name = f"circuit_breaker.{operation}"
        
        # Create span attributes
        span_attrs = {
            "circuit_breaker.name": cb_name,
            "circuit_breaker.operation": operation
        }
        
        # Add provider ID if available
        if provider_id:
            span_attrs["provider.id"] = provider_id
        
        # Add circuit breaker state if available
        if hasattr(circuit_breaker, "state"):
            span_attrs["circuit_breaker.state"] = circuit_breaker.state.value
        
        # Add additional attributes
        if attributes:
            span_attrs.update(attributes)
        
        # Use tracing service if available
        if self.tracing_service:
            with self.tracing_service.start_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes=span_attrs
            ) as span:
                try:
                    yield span
                except Exception as e:
                    # Add exception details to span
                    if isinstance(e, CircuitOpenError):
                        span.set_attribute("circuit_breaker.open", True)
                    
                    # Re-raise the exception
                    raise
        else:
            # Use tracer directly
            with self.tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes=span_attrs
            ) as span:
                try:
                    yield span
                except Exception as e:
                    # Add exception details to span
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, str(e))
                    
                    if isinstance(e, CircuitOpenError):
                        span.set_attribute("circuit_breaker.open", True)
                    
                    # Re-raise the exception
                    raise
    
    def record_state_change(
        self,
        circuit_breaker: Union[EnhancedCircuitBreaker, str],
        old_state: CircuitState,
        new_state: CircuitState,
        provider_id: Optional[str] = None,
        failure_category: Optional[FailureCategory] = None
    ) -> None:
        """
        Record circuit breaker state change in the current span.
        
        Args:
            circuit_breaker: Circuit breaker instance or name
            old_state: Old circuit breaker state
            new_state: New circuit breaker state
            provider_id: Provider ID (for metrics)
            failure_category: Failure category (if applicable)
        """
        # Get circuit breaker name
        cb_name = circuit_breaker.name if hasattr(circuit_breaker, "name") else str(circuit_breaker)
        
        # Get current span
        current_span = trace.get_current_span()
        if not current_span:
            return
        
        # Add state change event
        current_span.add_event(
            name="circuit_breaker.state_change",
            attributes={
                "circuit_breaker.name": cb_name,
                "circuit_breaker.old_state": old_state.value,
                "circuit_breaker.new_state": new_state.value,
                "provider.id": provider_id or "unknown",
                "failure_category": failure_category.value if failure_category else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update span attributes
        current_span.set_attribute("circuit_breaker.state", new_state.value)
        
        # Set span status for OPEN state
        if new_state == CircuitState.OPEN:
            current_span.set_status(
                StatusCode.ERROR,
                f"Circuit breaker {cb_name} opened"
            )
    
    def record_failure(
        self,
        circuit_breaker: Union[EnhancedCircuitBreaker, str],
        failure_count: int,
        failure_category: Optional[FailureCategory] = None,
        provider_id: Optional[str] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """
        Record circuit breaker failure in the current span.
        
        Args:
            circuit_breaker: Circuit breaker instance or name
            failure_count: Current failure count
            failure_category: Failure category
            provider_id: Provider ID (for metrics)
            exception: Exception that caused the failure
        """
        # Get circuit breaker name
        cb_name = circuit_breaker.name if hasattr(circuit_breaker, "name") else str(circuit_breaker)
        
        # Get current span
        current_span = trace.get_current_span()
        if not current_span:
            return
        
        # Add failure event
        current_span.add_event(
            name="circuit_breaker.failure",
            attributes={
                "circuit_breaker.name": cb_name,
                "circuit_breaker.failure_count": failure_count,
                "failure_category": failure_category.value if failure_category else "unknown",
                "provider.id": provider_id or "unknown",
                "exception": str(exception) if exception else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update span attributes
        current_span.set_attribute("circuit_breaker.failure_count", failure_count)
        
        # Record exception if available
        if exception:
            current_span.record_exception(exception)
            current_span.set_status(StatusCode.ERROR, str(exception))
    
    def record_success(
        self,
        circuit_breaker: Union[EnhancedCircuitBreaker, str],
        provider_id: Optional[str] = None
    ) -> None:
        """
        Record circuit breaker success in the current span.
        
        Args:
            circuit_breaker: Circuit breaker instance or name
            provider_id: Provider ID (for metrics)
        """
        # Get circuit breaker name
        cb_name = circuit_breaker.name if hasattr(circuit_breaker, "name") else str(circuit_breaker)
        
        # Get current span
        current_span = trace.get_current_span()
        if not current_span:
            return
        
        # Add success event
        current_span.add_event(
            name="circuit_breaker.success",
            attributes={
                "circuit_breaker.name": cb_name,
                "provider.id": provider_id or "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def record_retry(
        self,
        operation: str,
        attempt: int,
        max_attempts: int,
        provider_id: Optional[str] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """
        Record retry attempt in the current span.
        
        Args:
            operation: Operation being retried
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            provider_id: Provider ID (for metrics)
            exception: Exception that caused the retry
        """
        # Get current span
        current_span = trace.get_current_span()
        if not current_span:
            return
        
        # Add retry event
        current_span.add_event(
            name="resilience.retry",
            attributes={
                "retry.operation": operation,
                "retry.attempt": attempt,
                "retry.max_attempts": max_attempts,
                "provider.id": provider_id or "unknown",
                "exception": str(exception) if exception else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update span attributes
        current_span.set_attribute("retry.attempt", attempt)
        
        # Record exception if available
        if exception:
            current_span.record_exception(exception)
            current_span.set_status(StatusCode.ERROR, str(exception))
    
    def record_timeout(
        self,
        operation: str,
        timeout_seconds: float,
        actual_duration_ms: Optional[int] = None,
        provider_id: Optional[str] = None
    ) -> None:
        """
        Record timeout in the current span.
        
        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout in seconds
            actual_duration_ms: Actual duration in milliseconds
            provider_id: Provider ID (for metrics)
        """
        # Get current span
        current_span = trace.get_current_span()
        if not current_span:
            return
        
        # Add timeout event
        current_span.add_event(
            name="resilience.timeout",
            attributes={
                "timeout.operation": operation,
                "timeout.seconds": timeout_seconds,
                "timeout.actual_duration_ms": actual_duration_ms,
                "provider.id": provider_id or "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update span attributes
        current_span.set_attribute("timeout.seconds", timeout_seconds)
        if actual_duration_ms:
            current_span.set_attribute("timeout.actual_duration_ms", actual_duration_ms)
        
        # Set span status
        current_span.set_status(
            StatusCode.ERROR,
            f"Operation {operation} timed out after {timeout_seconds} seconds"
        )


def with_circuit_breaker_tracing(
    operation: str,
    provider_id: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for tracing circuit breaker operations.
    
    Args:
        operation: Operation being performed
        provider_id: Provider ID (for metrics)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Get function name for logging
        func_name = func.__qualname__
        
        # Get resilience tracing
        tracing = get_resilience_tracing()
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Get circuit breaker
                circuit_breaker = getattr(self, "name", self)
                
                # Get provider ID
                provider = getattr(self, "provider_id", provider_id)
                
                # Create span attributes
                attributes = {
                    "function": func_name,
                    "operation": operation
                }
                
                # Create span
                with tracing.circuit_breaker_span(
                    circuit_breaker=circuit_breaker,
                    operation=operation,
                    provider_id=provider,
                    attributes=attributes
                ) as span:
                    # Call the function
                    return await func(self, *args, **kwargs)
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Get circuit breaker
                circuit_breaker = getattr(self, "name", self)
                
                # Get provider ID
                provider = getattr(self, "provider_id", provider_id)
                
                # Create span attributes
                attributes = {
                    "function": func_name,
                    "operation": operation
                }
                
                # Create span
                with tracing.circuit_breaker_span(
                    circuit_breaker=circuit_breaker,
                    operation=operation,
                    provider_id=provider,
                    attributes=attributes
                ) as span:
                    # Call the function
                    return func(self, *args, **kwargs)
            
            return cast(F, sync_wrapper)
    
    return decorator


# Singleton instance
_resilience_tracing = None


def get_resilience_tracing() -> ResilienceTracing:
    """
    Get the singleton instance of the ResilienceTracing.
    
    Returns:
        ResilienceTracing instance
    """
    global _resilience_tracing
    if _resilience_tracing is None:
        _resilience_tracing = ResilienceTracing()
    return _resilience_tracing
