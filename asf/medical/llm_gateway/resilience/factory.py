"""
Factory for creating resilience components.

This module provides factories for creating resilience components,
such as circuit breakers, retries, and timeouts.
"""

import structlog
from typing import Dict, Any, Optional, Type, Union

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker, CircuitState
from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    RecoveryStrategy,
    FailureCategory
)
from asf.medical.llm_gateway.resilience.metrics import get_resilience_metrics

logger = structlog.get_logger("llm_gateway.resilience.factory")


class ResilienceFactory:
    """
    Factory for creating resilience components.
    
    This class provides methods for creating resilience components,
    such as circuit breakers, retries, and timeouts.
    """
    
    def __init__(self, metrics_service: Optional[Any] = None):
        """
        Initialize the resilience factory.
        
        Args:
            metrics_service: Optional metrics service for recording metrics
        """
        self.metrics_service = metrics_service
        self.resilience_metrics = get_resilience_metrics()
        self.circuit_breaker_registry = get_circuit_breaker_registry(metrics_service)
        
        self.logger = logger.bind(component="resilience_factory")
        self.logger.info("Initialized resilience factory")
    
    def create_circuit_breaker(
        self,
        name: str,
        provider_id: Optional[str] = None,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 1,
        reset_timeout: int = 600,  # 10 minutes
        enhanced: bool = True,
        recovery_strategy: Union[RecoveryStrategy, str] = RecoveryStrategy.EXPONENTIAL,
        max_recovery_timeout: int = 1800,  # 30 minutes
        min_recovery_timeout: int = 1,  # 1 second
        jitter_factor: float = 0.2,  # 20% jitter
        failure_timeout_multipliers: Optional[Dict[Union[FailureCategory, str], float]] = None,
        on_state_change: Optional[callable] = None
    ) -> Union[CircuitBreaker, EnhancedCircuitBreaker]:
        """
        Create a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider (for metrics)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Base seconds to wait before testing recovery
            half_open_max_calls: Max calls to allow in HALF_OPEN before deciding
            reset_timeout: Seconds after which to reset failure count in CLOSED state
            enhanced: Whether to create an enhanced circuit breaker
            recovery_strategy: Strategy for determining recovery timeout
            max_recovery_timeout: Maximum recovery timeout in seconds
            min_recovery_timeout: Minimum recovery timeout in seconds
            jitter_factor: Factor for jitter in recovery timeout (0-1)
            failure_timeout_multipliers: Multipliers for recovery timeout by failure category
            on_state_change: Callback for state change notifications
            
        Returns:
            Circuit breaker instance
        """
        # Convert string recovery strategy to enum if needed
        if isinstance(recovery_strategy, str):
            recovery_strategy = RecoveryStrategy(recovery_strategy)
        
        # Convert string failure categories to enums if needed
        if failure_timeout_multipliers:
            converted_multipliers = {}
            for category, multiplier in failure_timeout_multipliers.items():
                if isinstance(category, str):
                    category = FailureCategory(category)
                converted_multipliers[category] = multiplier
            failure_timeout_multipliers = converted_multipliers
        
        # Create circuit breaker
        if enhanced:
            # Check if circuit breaker already exists in registry
            circuit_breaker = self.circuit_breaker_registry.get(name)
            if circuit_breaker:
                self.logger.info(f"Using existing circuit breaker from registry: {name}")
                return circuit_breaker
            
            # Create new enhanced circuit breaker
            self.logger.info(f"Creating enhanced circuit breaker: {name}")
            
            # Create a state change callback that updates metrics
            original_callback = on_state_change
            
            def state_change_callback(old_state: CircuitState, new_state: CircuitState):
                # Call original callback if provided
                if original_callback:
                    original_callback(old_state, new_state)
                
                # Update metrics
                is_open = new_state == CircuitState.OPEN
                self.resilience_metrics.record_circuit_breaker_state(
                    name=name,
                    provider_id=provider_id or "unknown",
                    is_open=is_open
                )
            
            # Create the circuit breaker
            circuit_breaker = EnhancedCircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                reset_timeout=reset_timeout,
                on_state_change=state_change_callback,
                metrics_service=self.resilience_metrics,
                recovery_strategy=recovery_strategy,
                max_recovery_timeout=max_recovery_timeout,
                min_recovery_timeout=min_recovery_timeout,
                jitter_factor=jitter_factor,
                failure_timeout_multipliers=failure_timeout_multipliers,
                registry=self.circuit_breaker_registry
            )
            
            # Set provider ID for metrics
            if provider_id:
                setattr(circuit_breaker, "provider_id", provider_id)
            
            return circuit_breaker
        else:
            # Create basic circuit breaker
            self.logger.info(f"Creating basic circuit breaker: {name}")
            
            # Create a state change callback that updates metrics
            original_callback = on_state_change
            
            def state_change_callback(old_state: CircuitState, new_state: CircuitState):
                # Call original callback if provided
                if original_callback:
                    original_callback(old_state, new_state)
                
                # Update metrics
                is_open = new_state == CircuitState.OPEN
                self.resilience_metrics.record_circuit_breaker_state(
                    name=name,
                    provider_id=provider_id or "unknown",
                    is_open=is_open
                )
            
            # Create the circuit breaker
            circuit_breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                reset_timeout=reset_timeout,
                on_state_change=state_change_callback
            )
            
            # Set provider ID for metrics
            if provider_id:
                setattr(circuit_breaker, "provider_id", provider_id)
            
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[EnhancedCircuitBreaker]:
        """
        Get a circuit breaker by name from the registry.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Circuit breaker or None if not found
        """
        return self.circuit_breaker_registry.get(name)
    
    def get_or_create_circuit_breaker(
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
        circuit_breaker = self.get_circuit_breaker(name)
        if circuit_breaker:
            return circuit_breaker
        
        return self.create_circuit_breaker(name=name, enhanced=True, **kwargs)


# Singleton instance
_resilience_factory = None


def get_resilience_factory(metrics_service: Optional[Any] = None) -> ResilienceFactory:
    """
    Get the singleton instance of the ResilienceFactory.
    
    Args:
        metrics_service: Optional metrics service for recording metrics
        
    Returns:
        ResilienceFactory instance
    """
    global _resilience_factory
    if _resilience_factory is None:
        _resilience_factory = ResilienceFactory(metrics_service)
    return _resilience_factory
