"""Circuit Breaker Pattern Example

This module demonstrates how to use the circuit breaker pattern in both
synchronous and asynchronous code.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any

from asf.medical.ml.dspy.circuit_breaker import CircuitBreaker, AsyncCircuitBreaker, CircuitOpenError
from asf.medical.ml.dspy.circuit_breaker_registry import get_circuit_breaker_registry, get_async_circuit_breaker_registry
from asf.medical.ml.dspy.circuit_breaker_decorators import (
    circuit_breaker,
    async_circuit_breaker,
    with_fallback,
    async_with_fallback
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Using CircuitBreaker directly

def unreliable_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Simulate an unreliable service that sometimes fails.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response
        
    Raises:
        RuntimeError: If the service fails
    """
    if random.random() < fail_rate:
        logger.warning("Service failed!")
        raise RuntimeError("Service unavailable")
    
    logger.info("Service succeeded!")
    return {"status": "success", "data": "Some data"}


def call_with_circuit_breaker() -> None:
    """Call an unreliable service with a circuit breaker."""
    # Create a circuit breaker
    cb = CircuitBreaker(
        name="example_service",
        failure_threshold=3,
        reset_timeout=5.0,
        success_threshold=2
    )
    
    # Make several calls
    for i in range(10):
        try:
            with cb:
                result = unreliable_service(fail_rate=0.7)
                logger.info(f"Call {i+1} succeeded: {result}")
        except CircuitOpenError as e:
            logger.warning(f"Call {i+1} skipped: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Call {i+1} failed: {str(e)}")
        
        # Wait a bit between calls
        time.sleep(1)


# Example 2: Using circuit_breaker decorator

@circuit_breaker("decorated_service", failure_threshold=3, reset_timeout=5.0)
def decorated_unreliable_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Unreliable service with circuit breaker decorator.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response
        
    Raises:
        RuntimeError: If the service fails
    """
    if random.random() < fail_rate:
        logger.warning("Decorated service failed!")
        raise RuntimeError("Decorated service unavailable")
    
    logger.info("Decorated service succeeded!")
    return {"status": "success", "data": "Some decorated data"}


def fallback_service(*args, **kwargs) -> Dict[str, Any]:
    """Fallback service when the main service is unavailable.
    
    Returns:
        Dict[str, Any]: Fallback response
    """
    logger.info("Using fallback service")
    return {"status": "fallback", "data": "Fallback data"}


@with_fallback(fallback_service, handle_exceptions=[CircuitOpenError, RuntimeError])
@circuit_breaker("fallback_decorated_service", failure_threshold=3, reset_timeout=5.0)
def fallback_decorated_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Unreliable service with circuit breaker and fallback decorators.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response or fallback response
    """
    if random.random() < fail_rate:
        logger.warning("Fallback decorated service failed!")
        raise RuntimeError("Fallback decorated service unavailable")
    
    logger.info("Fallback decorated service succeeded!")
    return {"status": "success", "data": "Some fallback decorated data"}


def call_decorated_services() -> None:
    """Call services with circuit breaker decorators."""
    # Call decorated service
    logger.info("\n--- Calling decorated service ---")
    for i in range(10):
        try:
            result = decorated_unreliable_service(fail_rate=0.7)
            logger.info(f"Call {i+1} succeeded: {result}")
        except CircuitOpenError as e:
            logger.warning(f"Call {i+1} skipped: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Call {i+1} failed: {str(e)}")
        
        # Wait a bit between calls
        time.sleep(1)
    
    # Call fallback decorated service
    logger.info("\n--- Calling fallback decorated service ---")
    for i in range(10):
        result = fallback_decorated_service(fail_rate=0.7)
        logger.info(f"Call {i+1} result: {result}")
        
        # Wait a bit between calls
        time.sleep(1)


# Example 3: Using AsyncCircuitBreaker

async def async_unreliable_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Simulate an unreliable async service that sometimes fails.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response
        
    Raises:
        RuntimeError: If the service fails
    """
    # Simulate some async work
    await asyncio.sleep(0.1)
    
    if random.random() < fail_rate:
        logger.warning("Async service failed!")
        raise RuntimeError("Async service unavailable")
    
    logger.info("Async service succeeded!")
    return {"status": "success", "data": "Some async data"}


async def call_with_async_circuit_breaker() -> None:
    """Call an unreliable async service with a circuit breaker."""
    # Create an async circuit breaker
    cb = AsyncCircuitBreaker(
        name="async_example_service",
        failure_threshold=3,
        reset_timeout=5.0,
        success_threshold=2
    )
    
    # Make several calls
    for i in range(10):
        try:
            async with cb:
                result = await async_unreliable_service(fail_rate=0.7)
                logger.info(f"Async call {i+1} succeeded: {result}")
        except CircuitOpenError as e:
            logger.warning(f"Async call {i+1} skipped: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Async call {i+1} failed: {str(e)}")
        
        # Wait a bit between calls
        await asyncio.sleep(1)


# Example 4: Using async_circuit_breaker decorator

@async_circuit_breaker("async_decorated_service", failure_threshold=3, reset_timeout=5.0)
async def async_decorated_unreliable_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Unreliable async service with circuit breaker decorator.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response
        
    Raises:
        RuntimeError: If the service fails
    """
    # Simulate some async work
    await asyncio.sleep(0.1)
    
    if random.random() < fail_rate:
        logger.warning("Async decorated service failed!")
        raise RuntimeError("Async decorated service unavailable")
    
    logger.info("Async decorated service succeeded!")
    return {"status": "success", "data": "Some async decorated data"}


async def async_fallback_service(*args, **kwargs) -> Dict[str, Any]:
    """Fallback async service when the main service is unavailable.
    
    Returns:
        Dict[str, Any]: Fallback response
    """
    logger.info("Using async fallback service")
    return {"status": "fallback", "data": "Async fallback data"}


@async_with_fallback(async_fallback_service, handle_exceptions=[CircuitOpenError, RuntimeError])
@async_circuit_breaker("async_fallback_decorated_service", failure_threshold=3, reset_timeout=5.0)
async def async_fallback_decorated_service(fail_rate: float = 0.5) -> Dict[str, Any]:
    """Unreliable async service with circuit breaker and fallback decorators.
    
    Args:
        fail_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: Service response or fallback response
    """
    # Simulate some async work
    await asyncio.sleep(0.1)
    
    if random.random() < fail_rate:
        logger.warning("Async fallback decorated service failed!")
        raise RuntimeError("Async fallback decorated service unavailable")
    
    logger.info("Async fallback decorated service succeeded!")
    return {"status": "success", "data": "Some async fallback decorated data"}


async def call_async_decorated_services() -> None:
    """Call async services with circuit breaker decorators."""
    # Call async decorated service
    logger.info("\n--- Calling async decorated service ---")
    for i in range(10):
        try:
            result = await async_decorated_unreliable_service(fail_rate=0.7)
            logger.info(f"Async call {i+1} succeeded: {result}")
        except CircuitOpenError as e:
            logger.warning(f"Async call {i+1} skipped: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Async call {i+1} failed: {str(e)}")
        
        # Wait a bit between calls
        await asyncio.sleep(1)
    
    # Call async fallback decorated service
    logger.info("\n--- Calling async fallback decorated service ---")
    for i in range(10):
        result = await async_fallback_decorated_service(fail_rate=0.7)
        logger.info(f"Async call {i+1} result: {result}")
        
        # Wait a bit between calls
        await asyncio.sleep(1)


# Example 5: Using circuit breaker registry

async def demonstrate_registry() -> None:
    """Demonstrate using the circuit breaker registry."""
    # Get the registry
    sync_registry = get_circuit_breaker_registry()
    async_registry = await get_async_circuit_breaker_registry()
    
    # Create some circuit breakers
    cb1 = sync_registry.get_or_create("service1", failure_threshold=3)
    cb2 = sync_registry.get_or_create("service2", failure_threshold=5)
    
    async_cb1 = await async_registry.get_or_create("async_service1", failure_threshold=3)
    async_cb2 = await async_registry.get_or_create("async_service2", failure_threshold=5)
    
    # Get status
    sync_status = sync_registry.get_status()
    async_status = await async_registry.get_status()
    
    logger.info(f"Sync circuit breakers: {sync_status}")
    logger.info(f"Async circuit breakers: {async_status}")
    
    # Reset a specific circuit breaker
    sync_registry.reset("service1")
    await async_registry.reset("async_service1")
    
    # Reset all circuit breakers
    sync_registry.reset()
    await async_registry.reset()


async def main() -> None:
    """Run all examples."""
    # Example 1: Using CircuitBreaker directly
    logger.info("\n=== Example 1: Using CircuitBreaker directly ===")
    call_with_circuit_breaker()
    
    # Example 2: Using circuit_breaker decorator
    logger.info("\n=== Example 2: Using circuit_breaker decorator ===")
    call_decorated_services()
    
    # Example 3: Using AsyncCircuitBreaker
    logger.info("\n=== Example 3: Using AsyncCircuitBreaker ===")
    await call_with_async_circuit_breaker()
    
    # Example 4: Using async_circuit_breaker decorator
    logger.info("\n=== Example 4: Using async_circuit_breaker decorator ===")
    await call_async_decorated_services()
    
    # Example 5: Using circuit breaker registry
    logger.info("\n=== Example 5: Using circuit breaker registry ===")
    await demonstrate_registry()


if __name__ == "__main__":
    asyncio.run(main())
