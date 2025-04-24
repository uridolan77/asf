"""
Demo script for enhanced circuit breaker.

This script demonstrates the enhanced circuit breaker with metrics integration,
adaptive recovery, and centralized registry.
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import structlog

from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    RecoveryStrategy,
    FailureCategory
)
from asf.medical.llm_gateway.resilience.factory import get_resilience_factory
from asf.medical.llm_gateway.resilience.metrics import get_resilience_metrics
from asf.medical.llm_gateway.resilience.decorators import (
    with_circuit_breaker,
    with_retry,
    with_timeout,
    CircuitOpenError
)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger("demo")


class DemoService:
    """Demo service for testing circuit breakers."""
    
    def __init__(self, service_id: str):
        """
        Initialize the demo service.
        
        Args:
            service_id: Service ID
        """
        self.service_id = service_id
        self.resilience_factory = get_resilience_factory()
        self.resilience_metrics = get_resilience_metrics()
        self.circuit_breaker_registry = get_circuit_breaker_registry()
        
        # Create circuit breakers
        self.main_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{service_id}_main",
            provider_id=service_id,
            failure_threshold=3,
            recovery_timeout=5,
            recovery_strategy=RecoveryStrategy.EXPONENTIAL
        )
        
        self.timeout_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{service_id}_timeout",
            provider_id=service_id,
            failure_threshold=2,
            recovery_timeout=5,
            recovery_strategy=RecoveryStrategy.ADAPTIVE,
            failure_timeout_multipliers={
                FailureCategory.TIMEOUT: 3.0,
                FailureCategory.RATE_LIMIT: 5.0
            }
        )
        
        self.rate_limit_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{service_id}_rate_limit",
            provider_id=service_id,
            failure_threshold=2,
            recovery_timeout=10,
            recovery_strategy=RecoveryStrategy.JITTERED
        )
        
        logger.info(
            f"Initialized demo service: {service_id}",
            circuit_breakers=[
                self.main_circuit_breaker.name,
                self.timeout_circuit_breaker.name,
                self.rate_limit_circuit_breaker.name
            ]
        )
    
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=5)
    async def call_service(self, request_id: str, fail_probability: float = 0.3) -> Dict[str, Any]:
        """
        Call a service with circuit breaker protection.
        
        Args:
            request_id: Request ID
            fail_probability: Probability of failure
            
        Returns:
            Response data
        """
        logger.info(f"Calling service: {self.service_id}", request_id=request_id)
        
        # Simulate service call
        await asyncio.sleep(0.1)
        
        # Randomly fail
        if random.random() < fail_probability:
            logger.error(f"Service call failed: {self.service_id}", request_id=request_id)
            raise Exception(f"Service call failed: {self.service_id}")
        
        # Return success
        return {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    
    @with_circuit_breaker(
        circuit_breaker="timeout_circuit_breaker",
        failure_category_mapping={
            asyncio.TimeoutError: FailureCategory.TIMEOUT
        }
    )
    @with_timeout(timeout_seconds=0.5)
    async def call_slow_service(self, request_id: str, delay: float = 1.0) -> Dict[str, Any]:
        """
        Call a slow service with timeout and circuit breaker protection.
        
        Args:
            request_id: Request ID
            delay: Delay in seconds
            
        Returns:
            Response data
        """
        logger.info(
            f"Calling slow service: {self.service_id}",
            request_id=request_id,
            delay=delay
        )
        
        # Simulate slow service call
        await asyncio.sleep(delay)
        
        # Return success
        return {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "delay": delay
        }
    
    @with_circuit_breaker(
        circuit_breaker="rate_limit_circuit_breaker",
        failure_category_mapping={
            ValueError: FailureCategory.RATE_LIMIT
        }
    )
    @with_retry(max_attempts=3, backoff_factor=1.0)
    async def call_rate_limited_service(
        self,
        request_id: str,
        rate_limit_probability: float = 0.5
    ) -> Dict[str, Any]:
        """
        Call a rate-limited service with retry and circuit breaker protection.
        
        Args:
            request_id: Request ID
            rate_limit_probability: Probability of rate limit
            
        Returns:
            Response data
        """
        logger.info(
            f"Calling rate-limited service: {self.service_id}",
            request_id=request_id,
            rate_limit_probability=rate_limit_probability
        )
        
        # Simulate service call
        await asyncio.sleep(0.1)
        
        # Randomly hit rate limit
        if random.random() < rate_limit_probability:
            logger.warning(
                f"Rate limit hit: {self.service_id}",
                request_id=request_id
            )
            raise ValueError(f"Rate limit exceeded: {self.service_id}")
        
        # Return success
        return {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    
    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get status of all circuit breakers.
        
        Returns:
            Circuit breaker status
        """
        return {
            "service_id": self.service_id,
            "circuit_breakers": {
                "main": self.main_circuit_breaker.get_metrics(),
                "timeout": self.timeout_circuit_breaker.get_metrics(),
                "rate_limit": self.rate_limit_circuit_breaker.get_metrics()
            }
        }


async def run_demo():
    """Run the demo."""
    logger.info("Starting enhanced circuit breaker demo")
    
    # Create demo service
    service = DemoService("demo_service")
    
    # Test normal service calls
    logger.info("Testing normal service calls")
    for i in range(10):
        try:
            response = await service.call_service(f"req_{i}", fail_probability=0.3)
            logger.info(f"Service call succeeded", response=response)
        except Exception as e:
            logger.error(f"Service call failed: {str(e)}")
    
    # Print circuit breaker status
    status = await service.get_circuit_breaker_status()
    logger.info("Circuit breaker status after normal calls", status=status)
    
    # Test timeout service calls
    logger.info("Testing timeout service calls")
    for i in range(5):
        try:
            response = await service.call_slow_service(f"slow_req_{i}", delay=1.0)
            logger.info(f"Slow service call succeeded", response=response)
        except asyncio.TimeoutError:
            logger.error(f"Slow service call timed out")
        except CircuitOpenError as e:
            logger.warning(f"Circuit open: {str(e)}")
        except Exception as e:
            logger.error(f"Slow service call failed: {str(e)}")
    
    # Print circuit breaker status
    status = await service.get_circuit_breaker_status()
    logger.info("Circuit breaker status after timeout calls", status=status)
    
    # Wait for circuit to recover
    logger.info("Waiting for circuit to recover")
    await asyncio.sleep(10)
    
    # Test rate-limited service calls
    logger.info("Testing rate-limited service calls")
    for i in range(5):
        try:
            response = await service.call_rate_limited_service(
                f"rate_req_{i}",
                rate_limit_probability=0.7
            )
            logger.info(f"Rate-limited service call succeeded", response=response)
        except CircuitOpenError as e:
            logger.warning(f"Circuit open: {str(e)}")
        except Exception as e:
            logger.error(f"Rate-limited service call failed: {str(e)}")
    
    # Print circuit breaker status
    status = await service.get_circuit_breaker_status()
    logger.info("Circuit breaker status after rate-limited calls", status=status)
    
    # Print registry status
    registry = get_circuit_breaker_registry()
    registry_status = registry.get_metrics()
    logger.info("Circuit breaker registry status", registry_status=registry_status)
    
    # Print health status
    health_status = registry.get_health()
    logger.info("Circuit breaker health status", health_status=health_status)
    
    # Print open circuits
    open_circuits = registry.get_open_circuits()
    logger.info("Open circuits", open_circuits=open_circuits)
    
    logger.info("Demo completed")


if __name__ == "__main__":
    asyncio.run(run_demo())
