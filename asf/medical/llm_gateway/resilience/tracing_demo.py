"""
Demo script for circuit breaker with distributed tracing.

This script demonstrates the integration of circuit breakers with
distributed tracing for better visibility into failures.
"""

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.trace import SpanKind, StatusCode

from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    RecoveryStrategy,
    FailureCategory
)
from asf.medical.llm_gateway.resilience.factory import get_resilience_factory
from asf.medical.llm_gateway.resilience.metrics import get_resilience_metrics
from asf.medical.llm_gateway.resilience.tracing import (
    get_resilience_tracing,
    ResilienceTracing
)
from asf.medical.llm_gateway.resilience.traced_decorators import (
    with_traced_circuit_breaker,
    with_traced_retry,
    with_traced_timeout
)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger("tracing_demo")


def setup_tracing():
    """Set up OpenTelemetry tracing."""
    # Create resource with service info
    resource = Resource.create({
        SERVICE_NAME: "resilience_tracing_demo",
        "service.version": "1.0.0",
        "deployment.environment": "demo"
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Add console exporter
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Get a tracer
    return trace.get_tracer(
        "resilience_tracing_demo",
        schema_url="https://opentelemetry.io/schemas/1.9.0"
    )


class TracedDemoService:
    """Demo service for testing circuit breakers with tracing."""
    
    def __init__(self, service_id: str):
        """
        Initialize the demo service.
        
        Args:
            service_id: Service ID
        """
        self.service_id = service_id
        self.provider_id = service_id  # For compatibility with decorators
        
        # Get resilience components
        self.resilience_factory = get_resilience_factory()
        self.resilience_metrics = get_resilience_metrics()
        self.resilience_tracing = get_resilience_tracing()
        
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
            f"Initialized traced demo service: {service_id}",
            circuit_breakers=[
                self.main_circuit_breaker.name,
                self.timeout_circuit_breaker.name,
                self.rate_limit_circuit_breaker.name
            ]
        )
    
    @with_traced_circuit_breaker(
        failure_threshold=3,
        recovery_timeout=5,
        provider_id="demo_service"
    )
    async def call_service(self, request_id: str, fail_probability: float = 0.3) -> Dict[str, Any]:
        """
        Call a service with circuit breaker protection and tracing.
        
        Args:
            request_id: Request ID
            fail_probability: Probability of failure
            
        Returns:
            Response data
        """
        logger.info(f"Calling service: {self.service_id}", request_id=request_id)
        
        # Get current span
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("request.id", request_id)
            current_span.set_attribute("service.id", self.service_id)
            current_span.set_attribute("fail_probability", fail_probability)
        
        # Simulate service call
        await asyncio.sleep(0.1)
        
        # Randomly fail
        if random.random() < fail_probability:
            logger.error(f"Service call failed: {self.service_id}", request_id=request_id)
            
            # Add event to span
            if current_span:
                current_span.add_event(
                    "service.failure",
                    {
                        "service.id": self.service_id,
                        "request.id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            raise Exception(f"Service call failed: {self.service_id}")
        
        # Return success
        response = {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success"
        }
        
        # Add event to span
        if current_span:
            current_span.add_event(
                "service.success",
                {
                    "service.id": self.service_id,
                    "request.id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        return response
    
    @with_traced_circuit_breaker(
        circuit_breaker="timeout_circuit_breaker",
        failure_category_mapping={
            asyncio.TimeoutError: FailureCategory.TIMEOUT
        },
        provider_id="demo_service"
    )
    @with_traced_timeout(timeout_seconds=0.5)
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
        
        # Get current span
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("request.id", request_id)
            current_span.set_attribute("service.id", self.service_id)
            current_span.set_attribute("delay", delay)
        
        # Simulate slow service call
        await asyncio.sleep(delay)
        
        # Return success
        response = {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "delay": delay
        }
        
        # Add event to span
        if current_span:
            current_span.add_event(
                "service.success",
                {
                    "service.id": self.service_id,
                    "request.id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "delay": delay
                }
            )
        
        return response
    
    @with_traced_circuit_breaker(
        circuit_breaker="rate_limit_circuit_breaker",
        failure_category_mapping={
            ValueError: FailureCategory.RATE_LIMIT
        },
        provider_id="demo_service"
    )
    @with_traced_retry(max_attempts=3, backoff_factor=1.0)
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
        
        # Get current span
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("request.id", request_id)
            current_span.set_attribute("service.id", self.service_id)
            current_span.set_attribute("rate_limit_probability", rate_limit_probability)
        
        # Simulate service call
        await asyncio.sleep(0.1)
        
        # Randomly hit rate limit
        if random.random() < rate_limit_probability:
            logger.warning(
                f"Rate limit hit: {self.service_id}",
                request_id=request_id
            )
            
            # Add event to span
            if current_span:
                current_span.add_event(
                    "service.rate_limit",
                    {
                        "service.id": self.service_id,
                        "request.id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            raise ValueError(f"Rate limit exceeded: {self.service_id}")
        
        # Return success
        response = {
            "service_id": self.service_id,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success"
        }
        
        # Add event to span
        if current_span:
            current_span.add_event(
                "service.success",
                {
                    "service.id": self.service_id,
                    "request.id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        return response


async def run_demo():
    """Run the tracing demo."""
    # Set up tracing
    tracer = setup_tracing()
    
    # Create demo service
    service = TracedDemoService("demo_service")
    
    # Create root span
    with tracer.start_as_current_span(
        "tracing_demo",
        kind=SpanKind.INTERNAL,
        attributes={
            "demo.type": "circuit_breaker_tracing",
            "demo.version": "1.0.0"
        }
    ) as root_span:
        logger.info("Starting circuit breaker tracing demo")
        
        # Test normal service calls
        with tracer.start_as_current_span(
            "normal_service_calls",
            attributes={"test.type": "normal"}
        ) as span:
            logger.info("Testing normal service calls")
            span.set_attribute("test.description", "Testing normal service calls with circuit breaker and tracing")
            
            for i in range(10):
                try:
                    response = await service.call_service(f"req_{i}", fail_probability=0.3)
                    logger.info(f"Service call succeeded", response=response)
                except Exception as e:
                    logger.error(f"Service call failed: {str(e)}")
        
        # Test timeout service calls
        with tracer.start_as_current_span(
            "timeout_service_calls",
            attributes={"test.type": "timeout"}
        ) as span:
            logger.info("Testing timeout service calls")
            span.set_attribute("test.description", "Testing timeout service calls with circuit breaker and tracing")
            
            for i in range(5):
                try:
                    response = await service.call_slow_service(f"slow_req_{i}", delay=1.0)
                    logger.info(f"Slow service call succeeded", response=response)
                except asyncio.TimeoutError:
                    logger.error(f"Slow service call timed out")
                except Exception as e:
                    logger.error(f"Slow service call failed: {str(e)}")
        
        # Wait for circuit to recover
        logger.info("Waiting for circuit to recover")
        await asyncio.sleep(10)
        
        # Test rate-limited service calls
        with tracer.start_as_current_span(
            "rate_limited_service_calls",
            attributes={"test.type": "rate_limit"}
        ) as span:
            logger.info("Testing rate-limited service calls")
            span.set_attribute("test.description", "Testing rate-limited service calls with circuit breaker and tracing")
            
            for i in range(5):
                try:
                    response = await service.call_rate_limited_service(
                        f"rate_req_{i}",
                        rate_limit_probability=0.7
                    )
                    logger.info(f"Rate-limited service call succeeded", response=response)
                except Exception as e:
                    logger.error(f"Rate-limited service call failed: {str(e)}")
        
        logger.info("Demo completed")


if __name__ == "__main__":
    asyncio.run(run_demo())
