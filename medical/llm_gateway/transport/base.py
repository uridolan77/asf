"""
Base transport for MCP.

This module provides the base transport interface for MCP,
with support for rate limiting, circuit breaking, and metrics.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from asf.medical.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig
from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Base transport configuration."""
    transport_type: str
    timeout_seconds: float = 60.0
    enable_rate_limiting: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None
    enable_circuit_breaker: bool = True
    circuit_breaker_config: Optional[Dict[str, Any]] = None


@dataclass
class TransportResponse:
    """Transport response."""
    data: Any
    metadata: Dict[str, Any]
    latency_ms: float


class Transport(ABC):
    """
    Base transport interface.
    
    This class provides the base transport interface for MCP,
    with support for rate limiting, circuit breaking, and metrics.
    """
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the transport.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the transport.
        """
        pass
    
    @abstractmethod
    async def send_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TransportResponse:
        """
        Send a request.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Transport response
        """
        pass
    
    @abstractmethod
    async def send_streaming_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> AsyncIterator[TransportResponse]:
        """
        Send a streaming request.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Iterator of transport responses
        """
        pass
    
    @abstractmethod
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        pass


class BaseTransport(Transport):
    """
    Base transport implementation.
    
    This class provides a base implementation of the transport interface,
    with support for rate limiting, circuit breaking, and metrics.
    """
    
    def __init__(
        self,
        provider_id: str,
        config: TransportConfig,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the base transport.
        
        Args:
            provider_id: Provider ID
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        self.config = config
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Create rate limiter
        self.rate_limiter = None
        if config.enable_rate_limiting:
            rate_limit_config = config.rate_limit_config or RateLimitConfig()
            self.rate_limiter = RateLimiter(provider_id, rate_limit_config)
        
        # Create circuit breaker
        self.circuit_breaker = None
        if config.enable_circuit_breaker:
            circuit_breaker_config = config.circuit_breaker_config or {}
            self.circuit_breaker = CircuitBreaker(**circuit_breaker_config)
    
    async def start(self) -> None:
        """
        Start the transport.
        """
        pass
    
    async def stop(self) -> None:
        """
        Stop the transport.
        """
        pass
    
    async def _check_rate_limit(self) -> None:
        """
        Check rate limit.
        
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        if self.rate_limiter:
            await self.rate_limiter.wait()
    
    async def _check_circuit_breaker(self) -> None:
        """
        Check circuit breaker.
        
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for provider {self.provider_id}"
            )
    
    async def _record_success(self) -> None:
        """
        Record a successful request.
        """
        if self.rate_limiter:
            await self.rate_limiter.record_success()
        
        if self.circuit_breaker:
            self.circuit_breaker.record_success()
    
    async def _record_failure(self) -> None:
        """
        Record a failed request.
        """
        if self.rate_limiter:
            await self.rate_limiter.record_failure()
        
        if self.circuit_breaker:
            self.circuit_breaker.record_failure()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        stats = {
            "provider_id": self.provider_id,
            "transport_type": self.config.transport_type
        }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        if self.circuit_breaker:
            stats["circuit_breaker"] = {
                "is_open": self.circuit_breaker.is_open(),
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "reset_timeout_seconds": self.circuit_breaker.reset_timeout_seconds
            }
        
        return stats


class TransportError(Exception):
    """
    Transport error.
    
    This exception is raised when a transport error occurs.
    """
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the transport error.
        
        Args:
            message: Error message
            code: Error code
            details: Error details
        """
        self.message = message
        self.code = code or "TRANSPORT_ERROR"
        self.details = details or {}
        super().__init__(message)


class RateLimitExceededError(TransportError):
    """
    Rate limit exceeded error.
    
    This exception is raised when a rate limit is exceeded.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the rate limit exceeded error.
        
        Args:
            message: Error message
            details: Error details
        """
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class CircuitBreakerOpenError(TransportError):
    """
    Circuit breaker open error.
    
    This exception is raised when a circuit breaker is open.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the circuit breaker open error.
        
        Args:
            message: Error message
            details: Error details
        """
        super().__init__(message, "CIRCUIT_BREAKER_OPEN", details)
