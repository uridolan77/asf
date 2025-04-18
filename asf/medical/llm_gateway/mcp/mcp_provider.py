"""
Enhanced MCP Provider with Production-Grade Features

This module implements a robust MCP provider that leverages the enhanced
transport layer components to provide reliable and efficient LLM integration.
"""

import asyncio
import logging
import time
import uuid
import copy
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union, Callable, Type, Tuple

# Metrics and monitoring imports
import structlog
from opentelemetry import trace, metrics
from opentelemetry.trace import SpanKind

# Internal imports
from asf.medical.llm_gateway.core.models import (
    ContentItem, ErrorDetails, ErrorLevel, FinishReason, GatewayConfig,
    InterventionContext, LLMConfig, LLMRequest, LLMResponse,
    MCPContentType as GatewayMCPContentType, MCPMetadata, MCPRole as GatewayMCPRole,
    PerformanceMetrics, ProviderConfig, StreamChunk, ToolDefinition, ToolFunction,
    ToolUseRequest, UsageStats, MCPUsage
)
from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.mcp.transport.base import BaseTransport, TransportError
from asf.medical.llm_gateway.mcp.transport.factory import TransportFactory
from asf.medical.llm_gateway.mcp.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.mcp.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.mcp.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
from asf.medical.llm_gateway.mcp.observability.tracing import TracingService
from asf.medical.llm_gateway.mcp.transport.base import SessionState, SessionPriority

# Create structured logger
logger = structlog.get_logger("mcp_provider")

# OpenTelemetry tracer and metrics
tracer = trace.get_tracer("mcp_provider")
meter = metrics.get_meter("mcp_provider")

# Define metrics
request_counter = meter.create_counter(
    "mcp_requests",
    description="Count of requests to MCP provider"
)
error_counter = meter.create_counter(
    "mcp_errors",
    description="Count of errors from MCP provider"
)
request_latency = meter.create_histogram(
    "mcp_request_duration",
    description="Duration of MCP requests",
    unit="ms"
)


class MCPProvider(BaseProvider):
    """
    Enhanced MCP Provider with production-grade features.
    
    This implementation leverages the enhanced transport layer components
    to provide reliable and efficient integration with MCP-compatible LLMs.
    
    Features:
    - Multi-transport support (stdio, gRPC, HTTP, WebSocket)
    - Advanced session management with health monitoring
    - Circuit breaker for rapid failure detection
    - Exponential backoff with jitter for retries
    - Comprehensive observability with metrics and tracing
    - Content and error standardization across transports
    - Full streaming support with backpressure
    """
    
    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """
        Initialize the enhanced MCP provider.
        
        Args:
            provider_config: Provider configuration from gateway
            gateway_config: Global gateway configuration
        """
        super().__init__(provider_config, gateway_config)
        
        # Initialize telemetry
        self.tracing_service = TracingService()
        self.metrics_service = MetricsService()
        self.prometheus = get_prometheus_exporter()
        
        # Extract connection parameters
        self.connection_config = provider_config.connection_params
        
        # Register provider with Prometheus
        self.prometheus.update_provider_info(
            self.provider_id,
            {
                "provider_type": self.provider_config.provider_type,
                "transport_type": self.connection_config.get("transport_type", "stdio"),
                "display_name": getattr(self.provider_config, 'display_name', self.provider_id),
            }
        )
        
        # Initialize operational parameters
        self._enable_streaming = self.connection_config.get("enable_streaming", True)
        self._max_retry_attempts = self.connection_config.get("max_retries", 3)
        self._retry_delay_seconds = self.connection_config.get("retry_delay_seconds", 1.0)
        self._timeout_seconds = self.connection_config.get("timeout_seconds", 60.0)
        
        # Initialize transport
        self.transport_factory = TransportFactory(
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus
        )
        
        # Create transport instance
        transport_type = self.connection_config.get("transport_type", "stdio")
        self.transport = self.transport_factory.create_transport(
            transport_type=transport_type,
            config=self.connection_config,
            provider_id=self.provider_id,
            singleton=True  # Reuse transport instances when possible
        )
        
        # Create retry policy
        self.retry_policy = RetryPolicy(
            max_retries=self._max_retry_attempts,
            base_delay=self._retry_delay_seconds,
            max_delay=30.0,  # Cap at 30 seconds
            jitter_factor=0.2  # 20% jitter
        )
        
        # Create circuit breaker for the provider
        self.circuit_breaker = CircuitBreaker(
            name=f"mcp_provider_{self.provider_id}",
            failure_threshold=self.connection_config.get("circuit_breaker_threshold", 5),
            recovery_timeout=self.connection_config.get("circuit_breaker_recovery_timeout", 30)
        )
        
        # Session tags for workload differentiation
        self.session_tags = set(self.connection_config.get("session_tags", []))
        self.session_capabilities = set(self.connection_config.get("session_capabilities", []))
        
        logger.info(
            "Initialized enhanced MCP provider",
            provider_id=self.provider_id,
            transport_type=transport_type,
            streaming_enabled=self._enable_streaming
        )
    
    async def initialize_async(self) -> None:
        """Perform async initialization tasks."""
        try:
            # Start the transport if it has a start method
            if hasattr(self.transport, "start") and callable(self.transport.start):
                await self.transport.start()
            
            logger.info(f"Initialized transport for MCP provider {self.provider_id}")
            
            # Initialize the session pool if using
            # (this would be integrated with the transport implementation)
        except Exception as e:
            logger.error(
                f"Error initializing MCP provider {self.provider_id}",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop the transport if it has a stop method
            if hasattr(self.transport, "stop") and callable(self.transport.stop):
                await self.transport.stop()
            
            logger.info(f"Cleaned up MCP provider {self.provider_id}")
        except Exception as e:
            logger.error(
                f"Error cleaning up MCP provider {self.provider_id}",
                error=str(e),
                exc_info=True
            )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response using the MCP provider.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        # Record request metrics
        request_counter.add(1, {"provider": self.provider_id, "model": request.config.model_identifier})
        
        start_time = time.time()
        context = request.initial_context
        
        # Create span for tracing
        with tracer.start_as_current_span(
            "mcp_generate",
            kind=SpanKind.CLIENT,
            attributes={
                "provider.id": self.provider_id,
                "request.id": context.request_id,
                "model.id": request.config.model_identifier
            }
        ) as span:
            # Check if circuit breaker is open
            if self.circuit_breaker.is_open():
                logger.warning(f"Circuit breaker open for provider {self.provider_id}")
                
                # Create error response
                error_details = ErrorDetails(
                    code="CIRCUIT_BREAKER_OPEN",
                    message=f"Circuit breaker open for provider {self.provider_id}",
                    level=ErrorLevel.ERROR,
                    retryable=False
                )
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Return error response
                return self._create_error_response(
                    request=request,
                    error_details=error_details,
                    total_duration_ms=duration_ms
                )
            
            try:
                # Prepare MCP request
                mcp_request = self._prepare_mcp_request(request)
                
                # Apply retry policy
                for attempt in range(self.retry_policy.max_retries + 1):
                    is_last_attempt = attempt == self.retry_policy.max_retries
                    
                    try:
                        # Send request to MCP provider via transport
                        llm_start_time = time.time()
                        
                        # Use appropriate session tags based on request
                        request_tags = self._get_request_tags(request)
                        
                        # Get a session from the transport
                        async with self.transport.connect() as session:
                            # Send request through transport
                            mcp_response = await self.transport.send_message(mcp_request)
                            
                            # Calculate latency
                            llm_latency_ms = (time.time() - llm_start_time) * 1000
                            
                            # Reset circuit breaker on success
                            self.circuit_breaker.record_success()
                            
                            # Calculate total duration
                            total_duration_ms = (time.time() - start_time) * 1000
                            
                            # Create LLM response
                            response = self._create_response_from_mcp(
                                mcp_response=mcp_response,
                                request=request,
                                llm_latency_ms=llm_latency_ms,
                                total_duration_ms=total_duration_ms
                            )
                            
                            # Record latency metric
                            request_latency.record(
                                total_duration_ms,
                                {
                                    "provider": self.provider_id,
                                    "model": request.config.model_identifier,
                                    "success": "true"
                                }
                            )
                            
                            return response
                    
                    except TransportError as e:
                        # Check if retryable and not last attempt
                        if e.code in self.retry_policy.retry_codes and not is_last_attempt:
                            # Calculate retry delay
                            delay = self.retry_policy.calculate_delay(attempt + 1)
                            
                            logger.warning(
                                f"Retryable error in MCP request, will retry",
                                provider_id=self.provider_id,
                                attempt=attempt + 1,
                                max_attempts=self.retry_policy.max_retries + 1,
                                delay=delay,
                                error_code=e.code,
                                error_message=e.message
                            )
                            
                            # Add retry info to span
                            span.add_event(
                                "retry",
                                {
                                    "attempt": attempt + 1,
                                    "delay": delay,
                                    "error_code": e.code,
                                    "error_message": e.message
                                }
                            )
                            
                            # Wait before retry
                            await asyncio.sleep(delay)
                            continue
                        
                        # Not retryable or last attempt
                        # Record failure in circuit breaker
                        self.circuit_breaker.record_failure()
                        
                        # Create error details
                        error_details = ErrorDetails(
                            code=e.code,
                            message=e.message,
                            level=ErrorLevel.ERROR,
                            retryable=e.code in self.retry_policy.retry_codes,
                            provider_error_details=e.details
                        )
                        
                        # Record error counter
                        error_counter.add(
                            1,
                            {
                                "provider": self.provider_id,
                                "model": request.config.model_identifier,
                                "error_code": e.code
                            }
                        )
                        
                        # Calculate duration
                        total_duration_ms = (time.time() - start_time) * 1000
                        
                        # Record latency metric
                        request_latency.record(
                            total_duration_ms,
                            {
                                "provider": self.provider_id,
                                "model": request.config.model_identifier,
                                "success": "false",
                                "error_code": e.code
                            }
                        )
                        
                        # Add error info to span
                        span.set_attribute("error", True)
                        span.set_attribute("error.code", e.code)
                        span.set_attribute("error.message", e.message)
                        
                        # Return error response
                        return self._create_error_response(
                            request=request,
                            error_details=error_details,
                            total_duration_ms=total_duration_ms
                        )
            
            except Exception as e:
                # Record failure in circuit breaker
                self.circuit_breaker.record_failure()
                
                # Create error details
                error_details = ErrorDetails(
                    code="UNEXPECTED_ERROR",
                    message=f"Unexpected error: {str(e)}",
                    level=ErrorLevel.ERROR,
                    retryable=False,
                    provider_error_details={"error_type": type(e).__name__}
                )
                
                # Record error counter
                error_counter.add(
                    1,
                    {
                        "provider": self.provider_id,
                        "model": request.config.model_identifier,
                        "error_code": "UNEXPECTED_ERROR"
                    }
                )
                
                # Calculate duration
                total_duration_ms = (time.time() - start_time) * 1000
                
                # Record latency metric
                request_latency.record(
                    total_duration_ms,
                    {
                        "provider": self.provider_id,
                        "model": request.config.model_identifier,
                        "success": "false",
                        "error_code": "UNEXPECTED_ERROR"
                    }
                )
                
                # Add error info to span
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                # Log the error
                logger.error(
                    f"Unexpected error in MCP request",
                    provider_id=self.provider_id,
                    error=str(e),
                    exc_info=True
                )
                
                # Return error response
                return self._create_error_response(
                    request=request,
                    error_details=error_details,
                    total_duration_ms=total_duration_ms
                )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response using the MCP provider.
        
        Args:
            request: The LLM request
            
        Yields:
            Stream chunks of the response
        """
        # Record request metrics
        request_counter.add(
            1, 
            {
                "provider": self.provider_id, 
                "model": request.config.model_identifier,
                "streaming": "true"
            }
        )
        
        # Check if streaming is enabled
        if not self._enable_streaming:
            logger.warning(
                "Streaming requested but disabled for provider, using fallback",
                provider_id=self.provider_id
            )
            
            # Fallback to non-streaming
            response = await self.generate(request)
            yield self._create_response_chunk(response, request.initial_context.request_id, 0)
            return
        
        start_time = time.time()
        context = request.initial_context
        
        # Create span for tracing
        with tracer.start_as_current_span(
            "mcp_generate_stream",
            kind=SpanKind.CLIENT,
            attributes={
                "provider.id": self.provider_id,
                "request.id": context.request_id,
                "model.id": request.config.model_identifier,
                "streaming": True
            }
        ) as span:
            # Check if circuit breaker is open
            if self.circuit_breaker.is_open():
                logger.warning(f"Circuit breaker open for provider {self.provider_id}")
                
                # Create error chunk
                yield StreamChunk(
                    chunk_id=0,
                    request_id=context.request_id,
                    finish_reason=FinishReason.ERROR,
                    provider_specific_data={
                        "error": {
                            "code": "CIRCUIT_BREAKER_OPEN",
                            "message": f"Circuit breaker open for provider {self.provider_id}"
                        }
                    }
                )
                return
            
            try:
                # Prepare MCP request with streaming enabled
                mcp_request = self._prepare_mcp_request(request, streaming=True)
                
                # Apply retry policy
                for attempt in range(self.retry_policy.max_retries + 1):
                    is_last_attempt = attempt == self.retry_policy.max_retries
                    
                    try:
                        # Get request tags
                        request_tags = self._get_request_tags(request)
                        
                        # Get a session from the transport
                        async with self.transport.connect() as session:
                            # Send streaming request through transport
                            chunk_index = 0
                            async for chunk in self.transport.send_message_stream(mcp_request):
                                # Convert MCP chunk to StreamChunk
                                stream_chunk = self._convert_mcp_chunk_to_stream_chunk(
                                    chunk=chunk,
                                    request_id=context.request_id,
                                    chunk_id=chunk_index
                                )
                                
                                # Increment chunk index
                                chunk_index += 1
                                
                                # Yield the chunk
                                yield stream_chunk
                                
                                # Check if this is the last chunk
                                if stream_chunk.finish_reason is not None:
                                    break
                            
                            # Reset circuit breaker on success
                            self.circuit_breaker.record_success()
                            
                            # Calculate total duration
                            total_duration_ms = (time.time() - start_time) * 1000
                            
                            # Record latency metric
                            request_latency.record(
                                total_duration_ms,
                                {
                                    "provider": self.provider_id,
                                    "model": request.config.model_identifier,
                                    "streaming": "true",
                                    "success": "true"
                                }
                            )
                            
                            # Break out of retry loop on success
                            break
                    
                    except TransportError as e:
                        # Check if retryable and not last attempt
                        if e.code in self.retry_policy.retry_codes and not is_last_attempt:
                            # Calculate retry delay
                            delay = self.retry_policy.calculate_delay(attempt + 1)
                            
                            logger.warning(
                                f"Retryable error in MCP stream, will retry",
                                provider_id=self.provider_id,
                                attempt=attempt + 1,
                                max_attempts=self.retry_policy.max_retries + 1,
                                delay=delay,
                                error_code=e.code,
                                error_message=e.message
                            )
                            
                            # Add retry info to span
                            span.add_event(
                                "retry",
                                {
                                    "attempt": attempt + 1,
                                    "delay": delay,
                                    "error_code": e.code,
                                    "error_message": e.message
                                }
                            )
                            
                            # Wait before retry
                            await asyncio.sleep(delay)
                            continue
                        
                        # Not retryable or last attempt
                        # Record failure in circuit breaker
                        self.circuit_breaker.record_failure()
                        
                        # Record error counter
                        error_counter.add(
                            1,
                            {
                                "provider": self.provider_id,
                                "model": request.config.model_identifier,
                                "error_code": e.code,
                                "streaming": "true"
                            }
                        )
                        
                        # Calculate duration
                        total_duration_ms = (time.time() - start_time) * 1000
                        
                        # Record latency metric
                        request_latency.record(
                            total_duration_ms,
                            {
                                "provider": self.provider_id,
                                "model": request.config.model_identifier,
                                "streaming": "true",
                                "success": "false",
                                "error_code": e.code
                            }
                        )
                        
                        # Add error info to span
                        span.set_attribute("error", True)
                        span.set_attribute("error.code", e.code)
                        span.set_attribute("error.message", e.message)
                        
                        # Yield error chunk
                        yield StreamChunk(
                            chunk_id=999,
                            request_id=context.request_id,
                            finish_reason=FinishReason.ERROR,
                            provider_specific_data={
                                "error": {
                                    "code": e.code,
                                    "message": e.message,
                                    "details": e.details
                                }
                            }
                        )
                        
                        # Return from function
                        return
            
            except Exception as e:
                # Record failure in circuit breaker
                self.circuit_breaker.record_failure()
                
                # Record error counter
                error_counter.add(
                    1,
                    {
                        "provider": self.provider_id,
                        "model": request.config.model_identifier,
                        "error_code": "UNEXPECTED_ERROR",
                        "streaming": "true"
                    }
                )
                
                # Calculate duration
                total_duration_ms = (time.time() - start_time) * 1000
                
                # Record latency metric
                request_latency.record(
                    total_duration_ms,
                    {
                        "provider": self.provider_id,
                        "model": request.config.model_identifier,
                        "streaming": "true",
                        "success": "false",
                        "error_code": "UNEXPECTED_ERROR"
                    }
                )
                
                # Add error info to span
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                # Log the error
                logger.error(
                    f"Unexpected error in MCP stream",
                    provider_id=self.provider_id,
                    error=str(e),
                    exc_info=True
                )
                
                # Yield error chunk
                yield StreamChunk(
                    chunk_id=999,
                    request_id=context.request_id,
                    finish_reason=FinishReason.ERROR,
                    provider_specific_data={
                        "error": {
                            "code": "UNEXPECTED_ERROR",
                            "message": f"Unexpected error: {str(e)}",
                            "details": {"error_type": type(e).__name__}
                        }
                    }
                )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the MCP provider.
        
        Returns:
            Health check results
        """
        start_time = time.time()
        
        # Check if circuit breaker is open
        if self.circuit_breaker.is_open():
            status_data = {
                "provider_id": self.provider_id,
                "status": "unavailable",
                "provider_type": self.provider_config.provider_type,
                "checked_at": datetime.utcnow().isoformat(),
                "message": f"Circuit breaker open until {self.circuit_breaker.recovery_time.isoformat() if self.circuit_breaker.recovery_time else 'unknown'}",
                "circuit_breaker": {
                    "state": "open",
                    "failure_count": self.circuit_breaker.failure_count,
                    "recovery_time": self.circuit_breaker.recovery_time.isoformat() if self.circuit_breaker.recovery_time else None,
                    "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
                }
            }
            
            return status_data
        
        try:
            # Check transport health
            if hasattr(self.transport, "health_check") and callable(self.transport.health_check):
                transport_health = await self.transport.health_check()
                
                # Get transport status
                transport_status = transport_health.get("status", "unknown")
                
                if transport_status == "available":
                    status = "available"
                    message = f"Transport is healthy: {transport_health.get('message', '')}"
                    self._is_healthy = True
                else:
                    status = "degraded"
                    message = f"Transport is in {transport_status} state: {transport_health.get('message', '')}"
                    self._is_healthy = False
            else:
                # Try a simple ping to check health
                try:
                    # Create a simple ping request
                    ping_request = {
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Attempt to get a session
                    async with self.transport.connect() as session:
                        # Consider transport as healthy if session can be acquired
                        status = "available"
                        message = "Successfully connected to transport"
                        self._is_healthy = True
                except Exception as e:
                    status = "unavailable"
                    message = f"Failed to connect to transport: {str(e)}"
                    self._is_healthy = False
        
        except Exception as e:
            status = "error"
            message = f"Error checking health: {str(e)}"
            self._is_healthy = False
        
        # Get pool stats if available
        pool_stats = None
        if hasattr(self.transport, "get_pool_stats") and callable(self.transport.get_pool_stats):
            pool_stats = self.transport.get_pool_stats()
        
        # Get supported models
        models = []
        for model_id, model_config in self.provider_config.models.items():
            models.append({
                "id": model_id,
                "display_name": model_config.get("display_name", model_id),
                "context_window": model_config.get("context_window", 0),
                "max_tokens": model_config.get("max_tokens", 0)
            })
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Construct status response
        status_data = {
            "provider_id": self.provider_id,
            "display_name": getattr(self.provider_config, 'display_name', self.provider_id),
            "status": status,
            "provider_type": self.provider_config.provider_type,
            "transport_type": self.connection_config.get("transport_type", "stdio"),
            "checked_at": datetime.utcnow().isoformat(),
            "message": message,
            "duration_ms": duration_ms,
            "circuit_breaker": {
                "state": "open" if self.circuit_breaker.is_open() else "closed",
                "failure_count": self.circuit_breaker.failure_count,
                "recovery_time": self.circuit_breaker.recovery_time.isoformat() if self.circuit_breaker.recovery_time else None,
                "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
            },
            "models": models,
            "pool_stats": pool_stats
        }
        
        return status_data
    
    def _prepare_mcp_request(self, request: LLMRequest, streaming: bool = False) -> Dict[str, Any]:
        """
        Prepare request for MCP provider.
        
        Args:
            request: The LLM request
            streaming: Whether this is a streaming request
            
        Returns:
            MCP request dictionary
        """
        # Create base request
        mcp_request = {
            "model": request.config.model_identifier,
            "request_id": request.initial_context.request_id,
            "stream": streaming,
            "messages": []
        }
        
        # Add system prompt if provided
        if request.config.system_prompt:
            mcp_request["messages"].append({
                "role": "system",
                "content": request.config.system_prompt
            })
        
        # Add conversation history
        for turn in request.initial_context.conversation_history:
            # Skip system messages if we already added a system prompt
            if turn.role == "system" and request.config.system_prompt:
                continue
            
            # Convert turn to MCP message
            if isinstance(turn.content, str):
                # Simple text message
                mcp_request["messages"].append({
                    "role": turn.role,
                    "content": turn.content
                })
            elif isinstance(turn.content, list):
                # Content items list
                message_content = []
                
                for item in turn.content:
                    if isinstance(item, ContentItem):
                        # Convert ContentItem to MCP format
                        content_item = self._convert_content_item_to_mcp(item)
                        if content_item:
                            message_content.append(content_item)
                    elif isinstance(item, dict) and "type" in item:
                        # Already in MCP-like format
                        message_content.append(item)
                
                mcp_request["messages"].append({
                    "role": turn.role,
                    "content": message_content
                })
            elif isinstance(turn.content, dict):
                # Single content item as dict
                if "type" in turn.content:
                    mcp_request["messages"].append({
                        "role": turn.role,
                        "content": [turn.content]
                    })
                else:
                    # Unknown format, convert to string
                    mcp_request["messages"].append({
                        "role": turn.role,
                        "content": str(turn.content)
                    })
        
        # Add prompt content
        if isinstance(request.prompt_content, str):
            # Simple text prompt
            mcp_request["messages"].append({
                "role": "user",
                "content": request.prompt_content
            })
        elif isinstance(request.prompt_content, list):
            # Content items list
            message_content = []
            
            for item in request.prompt_content:
                if isinstance(item, ContentItem):
                    # Convert ContentItem to MCP format
                    content_item = self._convert_content_item_to_mcp(item)
                    if content_item:
                        message_content.append(content_item)
                elif isinstance(item, dict) and "type" in item:
                    # Already in MCP-like format
                    message_content.append(item)
            
            mcp_request["messages"].append({
                "role": "user",
                "content": message_content
            })
        
        # Add LLM parameters
        if request.config.temperature is not None:
            mcp_request["temperature"] = request.config.temperature
        
        if request.config.max_tokens is not None:
            mcp_request["max_tokens"] = request.config.max_tokens
        
        if request.config.top_p is not None:
            mcp_request["top_p"] = request.config.top_p
        
        if request.config.stop_sequences is not None:
            mcp_request["stop_sequences"] = request.config.stop_sequences
        
        # Add tools if provided
        if request.tools:
            tools = []
            
            for tool_def in request.tools:
                if hasattr(tool_def, "function") and tool_def.function:
                    tool = {
                        "type": "function",
                        "function": {
                            "name": tool_def.function.name,
                            "description": tool_def.function.description,
                            "parameters": tool_def.function.parameters
                        }
                    }
                    tools.append(tool)
            
            if tools:
                mcp_request["tools"] = tools
        
        return mcp_request
    
    def _convert_content_item_to_mcp(self, item: ContentItem) -> Optional[Dict[str, Any]]:
        """
        Convert ContentItem to MCP format.
        
        Args:
            item: ContentItem to convert
            
        Returns:
            MCP content item or None if conversion fails
        """
        try:
            if item.type == GatewayMCPContentType.TEXT:
                return {
                    "type": "text",
                    "text": item.text_content or item.data.get("text", "")
                }
            
            elif item.type == GatewayMCPContentType.IMAGE:
                image_data = item.data.get("image", {}).get("source", {})
                
                if "url" in image_data:
                    return {
                        "type": "image",
                        "image": {
                            "source": {
                                "type": "url",
                                "url": image_data["url"]
                            }
                        }
                    }
                elif "data" in image_data:
                    return {
                        "type": "image",
                        "image": {
                            "source": {
                                "type": "base64",
                                "data": image_data["data"]
                            }
                        }
                    }
            
            elif item.type == GatewayMCPContentType.TOOL_USE:
                tool_use = item.data.get("tool_use", {})
                return {
                    "type": "tool_use",
                    "id": tool_use.get("id", str(uuid.uuid4())),
                    "name": tool_use.get("name", ""),
                    "input": tool_use.get("input", {})
                }
            
            elif item.type == GatewayMCPContentType.TOOL_RESULT:
                tool_result = item.data.get("tool_result", {})
                return {
                    "type": "tool_result",
                    "id": tool_result.get("id", ""),
                    "output": tool_result.get("output", "")
                }
            
            # Unsupported type
            return None
        
        except Exception as e:
            logger.warning(f"Error converting content item to MCP: {str(e)}")
            return None
    
    def _convert_mcp_chunk_to_stream_chunk(
        self,
        chunk: Dict[str, Any],
        request_id: str,
        chunk_id: int
    ) -> StreamChunk:
        """
        Convert MCP chunk to StreamChunk.
        
        Args:
            chunk: MCP chunk
            request_id: Request ID
            chunk_id: Chunk ID
            
        Returns:
            Stream chunk
        """
        # Extract delta text
        delta_text = None
        if "delta" in chunk and "content" in chunk["delta"]:
            delta_text = chunk["delta"]["content"]
        elif "content" in chunk:
            if isinstance(chunk["content"], str):
                delta_text = chunk["content"]
            elif isinstance(chunk["content"], list):
                # Concatenate text content items
                content_items = chunk["content"]
                text_items = []
                
                for item in content_items:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_items.append(item.get("text", ""))
                
                if text_items:
                    delta_text = "".join(text_items)
        
        # Extract tool calls
        delta_tool_calls = None
        if "delta" in chunk and "tool_calls" in chunk["delta"]:
            # Convert tool calls to ToolUseRequest
            tool_calls = []
            
            for tool_call in chunk["delta"]["tool_calls"]:
                if "function" in tool_call:
                    tool_use = ToolUseRequest(
                        id=tool_call.get("id", str(uuid.uuid4())),
                        type="function",
                        function=ToolFunction(
                            name=tool_call["function"].get("name", ""),
                            description=tool_call["function"].get("description", ""),
                            parameters=tool_call["function"].get("parameters", {})
                        )
                    )
                    tool_calls.append(tool_use)
            
            if tool_calls:
                delta_tool_calls = tool_calls
        
        # Extract finish reason
        finish_reason = None
        if "stop_reason" in chunk and chunk["stop_reason"]:
            finish_reason = self._map_mcp_stop_reason(chunk["stop_reason"])
        
        # Extract usage
        usage_update = None
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            usage_update = UsageStats(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0)
            )
        
        # Create stream chunk
        return StreamChunk(
            chunk_id=chunk_id,
            request_id=request_id,
            delta_text=delta_text,
            delta_tool_calls=delta_tool_calls,
            finish_reason=finish_reason,
            usage_update=usage_update,
            provider_specific_data={"raw_chunk_id": chunk.get("id")}
        )
    
    def _create_response_from_mcp(
        self,
        mcp_response: Dict[str, Any],
        request: LLMRequest,
        llm_latency_ms: float,
        total_duration_ms: float
    ) -> LLMResponse:
        """
        Create LLMResponse from MCP response.
        
        Args:
            mcp_response: MCP response
            request: Original request
            llm_latency_ms: LLM latency in milliseconds
            total_duration_ms: Total duration in milliseconds
            
        Returns:
            LLM response
        """
        # Extract generated content
        content_items = []
        
        if "content" in mcp_response:
            content = mcp_response["content"]
            
            if isinstance(content, str):
                # Simple text response
                content_items.append(ContentItem.from_text(content))
            elif isinstance(content, list):
                # Content items list
                for item in content:
                    if isinstance(item, dict) and "type" in item:
                        content_item = self._convert_mcp_to_content_item(item)
                        if content_item:
                            content_items.append(content_item)
        
        # Extract usage
        usage = None
        if "usage" in mcp_response and mcp_response["usage"]:
            usage_data = mcp_response["usage"]
            usage = UsageStats(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0)
            )
        
        # Extract finish reason
        finish_reason = None
        if "stop_reason" in mcp_response and mcp_response["stop_reason"]:
            finish_reason = self._map_mcp_stop_reason(mcp_response["stop_reason"])
        
        # Extract tool use requests
        tool_use_requests = None
        if "tool_calls" in mcp_response and mcp_response["tool_calls"]:
            # Convert tool calls to ToolUseRequest
            tool_calls = []
            
            for tool_call in mcp_response["tool_calls"]:
                if "function" in tool_call:
                    tool_use = ToolUseRequest(
                        id=tool_call.get("id", str(uuid.uuid4())),
                        type="function",
                        function=ToolFunction(
                            name=tool_call["function"].get("name", ""),
                            description=tool_call["function"].get("description", ""),
                            parameters=tool_call["function"].get("parameters", {})
                        )
                    )
                    tool_calls.append(tool_use)
            
            if tool_calls:
                tool_use_requests = tool_calls
        
        # Create MCP metadata
        mcp_metadata = MCPMetadata(
            mcp_version="1.0",
            model_version_reported=mcp_response.get("model_version"),
            context_id=mcp_response.get("context_id"),
            provider_usage=MCPUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0
            ) if usage else None
        )
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms
        )
        
        # Create final context from initial context
        final_context = copy.deepcopy(request.initial_context)
        
        # Create response
        return LLMResponse(
            request_id=request.initial_context.request_id,
            generated_content=content_items,
            finish_reason=finish_reason,
            tool_use_requests=tool_use_requests,
            usage=usage,
            final_context=final_context,
            performance_metrics=performance_metrics,
            mcp_metadata=mcp_metadata
        )
    
    def _convert_mcp_to_content_item(self, mcp_item: Dict[str, Any]) -> Optional[ContentItem]:
        """
        Convert MCP content item to ContentItem.
        
        Args:
            mcp_item: MCP content item
            
        Returns:
            ContentItem or None if conversion fails
        """
        try:
            item_type = mcp_item.get("type")
            
            if item_type == "text":
                return ContentItem.from_text(mcp_item.get("text", ""))
            
            elif item_type == "image":
                image_data = mcp_item.get("image", {}).get("source", {})
                
                if "url" in image_data:
                    return ContentItem.from_image_url(image_data["url"])
                elif "data" in image_data:
                    mime_type = mcp_item.get("image", {}).get("mime_type", "image/jpeg")
                    return ContentItem.from_image_base64(image_data["data"], mime_type)
            
            elif item_type == "tool_use":
                return ContentItem(
                    type=GatewayMCPContentType.TOOL_USE,
                    data={
                        "tool_use": {
                            "id": mcp_item.get("id", str(uuid.uuid4())),
                            "name": mcp_item.get("name", ""),
                            "input": mcp_item.get("input", {})
                        }
                    }
                )
            
            elif item_type == "tool_result":
                return ContentItem(
                    type=GatewayMCPContentType.TOOL_RESULT,
                    data={
                        "tool_result": {
                            "id": mcp_item.get("id", ""),
                            "output": mcp_item.get("output", "")
                        }
                    }
                )
            
            # Unsupported type
            return None
        
        except Exception as e:
            logger.warning(f"Error converting MCP item to ContentItem: {str(e)}")
            return None
    
    def _map_mcp_stop_reason(self, stop_reason: str) -> FinishReason:
        """
        Map MCP stop reason to Gateway finish reason.
        
        Args:
            stop_reason: MCP stop reason
            
        Returns:
            Gateway finish reason
        """
        mapping = {
            "stop": FinishReason.STOP,
            "endTurn": FinishReason.STOP,
            "maxTokens": FinishReason.LENGTH,
            "stopSequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "content_filtered": FinishReason.CONTENT_FILTERED,
            "error": FinishReason.ERROR
        }
        
        return mapping.get(stop_reason, FinishReason.UNKNOWN)
    
    def _create_error_response(
        self,
        request: LLMRequest,
        error_details: ErrorDetails,
        total_duration_ms: float
    ) -> LLMResponse:
        """
        Create error response.
        
        Args:
            request: Original request
            error_details: Error details
            total_duration_ms: Total duration in milliseconds
            
        Returns:
            Error response
        """
        # Create final context from initial context
        final_context = copy.deepcopy(request.initial_context)
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms
        )
        
        # Create response
        return LLMResponse(
            request_id=request.initial_context.request_id,
            generated_content=None,
            finish_reason=FinishReason.ERROR,
            error_details=error_details,
            final_context=final_context,
            performance_metrics=performance_metrics
        )
    
    def _create_response_chunk(
        self,
        response: LLMResponse,
        request_id: str,
        chunk_id: int
    ) -> StreamChunk:
        """
        Create stream chunk from LLMResponse for fallback streaming.
        
        Args:
            response: LLM response
            request_id: Request ID
            chunk_id: Chunk ID
            
        Returns:
            Stream chunk
        """
        if response.error_details:
            return StreamChunk(
                chunk_id=chunk_id,
                request_id=request_id,
                finish_reason=FinishReason.ERROR,
                provider_specific_data={"error": response.error_details.model_dump()}
            )
        
        delta_text = None
        delta_content_items = None
        
        if isinstance(response.generated_content, str):
            delta_text = response.generated_content
        elif isinstance(response.generated_content, list):
            delta_content_items = response.generated_content
        
        return StreamChunk(
            chunk_id=chunk_id,
            request_id=request_id,
            delta_text=delta_text,
            delta_content_items=delta_content_items,
            delta_tool_calls=response.tool_use_requests,
            finish_reason=response.finish_reason,
            usage_update=response.usage,
            provider_specific_data=response.mcp_metadata.model_dump() if response.mcp_metadata else None
        )
    
    def _get_request_tags(self, request: LLMRequest) -> Set[str]:
        """
        Get session tags for the request.
        
        Args:
            request: LLM request
            
        Returns:
            Set of tags
        """
        tags = set(self.session_tags)
        
        # Add model-specific tag
        model_id = request.config.model_identifier
        if model_id:
            tags.add(f"model:{model_id}")
        
        # Add priority tag based on size
        token_estimate = 0
        
        # Estimate prompt tokens
        if isinstance(request.prompt_content, str):
            token_estimate += len(request.prompt_content.split()) // 3
        
        # Adjust for expected response size
        max_tokens = request.config.max_tokens or 1024
        token_estimate += max_tokens
        
        # Set priority tag based on size
        if token_estimate > 10000:
            tags.add("priority:high")
        elif token_estimate > 1000:
            tags.add("priority:normal")
        else:
            tags.add("priority:low")
        
        # Add stream tag if streaming
        if request.stream:
            tags.add("stream:true")
        
        # Add tools tag if tools are present
        if request.tools:
            tags.add("tools:true")
        
        return tags