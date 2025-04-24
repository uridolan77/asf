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
from asf.medical.llm_gateway.transport.base import Transport as BaseTransport, TransportError
from asf.medical.llm_gateway.transport.factory import TransportFactory
from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
from asf.medical.llm_gateway.observability.tracing import TracingService
from asf.medical.llm_gateway.mcp.session_pool import SessionState, SessionPriority

# Import cache components
from asf.medical.llm_gateway.cache.cache_manager import get_cache_manager

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
cache_hit_counter = meter.create_counter(
    "mcp_cache_hits",
    description="Count of cache hits"
)
cache_miss_counter = meter.create_counter(
    "mcp_cache_misses",
    description="Count of cache misses"
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
    - Semantic caching for improved performance and cost efficiency
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

        # Initialize cache manager
        self.enable_caching = gateway_config.get("caching_enabled", True)
        if self.enable_caching:
            self.cache_ttl_seconds = gateway_config.get("cache_default_ttl_seconds", 3600)
            self.cache_manager = get_cache_manager(
                enable_caching=True,
                ttl_seconds=self.cache_ttl_seconds
            )
            logger.info(
                "Semantic caching enabled for provider",
                provider_id=self.provider_id,
                ttl_seconds=self.cache_ttl_seconds
            )
        else:
            self.cache_manager = get_cache_manager(enable_caching=False)
            logger.info(
                "Caching disabled for provider",
                provider_id=self.provider_id
            )

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
            # Check cache first if caching is enabled
            if self.enable_caching:
                span.set_attribute("cache.enabled", True)

                # Try to get cached response
                cached_response = await self.cache_manager.get_response(request)

                if cached_response:
                    # Record cache hit
                    duration_ms = (time.time() - start_time) * 1000

                    # Add cache info to span
                    span.set_attribute("cache.hit", True)
                    span.set_attribute("duration_ms", duration_ms)

                    # Record cache hit metric
                    cache_hit_counter.add(
                        1,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier
                        }
                    )

                    # Record latency metric
                    request_latency.record(
                        duration_ms,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier,
                            "success": "true",
                            "cached": "true"
                        }
                    )

                    logger.info(
                        "Cache hit for request",
                        provider_id=self.provider_id,
                        model=request.config.model_identifier,
                        request_id=context.request_id,
                        duration_ms=duration_ms
                    )

                    return cached_response
                else:
                    # Record cache miss metric
                    cache_miss_counter.add(
                        1,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier
                        }
                    )
                    span.set_attribute("cache.hit", False)
            else:
                span.set_attribute("cache.enabled", False)

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

                            # Store in cache if enabled and response is valid
                            if self.enable_caching:
                                await self.cache_manager.store_response(request, response)

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
            Stream chunks containing response content
        """
        # Record request metrics
        request_counter.add(1, {"provider": self.provider_id, "model": request.config.model_identifier, "streaming": "true"})

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
            # Check cache first if caching is enabled
            if self.enable_caching:
                span.set_attribute("cache.enabled", True)

                # Try to get cached response - using exact match only for streams to ensure quality
                cached_response = await self.cache_manager.get_response(request, check_exact_only=True)

                if cached_response:
                    # Record cache hit
                    duration_ms = (time.time() - start_time) * 1000

                    # Add cache info to span
                    span.set_attribute("cache.hit", True)
                    span.set_attribute("duration_ms", duration_ms)

                    # Record cache hit metric
                    cache_hit_counter.add(
                        1,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier,
                            "streaming": "true"
                        }
                    )

                    # Record latency metric
                    request_latency.record(
                        duration_ms,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier,
                            "streaming": "true",
                            "success": "true",
                            "cached": "true"
                        }
                    )

                    logger.info(
                        "Cache hit for streaming request",
                        provider_id=self.provider_id,
                        model=request.config.model_identifier,
                        request_id=context.request_id,
                        duration_ms=duration_ms
                    )

                    # Convert cached response to stream chunks
                    # For streaming, we'll simulate progressive generation by yielding content in chunks
                    yield await self._convert_cached_response_to_stream(
                        cached_response, context.request_id, simulate_typing=True
                    )
                    return
                else:
                    # Record cache miss metric
                    cache_miss_counter.add(
                        1,
                        {
                            "provider": self.provider_id,
                            "model": request.config.model_identifier,
                            "streaming": "true"
                        }
                    )
                    span.set_attribute("cache.hit", False)
            else:
                span.set_attribute("cache.enabled", False)

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
                            # For caching streamed responses, we'll collect the full response
                            if self.enable_caching:
                                # Initialize collection of content
                                full_content = []
                                finish_reason = None
                                usage = None
                                tool_calls = []

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

                                    # Collect content for caching
                                    if stream_chunk.delta_text:
                                        full_content.append(stream_chunk.delta_text)

                                    # Track finish reason
                                    if stream_chunk.finish_reason:
                                        finish_reason = stream_chunk.finish_reason

                                    # Track usage
                                    if stream_chunk.usage_update:
                                        usage = stream_chunk.usage_update

                                    # Track tool calls
                                    if stream_chunk.delta_tool_calls:
                                        tool_calls.extend(stream_chunk.delta_tool_calls)

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

                                # Create a complete response for caching
                                from asf.medical.llm_gateway.core.models import ContentItem

                                # Only cache if we have content
                                if full_content:
                                    # Create a complete response object
                                    complete_response = LLMResponse(
                                        request_id=context.request_id,
                                        generated_content=[ContentItem.from_text(''.join(full_content))],
                                        finish_reason=finish_reason,
                                        tool_use_requests=tool_calls if tool_calls else None,
                                        usage=usage,
                                        final_context=copy.deepcopy(request.initial_context),
                                        performance_metrics=PerformanceMetrics(
                                            total_duration_ms=total_duration_ms,
                                            llm_latency_ms=total_duration_ms - 5  # Rough estimate
                                        )
                                    )

                                    # Store in cache
                                    await self.cache_manager.store_response(request, complete_response)
                            else:
                                # Regular streaming without caching
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

    async def _convert_cached_response_to_stream(
        self,
        cached_response: LLMResponse,
        request_id: str,
        simulate_typing: bool = True,
        words_per_chunk: int = 4,
        delay_ms: int = 30
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Convert a cached response to a stream of chunks.

        Args:
            cached_response: The cached LLM response
            request_id: The request ID for the stream chunks
            simulate_typing: Whether to simulate typing by breaking up content and delaying
            words_per_chunk: Number of words per chunk when simulating typing
            delay_ms: Delay between chunks in milliseconds when simulating typing

        Yields:
            Stream chunks containing response content
        """
        if not cached_response.generated_content:
            # Handle empty response case
            yield StreamChunk(
                chunk_id=0,
                request_id=request_id,
                delta_text="",
                finish_reason=cached_response.finish_reason or FinishReason.STOP,
                provider_specific_data={"cached": True}
            )
            return

        # Get full text from content items
        full_text = ""
        for item in cached_response.generated_content:
            if hasattr(item, 'text'):
                full_text += item.text
            elif isinstance(item, dict) and 'text' in item:
                full_text += item['text']
            elif isinstance(item, str):
                full_text += item

        # Get tool calls if any
        tool_calls = cached_response.tool_use_requests

        if simulate_typing:
            # Break text into words
            words = full_text.split()
            chunks = []

            # Group words into chunks
            for i in range(0, len(words), words_per_chunk):
                end = min(i + words_per_chunk, len(words))
                chunk_text = ' '.join(words[i:end])
                # Add space to end of chunk if not the last chunk and not ending with punctuation
                if end < len(words) and not chunk_text.endswith(('.', ',', '!', '?', ':', ';', '-')):
                    chunk_text += ' '
                chunks.append(chunk_text)

            # Yield chunks with delay
            for i, chunk_text in enumerate(chunks):
                # Last chunk gets finish reason
                if i == len(chunks) - 1:
                    yield StreamChunk(
                        chunk_id=i,
                        request_id=request_id,
                        delta_text=chunk_text,
                        finish_reason=cached_response.finish_reason,
                        usage_update=cached_response.usage,
                        delta_tool_calls=tool_calls if i == len(chunks) - 1 else None,
                        provider_specific_data={"cached": True}
                    )
                else:
                    yield StreamChunk(
                        chunk_id=i,
                        request_id=request_id,
                        delta_text=chunk_text,
                        provider_specific_data={"cached": True}
                    )

                    # Add realistic delay between chunks
                    if i < len(chunks) - 1:
                        await asyncio.sleep(delay_ms / 1000)
        else:
            # Just yield the full response as a single chunk
            yield StreamChunk(
                chunk_id=0,
                request_id=request_id,
                delta_text=full_text,
                finish_reason=cached_response.finish_reason,
                usage_update=cached_response.usage,
                delta_tool_calls=tool_calls,
                provider_specific_data={"cached": True}
            )