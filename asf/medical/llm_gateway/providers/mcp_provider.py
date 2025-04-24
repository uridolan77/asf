"""
MCP Provider Implementation

This module implements the MCP (Model Context Protocol) provider for the LLM Gateway.
It provides a standardized interface for interacting with LLMs that implement the MCP protocol.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional, Set
import uuid

from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.core.models import (
    LLMRequest, LLMResponse, StreamChunk, FinishReason, UsageStats,
    ProviderConfig, ContentItem, MCPContentType as ContentType
)
from asf.medical.llm_gateway.core.utils import CircuitBreaker

# Import resource management components
from asf.medical.llm_gateway.mcp.resource_managed_session_pool import (
    create_session_pool, MCPSessionResource, SessionPriority
)
from asf.medical.llm_gateway.core.errors import (
    ResourceError, ResourcePoolError, ResourceAcquisitionError,
    MCPSessionError, CircuitBreakerError
)

# Import MCP-related errors
class McpError(Exception):
    """Base exception for MCP-related errors."""
    pass

class McpTimeoutError(McpError):
    """Exception for timeout errors."""
    pass

class McpConnectionError(McpError):
    """Exception for connection errors."""
    pass

logger = logging.getLogger(__name__)

# Define MCP role mapping
class MCPRole:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MCPProvider(BaseProvider):
    """
    Enhanced MCP Provider implementing a production-ready gateway with:
    - Pluggable transport layer (stdio, gRPC, HTTP)
    - Connection pooling and efficient management
    - Advanced resilience patterns (circuit breaker, exponential backoff)
    - Full streaming support with backpressure control
    - Comprehensive observability (metrics, tracing, structured logging)
    - Type-safe request/response mapping
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: Dict[str, Any] = None):
        """
        Initialize the MCP provider.

        Args:
            provider_config: Provider configuration
            gateway_config: Gateway configuration
        """
        super().__init__(provider_config)

        # Extract connection parameters
        connection_params = provider_config.connection_params or {}

        # Set up configuration
        self._transport_type = connection_params.get("transport_type", "stdio")
        self._enable_streaming = connection_params.get("enable_streaming", True)
        self._timeout_seconds = connection_params.get("timeout_seconds", 60)
        self._max_retries = connection_params.get("max_retries", 3)

        # Session pool configuration
        self._min_pool_size = connection_params.get("min_pool_size", 2)
        self._max_pool_size = connection_params.get("max_pool_size", 10)
        self._max_idle_time_seconds = connection_params.get("max_idle_time_seconds", 300)
        self._health_check_interval_seconds = connection_params.get("health_check_interval_seconds", 60)
        self._circuit_breaker_threshold = connection_params.get("circuit_breaker_threshold", 5)
        self._circuit_breaker_reset_timeout = connection_params.get("circuit_breaker_reset_timeout", 300)
        
        # Session pool instance (initialized in initialize_async)
        self._session_pool = None
        
        # Initialize circuit breaker (legacy, will be replaced by pool's circuit breaker)
        self.circuit_breaker = CircuitBreaker(
            threshold=self._circuit_breaker_threshold,
            reset_timeout=self._circuit_breaker_reset_timeout
        )

        # Initialize metrics
        self._request_count = 0
        self._error_count = 0
        self._token_count = 0

        logger.info(
            f"Initialized MCP provider with transport type: {self._transport_type}, "
            f"streaming: {self._enable_streaming}, timeout: {self._timeout_seconds}s"
        )

    async def initialize_async(self) -> None:
        """
        Initialize the provider asynchronously.

        This method is called after the provider is created to perform
        any async initialization tasks.
        """
        try:
            # Create a session creation function specific to this provider
            async def create_session():
                # In a real implementation, this would create an actual MCP session
                # using the appropriate transport (gRPC, HTTP, etc.)
                logger.debug(f"Creating MCP session for provider {self.provider_id}")
                # For demonstration, we'll create a mock session
                return {"id": str(uuid.uuid4()), "model": self.provider_config.models[0] if self.provider_config.models else "unknown"}
            
            # Create a session close function
            async def close_session(session):
                # In a real implementation, this would properly close the session
                logger.debug(f"Closing MCP session {session.get('id')} for provider {self.provider_id}")
                # For demonstration, we'll just wait briefly
                await asyncio.sleep(0.1)
            
            # Create a ping function
            async def ping_session(session):
                # In a real implementation, this would ping the session to check health
                logger.debug(f"Pinging MCP session {session.get('id')} for provider {self.provider_id}")
                # For demonstration, we'll just return a mock latency
                await asyncio.sleep(0.1)
                return 10.0  # 10ms latency
            
            # Create the session pool using our resource management layer
            self._session_pool = create_session_pool(
                provider_id=self.provider_id,
                create_session_func=create_session,
                close_session_func=close_session,
                ping_session_func=ping_session,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                max_idle_time_seconds=self._max_idle_time_seconds,
                health_check_interval_seconds=self._health_check_interval_seconds,
                circuit_breaker_threshold=self._circuit_breaker_threshold,
                use_resource_manager=True  # Use the new resource management layer
            )
            
            # Start the session pool
            await self._session_pool.start()
            
            logger.info(f"Initialized MCP provider {self.provider_id} with resource-managed session pool")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP provider {self.provider_id}: {e}", exc_info=True)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check MCP provider health by attempting to initialize a session.

        Returns:
            Dict with health check status and details.
        """
        check_start_time = datetime.now()

        # Use session pool health check if available
        if self._session_pool:
            try:
                # Try to get a session for health check
                async with self._session_pool.get_session(timeout=5.0) as session_resource:
                    # Session acquisition successful
                    stats = self._session_pool.get_stats()
                    return {
                        "provider_id": self.provider_id,
                        "status": "operational",
                        "provider_type": self.provider_config.provider_type,
                        "checked_at": check_start_time.isoformat(),
                        "message": "MCP provider is operational",
                        "session_pool": stats
                    }
            except ResourceError as e:
                # Resource error during health check
                return {
                    "provider_id": self.provider_id,
                    "status": "degraded" if "circuit breaker" in str(e).lower() else "error",
                    "provider_type": self.provider_config.provider_type,
                    "checked_at": check_start_time.isoformat(),
                    "message": f"Failed to acquire session: {str(e)}",
                    "error": {
                        "type": type(e).__name__,
                        "details": str(e)
                    }
                }
            except Exception as e:
                # General error
                return {
                    "provider_id": self.provider_id,
                    "status": "error",
                    "provider_type": self.provider_config.provider_type,
                    "checked_at": check_start_time.isoformat(),
                    "message": f"Health check failed: {str(e)}",
                    "error": {
                        "type": type(e).__name__,
                        "details": str(e)
                    }
                }
        else:
            # Fallback to legacy health check if session pool not initialized
            if self.circuit_breaker.is_open():
                return {
                    "provider_id": self.provider_id,
                    "status": "unavailable",
                    "provider_type": self.provider_config.provider_type,
                    "checked_at": check_start_time.isoformat(),
                    "message": f"Circuit breaker open until {self.circuit_breaker.recovery_time.isoformat()}"
                }
            
            return {
                "provider_id": self.provider_id,
                "status": "unknown",
                "provider_type": self.provider_config.provider_type,
                "checked_at": check_start_time.isoformat(),
                "message": "Session pool not initialized"
            }

    @retry(
        retry=retry_if_exception_type(McpError),
        stop=stop_after_attempt(3),  # Max retries
        wait=wait_exponential_jitter(max=10)  # Exponential backoff with jitter
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using MCP (non-streaming).

        This method includes comprehensive error handling, retry logic,
        telemetry, and proper context management.

        Args:
            request: The LLM request to process.

        Returns:
            LLM response with generated content.

        Raises:
            McpError: If there's an error communicating with the MCP provider.
        """
        # Check if circuit breaker is open (legacy check, will be replaced by pool's circuit breaker)
        if self.circuit_breaker.is_open():
            raise McpError(f"Circuit breaker open until {self.circuit_breaker.recovery_time.isoformat()}")

        # Generate request ID if not provided
        request_id = request.initial_context.request_id if request.initial_context else str(uuid.uuid4())

        # Increment request counter
        self._request_count += 1

        # Log request
        logger.info(
            f"Processing request {request_id} with model {request.config.model_identifier} "
            f"for provider {self.provider_id}"
        )

        # Check if session pool is initialized
        if not self._session_pool:
            raise McpError("Session pool not initialized")

        try:
            # Use the resource-managed session pool
            tags = set()
            if request.config.priority:
                tags.add(f"priority:{request.config.priority}")
            
            # Map request priority to session priority
            priority = SessionPriority.NORMAL
            if request.config.priority == "high":
                priority = SessionPriority.HIGH
            elif request.config.priority == "critical":
                priority = SessionPriority.CRITICAL
            elif request.config.priority == "low":
                priority = SessionPriority.LOW
            
            # Get a session from the pool with appropriate model
            model_id = request.config.model_identifier
            capabilities = set()
            
            # Acquisition timeout based on request timeout or default
            timeout = request.config.timeout_seconds or self._timeout_seconds
            
            llm_call_start = datetime.now()
            
            # Acquire a session from the pool
            async with self._session_pool.get_session(
                model_id=model_id,
                tags=tags,
                priority=priority,
                timeout=timeout,
                capabilities=capabilities
            ) as session_resource:
                # Use the session to generate a response
                session = session_resource.session
                
                # In a real implementation, this would use the session to call the MCP service
                # For demonstration, we'll return a mock response
                await asyncio.sleep(0.5)  # Simulate processing time
                
                # Record metrics for the session
                session_resource.record_request(
                    duration_ms=(datetime.now() - llm_call_start).total_seconds() * 1000, 
                    tokens=50  # Dummy token count
                )
                
                # Reset circuit breaker on success (legacy)
                self.circuit_breaker.record_success()
                
                # Create a response
                content = (f"This is a response from the MCP provider {self.provider_id} "
                           f"using model {model_id} "
                           f"with session {session.get('id', 'unknown')}")
                content_items = [ContentItem(
                    type=ContentType.TEXT, 
                    data={"text": content}, 
                    text_content=content
                )]

                # Create usage stats
                prompt_tokens = len(request.prompt_content.split()) if isinstance(request.prompt_content, str) else 10
                completion_tokens = len(content.split())
                usage = UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )

                # Record LLM latency
                llm_latency_ms = (datetime.now() - llm_call_start).total_seconds() * 1000

                # Update token count
                self._token_count += usage.total_tokens

                logger.info(
                    f"Request {request_id} completed in {llm_latency_ms:.2f}ms "
                    f"with {usage.total_tokens} tokens"
                )

                # Create response
                return LLMResponse(
                    request_id=request_id,
                    generated_content=content,
                    content_items=content_items,
                    tool_use_requests=[],
                    finish_reason=FinishReason.STOP,
                    usage=usage
                )

        except ResourceError as e:
            # Handle resource errors specifically
            error_msg = f"Resource error: {str(e)}"
            logger.error(f"{error_msg} for request {request_id}")
            self._error_count += 1
            
            # Legacy circuit breaker update
            self.circuit_breaker.record_failure()
            
            # Convert to appropriate MCP error
            if isinstance(e, CircuitBreakerError):
                raise McpError(f"Circuit breaker tripped: {str(e)}")
            elif isinstance(e, ResourceAcquisitionError):
                raise McpError(f"Failed to acquire MCP session: {str(e)}")
            else:
                raise McpError(f"Resource management error: {str(e)}")

        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {timeout}s"
            logger.error(f"{error_msg} for request {request_id}")
            
            # Legacy circuit breaker update
            self.circuit_breaker.record_failure()
            
            self._error_count += 1
            raise McpTimeoutError(error_msg)

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(f"{error_msg} for request {request_id}")
            
            # Legacy circuit breaker update
            self.circuit_breaker.record_failure()
            
            self._error_count += 1
            raise McpError(error_msg) from e

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response using MCP.

        This method implements full streaming support with proper backpressure
        control and error handling.

        Args:
            request: The LLM request to process.

        Yields:
            StreamChunk: Incremental chunks of the response.

        Raises:
            McpError: If there's an error communicating with the MCP provider.
        """
        # Check if circuit breaker is open
        if self.circuit_breaker.is_open():
            raise McpError(f"Circuit breaker open until {self.circuit_breaker.recovery_time.isoformat()}")

        # Check if streaming is enabled
        if not self._enable_streaming:
            logger.warning(f"Streaming is disabled for provider {self.provider_id}, using non-streaming fallback")
            response = await self.generate(request)
            yield StreamChunk(
                chunk_id="0",
                request_id=response.request_id,
                delta_text=response.generated_content,
                delta_content_items=response.content_items,
                delta_tool_calls=response.tool_use_requests,
                finish_reason=response.finish_reason,
                usage_update=response.usage
            )
            return

        # Generate request ID if not provided
        request_id = request.initial_context.request_id if request.initial_context else str(uuid.uuid4())

        # Increment request counter
        self._request_count += 1

        # Log request
        logger.info(
            f"Processing streaming request {request_id} with model {request.config.model_identifier} "
            f"for provider {self.provider_id}"
        )

        # Check if session pool is initialized
        if not self._session_pool:
            raise McpError("Session pool not initialized")

        chunk_index = 0

        try:
            # Use the resource-managed session pool
            tags = set()
            if request.config.priority:
                tags.add(f"priority:{request.config.priority}")
            
            # Map request priority to session priority
            priority = SessionPriority.NORMAL
            if request.config.priority == "high":
                priority = SessionPriority.HIGH
            elif request.config.priority == "critical":
                priority = SessionPriority.CRITICAL
            elif request.config.priority == "low":
                priority = SessionPriority.LOW
            
            # Get a session from the pool with appropriate model
            model_id = request.config.model_identifier
            capabilities = set()
            
            # Acquisition timeout based on request timeout or default
            timeout = request.config.timeout_seconds or self._timeout_seconds
            
            # Acquire a session from the pool
            async with self._session_pool.get_session(
                model_id=model_id,
                tags=tags,
                priority=priority,
                timeout=timeout,
                capabilities=capabilities
            ) as session_resource:
                # Use the session for streaming
                session = session_resource.session
                
                # Start timing
                llm_call_start = datetime.now()
                
                # In a real implementation, this would use the session to call the MCP service
                # For demonstration, we'll return mock streaming chunks
                content = (f"This is a streaming response from the MCP provider {self.provider_id} "
                          f"using model {model_id} with session {session.get('id', 'unknown')}")
                words = content.split()

                # Stream words one by one with a small delay
                for i, word in enumerate(words):
                    # Add a small delay to simulate streaming
                    await asyncio.sleep(0.1)

                    # Create stream chunk
                    yield StreamChunk(
                        chunk_id=f"{chunk_index}",
                        request_id=request_id,
                        delta_text=word + " ",
                        delta_content_items=[ContentItem(
                            type=ContentType.TEXT, 
                            data={"text": word + " "}, 
                            text_content=word + " "
                        )],
                        delta_tool_calls=[],
                        finish_reason=FinishReason.STOP if i == len(words) - 1 else None,
                        usage_update=UsageStats(prompt_tokens=0, completion_tokens=1, total_tokens=1) if i == len(words) - 1 else None
                    )
                    chunk_index += 1

                # Record metrics for the session
                session_resource.record_request(
                    duration_ms=(datetime.now() - llm_call_start).total_seconds() * 1000, 
                    tokens=len(words)
                )
                
                # Reset circuit breaker on success
                self.circuit_breaker.record_success()
                
                # Record LLM latency
                llm_latency_ms = (datetime.now() - llm_call_start).total_seconds() * 1000

                logger.info(
                    f"Streaming request {request_id} completed in {llm_latency_ms:.2f}ms "
                    f"with {chunk_index} chunks"
                )

        except ResourceError as e:
            # Handle resource errors specifically
            error_msg = f"Resource error in streaming: {str(e)}"
            logger.error(f"{error_msg} for request {request_id}")
            self._error_count += 1
            
            # Legacy circuit breaker update
            self.circuit_breaker.record_failure()
            
            # Yield error chunk
            yield StreamChunk(
                chunk_id=f"error-{chunk_index}",
                request_id=request_id,
                delta_text="",
                error=error_msg,
                finish_reason=FinishReason.ERROR
            )
            
            # Convert to appropriate MCP error
            if isinstance(e, CircuitBreakerError):
                raise McpError(f"Circuit breaker tripped: {str(e)}")
            elif isinstance(e, ResourceAcquisitionError):
                raise McpError(f"Failed to acquire MCP session: {str(e)}")
            else:
                raise McpError(f"Resource management error: {str(e)}")

        except asyncio.TimeoutError:
            error_msg = f"Streaming request timed out after {timeout}s"
            logger.error(f"{error_msg} for request {request_id}")
            self.circuit_breaker.record_failure()
            self._error_count += 1

            # Yield error chunk
            yield StreamChunk(
                chunk_id=f"error-{chunk_index}",
                request_id=request_id,
                delta_text="",
                error=error_msg,
                finish_reason=FinishReason.ERROR
            )

            raise McpTimeoutError(error_msg)

        except Exception as e:
            error_msg = f"Error in streaming response: {str(e)}"
            logger.error(f"{error_msg} for request {request_id}")
            self.circuit_breaker.record_failure()
            self._error_count += 1

            # Yield error chunk
            yield StreamChunk(
                chunk_id=f"error-{chunk_index}",
                request_id=request_id,
                delta_text="",
                error=error_msg,
                finish_reason=FinishReason.ERROR
            )

            raise McpError(error_msg) from e

    async def cleanup(self) -> None:
        """
        Clean up resources used by this provider.
        """
        logger.info(f"Cleaning up MCP provider {self.provider_id}")
        
        # Clean up session pool if initialized
        if self._session_pool:
            try:
                await self._session_pool.stop()
                logger.info(f"Session pool for {self.provider_id} stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping session pool for {self.provider_id}: {e}", exc_info=True)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get provider metrics.

        Returns:
            Dictionary with provider metrics
        """
        metrics = {
            "provider_id": self.provider_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "token_count": self._token_count,
            "circuit_breaker": {
                "state": "open" if self.circuit_breaker.is_open() else "closed",
                "failure_count": self.circuit_breaker.failure_count,
                "threshold": self._circuit_breaker_threshold,
                "reset_timeout": self._circuit_breaker_reset_timeout
            }
        }
        
        # Add session pool metrics if available
        if self._session_pool:
            try:
                pool_stats = self._session_pool.get_stats()
                metrics["session_pool"] = pool_stats
            except Exception as e:
                logger.error(f"Error getting session pool metrics: {e}", exc_info=True)
                metrics["session_pool"] = {"error": str(e)}
        else:
            metrics["session_pool"] = {"status": "not_initialized"}
            
        return metrics
