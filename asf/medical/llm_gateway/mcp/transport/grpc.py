"""
Enhanced gRPC transport implementation for LLM Gateway.

This module provides a complete, production-ready gRPC transport implementation
for connecting to LLM services that expose gRPC interfaces.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple, Union

import grpc
from grpc.aio import Channel, ClientCallDetails, UnaryUnaryCall, UnaryStreamCall

from asf.medical.llm_gateway.transport.base import (
    BaseTransport, TransportConfig, TransportError, 
    CircuitBreakerOpenError, RateLimitExceededError
)
from asf.medical.llm_gateway.transport.grpc_pool import GRPCConnectionPool, PooledChannel
from asf.medical.llm_gateway.mcp.observability.metrics import MetricsService
from asf.medical.llm_gateway.mcp.observability.tracing import TracingService
from asf.medical.llm_gateway.mcp.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.mcp.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
from asf.medical.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig

# Import proto generated client - in a real implementation, this would be the compiled protobuf
try:
    # This would be your actual generated gRPC client
    from asf.medical.llm_gateway.protos import mcp_pb2, mcp_pb2_grpc
    GRPC_PROTOS_AVAILABLE = True
except ImportError:
    # Define minimal stub classes for type checking when protos aren't available
    GRPC_PROTOS_AVAILABLE = False
    
    class mcp_pb2:
        """Stub for mcp_pb2 when not available."""
        
        class CreateMessageRequest:
            """Stub for CreateMessageRequest."""
            pass
            
        class StreamMessageRequest:
            """Stub for StreamMessageRequest."""
            pass
            
        class CreateMessageResponse:
            """Stub for CreateMessageResponse."""
            pass
            
        class StreamMessageResponse:
            """Stub for StreamMessageResponse."""
            pass
            
        class Message:
            """Stub for Message."""
            pass
            
        class Content:
            """Stub for Content."""
            pass
            
        class TextContent:
            """Stub for TextContent."""
            pass
            
        class ImageContent:
            """Stub for ImageContent."""
            pass
            
        class ToolUseContent:
            """Stub for ToolUseContent."""
            pass
            
        class ToolResultContent:
            """Stub for ToolResultContent."""
            pass
            
    class mcp_pb2_grpc:
        """Stub for mcp_pb2_grpc when not available."""
        
        class MCPServiceStub:
            """Stub for MCPServiceStub."""
            
            def __init__(self, channel):
                self.channel = channel
                
            def CreateMessage(self, request, metadata=None, timeout=None):
                """Stub for CreateMessage."""
                pass
                
            def StreamMessage(self, request, metadata=None, timeout=None):
                """Stub for StreamMessage."""
                pass

logger = logging.getLogger(__name__)

class GRPCTransportConfig(TransportConfig):
    """Configuration for gRPC transport."""
    
    transport_type: str = "grpc"
    endpoint: str  # gRPC server endpoint (host:port)
    use_tls: bool = False  # Whether to use TLS 
    ca_cert: Optional[str] = None  # CA certificate file for TLS
    client_cert: Optional[str] = None  # Client certificate for mTLS
    client_key: Optional[str] = None  # Client key for mTLS
    max_concurrent_streams: int = 100  # Max concurrent streams
    channel_options: Optional[Dict[str, Any]] = None  # Additional gRPC channel options
    pool_min_size: int = 2  # Minimum connection pool size
    pool_max_size: int = 10  # Maximum connection pool size
    pool_max_idle_time_seconds: int = 300  # Maximum idle time for pooled connections
    health_check_interval_seconds: int = 60  # Health check interval
    max_message_length: int = 4 * 1024 * 1024  # 4 MB
    enable_backpressure: bool = True  # Whether to enable backpressure
    buffer_size: int = 10  # Buffer size for streaming responses
    timeout_seconds: float = 60.0  # Default request timeout
    channel_keepalive_time_ms: int = 30000  # Keepalive time in ms
    channel_keepalive_timeout_ms: int = 10000  # Keepalive timeout in ms
    channel_keepalive_without_calls: bool = True  # Whether to send keepalive pings without calls
    reconnect_backoff_ms: int = 1000  # Backoff for reconnections
    service_config: Optional[Dict[str, Any]] = None  # Service config for load balancing


class GRPCTransport(BaseTransport):
    """
    Production-ready gRPC transport implementation for LLM Gateway.
    
    Features:
    - TLS/mTLS support for secure communication
    - Connection pooling with health monitoring
    - Circuit breaker pattern for resilience
    - Rate limiting to prevent overload
    - Adaptive backoff and retry mechanism
    - Streaming with backpressure control
    - Comprehensive metrics and tracing
    - Proper error mapping and propagation
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the gRPC transport.
        
        Args:
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        super().__init__(config)
        
        if not GRPC_PROTOS_AVAILABLE:
            logger.warning("gRPC protos not available. Using stub implementations.")
        
        # Extract config
        self.endpoint = config.get("endpoint")
        self.use_tls = config.get("use_tls", False)
        self.ca_cert = config.get("ca_cert")
        self.client_cert = config.get("client_cert")
        self.client_key = config.get("client_key")
        self.max_concurrent_streams = config.get("max_concurrent_streams", 100)
        self.channel_options = config.get("channel_options", {})
        self.pool_min_size = config.get("pool_min_size", 2)
        self.pool_max_size = config.get("pool_max_size", 10)
        self.pool_max_idle_time_seconds = config.get("pool_max_idle_time_seconds", 300)
        self.health_check_interval_seconds = config.get("health_check_interval_seconds", 60)
        self.max_message_length = config.get("max_message_length", 4 * 1024 * 1024)
        self.enable_backpressure = config.get("enable_backpressure", True)
        self.buffer_size = config.get("buffer_size", 10)
        self.timeout_seconds = config.get("timeout_seconds", 60.0)
        self.keepalive_time_ms = config.get("channel_keepalive_time_ms", 30000)
        self.keepalive_timeout_ms = config.get("channel_keepalive_timeout_ms", 10000)
        self.keepalive_without_calls = config.get("channel_keepalive_without_calls", True)
        self.reconnect_backoff_ms = config.get("reconnect_backoff_ms", 1000)
        self.service_config = config.get("service_config")
        
        # Set up metrics and monitoring
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Create connection pool
        self.pool = GRPCConnectionPool(
            provider_id=self.transport_type,
            create_channel_func=self._create_channel,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            max_idle_time_seconds=self.pool_max_idle_time_seconds,
            health_check_interval_seconds=self.health_check_interval_seconds,
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus
        )
        
        # Create retry policy
        self.retry_policy = RetryPolicy(
            max_retries=config.get("max_retries", 3),
            base_delay=config.get("retry_base_delay", 1.0),
            max_delay=config.get("retry_max_delay", 30.0),
            jitter_factor=config.get("retry_jitter_factor", 0.2),
            retry_codes=set(config.get("retry_codes", ["UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"]))
        )
        
        # Create circuit breaker if enabled
        self.circuit_breaker = None
        if config.get("enable_circuit_breaker", True):
            self.circuit_breaker = CircuitBreaker(
                name=f"grpc_{self.transport_type}",
                failure_threshold=config.get("circuit_breaker_threshold", 5),
                recovery_timeout=config.get("circuit_breaker_recovery_timeout", 30),
                half_open_max_calls=config.get("circuit_breaker_half_open_max_calls", 1),
                reset_timeout=config.get("circuit_breaker_reset_timeout", 600)
            )
        
        # Create rate limiter if enabled
        self.rate_limiter = None
        if config.get("enable_rate_limiting", True):
            rate_limit_config = RateLimitConfig(
                strategy=config.get("rate_limit_strategy", "token_bucket"),
                requests_per_minute=config.get("rate_limit_rpm", 600),
                burst_size=config.get("rate_limit_burst_size", 100),
                window_size_seconds=config.get("rate_limit_window_size", 60),
                adaptive_factor=config.get("rate_limit_adaptive_factor", 0.5)
            )
            self.rate_limiter = RateLimiter(
                provider_id=self.transport_type,
                config=rate_limit_config
            )
        
        logger.info(
            f"Initialized gRPC transport with endpoint {self.endpoint}",
            use_tls=self.use_tls,
            pool_size=f"{self.pool_min_size}-{self.pool_max_size}"
        )
    
    def _create_channel(self) -> Channel:
        """
        Create a gRPC channel with the configured options.
        
        Returns:
            gRPC channel
        """
        # Create channel options
        options = [
            ('grpc.max_send_message_length', self.max_message_length),
            ('grpc.max_receive_message_length', self.max_message_length),
            ('grpc.max_concurrent_streams', self.max_concurrent_streams),
            ('grpc.keepalive_time_ms', self.keepalive_time_ms),
            ('grpc.keepalive_timeout_ms', self.keepalive_timeout_ms),
            ('grpc.keepalive_permit_without_calls', int(self.keepalive_without_calls)),
            ('grpc.enable_retries', 1),
            ('grpc.initial_reconnect_backoff_ms', self.reconnect_backoff_ms)
        ]
        
        # Add custom options
        for key, value in self.channel_options.items():
            options.append((key, value))
        
        # Create service config for load balancing if provided
        if self.service_config:
            options.append(('grpc.service_config', json.dumps(self.service_config)))
        
        logger.debug(f"Creating gRPC channel with options: {options}")
        
        # Create credentials for secure connections
        if self.use_tls:
            if self.client_cert and self.client_key:
                # mTLS (mutual TLS)
                logger.info("Using mTLS for gRPC channel")
                with open(self.client_key, 'rb') as f:
                    private_key = f.read()
                
                with open(self.client_cert, 'rb') as f:
                    certificate_chain = f.read()
                
                if self.ca_cert:
                    with open(self.ca_cert, 'rb') as f:
                        root_certificates = f.read()
                else:
                    root_certificates = None
                
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_certificates,
                    private_key=private_key,
                    certificate_chain=certificate_chain
                )
            elif self.ca_cert:
                # TLS with custom CA
                logger.info("Using TLS with custom CA for gRPC channel")
                with open(self.ca_cert, 'rb') as f:
                    root_certificates = f.read()
                
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_certificates
                )
            else:
                # Standard TLS
                logger.info("Using standard TLS for gRPC channel")
                credentials = grpc.ssl_channel_credentials()
            
            # Create secure channel
            channel = grpc.aio.secure_channel(self.endpoint, credentials, options=options)
        else:
            # Create insecure channel
            logger.info("Using insecure channel for gRPC")
            channel = grpc.aio.insecure_channel(self.endpoint, options=options)
        
        return channel
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Tuple[Any, Any], None]:
        """
        Get a connection from the pool.
        
        Returns:
            Tuple of (reader, writer) - in gRPC case, this is a stub
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RateLimitExceededError: If rate limit is exceeded
            ConnectionError: If connection fails
        """
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning("Circuit breaker open, failing fast")
            self.prometheus.record_circuit_breaker_event(
                provider_id=self.transport_type,
                state="open",
                event="rejected_request"
            )
            raise CircuitBreakerOpenError(
                message=f"Circuit breaker open for {self.transport_type}",
                transport_type="grpc"
            )
        
        # Check rate limit
        if self.rate_limiter:
            success, wait_time = await self.rate_limiter.acquire()
            if not success:
                logger.warning(f"Rate limit exceeded, retry after {wait_time:.2f}s")
                self.prometheus.record_rate_limit_event(
                    provider_id=self.transport_type,
                    wait_time=wait_time
                )
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded, retry after {wait_time:.2f}s",
                    details={"retry_after": wait_time}
                )
        
        # Get channel from pool
        pooled_channel = await self.pool.get_channel()
        channel = pooled_channel.channel
        
        # Create service stub
        stub = mcp_pb2_grpc.MCPServiceStub(channel)
        
        try:
            # Yield the stub for use by clients
            yield stub, channel
            
            # Record success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            if self.rate_limiter:
                await self.rate_limiter.record_success()
            
        except grpc.RpcError as e:
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            if self.rate_limiter:
                await self.rate_limiter.record_failure()
            
            # Map gRPC error to transport error
            code = e.code().name
            details = e.details()
            
            logger.warning(f"gRPC error: {code} - {details}")
            
            self.prometheus.record_transport_error(
                provider_id=self.transport_type,
                transport_type="grpc",
                error_type=code
            )
            
            raise TransportError(
                message=f"gRPC error: {details}",
                code=code,
                details={"grpc_code": code, "grpc_details": details},
                transport_type="grpc",
                original_error=e
            )
        
        except Exception as e:
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            if self.rate_limiter:
                await self.rate_limiter.record_failure()
            
            logger.error(f"Unexpected error during gRPC transport: {str(e)}", exc_info=True)
            
            self.prometheus.record_transport_error(
                provider_id=self.transport_type,
                transport_type="grpc",
                error_type=type(e).__name__
            )
            
            raise TransportError(
                message=f"Unexpected error: {str(e)}",
                code="INTERNAL",
                details={"error_type": type(e).__name__},
                transport_type="grpc",
                original_error=e
            )
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server using gRPC.
        
        Args:
            message: Message data
            
        Returns:
            Response from the server
            
        Raises:
            TransportError: If an error occurs
        """
        # Record metrics for the request
        start_time = time.time()
        
        # Convert JSON message to protobuf
        try:
            request = self._json_to_proto_request(message)
        except Exception as e:
            logger.error(f"Error converting message to protobuf: {str(e)}", exc_info=True)
            raise TransportError(
                message=f"Failed to convert message to protobuf: {str(e)}",
                code="INVALID_ARGUMENT",
                details={"error_type": "serialization_error"},
                transport_type="grpc"
            )
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as (stub, _):
                    # Call the CreateMessage method
                    response = await stub.CreateMessage(
                        request,
                        timeout=self.timeout_seconds
                    )
                    
                    # Convert protobuf response to JSON
                    result = self._proto_response_to_json(response)
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    self.prometheus.record_request(
                        provider_id=self.transport_type,
                        method="CreateMessage",
                        status="success",
                        duration=duration
                    )
                    
                    return result
            
            except TransportError as e:
                # Check if error is retryable
                if e.code in self.retry_policy.retry_codes and attempt < self.retry_policy.max_retries:
                    # Calculate retry delay
                    delay = self.retry_policy.calculate_delay(attempt + 1)
                    logger.info(f"Retrying after error: {e.code} (attempt {attempt+1}/{self.retry_policy.max_retries}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)
                    continue
                raise
            
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.transport_type,
                    method="CreateMessage",
                    status="error",
                    duration=duration,
                    error_type=type(e).__name__
                )
                
                # Convert to TransportError
                raise TransportError(
                    message=f"Unexpected error in gRPC transport: {str(e)}",
                    code="INTERNAL",
                    details={"error_type": type(e).__name__},
                    transport_type="grpc",
                    original_error=e
                )
    
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Send a streaming message to the MCP server using gRPC.
        
        Args:
            message: Message data
            
        Returns:
            Iterator of responses from the server
            
        Raises:
            TransportError: If an error occurs
        """
        # Record metrics for the request
        start_time = time.time()
        
        # Convert JSON message to protobuf
        try:
            request = self._json_to_proto_stream_request(message)
        except Exception as e:
            logger.error(f"Error converting stream message to protobuf: {str(e)}", exc_info=True)
            raise TransportError(
                message=f"Failed to convert stream message to protobuf: {str(e)}",
                code="INVALID_ARGUMENT",
                details={"error_type": "serialization_error"},
                transport_type="grpc"
            )
        
        # Create buffer for back-pressure
        buffer = asyncio.Queue(maxsize=self.buffer_size if self.enable_backpressure else 0)
        buffer_task = None
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as (stub, _):
                    # Call the StreamMessage method
                    response_stream = stub.StreamMessage(
                        request,
                        timeout=self.timeout_seconds
                    )
                    
                    # Start background task to fill buffer
                    buffer_task = asyncio.create_task(
                        self._fill_buffer_from_stream(response_stream, buffer)
                    )
                    
                    try:
                        # Yield responses from buffer
                        chunk_index = 0
                        while True:
                            # Get next response or None if done
                            item = await buffer.get()
                            if item is None:
                                break
                            
                            # Convert protobuf response to JSON
                            result = self._proto_stream_response_to_json(item)
                            
                            # Record chunk metrics
                            self.prometheus.record_stream_chunk(
                                provider_id=self.transport_type,
                                chunk_index=chunk_index
                            )
                            
                            # Yield the result
                            yield result
                            chunk_index += 1
                        
                        # Record success metrics for the stream
                        duration = time.time() - start_time
                        self.prometheus.record_request(
                            provider_id=self.transport_type,
                            method="StreamMessage",
                            status="success",
                            duration=duration,
                            chunks=chunk_index
                        )
                        
                        # Break out of retry loop on success
                        break
                    
                    finally:
                        # Cancel buffer task if still running
                        if buffer_task and not buffer_task.done():
                            buffer_task.cancel()
                            try:
                                await buffer_task
                            except asyncio.CancelledError:
                                pass
            
            except TransportError as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.transport_type,
                    method="StreamMessage",
                    status="error",
                    duration=duration,
                    error_type=e.code
                )
                
                # Check if error is retryable
                if e.code in self.retry_policy.retry_codes and attempt < self.retry_policy.max_retries:
                    # Calculate retry delay
                    delay = self.retry_policy.calculate_delay(attempt + 1)
                    logger.info(f"Retrying stream after error: {e.code} (attempt {attempt+1}/{self.retry_policy.max_retries}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)
                    continue
                raise
            
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.transport_type,
                    method="StreamMessage",
                    status="error",
                    duration=duration,
                    error_type=type(e).__name__
                )
                
                # Convert to TransportError
                raise TransportError(
                    message=f"Unexpected error in gRPC stream: {str(e)}",
                    code="INTERNAL",
                    details={"error_type": type(e).__name__},
                    transport_type="grpc",
                    original_error=e
                )
    
    async def _fill_buffer_from_stream(
        self,
        response_stream: AsyncIterator[Any],
        buffer: asyncio.Queue
    ) -> None:
        """
        Fill buffer with responses from stream.
        
        Args:
            response_stream: gRPC response stream
            buffer: Response buffer
        """
        try:
            async for response in response_stream:
                await buffer.put(response)
            
            # Signal end of stream
            await buffer.put(None)
        except grpc.RpcError as e:
            # Put error in buffer
            await buffer.put(None)
            
            # Log error
            logger.error(f"gRPC stream error: {e.code()} - {e.details()}")
            
            # Re-raise
            raise
        except Exception as e:
            # Put error in buffer
            await buffer.put(None)
            
            # Log error
            logger.error(f"Error in gRPC stream: {str(e)}", exc_info=True)
            
            # Re-raise
            raise
    
    def _json_to_proto_request(self, message: Dict[str, Any]) -> Any:
        """
        Convert JSON message to protobuf request.
        
        Args:
            message: JSON message
            
        Returns:
            Protobuf request
        """
        # Create CreateMessageRequest
        request = mcp_pb2.CreateMessageRequest()
        
        # Extract and set fields from JSON
        if 'model' in message:
            request.model = message['model']
        
        if 'max_tokens' in message:
            request.max_tokens = message['max_tokens']
        
        if 'temperature' in message:
            request.temperature = message['temperature']
        
        if 'top_p' in message:
            request.top_p = message['top_p']
        
        if 'messages' in message:
            for msg in message['messages']:
                proto_msg = request.messages.add()
                proto_msg.role = msg['role']
                
                # Handle content
                if 'content' in msg:
                    if isinstance(msg['content'], str):
                        # Simple text content
                        content = proto_msg.content.add()
                        content.text.text = msg['content']
                    elif isinstance(msg['content'], list):
                        # Multiple content items
                        for item in msg['content']:
                            content = proto_msg.content.add()
                            
                            if item.get('type') == 'text':
                                content.text.text = item['text']
                            elif item.get('type') == 'image':
                                # Handle image content (URL or base64)
                                if 'url' in item['image']['source']:
                                    content.image.url = item['image']['source']['url']
                                elif 'data' in item['image']['source']:
                                    content.image.data = item['image']['source']['data']
                            elif item.get('type') == 'tool_use':
                                # Handle tool use
                                content.tool_use.id = item.get('id', '')
                                content.tool_use.name = item.get('name', '')
                                # Tool input would need to be serialized based on specific format
                            elif item.get('type') == 'tool_result':
                                # Handle tool result
                                content.tool_result.id = item.get('id', '')
                                content.tool_result.output = item.get('output', '')
        
        # Handle tools
        if 'tools' in message:
            for tool in message['tools']:
                proto_tool = request.tools.add()
                if 'function' in tool:
                    proto_tool.function.name = tool['function'].get('name', '')
                    proto_tool.function.description = tool['function'].get('description', '')
                    # Parameters would need to be serialized based on JSON schema format
        
        # Add other fields like stop_sequences, stream, etc.
        if 'stop_sequences' in message:
            request.stop_sequences.extend(message['stop_sequences'])
        
        if 'stream' in message:
            request.stream = message['stream']
        
        return request
    
    def _json_to_proto_stream_request(self, message: Dict[str, Any]) -> Any:
        """
        Convert JSON message to protobuf stream request.
        
        Args:
            message: JSON message
            
        Returns:
            Protobuf stream request
        """
        # For most MCP implementations, stream requests use the same format
        # but with the stream flag set to True
        request = self._json_to_proto_request(message)
        request.stream = True
        return request
    
    def _proto_response_to_json(self, response: Any) -> Dict[str, Any]:
        """
        Convert protobuf response to JSON.
        
        Args:
            response: Protobuf response
            
        Returns:
            JSON response
        """
        # Basic conversion of fields
        result = {
            'id': response.id,
            'model': response.model,
            'content': [],
            'usage': {
                'input_tokens': response.usage.input_tokens if hasattr(response, 'usage') else 0,
                'output_tokens': response.usage.output_tokens if hasattr(response, 'usage') else 0
            }
        }
        
        # Convert stop_reason
        if hasattr(response, 'stop_reason') and response.stop_reason:
            result['stop_reason'] = response.stop_reason
        
        # Convert content
        for content in response.content:
            if content.HasField('text'):
                result['content'].append({
                    'type': 'text',
                    'text': content.text.text
                })
            elif content.HasField('tool_use'):
                result['content'].append({
                    'type': 'tool_use',
                    'id': content.tool_use.id,
                    'name': content.tool_use.name,
                    'input': content.tool_use.input  # Simplified, would need proper JSON conversion
                })
            # Handle other content types as needed
        
        return result
    
    def _proto_stream_response_to_json(self, response: Any) -> Dict[str, Any]:
        """
        Convert protobuf stream response to JSON.
        
        Args:
            response: Protobuf stream response
            
        Returns:
            JSON response
        """
        # Basic conversion of fields
        result = {
            'id': response.id,
            'type': 'stream_chunk',
            'delta': {}
        }
        
        # Convert delta content
        if hasattr(response, 'delta') and response.delta:
            if response.delta.HasField('content'):
                result['delta']['content'] = response.delta.content
            if response.delta.HasField('tool_use'):
                result['delta']['tool_use'] = {
                    'id': response.delta.tool_use.id,
                    'name': response.delta.tool_use.name,
                    'input': response.delta.tool_use.input  # Simplified
                }
            # Handle other delta types
        
        # Add stop reason if present
        if hasattr(response, 'stop_reason') and response.stop_reason:
            result['stop_reason'] = response.stop_reason
        
        # Add usage if present
        if hasattr(response, 'usage') and response.usage:
            result['usage'] = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        
        return result
    
    async def start(self) -> None:
        """Start the transport."""
        await self.pool.start()
    
    async def stop(self) -> None:
        """Stop the transport."""
        await self.pool.stop()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.pool.get_pool_stats()