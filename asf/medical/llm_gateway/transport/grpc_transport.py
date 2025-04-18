"""
gRPC transport for MCP.

This module provides a gRPC transport implementation for MCP,
with support for connection pooling, back-pressure, and buffer management.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple, Union
import grpc
from grpc.aio import Channel, ClientCallDetails, UnaryUnaryCall, UnaryStreamCall, StreamUnaryCall, StreamStreamCall

from asf.medical.llm_gateway.transport.base import Transport, TransportConfig, TransportResponse
from asf.medical.llm_gateway.transport.grpc_pool import GRPCConnectionPool, PooledChannel
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

logger = logging.getLogger(__name__)


class GRPCTransportConfig(TransportConfig):
    """Configuration for gRPC transport."""
    
    transport_type: str = "grpc"
    target: str
    options: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, Any]] = None
    compression: Optional[str] = None
    pool_min_size: int = 2
    pool_max_size: int = 10
    pool_max_idle_time_seconds: int = 300
    pool_health_check_interval_seconds: int = 60
    max_message_length: int = 4 * 1024 * 1024  # 4 MB
    enable_backpressure: bool = True
    buffer_size: int = 10
    timeout_seconds: float = 60.0


class GRPCTransport(Transport):
    """
    gRPC transport implementation.
    
    This class provides a gRPC transport implementation for MCP,
    with support for connection pooling, back-pressure, and buffer management.
    """
    
    def __init__(
        self,
        provider_id: str,
        config: GRPCTransportConfig,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the gRPC transport.
        
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
        
        # Create connection pool
        self.pool = GRPCConnectionPool(
            provider_id=provider_id,
            create_channel_func=self._create_channel,
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            max_idle_time_seconds=config.pool_max_idle_time_seconds,
            health_check_interval_seconds=config.pool_health_check_interval_seconds,
            metrics_service=metrics_service,
            prometheus_exporter=prometheus_exporter
        )
    
    def _create_channel(self) -> Channel:
        """
        Create a new gRPC channel.
        
        Returns:
            gRPC channel
        """
        options = []
        
        # Add options from config
        if self.config.options:
            for key, value in self.config.options.items():
                options.append((key, value))
        
        # Add max message length options
        options.extend([
            ('grpc.max_send_message_length', self.config.max_message_length),
            ('grpc.max_receive_message_length', self.config.max_message_length)
        ])
        
        # Create channel
        if self.config.credentials:
            # TODO: Implement credentials
            channel = grpc.aio.insecure_channel(self.config.target, options=options)
        else:
            channel = grpc.aio.insecure_channel(self.config.target, options=options)
        
        return channel
    
    async def start(self) -> None:
        """
        Start the transport.
        
        This initializes the connection pool.
        """
        await self.pool.start()
    
    async def stop(self) -> None:
        """
        Stop the transport.
        
        This closes all connections in the pool.
        """
        await self.pool.stop()
    
    async def send_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> TransportResponse:
        """
        Send a unary request.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Transport response
        """
        # Get timeout
        timeout = timeout or self.config.timeout_seconds
        
        # Get channel from pool
        pooled_channel = await self.pool.get_channel()
        
        # Prepare metadata
        grpc_metadata = []
        if metadata:
            for key, value in metadata.items():
                grpc_metadata.append((key, str(value)))
        
        # Send request
        start_time = time.time()
        try:
            async with pooled_channel as channel:
                # Get method callable
                method_callable = channel.unary_unary(
                    method,
                    request_serializer=None,  # Use default serializer
                    response_deserializer=None  # Use default deserializer
                )
                
                # Call method
                response = await method_callable(
                    request,
                    metadata=grpc_metadata,
                    timeout=timeout
                )
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Return response
                return TransportResponse(
                    data=response,
                    metadata={},
                    latency_ms=latency_ms
                )
        except grpc.RpcError as e:
            # Handle gRPC errors
            error_code = e.code()
            error_details = e.details()
            
            # Record error in metrics
            self.metrics_service.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=str(error_code),
                error_message=error_details
            )
            
            # Record error in Prometheus
            self.prometheus.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=str(error_code)
            )
            
            # Re-raise as transport error
            raise TransportError(
                code=str(error_code),
                message=error_details,
                details={"grpc_code": str(error_code)}
            )
        except Exception as e:
            # Handle other errors
            error_type = type(e).__name__
            error_message = str(e)
            
            # Record error in metrics
            self.metrics_service.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=error_type,
                error_message=error_message
            )
            
            # Record error in Prometheus
            self.prometheus.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=error_type
            )
            
            # Re-raise as transport error
            raise TransportError(
                code="UNKNOWN",
                message=error_message,
                details={"error_type": error_type}
            )
    
    async def send_streaming_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, str]] = None,
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
        # Get timeout
        timeout = timeout or self.config.timeout_seconds
        
        # Get channel from pool
        pooled_channel = await self.pool.get_channel()
        
        # Prepare metadata
        grpc_metadata = []
        if metadata:
            for key, value in metadata.items():
                grpc_metadata.append((key, str(value)))
        
        # Send request
        start_time = time.time()
        try:
            async with pooled_channel as channel:
                # Get method callable
                method_callable = channel.unary_stream(
                    method,
                    request_serializer=None,  # Use default serializer
                    response_deserializer=None  # Use default deserializer
                )
                
                # Call method
                response_iterator = await method_callable(
                    request,
                    metadata=grpc_metadata,
                    timeout=timeout
                )
                
                # Create buffer for back-pressure
                buffer = asyncio.Queue(maxsize=self.config.buffer_size if self.config.enable_backpressure else 0)
                
                # Start background task to fill buffer
                buffer_task = asyncio.create_task(self._fill_buffer(response_iterator, buffer))
                
                try:
                    # Yield responses from buffer
                    while True:
                        # Get next response or None if done
                        item = await buffer.get()
                        if item is None:
                            break
                        
                        # Calculate latency for this chunk
                        chunk_latency_ms = (time.time() - start_time) * 1000
                        
                        # Yield response
                        yield TransportResponse(
                            data=item,
                            metadata={},
                            latency_ms=chunk_latency_ms
                        )
                finally:
                    # Cancel buffer task if still running
                    if not buffer_task.done():
                        buffer_task.cancel()
                        try:
                            await buffer_task
                        except asyncio.CancelledError:
                            pass
        except grpc.RpcError as e:
            # Handle gRPC errors
            error_code = e.code()
            error_details = e.details()
            
            # Record error in metrics
            self.metrics_service.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=str(error_code),
                error_message=error_details
            )
            
            # Record error in Prometheus
            self.prometheus.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=str(error_code)
            )
            
            # Re-raise as transport error
            raise TransportError(
                code=str(error_code),
                message=error_details,
                details={"grpc_code": str(error_code)}
            )
        except Exception as e:
            # Handle other errors
            error_type = type(e).__name__
            error_message = str(e)
            
            # Record error in metrics
            self.metrics_service.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=error_type,
                error_message=error_message
            )
            
            # Record error in Prometheus
            self.prometheus.record_transport_error(
                provider_id=self.provider_id,
                transport_type="grpc",
                error_type=error_type
            )
            
            # Re-raise as transport error
            raise TransportError(
                code="UNKNOWN",
                message=error_message,
                details={"error_type": error_type}
            )
    
    async def _fill_buffer(
        self,
        response_iterator: AsyncIterator[Any],
        buffer: asyncio.Queue
    ) -> None:
        """
        Fill buffer with responses from iterator.
        
        Args:
            response_iterator: Response iterator
            buffer: Response buffer
        """
        try:
            async for response in response_iterator:
                await buffer.put(response)
            
            # Signal end of stream
            await buffer.put(None)
        except Exception as e:
            # Put error in buffer
            await buffer.put(None)
            
            # Log error
            logger.error(f"Error filling buffer: {str(e)}")
            
            # Re-raise
            raise
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        return self.pool.get_pool_stats()


class TransportError(Exception):
    """
    Transport error.
    
    This exception is raised when a transport error occurs.
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the transport error.
        
        Args:
            code: Error code
            message: Error message
            details: Error details
        """
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")
