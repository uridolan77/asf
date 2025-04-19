"""
gRPC connection pool for MCP transport.

This module provides a connection pool for gRPC transport,
with support for connection health monitoring and adaptive selection.
"""

import logging
import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import grpc
from grpc.aio import Channel, ClientCallDetails, UnaryUnaryCall, UnaryStreamCall, StreamUnaryCall, StreamStreamCall

from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

logger = logging.getLogger(__name__)


@dataclass
class ChannelStats:
    """Statistics for a gRPC channel."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    error_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Get the success rate for this channel."""
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count
    
    @property
    def avg_latency_ms(self) -> float:
        """Get the average latency for this channel."""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count
    
    def record_request(self, success: bool, latency_ms: float = 0.0) -> None:
        """
        Record a request.
        
        Args:
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
        """
        self.last_used_at = datetime.utcnow()
        self.request_count += 1
        
        if success:
            self.success_count += 1
            self.total_latency_ms += latency_ms
        else:
            self.error_count += 1


class PooledChannel:
    """
    A gRPC channel with connection pooling.
    
    This class wraps a gRPC channel and provides connection pooling,
    with support for connection health monitoring and adaptive selection.
    """
    
    def __init__(
        self,
        channel: Channel,
        channel_id: str,
        provider_id: str,
        stats: Optional[ChannelStats] = None
    ):
        """
        Initialize a pooled channel.
        
        Args:
            channel: gRPC channel
            channel_id: Channel ID
            provider_id: Provider ID
            stats: Channel statistics
        """
        self.channel = channel
        self.channel_id = channel_id
        self.provider_id = provider_id
        self.stats = stats or ChannelStats()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self) -> Channel:
        """
        Enter the context manager.
        
        Returns:
            gRPC channel
        """
        return self.channel
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Record request result
        if exc_type is not None:
            self.stats.record_request(False)
        else:
            self.stats.record_request(True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get channel statistics.
        
        Returns:
            Channel statistics
        """
        return {
            "id": self.channel_id,
            "provider_id": self.provider_id,
            "created_at": self.stats.created_at.isoformat(),
            "last_used_at": self.stats.last_used_at.isoformat(),
            "request_count": self.stats.request_count,
            "error_count": self.stats.error_count,
            "success_count": self.stats.success_count,
            "success_rate": self.stats.success_rate,
            "avg_latency_ms": self.stats.avg_latency_ms
        }


class ChannelInterceptor(grpc.aio.UnaryUnaryClientInterceptor,
                         grpc.aio.UnaryStreamClientInterceptor,
                         grpc.aio.StreamUnaryClientInterceptor,
                         grpc.aio.StreamStreamClientInterceptor):
    """
    gRPC channel interceptor for metrics and monitoring.
    
    This interceptor records metrics for gRPC requests,
    including latency, success rate, and error types.
    """
    
    def __init__(
        self,
        channel_id: str,
        provider_id: str,
        stats: ChannelStats,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the interceptor.
        
        Args:
            channel_id: Channel ID
            provider_id: Provider ID
            stats: Channel statistics
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.channel_id = channel_id
        self.provider_id = provider_id
        self.stats = stats
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
    
    async def _intercept(
        self,
        method: Callable,
        request: Any,
        call_details: ClientCallDetails,
        *args, **kwargs
    ) -> Any:
        """
        Intercept a gRPC request.
        
        Args:
            method: gRPC method
            request: Request data
            call_details: Call details
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            gRPC response
        """
        start_time = time.time()
        method_name = call_details.method.decode('utf-8') if hasattr(call_details, 'method') else 'unknown'
        
        try:
            response = await method(request, call_details, *args, **kwargs)
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.stats.record_request(True, latency_ms)
            
            # Record metrics
            self.metrics_service.record_grpc_request(
                provider_id=self.provider_id,
                method=method_name,
                success=True,
                latency_ms=latency_ms
            )
            
            # Record Prometheus metrics
            self.prometheus.record_grpc_request(
                provider_id=self.provider_id,
                channel_id=self.channel_id,
                method=method_name,
                status="success",
                duration_seconds=(time.time() - start_time)
            )
            
            return response
        except Exception as e:
            # Record error
            self.stats.record_request(False)
            
            # Record metrics
            error_type = type(e).__name__
            self.metrics_service.record_grpc_error(
                provider_id=self.provider_id,
                method=method_name,
                error_type=error_type
            )
            
            # Record Prometheus metrics
            self.prometheus.record_grpc_error(
                provider_id=self.provider_id,
                channel_id=self.channel_id,
                method=method_name,
                error_type=error_type
            )
            
            raise
    
    async def intercept_unary_unary(
        self,
        continuation: Callable[[ClientCallDetails, Any], Awaitable[UnaryUnaryCall]],
        client_call_details: ClientCallDetails,
        request: Any
    ) -> UnaryUnaryCall:
        """
        Intercept a unary-unary gRPC request.
        
        Args:
            continuation: Continuation function
            client_call_details: Call details
            request: Request data
            
        Returns:
            gRPC response
        """
        return await self._intercept(continuation, request, client_call_details)
    
    async def intercept_unary_stream(
        self,
        continuation: Callable[[ClientCallDetails, Any], Awaitable[UnaryStreamCall]],
        client_call_details: ClientCallDetails,
        request: Any
    ) -> UnaryStreamCall:
        """
        Intercept a unary-stream gRPC request.
        
        Args:
            continuation: Continuation function
            client_call_details: Call details
            request: Request data
            
        Returns:
            gRPC response
        """
        return await self._intercept(continuation, request, client_call_details)
    
    async def intercept_stream_unary(
        self,
        continuation: Callable[[ClientCallDetails, Any], Awaitable[StreamUnaryCall]],
        client_call_details: ClientCallDetails,
        request_iterator: Any
    ) -> StreamUnaryCall:
        """
        Intercept a stream-unary gRPC request.
        
        Args:
            continuation: Continuation function
            client_call_details: Call details
            request_iterator: Request iterator
            
        Returns:
            gRPC response
        """
        return await self._intercept(continuation, request_iterator, client_call_details)
    
    async def intercept_stream_stream(
        self,
        continuation: Callable[[ClientCallDetails, Any], Awaitable[StreamStreamCall]],
        client_call_details: ClientCallDetails,
        request_iterator: Any
    ) -> StreamStreamCall:
        """
        Intercept a stream-stream gRPC request.
        
        Args:
            continuation: Continuation function
            client_call_details: Call details
            request_iterator: Request iterator
            
        Returns:
            gRPC response
        """
        return await self._intercept(continuation, request_iterator, client_call_details)


class GRPCConnectionPool:
    """
    Connection pool for gRPC channels.
    
    This class provides a pool of gRPC channels,
    with support for connection health monitoring and adaptive selection.
    """
    
    def __init__(
        self,
        provider_id: str,
        create_channel_func: Callable[[], Channel],
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time_seconds: int = 300,
        health_check_interval_seconds: int = 60,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the connection pool.
        
        Args:
            provider_id: Provider ID
            create_channel_func: Function to create a new channel
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time_seconds: Maximum idle time in seconds
            health_check_interval_seconds: Health check interval in seconds
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        self.create_channel_func = create_channel_func
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time_seconds = max_idle_time_seconds
        self.health_check_interval_seconds = health_check_interval_seconds
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        self._channels: List[PooledChannel] = []
        self._lock = asyncio.Lock()
        self._health_check_task = None
        self._channel_counter = 0
    
    async def start(self) -> None:
        """
        Start the connection pool.
        
        This initializes the minimum number of channels and starts the health check task.
        """
        async with self._lock:
            # Initialize minimum number of channels
            for _ in range(self.min_size):
                await self._create_channel()
            
            # Start health check task
            if self._health_check_task is None:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Update Prometheus metrics
        self.prometheus.update_connection_pool(
            provider_id=self.provider_id,
            transport_type="grpc",
            pool_size=self.max_size,
            active_connections=len(self._channels)
        )
    
    async def stop(self) -> None:
        """
        Stop the connection pool.
        
        This closes all channels and stops the health check task.
        """
        # Cancel health check task
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        # Close all channels
        async with self._lock:
            for pooled_channel in self._channels:
                await pooled_channel.channel.close()
            self._channels = []
    
    async def get_channel(self) -> PooledChannel:
        """
        Get a channel from the pool.
        
        Returns:
            Pooled channel
        """
        async with self._lock:
            # If pool is empty, create a new channel
            if not self._channels:
                return await self._create_channel()
            
            # Select channel using adaptive selection
            return await self._select_channel()
    
    async def _create_channel(self) -> PooledChannel:
        """
        Create a new channel.
        
        Returns:
            Pooled channel
        """
        # Generate channel ID
        self._channel_counter += 1
        channel_id = f"{self.provider_id}-{self._channel_counter}"
        
        # Create channel statistics
        stats = ChannelStats()
        
        # Create interceptor
        interceptor = ChannelInterceptor(
            channel_id=channel_id,
            provider_id=self.provider_id,
            stats=stats,
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus
        )
        
        # Create channel with interceptor
        channel = self.create_channel_func()
        channel = grpc.aio.intercept_channel(channel, interceptor)
        
        # Create pooled channel
        pooled_channel = PooledChannel(
            channel=channel,
            channel_id=channel_id,
            provider_id=self.provider_id,
            stats=stats
        )
        
        # Add to pool
        self._channels.append(pooled_channel)
        
        # Update Prometheus metrics
        self.prometheus.update_connection_pool(
            provider_id=self.provider_id,
            transport_type="grpc",
            pool_size=self.max_size,
            active_connections=len(self._channels)
        )
        
        return pooled_channel
    
    async def _select_channel(self) -> PooledChannel:
        """
        Select a channel from the pool using adaptive selection.
        
        Returns:
            Pooled channel
        """
        # If only one channel, return it
        if len(self._channels) == 1:
            return self._channels[0]
        
        # Calculate weights based on success rate and latency
        weights = []
        for channel in self._channels:
            # Base weight
            weight = 1.0
            
            # Adjust weight based on success rate
            success_rate = channel.stats.success_rate
            weight *= max(0.1, success_rate)
            
            # Adjust weight based on latency
            avg_latency = channel.stats.avg_latency_ms
            if avg_latency > 0:
                # Normalize latency (lower is better)
                latency_factor = 1.0 / max(1.0, avg_latency / 100.0)
                weight *= latency_factor
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Equal weights if total is zero
            weights = [1.0 / len(self._channels) for _ in self._channels]
        
        # Select channel based on weights
        return random.choices(self._channels, weights=weights, k=1)[0]
    
    async def _health_check_loop(self) -> None:
        """
        Health check loop.
        
        This periodically checks the health of all channels and removes idle ones.
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval_seconds)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
    
    async def _perform_health_check(self) -> None:
        """
        Perform health check on all channels.
        
        This removes idle channels and ensures the minimum pool size.
        """
        now = datetime.utcnow()
        channels_to_remove = []
        
        async with self._lock:
            # Identify idle channels to remove
            for channel in self._channels:
                idle_time = (now - channel.stats.last_used_at).total_seconds()
                
                # Remove if idle for too long and we have more than min_size channels
                if idle_time > self.max_idle_time_seconds and len(self._channels) > self.min_size:
                    channels_to_remove.append(channel)
            
            # Remove idle channels
            for channel in channels_to_remove:
                self._channels.remove(channel)
                await channel.channel.close()
            
            # Ensure minimum pool size
            while len(self._channels) < self.min_size:
                await self._create_channel()
        
        # Update Prometheus metrics
        if channels_to_remove:
            self.prometheus.update_connection_pool(
                provider_id=self.provider_id,
                transport_type="grpc",
                pool_size=self.max_size,
                active_connections=len(self._channels)
            )
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Pool statistics
        """
        return {
            "provider_id": self.provider_id,
            "transport_type": "grpc",
            "pool_size": len(self._channels),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "max_idle_time_seconds": self.max_idle_time_seconds,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "connections": [channel.get_stats() for channel in self._channels]
        }
