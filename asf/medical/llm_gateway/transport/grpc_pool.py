"""
gRPC Connection Pool implementation using the resource management layer.

This module provides a specialized resource pool for managing gRPC client connections
with features like connection reuse, health checking, and automatic reconnection.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable, Generic, TypeVar
import grpc
from grpc.aio import Channel, ClientInterceptor

from ..core.resource_manager import ResourcePool, ResourceAcquisitionContext, ResourceType
from ..core.models import ResourcePoolConfig, ResourceLimits
from ..core.errors import ResourceError, GrpcTransportError

logger = logging.getLogger(__name__)

# Generic type for gRPC stub classes
S = TypeVar('S')

class GrpcClientResource:
    """Wrapper for a gRPC channel and stub."""
    
    def __init__(
        self,
        channel: grpc.aio.Channel,
        stub: Any,
        target: str
    ):
        self.channel = channel
        self.stub = stub
        self.target = target
        self.created_at = time.time()
    
    async def close(self) -> None:
        """Close the channel."""
        await self.channel.close()


class GrpcClientPool(ResourcePool[GrpcClientResource]):
    """Resource pool for gRPC client connections."""
    
    def __init__(
        self,
        config: ResourcePoolConfig,
        target: str,
        stub_class: Any,
        options: Optional[Dict[str, Any]] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        interceptors: Optional[list[ClientInterceptor]] = None
    ):
        # Override resource type to ensure correct classification
        config.resource_type = ResourceType.GRPC_CONNECTION
        
        super().__init__(config)
        
        # gRPC client configuration
        self.target = target
        self.stub_class = stub_class
        self.options = options or {}
        self.credentials = credentials
        self.interceptors = interceptors or []
    
    async def create_resource(self) -> GrpcClientResource:
        """Create a new gRPC client and channel."""
        try:
            if self.credentials:
                # Secure channel
                channel = grpc.aio.secure_channel(
                    self.target,
                    self.credentials,
                    options=[(key, value) for key, value in self.options.items()],
                    interceptors=self.interceptors
                )
            else:
                # Insecure channel
                channel = grpc.aio.insecure_channel(
                    self.target,
                    options=[(key, value) for key, value in self.options.items()],
                    interceptors=self.interceptors
                )
            
            # Create stub from the channel
            stub = self.stub_class(channel)
            
            # Create and return the resource
            return GrpcClientResource(
                channel=channel,
                stub=stub,
                target=self.target
            )
        except Exception as e:
            raise ResourceError(
                f"Failed to create gRPC client for {self.target}: {str(e)}",
                resource_type=ResourceType.GRPC_CONNECTION.value,
                operation="create",
                cause=e
            )
    
    async def validate_resource(self, resource: GrpcClientResource) -> bool:
        """Check if gRPC client is still usable."""
        channel = resource.channel
        
        try:
            # Check channel connectivity state
            state = channel.get_state(try_to_connect=False)
            
            # READY, IDLE, or CONNECTING states are considered valid
            valid_states = [
                grpc.ChannelConnectivity.READY,
                grpc.ChannelConnectivity.IDLE,
                grpc.ChannelConnectivity.CONNECTING
            ]
            
            if state in valid_states:
                return True
            elif state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                # For transient failures, try to connect and check again
                channel.get_state(try_to_connect=True)
                await asyncio.sleep(0.1)  # Small delay to allow connection attempt
                new_state = channel.get_state(try_to_connect=False)
                return new_state in valid_states
            else:
                return False
        except Exception as e:
            logger.warning(f"Error validating gRPC channel: {str(e)}")
            return False
    
    async def cleanup_resource(self, resource: GrpcClientResource) -> None:
        """Close the gRPC channel properly."""
        try:
            await resource.close()
        except Exception as e:
            logger.warning(f"Error closing gRPC channel: {str(e)}")


class GrpcTransportManager:
    """Manager for gRPC transport operations."""
    
    def __init__(self):
        self.client_pools: Dict[str, GrpcClientPool] = {}
    
    async def get_client(
        self,
        target: str,
        stub_class: Any,
        pool_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        interceptors: Optional[list[ClientInterceptor]] = None,
        max_pool_size: int = 10,
        min_pool_size: int = 1
    ) -> ResourceAcquisitionContext[GrpcClientResource]:
        """Get a gRPC client from a pool, creating the pool if needed."""
        # Generate pool ID if not provided
        if not pool_id:
            sanitized_target = target.replace(":", "_").replace("/", "_")
            stub_name = stub_class.__name__
            pool_id = f"grpc_pool_{sanitized_target}_{stub_name}_{uuid.uuid4().hex[:8]}"
        
        # Create pool if it doesn't exist
        if pool_id not in self.client_pools:
            config = ResourcePoolConfig(
                resource_type=ResourceType.GRPC_CONNECTION,
                pool_id=pool_id,
                limits=ResourceLimits(
                    max_pool_size=max_pool_size,
                    min_pool_size=min_pool_size,
                    max_idle_time_seconds=600,  # 10 minutes
                    circuit_breaker_threshold=5,
                    acquisition_timeout_seconds=10.0
                ),
                enable_health_checks=True,
                health_check_interval_seconds=60
            )
            
            # Create gRPC client pool
            client_pool = GrpcClientPool(
                config=config,
                target=target,
                stub_class=stub_class,
                options=options,
                credentials=credentials,
                interceptors=interceptors
            )
            
            await client_pool.start()
            self.client_pools[pool_id] = client_pool
        
        # Get client from pool
        return await self.client_pools[pool_id].acquire()
    
    async def shutdown(self):
        """Shutdown all gRPC client pools."""
        for pool_id, pool in list(self.client_pools.items()):
            await pool.stop()
        
        self.client_pools.clear()


# Create a singleton instance
grpc_transport_manager = GrpcTransportManager()


async def initialize_grpc_transport():
    """Initialize the gRPC transport system."""
    pass


async def shutdown_grpc_transport():
    """Shutdown the gRPC transport system."""
    await grpc_transport_manager.shutdown()


class StubManager(Generic[S]):
    """Generic manager for gRPC stubs with connection pooling."""
    
    def __init__(
        self,
        target: str,
        stub_class: Callable[[grpc.aio.Channel], S],
        options: Optional[Dict[str, Any]] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        interceptors: Optional[list[ClientInterceptor]] = None
    ):
        self.target = target
        self.stub_class = stub_class
        self.options = options or {}
        self.credentials = credentials
        self.interceptors = interceptors or []
        self.pool_id = f"grpc_{self.stub_class.__name__}_{uuid.uuid4().hex[:8]}"
    
    async def get_stub(self) -> ResourceAcquisitionContext[GrpcClientResource]:
        """Get a stub from the connection pool."""
        return await grpc_transport_manager.get_client(
            target=self.target,
            stub_class=self.stub_class,
            pool_id=self.pool_id,
            options=self.options,
            credentials=self.credentials,
            interceptors=self.interceptors
        )
    
    async def __call__(
        self,
        method_name: str,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[list] = None
    ) -> Any:
        """Convenience method to make a direct RPC call to the service."""
        async with await self.get_stub() as client:
            try:
                method = getattr(client.stub, method_name)
                if timeout is not None:
                    return await method(request, timeout=timeout, metadata=metadata)
                else:
                    return await method(request, metadata=metadata)
            except grpc.RpcError as e:
                # Convert gRPC errors to our error types
                status_code = e.code()
                raise GrpcTransportError(
                    f"gRPC call to {method_name} failed: {e.details()}",
                    provider_id=None,
                    grpc_code=status_code.value[0],
                    cause=e
                )
            except Exception as e:
                raise GrpcTransportError(
                    f"Error during gRPC call to {method_name}: {str(e)}",
                    provider_id=None,
                    cause=e
                )
