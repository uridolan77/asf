"""
gRPC transport implementation for MCP communication.

This module provides a transport that connects to an MCP server
using the gRPC protocol for high-performance communication.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

import structlog

from asf.medical.llm_gateway.transport.base import BaseTransport, ConnectionError, CommunicationError

# Conditional imports to handle missing grpc package gracefully
try:
    import grpc
    import grpc.aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logger = structlog.get_logger("mcp_transport.grpc")


class GrpcTransport(BaseTransport):
    """
    Transport implementation that communicates with an MCP server
    via gRPC for high-performance, bi-directional streaming.
    
    Features:
    - Secure connections with TLS
    - Connection pooling
    - Backpressure handling
    - Authentication via metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize gRPC transport.
        
        Args:
            config: Configuration dictionary with keys:
                - endpoint: gRPC server endpoint (host:port)
                - use_tls: Whether to use TLS (default: False)
                - ca_cert: CA certificate file for TLS (optional)
                - client_cert: Client certificate file for mTLS (optional)
                - client_key: Client key file for mTLS (optional)
                - timeout_seconds: Connection timeout
                - max_concurrent_streams: Max concurrent streams (default: 100)
                - api_key_env_var: Environment variable name for API key
                - metadata: Additional metadata for requests
        """
        super().__init__(config)
        
        if not GRPC_AVAILABLE:
            raise ImportError("grpc.aio package is required for GrpcTransport")
        
        # Extract configuration
        self.endpoint = config.get("endpoint")
        if not self.endpoint:
            raise ValueError("gRPC endpoint is required")
        
        self.use_tls = config.get("use_tls", False)
        self.ca_cert = config.get("ca_cert")
        self.client_cert = config.get("client_cert")
        self.client_key = config.get("client_key")
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.max_concurrent_streams = config.get("max_concurrent_streams", 100)
        self.api_key_env_var = config.get("api_key_env_var")
        self.metadata = config.get("metadata", {})
        
        # Channel state
        self.channel: Optional[grpc.aio.Channel] = None
        self.last_connected = None
        self.reconnect_interval = config.get("reconnect_interval", 60)  # seconds
        
        self.logger = logger.bind(
            endpoint=self.endpoint,
            transport_type="grpc"
        )
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Any, None]:
        """
        Connect to the gRPC MCP server.
        
        Returns:
            gRPC channel object
        
        Raises:
            ConnectionError: If connection fails
        """
        channel = await self._get_channel()
        
        try:
            # Update connection timestamp
            self.last_connected = datetime.utcnow()
            
            # Yield the channel to the caller
            yield channel
        
        except grpc.RpcError as e:
            # Handle gRPC-specific errors
            self.logger.error(
                "gRPC communication error",
                code=e.code() if hasattr(e, 'code') else "unknown",
                details=e.details() if hasattr(e, 'details') else str(e)
            )
            
            # Force channel recreation on next connect
            await self._close_channel()
            
            # Map to our exception hierarchy
            if self._is_connection_error(e):
                raise ConnectionError(
                    f"gRPC connection error: {str(e)}",
                    transport_type="grpc",
                    original_error=e
                )
            else:
                raise CommunicationError(
                    f"gRPC communication error: {str(e)}",
                    transport_type="grpc",
                    original_error=e
                )
        
        except Exception as e:
            # Handle other exceptions
            self.logger.error(
                "Unexpected error in gRPC transport",
                error=str(e),
                exc_info=True
            )
            
            # Force channel recreation on next connect
            await self._close_channel()
            
            raise ConnectionError(
                f"Unexpected gRPC error: {str(e)}",
                transport_type="grpc",
                original_error=e
            )
    
    async def _get_channel(self) -> grpc.aio.Channel:
        """
        Get or create a gRPC channel.
        
        Returns:
            Active gRPC channel
        """
        # Check if we need to reconnect
        if self.channel is not None and self.last_connected is not None:
            age = (datetime.utcnow() - self.last_connected).total_seconds()
            if age > self.reconnect_interval:
                self.logger.info(
                    "Channel exceeded max age, reconnecting",
                    age_seconds=age,
                    reconnect_interval=self.reconnect_interval
                )
                await self._close_channel()
        
        # Create new channel if needed
        if self.channel is None:
            self.logger.info(
                "Creating new gRPC channel",
                endpoint=self.endpoint,
                use_tls=self.use_tls
            )
            
            # Create channel options
            options = [
                ('grpc.max_send_message_length', -1),  # Unlimited
                ('grpc.max_receive_message_length', -1),  # Unlimited
                ('grpc.max_concurrent_streams', self.max_concurrent_streams),
            ]
            
            # Set up credentials
            if self.use_tls:
                credentials = await self._create_credentials()
                self.channel = grpc.aio.secure_channel(
                    self.endpoint,
                    credentials,
                    options=options
                )
            else:
                self.channel = grpc.aio.insecure_channel(
                    self.endpoint,
                    options=options
                )
            
            # Wait for channel to be ready
            try:
                await asyncio.wait_for(
                    self._wait_for_channel_ready(),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                await self._close_channel()
                raise ConnectionError(
                    f"Timeout waiting for gRPC channel to be ready after {self.timeout_seconds}s",
                    transport_type="grpc"
                )
            except Exception as e:
                await self._close_channel()
                raise ConnectionError(
                    f"Error connecting to gRPC endpoint: {str(e)}",
                    transport_type="grpc",
                    original_error=e
                )
        
        return self.channel
    
    async def _wait_for_channel_ready(self) -> None:
        """Wait for the channel to be ready."""
        if self.channel is None:
            return
        
        conn_timeout = grpc.aio.ChannelConnectivity.TRANSIENT_FAILURE
        
        while True:
            state = self.channel.get_state()
            
            if state == grpc.ChannelConnectivity.READY:
                self.logger.info("gRPC channel is ready")
                return
            
            if state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                self.logger.warning("gRPC channel in TRANSIENT_FAILURE state")
            
            if state == grpc.ChannelConnectivity.SHUTDOWN:
                raise ConnectionError(
                    "gRPC channel is in SHUTDOWN state",
                    transport_type="grpc"
                )
            
            # Wait for state change with timeout
            await asyncio.wait_for(
                self.channel.wait_for_state_change(state),
                timeout=5.0
            )
    
    async def _create_credentials(self) -> grpc.ChannelCredentials:
        """
        Create gRPC credentials for secure communication.
        
        Returns:
            grpc.ChannelCredentials object
        """
        # Handle TLS/mTLS credentials
        if self.client_cert and self.client_key:
            # mTLS (mutual TLS)
            with open(self.client_key, 'rb') as f:
                private_key = f.read()
            
            with open(self.client_cert, 'rb') as f:
                certificate_chain = f.read()
            
            if self.ca_cert:
                with open(self.ca_cert, 'rb') as f:
                    root_certificates = f.read()
            else:
                root_certificates = None
            
            return grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain
            )
        
        elif self.ca_cert:
            # TLS with custom CA
            with open(self.ca_cert, 'rb') as f:
                root_certificates = f.read()
            
            return grpc.ssl_channel_credentials(
                root_certificates=root_certificates
            )
        
        else:
            # Standard TLS using system CA store
            return grpc.ssl_channel_credentials()
    
    async def _close_channel(self) -> None:
        """Close the gRPC channel."""
        if self.channel is not None:
            try:
                await self.channel.close()
                self.logger.info("Closed gRPC channel")
            except Exception as e:
                self.logger.warning(
                    "Error closing gRPC channel",
                    error=str(e)
                )
            finally:
                self.channel = None
    
    def get_call_credentials(self) -> Optional[grpc.CallCredentials]:
        """
        Get call credentials for authentication.
        
        Returns:
            grpc.CallCredentials or None
        """
        # Collect auth from various sources
        metadata = self._get_auth_metadata()
        
        if not metadata:
            return None
        
        # Convert dict to list of tuples for gRPC
        metadata_list = [(key.lower(), value) for key, value in metadata.items()]
        
        async def auth_plugin(context, callback):
            callback(metadata_list, None)
        
        return grpc.metadata_call_credentials(auth_plugin)
    
    def _get_auth_metadata(self) -> Dict[str, str]:
        """
        Get authentication metadata for gRPC calls.
        
        Returns:
            Dict of metadata key-value pairs
        """
        import os
        
        auth_metadata = {}
        
        # Add API key if configured
        if self.api_key_env_var:
            api_key = os.environ.get(self.api_key_env_var)
            if api_key:
                auth_metadata["authorization"] = f"Bearer {api_key}"
                self.logger.debug(
                    "Using specific API key environment variable for auth",
                    env_var=self.api_key_env_var
                )
        
        # Fall back to generic key
        elif "MCP_SERVER_API_KEY" in os.environ:
            api_key = os.environ.get("MCP_SERVER_API_KEY")
            auth_metadata["authorization"] = f"Bearer {api_key}"
            self.logger.debug("Using generic MCP_SERVER_API_KEY for auth")
        
        # Add custom metadata from config
        auth_metadata.update(self.metadata)
        
        return auth_metadata
    
    def _is_connection_error(self, error: grpc.RpcError) -> bool:
        """
        Determine if a gRPC error is a connection error.
        
        Args:
            error: gRPC error
            
        Returns:
            True if error indicates connection issue
        """
        if not hasattr(error, 'code'):
            return False
        
        # gRPC connection-related error codes
        connection_codes = [
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.CANCELLED,
            grpc.StatusCode.UNKNOWN
        ]
        
        return error.code() in connection_codes
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the transport."""
        status = "unknown"
        message = "No connection attempt yet"
        channel_state = None
        
        if self.channel is not None:
            try:
                state = self.channel.get_state()
                channel_state = str(state).replace("ChannelConnectivity.", "")
                
                if state == grpc.ChannelConnectivity.READY:
                    status = "available"
                    message = "Channel is READY"
                elif state == grpc.ChannelConnectivity.CONNECTING:
                    status = "connecting"
                    message = "Channel is CONNECTING"
                elif state == grpc.ChannelConnectivity.IDLE:
                    status = "idle"
                    message = "Channel is IDLE"
                elif state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                    status = "degraded"
                    message = "Channel is in TRANSIENT_FAILURE"
                elif state == grpc.ChannelConnectivity.SHUTDOWN:
                    status = "unavailable"
                    message = "Channel is SHUTDOWN"
            except Exception as e:
                status = "error"
                message = f"Error checking channel state: {str(e)}"
        
        return {
            "status": status,
            "transport_type": "grpc",
            "message": message,
            "endpoint": self.endpoint,
            "channel_state": channel_state,
            "last_connected": self.last_connected.isoformat() if self.last_connected else None,
            "use_tls": self.use_tls
        }
    
    async def cleanup(self) -> None:
        """Clean up resources and close the channel."""
        await self._close_channel()