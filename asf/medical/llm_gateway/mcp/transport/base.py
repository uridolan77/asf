"""
Base transport layer for MCP communication.

This module provides the base abstractions for different transport mechanisms:
- stdio (subprocess-based)
- gRPC
- HTTP/REST

Each transport implements a common interface allowing them to be swapped dynamically.
"""

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

import structlog

logger = structlog.get_logger("mcp_transport")
# Add import at the top
from asf.medical.llm_gateway.mcp.session_pool import EnhancedSessionPool, SessionPoolConfig, Session, SessionPriority


class BaseTransport(ABC):
    """
    Abstract base class for MCP transport implementations.
    
    A transport is responsible for establishing and managing the
    underlying communication channel with an MCP server process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transport with configuration.
        
        Args:
            config: Configuration for the transport
        """
        self.config = config
        self.logger = logger.bind(transport_type=self.__class__.__name__)

        session_pool_config = SessionPoolConfig(
            min_size=config.get("pool_min_size", 2),
            max_size=config.get("pool_max_size", 10),
            max_idle_time_seconds=config.get("pool_max_idle_time_seconds", 300),
            health_check_interval_seconds=config.get("health_check_interval_seconds", 60),
            warmup_sessions=config.get("warmup_sessions", True),
            create_on_demand=config.get("create_on_demand", True),
            adaptive_sizing=config.get("adaptive_sizing", True)
        )
        
        # Create session pool
        self.session_pool = EnhancedSessionPool(
            provider_id=self.transport_type,
            create_session_func=self._create_session,
            close_session_func=self._close_session,
            ping_session_func=self._ping_session if hasattr(self, '_ping_session') else None,
            config=session_pool_config,
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus        
        )
    
    @asynccontextmanager
    @abstractmethod
    async def connect(self) -> AsyncGenerator[Union[Any, Tuple[Any, Any]], None]:
        """
        Establish a connection to the MCP server.
        
        The connection method returns either:
        1. A tuple of (reader, writer) streams for stdio-like transports
        2. A client object for RPC-based transports
        
        Returns:
            An object or tuple representing the connection
        """
        yield None
    
    @property
    def transport_type(self) -> str:
        """
        Get the type of transport.
        
        Returns:
            String identifier for this transport type
        """
        return self.__class__.__name__.replace("Transport", "").lower()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the transport.
        
        Returns:
            Dict with health status information
        """
        try:
            async with self.connect() as _:
                return {
                    "status": "available",
                    "transport_type": self.transport_type,
                    "message": "Successfully connected"
                }
        except Exception as e:
            return {
                "status": "unavailable",
                "transport_type": self.transport_type,
                "message": f"Failed to connect: {str(e)}"
            }


class TransportError(Exception):
    """Base exception for transport-related errors."""
    
    def __init__(self, message: str, transport_type: str, original_error: Optional[Exception] = None):
        self.transport_type = transport_type
        self.original_error = original_error
        super().__init__(f"{transport_type} transport error: {message}")


class ConnectionError(TransportError):
    """Error establishing a connection."""
    pass


class CommunicationError(TransportError):
    """Error during communication over an established connection."""
    pass


class TimeoutError(TransportError):
    """Timeout during transport operation."""
    pass