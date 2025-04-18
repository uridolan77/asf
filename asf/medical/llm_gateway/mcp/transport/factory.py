"""
Enhanced transport factory implementation for LLM Gateway.

This module provides a factory for creating transport instances,
with support for a variety of transport types and dynamic loading.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

from asf.medical.llm_gateway.mcp.transport.base import BaseTransport
from asf.medical.llm_gateway.mcp.transport.stdio import StdioTransport
from asf.medical.llm_gateway.mcp.transport.grpc import GRPCTransport
from asf.medical.llm_gateway.mcp.transport.http import HttpTransport
from asf.medical.llm_gateway.mcp.transport.websocket import WebSocketTransport
from asf.medical.llm_gateway.mcp.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

logger = logging.getLogger(__name__)


class TransportFactoryError(Exception):
    """Exception raised for transport factory errors."""
    pass


class TransportFactory:
    """
    Enhanced factory for creating transport instances.
    
    This factory supports built-in transport types (stdio, gRPC, HTTP, WebSocket)
    and dynamically loaded custom transports using a flexible registration system.
    
    Features:
    - Lazy loading of transport implementations
    - Custom transport registration
    - Dynamic transport discovery from configuration
    - Transport-specific metrics and monitoring
    - Singleton management for shared transports
    """
    
    # Registry of built-in transport types to their class paths
    BUILT_IN_TRANSPORTS = {
        "stdio": "asf.medical.llm_gateway.transport.stdio.StdioTransport",
        "grpc": "asf.medical.llm_gateway.transport.grpc_transport.GRPCTransport",
        "http": "asf.medical.llm_gateway.transport.http.HttpTransport",
        "websocket": "asf.medical.llm_gateway.transport.websocket.WebSocketTransport",
    }
    
    # Direct mapping for fast lookups after first use
    _transport_classes = {
        "stdio": StdioTransport,
        "grpc": GRPCTransport,
        "http": HttpTransport,
        "websocket": WebSocketTransport,
    }
    
    def __init__(
        self,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize the transport factory.
        
        Args:
            metrics_service: Metrics service for transport monitoring
            prometheus_exporter: Prometheus exporter for transport metrics
        """
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        self._custom_transports = {}
        self._transport_instances = {}
        self.logger = logger
    
    def register_transport(
        self,
        transport_type: str,
        transport_class: Optional[Type[BaseTransport]] = None,
        class_path: Optional[str] = None
    ) -> None:
        """
        Register a custom transport type.
        
        You can register either by providing a transport class directly,
        or by specifying a class path for lazy loading.
        
        Args:
            transport_type: Transport type identifier (e.g., "custom_http")
            transport_class: Transport class (optional)
            class_path: Fully qualified class path (e.g., "package.module.ClassName")
            
        Raises:
            ValueError: If neither transport_class nor class_path is provided
        """
        if not transport_class and not class_path:
            raise ValueError("Either transport_class or class_path must be provided")
        
        transport_type = transport_type.lower()
        
        if transport_class:
            # Ensure it's a BaseTransport subclass
            if not issubclass(transport_class, BaseTransport):
                raise ValueError(f"Transport class must inherit from BaseTransport")
            
            # Register the class directly
            self._transport_classes[transport_type] = transport_class
            self.logger.info(f"Registered transport type '{transport_type}' with class {transport_class.__name__}")
        else:
            # Register the class path for lazy loading
            self._custom_transports[transport_type] = class_path
            self.logger.info(f"Registered transport type '{transport_type}' with class path {class_path}")
    
    def create_transport(
        self,
        transport_type: str,
        config: Dict[str, Any],
        provider_id: str,
        singleton: bool = False
    ) -> BaseTransport:
        """
        Create a transport instance of the specified type.
        
        Args:
            transport_type: Type of transport (stdio, grpc, http, websocket, or custom)
            config: Configuration for the transport
            provider_id: Provider ID using this transport
            singleton: Whether to reuse existing instances with the same config
            
        Returns:
            Configured transport instance
            
        Raises:
            TransportFactoryError: If transport creation fails
        """
        transport_type = transport_type.lower()
        
        # Handle default transport type
        if not transport_type or transport_type == "default":
            # Use config.transport_type if available, otherwise default to stdio
            transport_type = config.get("transport_type", "stdio").lower()
        
        # Check for singleton reuse
        if singleton:
            # Create a key based on transport type, provider ID, and config
            instance_key = f"{transport_type}:{provider_id}:{hash(frozenset(config.items()))}"
            
            # Return existing instance if available
            if instance_key in self._transport_instances:
                self.logger.debug(f"Reusing existing transport instance for {provider_id}")
                return self._transport_instances[instance_key]
        
        # Get transport class
        transport_class = self._get_transport_class(transport_type)
        
        # Create the transport instance
        try:
            # Add provider_id to the config
            config_with_id = config.copy()
            config_with_id["provider_id"] = provider_id
            
            # Create the transport instance
            transport = transport_class(
                config=config_with_id,
                metrics_service=self.metrics_service,
                prometheus_exporter=self.prometheus
            )
            
            self.logger.info(
                f"Created {transport_type} transport for provider {provider_id}"
            )
            
            # Store instance if singleton
            if singleton:
                self._transport_instances[instance_key] = transport
            
            return transport
        
        except Exception as e:
            self.logger.error(
                f"Failed to create {transport_type} transport",
                error=str(e),
                exc_info=True
            )
            
            raise TransportFactoryError(
                f"Failed to create {transport_type} transport: {str(e)}"
            ) from e
    
    def _get_transport_class(self, transport_type: str) -> Type[BaseTransport]:
        """
        Get the transport class for the specified type.
        
        This either returns an already loaded class or dynamically
        loads it based on the registered class path.
        
        Args:
            transport_type: Transport type identifier
            
        Returns:
            Transport class
            
        Raises:
            TransportFactoryError: If transport type is not registered or class not found
        """
        # Check if already loaded
        if transport_type in self._transport_classes:
            return self._transport_classes[transport_type]
        
        # Check built-in transports
        if transport_type in self.BUILT_IN_TRANSPORTS:
            class_path = self.BUILT_IN_TRANSPORTS[transport_type]
            try:
                transport_class = self._load_class(class_path)
                self._transport_classes[transport_type] = transport_class
                return transport_class
            except Exception as e:
                raise TransportFactoryError(
                    f"Failed to load built-in transport '{transport_type}': {str(e)}"
                ) from e
        
        # Check custom transports
        if transport_type in self._custom_transports:
            class_path = self._custom_transports[transport_type]
            try:
                transport_class = self._load_class(class_path)
                self._transport_classes[transport_type] = transport_class
                return transport_class
            except Exception as e:
                raise TransportFactoryError(
                    f"Failed to load custom transport '{transport_type}': {str(e)}"
                ) from e
        
        # Try standard naming convention
        standard_path = f"asf.medical.llm_gateway.transport.{transport_type}.{transport_type.capitalize()}Transport"
        try:
            transport_class = self._load_class(standard_path)
            self._transport_classes[transport_type] = transport_class
            return transport_class
        except Exception:
            # Final attempt - check if class_path is in config
            pass
        
        # Not found
        available_types = list(self.BUILT_IN_TRANSPORTS.keys()) + list(self._custom_transports.keys())
        raise TransportFactoryError(
            f"Unknown transport type '{transport_type}'. Available types: {', '.join(available_types)}"
        )
    
    def _load_class(self, class_path: str) -> Type[BaseTransport]:
        """
        Load a class by its fully qualified path.
        
        Args:
            class_path: Fully qualified class path (e.g., "package.module.ClassName")
            
        Returns:
            Loaded class
            
        Raises:
            ImportError: If module or class cannot be loaded
            TypeError: If loaded class is not a BaseTransport subclass
        """
        try:
            # Split into module path and class name
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            transport_class = getattr(module, class_name)
            
            # Ensure it's a BaseTransport subclass
            if not issubclass(transport_class, BaseTransport):
                raise TypeError(f"Class {class_path} is not a BaseTransport subclass")
            
            return transport_class
        
        except (ImportError, AttributeError) as e:
            self.logger.error(
                f"Failed to load class {class_path}",
                error=str(e),
                exc_info=True
            )
            raise ImportError(f"Failed to load class {class_path}: {str(e)}")
    
    def get_transport_types(self) -> List[str]:
        """
        Get a list of available transport types.
        
        Returns:
            List of available transport types
        """
        return list(self.BUILT_IN_TRANSPORTS.keys()) + list(self._custom_transports.keys())
    
    async def cleanup(self) -> None:
        """
        Clean up all singleton transport instances.
        """
        cleanup_tasks = []
        for instance_key, transport in list(self._transport_instances.items()):
            if hasattr(transport, "stop") and callable(transport.stop):
                cleanup_tasks.append(transport.stop())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._transport_instances.clear()
        self.logger.info("Cleaned up all transport instances")

# Import asyncio here instead of at the top to avoid circular imports
import asyncio