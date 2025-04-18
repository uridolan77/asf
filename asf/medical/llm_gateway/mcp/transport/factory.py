"""
Factory for creating transport instances.

This module provides a factory pattern implementation to dynamically
create appropriate transport instances based on configuration.
"""

import importlib
from typing import Any, Dict, Type

import structlog

from asf.medical.llm_gateway.transport.base import BaseTransport
from asf.medical.llm_gateway.transport.stdio import StdioTransport

logger = structlog.get_logger("mcp_transport.factory")


class TransportFactory:
    """
    Factory for creating transport instances based on configuration.
    
    This factory supports built-in transports and dynamically loaded
    custom transport implementations.
    """
    
    # Registry of built-in transport types
    BUILT_IN_TRANSPORTS = {
        "stdio": "asf.medical.llm_gateway.transport.stdio.StdioTransport",
        "grpc": "asf.medical.llm_gateway.transport.grpc.GrpcTransport",
        "http": "asf.medical.llm_gateway.transport.http.HttpTransport",
    }
    
    def __init__(self):
        """Initialize the transport factory."""
        self.logger = logger
    
    def create_transport(self, transport_type: str, connection_config: Dict[str, Any]) -> BaseTransport:
        """
        Create a transport instance of the specified type.
        
        Args:
            transport_type: Type of transport (stdio, grpc, http, or custom)
            connection_config: Configuration for the transport
            
        Returns:
            Configured transport instance
            
        Raises:
            ValueError: If transport type is invalid or implementation not found
            ImportError: If transport implementation cannot be imported
        """
        transport_type = transport_type.lower()
        
        # Handle stdio as the default
        if transport_type == "default" or not transport_type:
            transport_type = "stdio"
        
        # Check if it's a built-in transport
        if transport_type in self.BUILT_IN_TRANSPORTS:
            transport_class = self._load_transport_class(
                self.BUILT_IN_TRANSPORTS[transport_type]
            )
        else:
            # Try to load a custom transport
            transport_class = self._load_custom_transport(transport_type, connection_config)
        
        # Create and return the transport instance
        transport = transport_class(connection_config)
        
        self.logger.info(
            "Created transport",
            transport_type=transport_type,
            transport_class=transport_class.__name__
        )
        
        return transport
    
    def _load_transport_class(self, class_path: str) -> Type[BaseTransport]:
        """
        Load a transport class by its fully qualified path.
        
        Args:
            class_path: Fully qualified class path (e.g., 'package.module.ClassName')
            
        Returns:
            Transport class
            
        Raises:
            ImportError: If module or class cannot be imported
            ValueError: If loaded class is not a BaseTransport subclass
        """
        try:
            # Split into module path and class name
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            transport_class = getattr(module, class_name)
            
            # Verify it's a BaseTransport subclass
            if not issubclass(transport_class, BaseTransport):
                raise ValueError(f"Class {class_path} is not a BaseTransport subclass")
            
            return transport_class
        
        except (ImportError, AttributeError) as e:
            self.logger.error(
                "Failed to load transport class",
                class_path=class_path,
                error=str(e),
                exc_info=True
            )
            raise ImportError(f"Failed to load transport class {class_path}: {str(e)}")
    
    def _load_custom_transport(self, transport_type: str, config: Dict[str, Any]) -> Type[BaseTransport]:
        """
        Load a custom transport implementation.
        
        Args:
            transport_type: Type identifier
            config: Transport configuration with optional 'class_path' key
            
        Returns:
            Transport class
            
        Raises:
            ValueError: If custom transport cannot be loaded
        """
        # Check if class path is specified in config
        class_path = config.get("class_path")
        
        if class_path:
            return self._load_transport_class(class_path)
        
        # Try standard naming convention
        standard_path = f"asf.medical.llm_gateway.transport.{transport_type}.{transport_type.capitalize()}Transport"
        
        try:
            return self._load_transport_class(standard_path)
        except ImportError:
            # One last attempt with more explicit naming
            try:
                return self._load_transport_class(f"asf.medical.llm_gateway.transport.{transport_type}")
            except ImportError:
                self.logger.error(
                    "Failed to load custom transport",
                    transport_type=transport_type
                )
                raise ValueError(
                    f"Could not find transport implementation for type '{transport_type}'. "
                    f"Please specify 'class_path' in the configuration."
                )